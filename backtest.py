import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config
from history import StockDataDownloader
from strategy import TradingStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Trade:
    symbol: str
    entry_date: datetime
    entry_price: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    signal_type: str = ''
    quantity: int = 0

    @property
    def is_open(self) -> bool:
        return self.exit_date is None

    @property
    def return_pct(self) -> float:
        if self.exit_price is None or self.entry_price == 0:
            return 0.0
        if self.signal_type == 'oversold':
            return (self.exit_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - self.exit_price) / self.entry_price

    @property
    def profit_loss(self) -> float:
        if self.exit_price is None:
            return 0.0
        return self.return_pct * self.entry_price * self.quantity


@dataclass
class BacktestResults:
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_return: float = 0.0
    win_rate: float = 0.0
    avg_return_per_trade: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    trades: List[Trade] = None
    portfolio_values: pd.DataFrame = None

    def __post_init__(self):
        if self.trades is None:
            self.trades = []
        if self.portfolio_values is None:
            self.portfolio_values = pd.DataFrame()


class TradingBacktest:
    def __init__(self, strategy: TradingStrategy,
                 initial_capital: float = 100000,
                 position_size: float = 0.02,
                 max_positions: int = 20,
                 hold_period_days: int = 10,
                 stop_loss_pct: float = -0.05,
                 take_profit_pct: float = 0.10):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.max_positions = max_positions
        self.hold_period_days = hold_period_days
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        self.downloader = StockDataDownloader()

    def run_backtest(self, start_date: str,
                     end_date: str,
                     categories: List[str] = None,
                     signal_types: List[str] = None) -> BacktestResults:
        if categories is None:
            categories = ['america', 'china']
        if signal_types is None:
            signal_types = ['oversold', 'overbought']

        logger.info(f"Starting backtest from {start_date} to {end_date}")

        historical_signals = self._get_historical_signals(start_date, end_date, categories, signal_types)

        if historical_signals.empty:
            logger.warning("No historical signals found for backtest period")
            return BacktestResults()

        results = self._simulate_trading(historical_signals, end_date)
        self._calculate_performance_metrics(results)

        logger.info(f"Backtest completed. Total trades: {results.total_trades}")
        return results

    def _get_historical_signals(self, start_date: str,
                                end_date: str,
                                categories: List[str],
                                signal_types: List[str]) -> pd.DataFrame:
        all_signals = []

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        while start_dt <= end_dt:
            if start_dt.weekday() < 5:
                date_signals = self._generate_signals_for_date(categories, signal_types)
                if not date_signals.empty:
                    date_signals['signal_date'] = start_dt
                    all_signals.append(date_signals)

            start_dt += timedelta(days=1)

        if all_signals:
            return pd.concat(all_signals, ignore_index=True)
        else:
            return pd.DataFrame()

    def _generate_signals_for_date(self, categories: List[str],
                                   signal_types: List[str]) -> pd.DataFrame:
        signals = []

        for category in categories:
            for signal_type in signal_types:
                try:
                    strategy_data = self.strategy.load_strategy_data(category, signal_type, 'daily')
                    if not strategy_data.empty:
                        strategy_data['category'] = category
                        strategy_data['signal_type'] = signal_type
                        signals.append(strategy_data)

                except Exception as e:
                    logger.error(f"Error loading signals for {category} {signal_type}: {e}")

        if signals:
            return pd.concat(signals, ignore_index=True)
        else:
            return pd.DataFrame()

    def _simulate_trading(self, signals_df: pd.DataFrame,
                          end_date: str) -> BacktestResults:
        results = BacktestResults()
        results.trades = []

        cash = self.initial_capital
        open_trades = []
        portfolio_history = []

        signals_by_date = signals_df.groupby('signal_date')

        for signal_date, day_signals in signals_by_date:
            open_trades, cash = self._process_exits(open_trades, cash, signal_date)

            if len(open_trades) < self.max_positions:
                new_trades = self._process_entries(day_signals, cash, signal_date, len(open_trades))

                for trade in new_trades:
                    if cash >= trade.entry_price * trade.quantity:
                        cash -= trade.entry_price * trade.quantity
                        open_trades.append(trade)
                        results.trades.append(trade)

            positions_value = sum(
                trade.quantity * self._get_current_price(trade.symbol, signal_date)
                for trade in open_trades
            )
            current_portfolio_value = cash + positions_value

            portfolio_history.append({
                'date': signal_date,
                'portfolio_value': current_portfolio_value,
                'cash': cash,
                'positions_value': positions_value,
                'open_positions': len(open_trades)
            })

        final_date = pd.to_datetime(end_date)
        for trade in open_trades:
            trade.exit_date = final_date
            trade.exit_price = self._get_current_price(trade.symbol, final_date)

        results.portfolio_values = pd.DataFrame(portfolio_history)
        results.total_trades = len(results.trades)

        return results

    def _process_exits(self,
                       open_trades: List[Trade],
                       cash: float,
                       current_date: datetime) -> Tuple[List[Trade], float]:
        remaining_trades = []

        for trade in open_trades:
            should_exit = False
            exit_price = self._get_current_price(trade.symbol, current_date)

            if exit_price is None:
                remaining_trades.append(trade)
                continue

            days_held = (current_date - trade.entry_date).days
            if days_held >= self.hold_period_days:
                should_exit = True

            current_return = (exit_price - trade.entry_price) / trade.entry_price
            if trade.signal_type == 'overbought':
                current_return = -current_return

            if current_return <= self.stop_loss_pct or current_return >= self.take_profit_pct:
                should_exit = True

            if should_exit:
                trade.exit_date = current_date
                trade.exit_price = exit_price
                cash += trade.exit_price * trade.quantity
            else:
                remaining_trades.append(trade)

        return remaining_trades, cash

    def _process_entries(self, signals: pd.DataFrame,
                         available_cash: float,
                         signal_date: datetime,
                         current_positions: int) -> List[Trade]:
        new_trades = []
        max_new_positions = self.max_positions - current_positions

        if max_new_positions <= 0:
            return new_trades

        if not signals.empty:
            for _, signal in signals.head(max_new_positions).iterrows():
                entry_price = signal.get(config.CLOSE, signal.get('Close', 0))
                if entry_price <= 0:
                    continue

                position_value = available_cash * self.position_size
                quantity = max(1, int(position_value / entry_price))

                if quantity * entry_price <= available_cash:
                    trade = Trade(
                        symbol=signal[config.SYMBOL],
                        entry_date=signal_date,
                        entry_price=entry_price,
                        signal_type=signal['signal_type'],
                        quantity=quantity
                    )
                    new_trades.append(trade)
                    available_cash -= quantity * entry_price

        return new_trades

    @staticmethod
    def _get_current_price(symbol, current_date) -> Optional[float]:
        try:
            base_price = 100
            volatility = 0.02
            random_return = np.random.normal(0, volatility)
            return base_price * (1 + random_return)

        except Exception:
            return None

    @staticmethod
    def _calculate_performance_metrics(results: BacktestResults) -> None:
        if not results.trades:
            return

        completed_trades = [t for t in results.trades if not t.is_open]

        if completed_trades:
            returns = [t.return_pct for t in completed_trades]

            results.winning_trades = len([r for r in returns if r > 0])
            results.losing_trades = len([r for r in returns if r < 0])
            results.win_rate = results.winning_trades / len(completed_trades)
            results.avg_return_per_trade = np.mean(returns)
            results.total_return = sum(returns)

        if not results.portfolio_values.empty:
            portfolio_returns = results.portfolio_values['portfolio_value'].pct_change().dropna()

            if len(portfolio_returns) > 0:
                if portfolio_returns.std() > 0:
                    results.sharpe_ratio = (portfolio_returns.mean() * 252 / portfolio_returns.std() / np.sqrt(252))

                portfolio_values = results.portfolio_values['portfolio_value']
                rolling_max = portfolio_values.expanding().max()
                drawdowns = (portfolio_values - rolling_max) / rolling_max
                results.max_drawdown = drawdowns.min()

    @staticmethod
    def generate_report(results: BacktestResults) -> str:
        report = []
        report.append("=" * 60)
        report.append("TRADING STRATEGY BACKTEST REPORT")
        report.append("=" * 60)
        report.append("")

        report.append("SUMMARY STATISTICS")
        report.append("-" * 30)
        report.append(f"Total Trades: {results.total_trades}")
        report.append(f"Winning Trades: {results.winning_trades}")
        report.append(f"Losing Trades: {results.losing_trades}")
        report.append(f"Win Rate: {results.win_rate:.2%}")
        report.append(f"Average Return per Trade: {results.avg_return_per_trade:.2%}")
        report.append(f"Total Return: {results.total_return:.2%}")
        report.append(f"Maximum Drawdown: {results.max_drawdown:.2%}")
        report.append(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
        report.append("")

        if results.trades:
            completed_trades = [t for t in results.trades if not t.is_open]
            if completed_trades:
                returns = [t.return_pct for t in completed_trades]
                report.append("TRADE ANALYSIS")
                report.append("-" * 30)
                report.append(f"Best Trade: {max(returns):.2%}")
                report.append(f"Worst Trade: {min(returns):.2%}")
                report.append(
                    f"Average Holding Period: {np.mean([(t.exit_date - t.entry_date).days for t in completed_trades if t.exit_date]):.1f} days")
                report.append("")

        if not results.portfolio_values.empty:
            initial_value = results.portfolio_values['portfolio_value'].iloc[0]
            final_value = results.portfolio_values['portfolio_value'].iloc[-1]
            total_return = (final_value - initial_value) / initial_value

            report.append("PORTFOLIO PERFORMANCE")
            report.append("-" * 30)
            report.append(f"Initial Portfolio Value: ${initial_value:,.2f}")
            report.append(f"Final Portfolio Value: ${final_value:,.2f}")
            report.append(f"Total Portfolio Return: {total_return:.2%}")
            report.append("")

        return "\n".join(report)

    @staticmethod
    def plot_results(results: BacktestResults, save_path: Optional[str] = None) -> None:
        if results.portfolio_values.empty:
            logger.warning("No portfolio data to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Trading Strategy Backtest Results', fontsize=16)

        axes[0, 0].plot(results.portfolio_values['date'], results.portfolio_values['portfolio_value'])
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].grid(True)

        if results.trades:
            completed_trades = [t for t in results.trades if not t.is_open]
            if completed_trades:
                returns = [t.return_pct * 100 for t in completed_trades]
                axes[0, 1].hist(returns, bins=20, alpha=0.7, edgecolor='black')
                axes[0, 1].set_title('Trade Returns Distribution')
                axes[0, 1].set_xlabel('Return (%)')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].grid(True)

        portfolio_values = results.portfolio_values['portfolio_value']
        rolling_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - rolling_max) / rolling_max * 100

        axes[1, 0].fill_between(results.portfolio_values['date'], drawdown, 0, color='red', alpha=0.3)
        axes[1, 0].set_title('Portfolio Drawdown')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Drawdown (%)')
        axes[1, 0].grid(True)

        if len(results.portfolio_values) > 30:
            portfolio_df = results.portfolio_values.set_index('date')
            monthly_returns = portfolio_df['portfolio_value'].resample('M').last().pct_change() * 100

            if len(monthly_returns) > 1:
                monthly_returns_pivot = monthly_returns.groupby([
                    monthly_returns.index.year,
                    monthly_returns.index.month
                ]).mean().unstack()

                sns.heatmap(monthly_returns_pivot, annot=True, fmt='.1f',
                            cmap='RdYlGn', center=0, ax=axes[1, 1])
                axes[1, 1].set_title('Monthly Returns Heatmap (%)')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")


def main():
    try:
        strategy = TradingStrategy()
        backtest = TradingBacktest(
            strategy=strategy,
            initial_capital=100000,
            position_size=0.02,
            max_positions=10,
            hold_period_days=7,
            stop_loss_pct=-0.05,
            take_profit_pct=0.10
        )

        results = backtest.run_backtest(
            start_date='2023-01-01',
            end_date='2023-12-31',
            categories=['america'],
            signal_types=['oversold',
                          'overbought']
        )

        report = backtest.generate_report(results)
        print(report)

        Path('images').mkdir(parents=True, exist_ok=True)
        backtest.plot_results(results, save_path='images/backtest_results.png')

    except Exception as e:
        logger.error(f"Error in main: {e}")


if __name__ == '__main__':
    main()
