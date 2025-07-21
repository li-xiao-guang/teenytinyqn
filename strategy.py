import logging
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

import config
from history import StockDataDownloader, DownloadConfig
from screener import StockScreener

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradingStrategy:

    def __init__(self, data_folder: str = 'data/strategy'):
        self.data_folder = Path(data_folder)
        self.data_folder.mkdir(parents=True, exist_ok=True)

        self.screener = StockScreener()
        self.downloader = StockDataDownloader()

    def download_all_data(self, log: List[str],
                          rsi_threshold: int = 25,
                          bbp_threshold: float = 0.0) -> bool:
        try:
            self._clear_existing_files()

            log.append('Downloading symbols...')
            symbols_data = self._download_all_symbols()

            if not symbols_data:
                log.append('Failed to download symbols')
                return False

            all_symbols = self._get_unique_symbols(symbols_data)
            log.append(f'Found {len(all_symbols)} unique symbols')

            log.append('Downloading daily price history...')
            daily_success = self._download_and_process_data(
                symbols_data, all_symbols, '1y', '1d', rsi_threshold, bbp_threshold
            )

            log.append('Downloading weekly price history...')
            weekly_success = self._download_and_process_data(
                symbols_data, all_symbols, '5y', '1wk', rsi_threshold, bbp_threshold
            )

            success = daily_success and weekly_success
            log.append(f'Data download {"completed successfully" if success else "completed with errors"}')
            return success

        except Exception as e:
            logger.error(f'Error in download_all_data: {e}')
            log.append(f'Error: {str(e)}')
            return False

    def load_strategy_data(self, category: str,
                           signal_type: str,
                           interval: str = 'daily') -> pd.DataFrame:
        try:
            interval_suffix = '1wk' if interval == 'weekly' else '1d'
            file_path = self._get_strategy_file_path(category, interval_suffix)

            if not file_path.exists():
                logger.warning(f'Strategy file not found: {file_path}')
                return pd.DataFrame()

            df = pd.read_csv(file_path)

            if signal_type.lower() == 'overbought':
                df = df[
                    (df[config.LEVEL] == config.OVERBOUGHT) |
                    (df[config.DIVERGENCE] == config.BEARISH)
                    ]
                df = df.sort_values(by=config.RSI, ascending=False)

            elif signal_type.lower() == 'oversold':
                df = df[
                    (df[config.LEVEL] == config.OVERSOLD) |
                    (df[config.DIVERGENCE] == config.BULLISH)
                    ]
                df = df.sort_values(by=config.RSI, ascending=True)

            return df.reset_index(drop=True)

        except Exception as e:
            logger.error(f'Error loading strategy data for {category}: {e}')
            return pd.DataFrame()

    def _clear_existing_files(self) -> None:
        try:
            for file_path in self.data_folder.glob('*.csv'):
                file_path.unlink()
                logger.info(f'Removed existing file: {file_path}')

        except Exception as e:
            logger.error(f'Error clearing existing files: {e}')

    def _download_all_symbols(self) -> Dict[str, pd.DataFrame]:
        symbols_data = {}
        categories = [config.AMERICA, config.CHINA]

        for category in categories:
            try:
                symbols_df = self.screener.download_symbols(category, count=100)
                if symbols_df is not None and not symbols_df.empty:
                    symbols_data[category] = symbols_df
                    logger.info(f'Downloaded {len(symbols_df)} symbols for {category}')
                else:
                    symbols_df = self.screener.load_symbols_from_csv(category)
                    if symbols_df is not None and not symbols_df.empty:
                        symbols_data[category] = symbols_df
                        logger.info(f'Loaded {len(symbols_df)} symbols for {category} from file')
                    else:
                        logger.warning(f'No symbols found for {category}')

            except Exception as e:
                logger.error(f'Error downloading symbols for {category}: {e}')

        return symbols_data

    @staticmethod
    def _get_unique_symbols(symbols_data: Dict[str, pd.DataFrame]) -> List[str]:
        all_symbols = set()

        for symbols_df in symbols_data.values():
            if config.SYMBOL in symbols_df.columns:
                all_symbols.update(symbols_df[config.SYMBOL].dropna().unique())

        return list(all_symbols)

    def _download_and_process_data(self, symbols_data: Dict[str, pd.DataFrame],
                                   all_symbols: List[str],
                                   period: str,
                                   interval: str,
                                   rsi_threshold: int,
                                   bbp_threshold: float) -> bool:
        try:
            # Configure download
            download_config = DownloadConfig(
                period=period,
                interval=interval,
                include_indicators=True,
                max_workers=3,
                retry_attempts=2
            )

            results = self.downloader.download_multiple_stocks(all_symbols, download_config)
            successful_data = self.downloader.get_successful_data(results)

            if not successful_data:
                logger.error('No successful downloads')
                return False

            for category, symbols_df in symbols_data.items():
                try:
                    category_symbols = symbols_df[config.SYMBOL].tolist()
                    category_data = {
                        symbol: data for symbol, data in successful_data.items()
                        if symbol in category_symbols
                    }

                    if not category_data:
                        logger.warning(f'No data available for category {category}')
                        continue

                    strategy_rows = self._calculate_strategy_signals(
                        category_data, rsi_threshold, bbp_threshold
                    )

                    if strategy_rows:
                        strategy_df = pd.DataFrame(strategy_rows)
                        final_df = symbols_df.merge(
                            strategy_df, on=config.SYMBOL, how='inner'
                        )

                        file_path = self._get_strategy_file_path(category, interval)
                        final_df.to_csv(file_path, index=False)
                        logger.info(f'Saved {len(final_df)} strategy results for {category} ({interval})')
                    else:
                        logger.warning(f'No strategy signals generated for {category}')

                except Exception as e:
                    logger.error(f'Error processing category {category}: {e}')
                    continue

            return True

        except Exception as e:
            logger.error(f'Error in download and process: {e}')
            return False

    def _calculate_strategy_signals(self, stock_data: Dict[str, pd.DataFrame],
                                    rsi_threshold: int,
                                    bbp_threshold: float) -> List[Dict]:
        strategy_rows = []

        for symbol, data in stock_data.items():
            try:
                if data.empty or len(data) < 20:
                    continue

                latest_row = data.iloc[-1].copy()

                row_dict = latest_row.to_dict()
                row_dict[config.SYMBOL] = symbol
                row_dict[config.LEVEL] = ''
                row_dict[config.DIVERGENCE] = ''

                if len(data) > 10:
                    bullish_points, bearish_points = self._calculate_rsi_divergence(data)

                    recent_bullish = any(point >= len(data) - 3 for point in bullish_points)
                    recent_bearish = any(point >= len(data) - 3 for point in bearish_points)

                    if recent_bullish:
                        row_dict[config.DIVERGENCE] = config.BULLISH
                    elif recent_bearish:
                        row_dict[config.DIVERGENCE] = config.BEARISH

                current_rsi = latest_row.get(config.RSI, 50)
                current_bbp = latest_row.get(config.BBP, 0.5)

                if pd.notna(current_rsi) and pd.notna(current_bbp):
                    if self._is_overbought(current_rsi, current_bbp, 100 - rsi_threshold, 1 - bbp_threshold):
                        row_dict[config.LEVEL] = config.OVERBOUGHT
                    elif self._is_oversold(current_rsi, current_bbp, rsi_threshold, bbp_threshold):
                        row_dict[config.LEVEL] = config.OVERSOLD

                strategy_rows.append(row_dict)

            except Exception as e:
                logger.error(f'Error calculating signals for {symbol}: {e}')
                continue

        return strategy_rows

    @staticmethod
    def _calculate_rsi_divergence(data: pd.DataFrame) -> Tuple[List[int], List[int]]:
        try:
            if len(data) < 10:
                return [], []

            price_lows = argrelextrema(data[config.LOW].values, np.less, order=5)[0]
            price_highs = argrelextrema(data[config.HIGH].values, np.greater, order=5)[0]

            rsi_values = data[config.RSI].values

            bullish_divergences = []
            bearish_divergences = []

            for i in range(1, len(price_lows)):
                current_low_idx = price_lows[i]
                previous_low_idx = price_lows[i - 1]

                if (data[config.LOW].iloc[current_low_idx] < data[config.LOW].iloc[previous_low_idx] and
                        rsi_values[current_low_idx] > rsi_values[previous_low_idx]):
                    bullish_divergences.append(current_low_idx)

            for i in range(1, len(price_highs)):
                current_high_idx = price_highs[i]
                previous_high_idx = price_highs[i - 1]

                if (data[config.HIGH].iloc[current_high_idx] > data[config.HIGH].iloc[previous_high_idx] and
                        rsi_values[current_high_idx] < rsi_values[previous_high_idx]):
                    bearish_divergences.append(current_high_idx)

            return bullish_divergences, bearish_divergences

        except Exception as e:
            logger.error(f'Error calculating RSI divergence: {e}')
            return [], []

    @staticmethod
    def _is_overbought(rsi: float,
                       bbp: float,
                       rsi_threshold: float,
                       bbp_threshold: float) -> bool:
        return rsi >= rsi_threshold and bbp >= bbp_threshold

    @staticmethod
    def _is_oversold(rsi: float,
                     bbp: float,
                     rsi_threshold: float,
                     bbp_threshold: float) -> bool:
        return rsi <= rsi_threshold and bbp <= bbp_threshold

    def _get_strategy_file_path(self, category: str,
                                interval: str) -> Path:
        return self.data_folder / f'{category}_{interval}_strategy.csv'


def main():
    try:
        strategy = TradingStrategy()
        log_messages = []

        print("Starting trading strategy analysis...")

        success = strategy.download_all_data(
            log=log_messages,
            rsi_threshold=30,
            bbp_threshold=0.2
        )

        for message in log_messages:
            print(f"LOG: {message}")

        if success:
            print("\nStrategy analysis completed successfully!")

            oversold_stocks = strategy.load_strategy_data('america', 'oversold', 'daily')
            print(f"\nFound {len(oversold_stocks)} oversold stocks (daily)")

            if not oversold_stocks.empty:
                print("\nTop 5 oversold stocks:")
                display_cols = [config.SYMBOL,
                                config.NAME,
                                config.RSI,
                                config.BBP,
                                config.LEVEL,
                                config.DIVERGENCE]
                available_cols = [col for col in display_cols if col in oversold_stocks.columns]
                print(oversold_stocks[available_cols].head())
        else:
            print("Strategy analysis completed with errors. Check logs for details.")

    except Exception as e:
        print(f"Error in main: {e}")
        logger.error(f"Error in main: {e}")


if __name__ == '__main__':
    main()
