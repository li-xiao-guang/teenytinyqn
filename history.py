import logging
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import yfinance as yf

import config
from indicator import TechnicalIndicators, TechnicalIndicatorError
from screener import StockScreener

warnings.filterwarnings('ignore', category=FutureWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DownloadConfig:
    period: str = '1y'
    interval: str = '1d'
    include_indicators: bool = True
    indicators_config: Dict = None
    max_workers: int = 5
    retry_attempts: int = 3
    retry_delay: float = 1.0

    def __post_init__(self):
        if self.indicators_config is None:
            self.indicators_config = {
                'sma': {'windows': [20, 50, 200]},
                'ema': {'windows': [12, 26]},
                'rsi': {'window': 14},
                'macd': {'fast': 12, 'slow': 26, 'signal': 9},
                'bollinger_bands': {'window': 20, 'num_std': 2.0}
            }


@dataclass
class DownloadResult:
    symbol: str
    success: bool
    data: Optional[pd.DataFrame] = None
    error: Optional[str] = None
    indicators_added: List[str] = None

    def __post_init__(self):
        if self.indicators_added is None:
            self.indicators_added = []


class StockDataDownloader:
    VALID_PERIODS = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
    VALID_INTERVALS = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']

    def __init__(self, data_folder: str = 'data/history'):

        self.data_folder = Path(data_folder)
        self.data_folder.mkdir(parents=True, exist_ok=True)
        self.indicators = TechnicalIndicators()

    def validate_parameters(self, period: str,
                            interval: str) -> None:

        if period not in self.VALID_PERIODS:
            raise ValueError(f"Invalid period '{period}'. Valid periods: {self.VALID_PERIODS}")

        if interval not in self.VALID_INTERVALS:
            raise ValueError(f"Invalid interval '{interval}'. Valid intervals: {self.VALID_INTERVALS}")

    def download_single_stock(self, symbol: str,
                              download_config: DownloadConfig) -> DownloadResult:

        for attempt in range(download_config.retry_attempts):
            try:
                self.validate_parameters(download_config.period, download_config.interval)

                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    period=download_config.period,
                    interval=download_config.interval,
                    actions=False,
                    auto_adjust=True,
                    back_adjust=False
                )

                if df.empty:
                    return DownloadResult(
                        symbol=symbol,
                        success=False,
                        error="No data returned from yfinance"
                    )

                df = self._standardize_columns(df)

                indicators_added = []
                if download_config.include_indicators:
                    indicators_added = self._add_technical_indicators(df, download_config.indicators_config)

                df = df.round(4)

                logger.info(f"Successfully downloaded {len(df)} records for {symbol}")

                return DownloadResult(
                    symbol=symbol,
                    success=True,
                    data=df,
                    indicators_added=indicators_added
                )

            except Exception as e:
                error_msg = f"Attempt {attempt + 1} failed for {symbol}: {str(e)}"
                logger.warning(error_msg)

                if attempt < download_config.retry_attempts - 1:
                    time.sleep(download_config.retry_delay * (attempt + 1))
                else:
                    logger.error(f"All retry attempts failed for {symbol}")
                    return DownloadResult(
                        symbol=symbol,
                        success=False,
                        error=str(e)
                    )

        return DownloadResult(symbol=symbol, success=False, error="Unknown error")

    def download_multiple_stocks(self, symbols: Union[List[str], pd.DataFrame],
                                 download_config: DownloadConfig = None) -> Dict[str, DownloadResult]:

        if download_config is None:
            download_config = DownloadConfig()

        if isinstance(symbols, pd.DataFrame):
            if config.SYMBOL not in symbols.columns:
                raise ValueError(f"DataFrame must contain '{config.SYMBOL}' column")
            symbol_list = symbols[config.SYMBOL].dropna().unique().tolist()
        else:
            symbol_list = list(symbols)

        if not symbol_list:
            logger.warning("No symbols provided for download")
            return {}

        logger.info(f"Starting download for {len(symbol_list)} symbols using {download_config.max_workers} workers")

        results = {}
        successful_downloads = 0
        failed_downloads = 0

        with ThreadPoolExecutor(max_workers=download_config.max_workers) as executor:
            future_to_symbol = {
                executor.submit(self.download_single_stock, symbol, download_config): symbol
                for symbol in symbol_list
            }

            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    results[symbol] = result

                    if result.success:
                        successful_downloads += 1
                    else:
                        failed_downloads += 1
                        logger.error(f"Failed to download {symbol}: {result.error}")

                except Exception as e:
                    error_msg = f"Unexpected error downloading {symbol}: {str(e)}"
                    logger.error(error_msg)
                    results[symbol] = DownloadResult(symbol=symbol, success=False, error=error_msg)
                    failed_downloads += 1

        logger.info(f"Download completed: {successful_downloads} successful, {failed_downloads} failed")
        return results

    @staticmethod
    def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:

        column_mapping = {
            'Open': config.OPEN,
            'High': config.HIGH,
            'Low': config.LOW,
            'Close': config.CLOSE,
            'Volume': config.VOLUME
        }

        existing_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
        return df.rename(columns=existing_mapping)

    def _add_technical_indicators(self, df: pd.DataFrame,
                                  indicators_config: Dict) -> List[str]:

        indicators_added = []

        try:
            if 'sma' in indicators_config:
                for window in indicators_config['sma']['windows']:
                    try:
                        sma_result = self.indicators.sma(df, window)
                        df[f"{config.SMA}_{window}"] = sma_result.values
                        indicators_added.append(f"SMA_{window}")

                    except TechnicalIndicatorError as e:
                        logger.warning(f"Failed to calculate SMA_{window}: {e}")

            if 'ema' in indicators_config:
                for window in indicators_config['ema']['windows']:
                    try:
                        ema_result = self.indicators.ema(df, window)
                        df[f"{config.EMA}_{window}"] = ema_result.values
                        indicators_added.append(f"EMA_{window}")

                    except TechnicalIndicatorError as e:
                        logger.warning(f"Failed to calculate EMA_{window}: {e}")

            if 'rsi' in indicators_config:
                try:
                    rsi_result = self.indicators.rsi(df, indicators_config['rsi']['window'])
                    df[config.RSI] = rsi_result.values
                    indicators_added.append("RSI")

                except TechnicalIndicatorError as e:
                    logger.warning(f"Failed to calculate RSI: {e}")

            if 'macd' in indicators_config:
                try:
                    macd_config = indicators_config['macd']
                    macd_result = self.indicators.macd(
                        df,
                        macd_config['fast'],
                        macd_config['slow'],
                        macd_config['signal']
                    )
                    df[config.MACD] = macd_result.macd
                    df[config.MACD_SIGNAL] = macd_result.signal
                    df[config.MACD_HISTOGRAM] = macd_result.histogram
                    indicators_added.extend(["MACD", "MACD_Signal", "MACD_Histogram"])

                except TechnicalIndicatorError as e:
                    logger.warning(f"Failed to calculate MACD: {e}")

            if 'bollinger_bands' in indicators_config:
                try:
                    bb_config = indicators_config['bollinger_bands']
                    bb_result = self.indicators.bollinger_bands(
                        df,
                        bb_config['window'],
                        bb_config['num_std']
                    )
                    df[config.BB_UPPER] = bb_result.upper_band
                    df[config.BB_MIDDLE] = bb_result.middle_band
                    df[config.BB_LOWER] = bb_result.lower_band
                    df[config.BBP] = bb_result.bbp
                    df[config.BB_WIDTH] = bb_result.bandwidth
                    indicators_added.extend(["BB_Upper", "BB_Middle", "BB_Lower", "BBP", "BB_Width"])

                except TechnicalIndicatorError as e:
                    logger.warning(f"Failed to calculate Bollinger Bands: {e}")

        except Exception as e:
            logger.error(f"Unexpected error adding indicators: {e}")

        return indicators_added

    def save_data(self, symbol: str,
                  data: pd.DataFrame,
                  file_format: str = 'csv') -> bool:

        try:
            clean_symbol = symbol.replace('/', '_').replace('\\', '_')

            if file_format.lower() == 'csv':
                file_path = self.data_folder / f"{clean_symbol}.csv"
                data.to_csv(file_path)
            elif file_format.lower() == 'parquet':
                file_path = self.data_folder / f"{clean_symbol}.parquet"
                data.to_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")

            logger.info(f"Saved data for {symbol} to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving data for {symbol}: {e}")
            return False

    @staticmethod
    def get_successful_data(results: Dict[str, DownloadResult]) -> Dict[str, pd.DataFrame]:

        return {
            symbol: result.data
            for symbol, result in results.items()
            if result.success and result.data is not None
        }

    @staticmethod
    def get_failed_downloads(results: Dict[str, DownloadResult]) -> Dict[str, str]:

        return {
            symbol: result.error
            for symbol, result in results.items()
            if not result.success
        }


def main():
    try:
        screener = StockScreener()
        downloader = StockDataDownloader()

        print("Loading symbols...")
        symbols_df = screener.load_symbols_from_csv('america')

        if symbols_df is None or symbols_df.empty:
            print("No symbols loaded, using sample symbols")
            symbols_df = pd.DataFrame({
                config.SYMBOL: ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
            })
        else:
            symbols_df = symbols_df.head(10)

        print(f"Downloaded symbols: {symbols_df[config.SYMBOL].tolist()}")

        download_config = DownloadConfig(
            period='1y',
            interval='1d',
            include_indicators=True,
            max_workers=3,
            retry_attempts=2
        )

        print("Starting downloads...")
        results = downloader.download_multiple_stocks(symbols_df, download_config)

        successful_data = downloader.get_successful_data(results)
        failed_downloads = downloader.get_failed_downloads(results)

        print(f"\nDownload Summary:")
        print(f"Successful: {len(successful_data)}")
        print(f"Failed: {len(failed_downloads)}")

        if successful_data:
            sample_symbol = next(iter(successful_data))
            sample_data = successful_data[sample_symbol]
            print(f"\nSample data for {sample_symbol}:")
            print(f"Shape: {sample_data.shape}")
            print(f"Columns: {list(sample_data.columns)}")
            print(f"Date range: {sample_data.index[0]} to {sample_data.index[-1]}")
            print("\nLast 3 rows:")
            print(sample_data.tail(3))

        if failed_downloads:
            print("\nFailed downloads:")
            for symbol, error in failed_downloads.items():
                print(f"  {symbol}: {error}")

        for symbol, data in successful_data.items():
            downloader.save_data(symbol, data)
        print("Data saved successfully!")

    except Exception as e:
        logger.error(f"Error in main: {e}")


if __name__ == '__main__':
    main()
