import logging
from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class IndicatorResult:
    values: Union[pd.Series, np.ndarray]
    name: str
    parameters: dict


@dataclass
class MACDResult:
    macd: pd.Series
    signal: pd.Series
    histogram: pd.Series
    parameters: dict


@dataclass
class BollingerBandsResult:
    upper_band: pd.Series
    middle_band: pd.Series
    lower_band: pd.Series
    bbp: pd.Series
    bandwidth: pd.Series
    parameters: dict


class TechnicalIndicatorError(Exception):
    pass


class TechnicalIndicators:

    def __init__(self, price_column: str = 'Close'):

        self.price_column = price_column

    def _validate_data(self, data: pd.DataFrame) -> pd.Series:

        if not isinstance(data, pd.DataFrame):
            raise TechnicalIndicatorError("Data must be a pandas DataFrame")

        if self.price_column not in data.columns:
            raise TechnicalIndicatorError(f"Column '{self.price_column}' not found in data")

        price_series = data[self.price_column].copy()

        if price_series.empty:
            raise TechnicalIndicatorError("Price series is empty")

        if price_series.isnull().all():
            raise TechnicalIndicatorError("Price series contains only null values")

        return price_series

    @staticmethod
    def _validate_window(window: int,
                         data_length: int,
                         indicator_name: str) -> None:

        if not isinstance(window, int) or window <= 0:
            raise TechnicalIndicatorError(f"{indicator_name}: Window must be a positive integer")

        if window >= data_length:
            raise TechnicalIndicatorError(
                f"{indicator_name}: Window size ({window}) must be smaller than data length ({data_length})"
            )

    def sma(self, data: pd.DataFrame,
            window: int = 20) -> IndicatorResult:

        try:
            price_series = self._validate_data(data)
            self._validate_window(window, len(price_series), "SMA")

            sma_values = price_series.rolling(window=window, min_periods=1).mean()

            return IndicatorResult(
                values=sma_values,
                name=f"SMA_{window}",
                parameters={"window": window}
            )

        except Exception as e:
            logger.error(f"Error calculating SMA: {e}")
            raise TechnicalIndicatorError(f"SMA calculation failed: {e}")

    def ema(self, data: pd.DataFrame,
            window: int = 20) -> IndicatorResult:

        try:
            price_series = self._validate_data(data)
            self._validate_window(window, len(price_series), "EMA")

            ema_values = price_series.ewm(span=window, adjust=False).mean()

            return IndicatorResult(
                values=ema_values,
                name=f"EMA_{window}",
                parameters={"window": window}
            )

        except Exception as e:
            logger.error(f"Error calculating EMA: {e}")
            raise TechnicalIndicatorError(f"EMA calculation failed: {e}")

    def rsi(self, data: pd.DataFrame,
            window: int = 14) -> IndicatorResult:

        try:
            price_series = self._validate_data(data)
            self._validate_window(window, len(price_series), "RSI")

            delta = price_series.diff()
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            avg_gains = gains.ewm(alpha=1 / window, adjust=False).mean()
            avg_losses = losses.ewm(alpha=1 / window, adjust=False).mean()
            rs = avg_gains / (avg_losses + 1e-10)
            rsi_values = 100 - (100 / (1 + rs))
            rsi_values.iloc[:window] = np.nan

            return IndicatorResult(
                values=rsi_values,
                name=f"RSI_{window}",
                parameters={"window": window}
            )

        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            raise TechnicalIndicatorError(f"RSI calculation failed: {e}")

    def macd(self, data: pd.DataFrame,
             fast_period: int = 12,
             slow_period: int = 26,
             signal_period: int = 9) -> MACDResult:

        try:
            price_series = self._validate_data(data)

            if fast_period >= slow_period:
                raise TechnicalIndicatorError("Fast period must be less than slow period")

            self._validate_window(slow_period, len(price_series), "MACD")

            fast_ema = price_series.ewm(span=fast_period, adjust=False).mean()
            slow_ema = price_series.ewm(span=slow_period, adjust=False).mean()
            macd_line = fast_ema - slow_ema
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
            histogram = macd_line - signal_line

            return MACDResult(
                macd=macd_line,
                signal=signal_line,
                histogram=histogram,
                parameters={
                    "fast_period": fast_period,
                    "slow_period": slow_period,
                    "signal_period": signal_period
                }
            )

        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            raise TechnicalIndicatorError(f"MACD calculation failed: {e}")

    def bollinger_bands(self, data: pd.DataFrame,
                        window: int = 20,
                        num_std: float = 2.0) -> BollingerBandsResult:

        try:
            price_series = self._validate_data(data)
            self._validate_window(window, len(price_series), "Bollinger Bands")

            if num_std <= 0:
                raise TechnicalIndicatorError("Number of standard deviations must be positive")

            sma_result = self.sma(data, window)
            middle_band = sma_result.values
            std_dev = price_series.rolling(window=window, min_periods=1).std()
            upper_band = middle_band + (num_std * std_dev)
            lower_band = middle_band - (num_std * std_dev)
            bbp = (price_series - lower_band) / (upper_band - lower_band + 1e-10)
            bandwidth = (upper_band - lower_band) / (middle_band + 1e-10)

            return BollingerBandsResult(
                upper_band=upper_band,
                middle_band=middle_band,
                lower_band=lower_band,
                bbp=bbp,
                bandwidth=bandwidth,
                parameters={"window": window, "num_std": num_std}
            )

        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            raise TechnicalIndicatorError(f"Bollinger Bands calculation failed: {e}")


def create_sample_data(length: int = 100) -> pd.DataFrame:
    np.random.seed(42)

    base_price = 100
    price_changes = np.random.normal(0, 2, length)
    prices = np.cumsum(price_changes) + base_price

    opens = prices + np.random.normal(0, 0.5, length)
    highs = np.maximum(opens, prices) + np.random.exponential(0.5, length)
    lows = np.minimum(opens, prices) - np.random.exponential(0.5, length)
    volumes = np.random.randint(1000, 10000, length)

    return pd.DataFrame({
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': prices,
        'Volume': volumes
    })


def main():
    try:
        data = create_sample_data(200)

        indicators = TechnicalIndicators('Close')
        print("Calculating technical indicators...")

        sma = indicators.sma(data, window=20)
        print(f"SMA: {sma.name}, Last 5 values: {sma.values.tail().tolist()}")

        ema = indicators.ema(data, window=20)
        print(f"EMA: {ema.name}, Last 5 values: {ema.values.tail().tolist()}")

        rsi = indicators.rsi(data, window=14)
        print(f"RSI: {rsi.name}, Last 5 values: {rsi.values.tail().tolist()}")

        macd = indicators.macd(data)
        print(f"MACD: Last 5 values: {macd.macd.tail().tolist()}")
        print(f"Signal: Last 5 values: {macd.signal.tail().tolist()}")

        bb = indicators.bollinger_bands(data)
        print(f"BB Upper: Last 5 values: {bb.upper_band.tail().tolist()}")
        print(f"BB Lower: Last 5 values: {bb.lower_band.tail().tolist()}")

        print("\nAll indicators calculated successfully!")

    except TechnicalIndicatorError as e:
        logger.error(f"Technical indicator error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


if __name__ == '__main__':
    main()
