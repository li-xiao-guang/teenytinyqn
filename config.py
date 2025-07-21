AMERICA = "america"
CHINA = "china"
INDICES = "indices"
WATCHLIST = "watchlist"
PORTFOLIO = "portfolio"

SYMBOL = "Symbol"
CODE = "Code"
NAME = "Name"
CLOSE = "Close"
OPEN = "Open"
HIGH = "High"
LOW = "Low"
VOLUME = "Volume"

SMA = "SMA"
EMA = "EMA"
RSI = "RSI"
MACD = "MACD"
MACD_SIGNAL = "MACD_Signal"
MACD_HISTOGRAM = "MACD_Histogram"
BB_UPPER = "BB_Upper"
BB_MIDDLE = "BB_Middle"
BB_LOWER = "BB_Lower"
BBP = "BBP"
BB_WIDTH = "BB_Width"

LEVEL = "Level"
DIVERGENCE = "Divergence"
OVERBOUGHT = "Overbought"
OVERSOLD = "Oversold"
BULLISH = "Bullish"
BEARISH = "Bearish"


def all_categories():
    return [AMERICA,
            CHINA,
            INDICES,
            WATCHLIST,
            PORTFOLIO]


def get_basic_columns():
    return [SYMBOL,
            OPEN,
            HIGH,
            LOW,
            CLOSE,
            VOLUME]


def get_indicator_columns():
    return [SMA,
            EMA,
            RSI,
            MACD,
            MACD_SIGNAL,
            MACD_HISTOGRAM,
            BB_UPPER,
            BB_MIDDLE,
            BB_LOWER,
            BBP,
            BB_WIDTH]


def get_strategy_columns():
    return [LEVEL,
            DIVERGENCE]
