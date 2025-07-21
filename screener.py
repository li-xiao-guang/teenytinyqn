import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict

import pandas as pd
import requests
from bs4 import BeautifulSoup

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ScrapingConfig:
    url: str
    parent_tag: Optional[str] = None
    parent_attrs: Optional[Dict] = None
    symbol_tag: str = "span"
    symbol_attrs: Optional[Dict] = None
    name_tag: str = "a"
    name_attrs: Optional[Dict] = None


@dataclass
class StockData:
    symbol: str
    code: str
    name: str


class FileManager:

    def __init__(self, folder: str = 'data/symbols'):

        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    def save_dataframe(self, category: str,
                       df: pd.DataFrame) -> bool:

        try:
            file_path = self._get_file_path(category)
            df.to_csv(file_path, index=False)
            logger.info(f"Saved {len(df)} symbols to {file_path}")
            return True

        except Exception as e:
            logger.error(f'Error saving symbols for {category}: {e}')
            return False

    def load_dataframe(self, category: str) -> Optional[pd.DataFrame]:

        try:
            file_path = self._get_file_path(category)
            if not file_path.exists():
                logger.warning(f"File does not exist: {file_path}")
                return None

            df = pd.read_csv(file_path, dtype=str)
            logger.info(f"Loaded {len(df)} symbols from {file_path}")
            return df

        except Exception as e:
            logger.error(f'Error reading symbols for {category}: {e}')
            return None

    def _get_file_path(self, category: str) -> Path:

        return self.folder / f"{category}.csv"


class WebScraper:
    DEFAULT_HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    @staticmethod
    def scrape_stocks(scraping_config: ScrapingConfig) -> Optional[pd.DataFrame]:

        try:
            response = requests.get(
                scraping_config.url,
                headers=WebScraper.DEFAULT_HEADERS,
                timeout=10
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            if scraping_config.parent_tag:
                if scraping_config.parent_attrs is None:
                    parent = soup.find(scraping_config.parent_tag)
                else:
                    parent = soup.find(scraping_config.parent_tag, scraping_config.parent_attrs)
                if not parent:
                    logger.warning(f"Parent tag not found: {scraping_config.parent_tag}")
                    return None

                soup = parent

            symbol_elements = soup.find_all(scraping_config.symbol_tag, scraping_config.symbol_attrs)
            name_elements = soup.find_all(scraping_config.name_tag, scraping_config.name_attrs)

            if not symbol_elements or not name_elements:
                logger.warning("No stock elements found")
                return None

            symbols = [elem.get_text(strip=True) for elem in symbol_elements if elem.get_text(strip=True)]
            names = []

            for elem in name_elements:
                name = elem.get('title', '').strip()
                if not name:
                    name = elem.get_text(strip=True)
                names.append(name)

            min_length = min(len(symbols), len(names))
            if min_length == 0:
                logger.warning("No valid stock data extracted")
                return None

            symbols = symbols[:min_length]
            names = names[:min_length]

            return pd.DataFrame({
                config.SYMBOL: symbols,
                config.CODE: symbols,
                config.NAME: names
            })

        except requests.RequestException as e:
            logger.error(f"Request failed for {scraping_config.url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error scraping stocks from {scraping_config.url}: {e}")
            return None


class StockDataProvider(ABC):

    @abstractmethod
    def get_stocks(self, count: int = 50) -> Optional[pd.DataFrame]:
        pass


class YahooStockProvider(StockDataProvider):

    def get_stocks(self, count: int = 50) -> Optional[pd.DataFrame]:
        config_obj = ScrapingConfig(
            url='https://finance.yahoo.com/markets/stocks/most-active/?start=0&count=100',
            parent_tag='table',
            symbol_tag='span',
            symbol_attrs={'class': 'symbol'},
            name_tag='a',
            name_attrs={'class': 'ticker'}
        )

        df = WebScraper.scrape_stocks(config_obj)
        return df.head(count) if df is not None else None


class TradingViewUSStockProvider(StockDataProvider):

    def get_stocks(self, count: int = 50) -> Optional[pd.DataFrame]:
        config_obj = ScrapingConfig(
            url='https://cn.tradingview.com/markets/stocks-usa/market-movers-active/',
            symbol_tag='a',
            symbol_attrs={'class': 'tickerNameBox-GrtoTeat'},
            name_tag='sup',
            name_attrs={'class': 'tickerDescription-GrtoTeat'}
        )

        df = WebScraper.scrape_stocks(config_obj)
        return df.head(count) if df is not None else None


class TradingViewChinaStockProvider(StockDataProvider):

    def get_stocks(self, count: int = 50) -> Optional[pd.DataFrame]:

        config_obj = ScrapingConfig(
            url='https://cn.tradingview.com/markets/stocks-china/market-movers-active/',
            symbol_tag='a',
            symbol_attrs={'class': 'tickerNameBox-GrtoTeat'},
            name_tag='sup',
            name_attrs={'class': 'tickerDescription-GrtoTeat'}
        )

        df = WebScraper.scrape_stocks(config_obj)
        if df is not None:
            df[config.SYMBOL] = df[config.SYMBOL].apply(self._add_exchange_suffix)
            df[config.CODE] = df[config.NAME].copy()
            df[config.NAME] = df[config.SYMBOL] + ' - ' + df[config.NAME]

        return df.head(count) if df is not None else None

    @staticmethod
    def _add_exchange_suffix(symbol: str) -> str:

        if symbol.startswith('6'):
            return f"{symbol}.SS"
        elif symbol.startswith(('0', '3')):
            return f"{symbol}.SZ"

        return symbol


class StockScreener:

    def __init__(self, data_folder: str = 'data/symbols'):

        self.file_manager = FileManager(data_folder)
        self._providers = {
            config.AMERICA: [YahooStockProvider(),
                             TradingViewUSStockProvider()],
            config.CHINA: [TradingViewChinaStockProvider()]
        }

    def download_symbols(self, category: str,
                         count: int = 50) -> Optional[pd.DataFrame]:

        category_lower = category.lower()

        if category_lower in {config.INDICES,
                              config.WATCHLIST,
                              config.PORTFOLIO}:
            return self.load_symbols_from_csv(category)

        if category_lower in self._providers:
            df = self._download_from_providers(category_lower, count)
            if df is not None:
                if category_lower == config.AMERICA:
                    df[config.NAME] = df[config.SYMBOL] + ' - ' + df[config.NAME]

                if self.file_manager.save_dataframe(category, df):
                    return df

        logger.warning(f"Unknown category: {category}")
        return None

    def _download_from_providers(self, category: str,
                                 count: int) -> Optional[pd.DataFrame]:

        combined_df = None

        for provider in self._providers[category]:
            df = provider.get_stocks(count)
            if df is not None:
                if combined_df is None:
                    combined_df = df
                else:
                    combined_df = pd.concat([combined_df, df]).drop_duplicates(
                        subset=config.SYMBOL, keep='first'
                    )

        return combined_df

    def load_symbols_from_csv(self, category: str) -> Optional[pd.DataFrame]:

        return self.file_manager.load_dataframe(category)

    def load_all_symbols(self) -> pd.DataFrame:

        watchlist_df = self.load_symbols_from_csv(config.WATCHLIST)
        portfolio_df = self.load_symbols_from_csv(config.PORTFOLIO)

        if watchlist_df is not None:
            watchlist_df = watchlist_df.copy()
            watchlist_df[config.WATCHLIST] = True

        if portfolio_df is not None:
            portfolio_df = portfolio_df.copy()
            portfolio_df[config.PORTFOLIO] = True

        if watchlist_df is not None and portfolio_df is not None:
            combined_df = pd.merge(
                watchlist_df,
                portfolio_df,
                on=[config.SYMBOL,
                    config.CODE,
                    config.NAME],
                how='outer'
            )
        elif watchlist_df is not None:
            combined_df = watchlist_df
        elif portfolio_df is not None:
            combined_df = portfolio_df
        else:
            combined_df = pd.DataFrame(columns=[config.SYMBOL,
                                                config.CODE,
                                                config.NAME])

        for category in [config.AMERICA, config.CHINA]:
            market_df = self.load_symbols_from_csv(category)
            if market_df is not None:
                combined_df = pd.concat([combined_df, market_df]).drop_duplicates(
                    subset=config.SYMBOL,
                    keep='first'
                )

        return combined_df.reset_index(drop=True)


def main():
    screener = StockScreener()
    categories = [config.AMERICA,
                  config.CHINA,
                  config.INDICES,
                  config.WATCHLIST,
                  config.PORTFOLIO]

    logger.info("Downloading symbols...")
    for category in categories:
        result = screener.download_symbols(category)
        if result is not None:
            logger.info(f"{category}: {len(result)} symbols downloaded")
        else:
            logger.warning(f"{category}: Download failed")

    logger.info("\nLoading symbols...")
    for category in categories:
        result = screener.load_symbols_from_csv(category)
        if result is not None:
            logger.info(f"{category}: {len(result)} symbols loaded")
        else:
            logger.warning(f"{category}: Load failed")

    logger.info("\nLoading all symbols...")
    all_symbols = screener.load_all_symbols()
    logger.info(f"Total symbols: {len(all_symbols)}")


if __name__ == '__main__':
    main()
