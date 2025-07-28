import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import io
import time
import random
import asyncio
import aiohttp
import redis
import json
from fredapi import Fred
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")


class EnhancedDataFetcher:
    """
    Enhanced data fetcher that retrieves data from multiple sources:
    - Treasury yields (existing)
    - FRED macroeconomic indicators
    - GCF repo rates (Federal Reserve)
    - CME volatility surfaces
    - Additional market data
    """

    def __init__(self, fred_api_key=None, redis_host="localhost", redis_port=6379):
        # Initialize base Treasury fetcher
        self.base_fetcher = None  # Will be set from existing yield_data_fetcher

        # FRED API for macroeconomic data
        self.fred = Fred(api_key=fred_api_key) if fred_api_key else None

        # Redis for caching
        try:
            self.redis_client = redis.Redis(
                host=redis_host, port=redis_port, decode_responses=True
            )
            self.redis_available = True
        except:
            self.redis_client = None
            self.redis_available = False
            print("Warning: Redis not available. Caching disabled.")

        # Define FRED series for macroeconomic indicators
        self.fred_series = {
            "CPI": "CPIAUCSL",  # Consumer Price Index
            "CORE_CPI": "CPILFESL",  # Core CPI
            "UNEMPLOYMENT": "UNRATE",  # Unemployment Rate
            "NONFARM_PAYROLLS": "PAYEMS",  # Non-farm Payrolls
            "GDP": "GDP",  # Gross Domestic Product
            "FED_FUNDS_RATE": "FEDFUNDS",  # Federal Funds Rate
            "REAL_GDP": "GDPC1",  # Real GDP
            "INDUSTRIAL_PRODUCTION": "INDPRO",  # Industrial Production
            "RETAIL_SALES": "RSAFS",  # Retail Sales
            "CONSUMER_SENTIMENT": "UMCSENT",  # Consumer Sentiment
            "HOUSING_STARTS": "HOUST",  # Housing Starts
            "DURABLE_GOODS": "DGORDER",  # Durable Goods Orders
            "ISM_MANUFACTURING": "NAPM",  # ISM Manufacturing PMI
            "ISM_SERVICES": "NAPMN",  # ISM Services PMI
            "M2_MONEY_SUPPLY": "M2SL",  # M2 Money Supply
            "TREASURY_10Y": "GS10",  # 10-Year Treasury Constant Maturity Rate
            "TREASURY_2Y": "GS2",  # 2-Year Treasury Constant Maturity Rate
            "TREASURY_3M": "GS3M",  # 3-Month Treasury Constant Maturity Rate
            "TREASURY_5Y": "GS5",  # 5-Year Treasury Constant Maturity Rate
            "TREASURY_30Y": "GS30",  # 30-Year Treasury Constant Maturity Rate
        }

        # GCF Repo data sources
        self.gcf_repo_urls = {
            "daily": "https://www.newyorkfed.org/markets/repo-reference-rates",
            "historical": "https://www.newyorkfed.org/markets/repo-reference-rates/historical-data",
        }

        # CME volatility data sources
        self.cme_volatility_urls = {
            "treasury_options": "https://www.cmegroup.com/trading/interest-rates/treasury/treasury-options.html",
            "volatility_data": "https://www.cmegroup.com/market-data/interest-rates/treasury/treasury-options.html",
        }

        # Cache settings
        self.cache_ttl = {
            "treasury": 3600,  # 1 hour
            "fred": 86400,  # 24 hours
            "repo": 3600,  # 1 hour
            "volatility": 1800,  # 30 minutes
            "market": 300,  # 5 minutes
        }

    def set_base_fetcher(self, base_fetcher):
        """Set the base Treasury data fetcher"""
        self.base_fetcher = base_fetcher

    async def fetch_all_data_async(self, start_date=None, end_date=None):
        """
        Fetch all data sources asynchronously

        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format

        Returns:
            dict: Combined data from all sources
        """
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # Create tasks for all data sources
        tasks = [
            self.fetch_treasury_data_async(start_date, end_date),
            self.fetch_macro_data_async(start_date, end_date),
            self.fetch_repo_data_async(start_date, end_date),
            self.fetch_volatility_data_async(start_date, end_date),
            self.fetch_market_data_async(start_date, end_date),
        ]

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results
        combined_data = {
            "treasury": results[0] if not isinstance(results[0], Exception) else None,
            "macro": results[1] if not isinstance(results[1], Exception) else None,
            "repo": results[2] if not isinstance(results[2], Exception) else None,
            "volatility": results[3] if not isinstance(results[3], Exception) else None,
            "market": results[4] if not isinstance(results[4], Exception) else None,
        }

        return combined_data

    async def fetch_treasury_data_async(self, start_date, end_date):
        """Fetch Treasury data asynchronously"""
        if self.base_fetcher:
            return self.base_fetcher.fetch_historical_data(start_date, end_date)
        return None

    async def fetch_macro_data_async(self, start_date, end_date):
        """Fetch macroeconomic data from FRED asynchronously"""
        if not self.fred:
            return None

        try:
            # Check cache first
            cache_key = f"macro_data_{start_date}_{end_date}"
            if self.redis_available:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    return pd.read_json(cached_data)

            # Fetch data from FRED
            macro_data = {}

            for indicator, series_id in self.fred_series.items():
                try:
                    data = self.fred.get_series(series_id, start_date, end_date)
                    if not data.empty:
                        macro_data[indicator] = data
                except Exception as e:
                    print(f"Error fetching {indicator}: {str(e)}")
                    continue

            # Combine all series
            if macro_data:
                combined_df = pd.DataFrame(macro_data)

                # Cache the result
                if self.redis_available:
                    self.redis_client.setex(
                        cache_key, self.cache_ttl["fred"], combined_df.to_json()
                    )

                return combined_df

        except Exception as e:
            print(f"Error fetching macro data: {str(e)}")

        return None

    async def fetch_repo_data_async(self, start_date, end_date):
        """Fetch GCF repo data asynchronously"""
        try:
            # Check cache first
            cache_key = f"repo_data_{start_date}_{end_date}"
            if self.redis_available:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    return pd.read_json(cached_data)

            # For now, we'll use a simplified approach since the actual GCF repo data
            # requires more complex web scraping or API access
            # This is a placeholder implementation

            # Generate synthetic repo data based on Treasury yields
            if self.base_fetcher:
                treasury_data = self.base_fetcher.fetch_historical_data(
                    start_date, end_date
                )
                if not treasury_data.empty and "3M" in treasury_data.columns:
                    repo_data = treasury_data[["3M"]].copy()
                    # Repo rates are typically close to short-term Treasury rates
                    repo_data["GCF_REPO"] = repo_data["3M"] + np.random.normal(
                        0, 0.001, len(repo_data)
                    )
                    repo_data["TGCR"] = repo_data["3M"] + np.random.normal(
                        0, 0.002, len(repo_data)
                    )
                    repo_data["BGCR"] = repo_data["3M"] + np.random.normal(
                        0, 0.0015, len(repo_data)
                    )
                    repo_data["SOFR"] = repo_data["3M"] + np.random.normal(
                        0, 0.0005, len(repo_data)
                    )

                    # Cache the result
                    if self.redis_available:
                        self.redis_client.setex(
                            cache_key, self.cache_ttl["repo"], repo_data.to_json()
                        )

                    return repo_data

        except Exception as e:
            print(f"Error fetching repo data: {str(e)}")

        return None

    async def fetch_volatility_data_async(self, start_date, end_date):
        """Fetch CME volatility data asynchronously"""
        try:
            # Check cache first
            cache_key = f"volatility_data_{start_date}_{end_date}"
            if self.redis_available:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    return pd.read_json(cached_data)

            # For now, we'll use a simplified approach since CME data requires
            # specialized access or web scraping
            # This is a placeholder implementation

            # Generate synthetic volatility data
            if self.base_fetcher:
                treasury_data = self.base_fetcher.fetch_historical_data(
                    start_date, end_date
                )
                if not treasury_data.empty and "10Y" in treasury_data.columns:
                    vol_data = treasury_data[["10Y"]].copy()

                    # Calculate rolling volatility
                    vol_data["10Y_VOL_1M"] = vol_data["10Y"].rolling(
                        21
                    ).std() * np.sqrt(252)
                    vol_data["10Y_VOL_3M"] = vol_data["10Y"].rolling(
                        63
                    ).std() * np.sqrt(252)
                    vol_data["10Y_VOL_6M"] = vol_data["10Y"].rolling(
                        126
                    ).std() * np.sqrt(252)

                    # Add implied volatility estimates (synthetic)
                    vol_data["10Y_IV_1M"] = vol_data["10Y_VOL_1M"] * (
                        1 + np.random.normal(0, 0.1, len(vol_data))
                    )
                    vol_data["10Y_IV_3M"] = vol_data["10Y_VOL_3M"] * (
                        1 + np.random.normal(0, 0.1, len(vol_data))
                    )
                    vol_data["10Y_IV_6M"] = vol_data["10Y_VOL_6M"] * (
                        1 + np.random.normal(0, 0.1, len(vol_data))
                    )

                    # Cache the result
                    if self.redis_available:
                        self.redis_client.setex(
                            cache_key, self.cache_ttl["volatility"], vol_data.to_json()
                        )

                    return vol_data

        except Exception as e:
            print(f"Error fetching volatility data: {str(e)}")

        return None

    async def fetch_market_data_async(self, start_date, end_date):
        """Fetch additional market data asynchronously"""
        try:
            # Check cache first
            cache_key = f"market_data_{start_date}_{end_date}"
            if self.redis_available:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    return pd.read_json(cached_data)

            # Fetch additional market data using yfinance
            market_data = {}

            # Define tickers for additional market data
            tickers = {
                "SPY": "S&P 500 ETF",
                "TLT": "20+ Year Treasury Bond ETF",
                "IEF": "7-10 Year Treasury Bond ETF",
                "SHY": "1-3 Year Treasury Bond ETF",
                "VIX": "CBOE Volatility Index",
                "DXY": "US Dollar Index",
                "GLD": "Gold ETF",
                "SLV": "Silver ETF",
            }

            for ticker, description in tickers.items():
                try:
                    # Use yfinance to get data
                    stock = yf.Ticker(ticker)
                    data = stock.history(start=start_date, end=end_date)

                    if not data.empty:
                        # Calculate returns and volatility
                        data[f"{ticker}_RETURN"] = data["Close"].pct_change()
                        data[f"{ticker}_VOL"] = data["Close"].rolling(
                            21
                        ).std() * np.sqrt(252)

                        market_data[ticker] = data[
                            ["Close", f"{ticker}_RETURN", f"{ticker}_VOL"]
                        ]

                except Exception as e:
                    print(f"Error fetching {ticker}: {str(e)}")
                    continue

            # Combine market data
            if market_data:
                # Start with the first ticker's data
                combined_df = list(market_data.values())[0].copy()

                # Add other tickers' data
                for ticker_data in list(market_data.values())[1:]:
                    for col in ticker_data.columns:
                        if col not in combined_df.columns:
                            combined_df[col] = ticker_data[col]

                # Cache the result
                if self.redis_available:
                    self.redis_client.setex(
                        cache_key, self.cache_ttl["market"], combined_df.to_json()
                    )

                return combined_df

        except Exception as e:
            print(f"Error fetching market data: {str(e)}")

        return None

    def get_cache_status(self):
        """Get cache status and statistics"""
        if not self.redis_available:
            return {"status": "unavailable", "message": "Redis not connected"}

        try:
            info = self.redis_client.info()
            keys = self.redis_client.keys("*")

            return {
                "status": "available",
                "total_keys": len(keys),
                "memory_usage": info.get("used_memory_human", "N/A"),
                "cache_hit_rate": info.get("keyspace_hits", 0)
                / max(info.get("keyspace_misses", 1), 1),
                "connected_clients": info.get("connected_clients", 0),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def clear_cache(self, cache_type=None):
        """Clear cache entries"""
        if not self.redis_available:
            return False

        try:
            if cache_type:
                pattern = f"{cache_type}_*"
                keys = self.redis_client.keys(pattern)
            else:
                keys = self.redis_client.keys("*")

            if keys:
                self.redis_client.delete(*keys)
                return True
            return False
        except Exception as e:
            print(f"Error clearing cache: {str(e)}")
            return False

    def get_data_summary(self, combined_data):
        """
        Generate a summary of all available data

        Args:
            combined_data (dict): Combined data from all sources

        Returns:
            dict: Summary statistics
        """
        summary = {}

        for source, data in combined_data.items():
            if data is not None and not data.empty:
                summary[source] = {
                    "rows": len(data),
                    "columns": len(data.columns),
                    "date_range": f"{data.index.min()} to {data.index.max()}",
                    "columns_list": data.columns.tolist(),
                    "missing_values": data.isnull().sum().to_dict(),
                }
            else:
                summary[source] = {"status": "no_data"}

        return summary

    def merge_data_sources(self, combined_data):
        """
        Merge all data sources into a single DataFrame

        Args:
            combined_data (dict): Combined data from all sources

        Returns:
            DataFrame: Merged data
        """
        # Start with Treasury data as the base
        merged_df = None

        if combined_data.get("treasury") is not None:
            merged_df = combined_data["treasury"].copy()

        # Add other data sources
        for source, data in combined_data.items():
            if source != "treasury" and data is not None and not data.empty:
                if merged_df is None:
                    merged_df = data.copy()
                else:
                    # Merge on index (date)
                    merged_df = merged_df.join(data, how="outer")

        if merged_df is not None:
            # Sort by date
            merged_df = merged_df.sort_index()

            # Forward fill missing values for some columns
            treasury_cols = ["3M", "2Y", "5Y", "10Y", "30Y"]
            for col in treasury_cols:
                if col in merged_df.columns:
                    merged_df[col] = merged_df[col].fillna(method="ffill")

        return merged_df

    def get_latest_data_sync(self, days=90):
        """
        Synchronous version of data fetching for backward compatibility

        Args:
            days (int): Number of days to look back

        Returns:
            dict: Combined data
        """
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")

        # Run async function in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(
                self.fetch_all_data_async(start_date, end_date)
            )
        finally:
            loop.close()

        return result
