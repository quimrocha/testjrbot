"""
Market Data Management Module
Handles fetching, storing, and retrieving cryptocurrency market data
"""

import asyncio
import ccxt
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from loguru import logger
import yaml
import os


class MarketDataManager:
    """Manages cryptocurrency market data fetching and storage"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the market data manager"""
        self.config = self._load_config(config_path)
        self.exchange = None
        self.db_path = self.config['database']['path']
        self._init_database()
        self._init_exchange()
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def _init_exchange(self):
        """Initialize the cryptocurrency exchange connection"""
        try:
            exchange_name = self.config['exchange']['default']
            exchange_class = getattr(ccxt, exchange_name)
            
            self.exchange = exchange_class({
                'apiKey': os.getenv('BINANCE_API_KEY', ''),
                'secret': os.getenv('BINANCE_API_SECRET', ''),
                'sandbox': self.config['exchange']['sandbox'],
                'enableRateLimit': True,
            })
            
            logger.info(f"Initialized {exchange_name} exchange connection")
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            raise
    
    def _init_database(self):
        """Initialize the SQLite database for storing market data"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timeframe, timestamp)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_symbol_timeframe_timestamp 
                ON market_data(symbol, timeframe, timestamp)
            """)
            
            logger.info("Database initialized successfully")
    
    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', 
                          limit: int = 1000, since: Optional[int] = None) -> pd.DataFrame:
        """Fetch OHLCV data from exchange"""
        try:
            if since is None:
                since = int((datetime.now() - timedelta(days=30)).timestamp() * 1000)
            
            ohlcv = await asyncio.to_thread(
                self.exchange.fetch_ohlcv, symbol, timeframe, since, limit
            )
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Fetched {len(df)} candles for {symbol} {timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            return pd.DataFrame()
    
    def store_market_data(self, symbol: str, timeframe: str, df: pd.DataFrame):
        """Store market data in the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                for timestamp, row in df.iterrows():
                    conn.execute("""
                        INSERT OR REPLACE INTO market_data 
                        (symbol, timeframe, timestamp, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol, timeframe, int(timestamp.timestamp()),
                        row['open'], row['high'], row['low'], row['close'], row['volume']
                    ))
                
                logger.info(f"Stored {len(df)} records for {symbol} {timeframe}")
                
        except Exception as e:
            logger.error(f"Error storing market data: {e}")
    
    def get_market_data(self, symbol: str, timeframe: str, 
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None,
                       limit: Optional[int] = None) -> pd.DataFrame:
        """Retrieve market data from the database"""
        try:
            query = """
                SELECT timestamp, open, high, low, close, volume
                FROM market_data 
                WHERE symbol = ? AND timeframe = ?
            """
            params = [symbol, timeframe]
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(int(start_date.timestamp()))
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(int(end_date.timestamp()))
            
            query += " ORDER BY timestamp ASC"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=params)
                
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving market data: {e}")
            return pd.DataFrame()
    
    async def update_all_symbols(self):
        """Update market data for all configured symbols and timeframes"""
        symbols = self.config['data']['symbols']
        timeframes = self.config['data']['timeframes']
        
        tasks = []
        for symbol in symbols:
            for timeframe in timeframes:
                task = self._update_symbol_timeframe(symbol, timeframe)
                tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("Completed market data update for all symbols")
    
    async def _update_symbol_timeframe(self, symbol: str, timeframe: str):
        """Update market data for a specific symbol and timeframe"""
        try:
            # Get the latest timestamp from database
            latest_data = self.get_market_data(symbol, timeframe, limit=1)
            
            if not latest_data.empty:
                since = int(latest_data.index[-1].timestamp() * 1000)
            else:
                since = int((datetime.now() - timedelta(days=30)).timestamp() * 1000)
            
            # Fetch new data
            df = await self.fetch_ohlcv(symbol, timeframe, since=since)
            
            if not df.empty:
                self.store_market_data(symbol, timeframe, df)
                
        except Exception as e:
            logger.error(f"Error updating {symbol} {timeframe}: {e}")
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get the latest price for a symbol"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            logger.error(f"Error fetching latest price for {symbol}: {e}")
            return None
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available trading symbols"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT DISTINCT symbol FROM market_data")
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting available symbols: {e}")
            return []
    
    def get_data_summary(self) -> Dict:
        """Get summary of available data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT symbol, timeframe, COUNT(*) as count,
                           MIN(timestamp) as start_time,
                           MAX(timestamp) as end_time
                    FROM market_data 
                    GROUP BY symbol, timeframe
                    ORDER BY symbol, timeframe
                """)
                
                summary = {}
                for row in cursor.fetchall():
                    symbol, timeframe, count, start_time, end_time = row
                    if symbol not in summary:
                        summary[symbol] = {}
                    
                    summary[symbol][timeframe] = {
                        'count': count,
                        'start_time': datetime.fromtimestamp(start_time),
                        'end_time': datetime.fromtimestamp(end_time)
                    }
                
                return summary
                
        except Exception as e:
            logger.error(f"Error getting data summary: {e}")
            return {}


# Async context manager for market data
class AsyncMarketDataManager:
    """Async wrapper for MarketDataManager"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.manager = MarketDataManager(config_path)
    
    async def __aenter__(self):
        return self.manager
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.manager.exchange:
            await self.manager.exchange.close()