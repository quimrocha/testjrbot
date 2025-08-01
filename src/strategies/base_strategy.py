"""
Base Strategy Class
All trading strategies inherit from this base class
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class SignalType(Enum):
    """Trading signal types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class TradingSignal:
    """Trading signal data structure"""
    symbol: str
    signal: SignalType
    price: float
    confidence: float  # 0.0 to 1.0
    timestamp: pd.Timestamp
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, name: str, config: Dict):
        """Initialize the strategy"""
        self.name = name
        self.config = config
        self.enabled = config.get('enabled', True)
        self.indicators = {}
        
    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the strategy"""
        pass
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """Generate trading signals based on market data and indicators"""
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate that the data is sufficient for strategy execution"""
        if data.empty:
            return False
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        return all(col in data.columns for col in required_columns)
    
    def get_required_history(self) -> int:
        """Return the minimum number of periods required for the strategy"""
        return 50  # Default value, override in specific strategies
    
    def get_strategy_info(self) -> Dict:
        """Return strategy information"""
        return {
            'name': self.name,
            'enabled': self.enabled,
            'config': self.config,
            'required_history': self.get_required_history()
        }
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data before strategy execution"""
        # Ensure data is sorted by timestamp
        data_copy = data.copy().sort_index()
        
        # Remove any NaN values
        data_copy = data_copy.dropna()
        
        return data_copy
    
    def execute(self, data: pd.DataFrame, symbol: str) -> List[TradingSignal]:
        """Main execution method for the strategy"""
        if not self.enabled:
            return []
        
        if not self.validate_data(data):
            return []
        
        # Preprocess data
        processed_data = self.preprocess_data(data)
        
        # Check if we have enough data
        if len(processed_data) < self.get_required_history():
            return []
        
        # Calculate indicators
        data_with_indicators = self.calculate_indicators(processed_data)
        
        # Generate signals
        signals = self.generate_signals(data_with_indicators)
        
        # Add symbol to all signals
        for signal in signals:
            signal.symbol = symbol
        
        return signals