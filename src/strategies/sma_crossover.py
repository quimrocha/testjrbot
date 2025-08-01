"""
Simple Moving Average (SMA) Crossover Strategy
Generates buy signals when short SMA crosses above long SMA
Generates sell signals when short SMA crosses below long SMA
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from .base_strategy import BaseStrategy, TradingSignal, SignalType


class SMAcrossoverStrategy(BaseStrategy):
    """Simple Moving Average Crossover Strategy"""
    
    def __init__(self, config: Dict):
        super().__init__("SMA Crossover", config)
        self.short_window = config.get('short_window', 20)
        self.long_window = config.get('long_window', 50)
    
    def get_required_history(self) -> int:
        """Return minimum periods needed"""
        return max(self.short_window, self.long_window) + 10
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate SMA indicators"""
        data_copy = data.copy()
        
        # Calculate moving averages
        data_copy[f'sma_{self.short_window}'] = data_copy['close'].rolling(
            window=self.short_window, min_periods=self.short_window
        ).mean()
        
        data_copy[f'sma_{self.long_window}'] = data_copy['close'].rolling(
            window=self.long_window, min_periods=self.long_window
        ).mean()
        
        # Calculate crossover signals
        data_copy['sma_diff'] = (
            data_copy[f'sma_{self.short_window}'] - data_copy[f'sma_{self.long_window}']
        )
        
        # Previous difference for crossover detection
        data_copy['sma_diff_prev'] = data_copy['sma_diff'].shift(1)
        
        # Detect crossovers
        data_copy['bullish_crossover'] = (
            (data_copy['sma_diff'] > 0) & (data_copy['sma_diff_prev'] <= 0)
        )
        
        data_copy['bearish_crossover'] = (
            (data_copy['sma_diff'] < 0) & (data_copy['sma_diff_prev'] >= 0)
        )
        
        return data_copy
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """Generate trading signals based on SMA crossovers"""
        signals = []
        
        # Get the last few rows to check for recent crossovers
        recent_data = data.tail(5)
        
        for timestamp, row in recent_data.iterrows():
            # Skip if we don't have valid indicator values
            if pd.isna(row[f'sma_{self.short_window}']) or pd.isna(row[f'sma_{self.long_window}']):
                continue
            
            signal_type = SignalType.HOLD
            confidence = 0.0
            metadata = {
                'short_sma': row[f'sma_{self.short_window}'],
                'long_sma': row[f'sma_{self.long_window}'],
                'sma_diff': row['sma_diff']
            }
            
            # Check for bullish crossover (buy signal)
            if row['bullish_crossover']:
                signal_type = SignalType.BUY
                confidence = self._calculate_confidence(row, 'bullish')
                metadata['crossover_type'] = 'bullish'
            
            # Check for bearish crossover (sell signal)
            elif row['bearish_crossover']:
                signal_type = SignalType.SELL
                confidence = self._calculate_confidence(row, 'bearish')
                metadata['crossover_type'] = 'bearish'
            
            # Only add non-HOLD signals
            if signal_type != SignalType.HOLD:
                signal = TradingSignal(
                    symbol="",  # Will be set by base class
                    signal=signal_type,
                    price=row['close'],
                    confidence=confidence,
                    timestamp=timestamp,
                    metadata=metadata
                )
                signals.append(signal)
        
        return signals
    
    def _calculate_confidence(self, row: pd.Series, crossover_type: str) -> float:
        """Calculate confidence score for the signal"""
        # Base confidence
        confidence = 0.6
        
        # Increase confidence based on the magnitude of the difference
        sma_diff_abs = abs(row['sma_diff'])
        price = row['close']
        
        # Normalize difference as percentage of price
        diff_percentage = sma_diff_abs / price
        
        # Add confidence based on the strength of the crossover
        confidence += min(diff_percentage * 100, 0.3)  # Max 0.3 additional confidence
        
        # Consider volume confirmation
        if 'volume' in row.index:
            # Higher volume increases confidence
            volume_sma = row.get('volume_sma_10', row['volume'])
            if row['volume'] > volume_sma:
                confidence += 0.1
        
        # Consider trend strength
        if crossover_type == 'bullish':
            # Check if we're in an uptrend
            if row[f'sma_{self.short_window}'] > row[f'sma_{self.long_window}']:
                confidence += 0.05
        else:  # bearish
            # Check if we're in a downtrend
            if row[f'sma_{self.short_window}'] < row[f'sma_{self.long_window}']:
                confidence += 0.05
        
        return min(confidence, 1.0)  # Cap at 1.0