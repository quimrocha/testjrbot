"""
MACD (Moving Average Convergence Divergence) Strategy
Generates signals based on MACD line crossovers and histogram analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from .base_strategy import BaseStrategy, TradingSignal, SignalType


class MACDStrategy(BaseStrategy):
    """MACD Strategy for trend following and momentum detection"""
    
    def __init__(self, config: Dict):
        super().__init__("MACD Strategy", config)
        self.fast_period = config.get('fast_period', 12)
        self.slow_period = config.get('slow_period', 26)
        self.signal_period = config.get('signal_period', 9)
    
    def get_required_history(self) -> int:
        """Return minimum periods needed"""
        return self.slow_period + self.signal_period + 10
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD indicators"""
        data_copy = data.copy()
        
        # Calculate exponential moving averages
        ema_fast = data_copy['close'].ewm(span=self.fast_period).mean()
        ema_slow = data_copy['close'].ewm(span=self.slow_period).mean()
        
        # Calculate MACD line
        data_copy['macd_line'] = ema_fast - ema_slow
        
        # Calculate signal line
        data_copy['macd_signal'] = data_copy['macd_line'].ewm(span=self.signal_period).mean()
        
        # Calculate MACD histogram
        data_copy['macd_histogram'] = data_copy['macd_line'] - data_copy['macd_signal']
        
        # Calculate previous values for crossover detection
        data_copy['macd_line_prev'] = data_copy['macd_line'].shift(1)
        data_copy['macd_signal_prev'] = data_copy['macd_signal'].shift(1)
        data_copy['macd_histogram_prev'] = data_copy['macd_histogram'].shift(1)
        
        # Detect MACD line crossovers
        data_copy['macd_bullish_crossover'] = (
            (data_copy['macd_line'] > data_copy['macd_signal']) & 
            (data_copy['macd_line_prev'] <= data_copy['macd_signal_prev'])
        )
        
        data_copy['macd_bearish_crossover'] = (
            (data_copy['macd_line'] < data_copy['macd_signal']) & 
            (data_copy['macd_line_prev'] >= data_copy['macd_signal_prev'])
        )
        
        # Detect zero line crossovers
        data_copy['macd_zero_bullish'] = (
            (data_copy['macd_line'] > 0) & (data_copy['macd_line_prev'] <= 0)
        )
        
        data_copy['macd_zero_bearish'] = (
            (data_copy['macd_line'] < 0) & (data_copy['macd_line_prev'] >= 0)
        )
        
        # Detect histogram divergence
        data_copy['histogram_increasing'] = data_copy['macd_histogram'] > data_copy['macd_histogram_prev']
        data_copy['histogram_decreasing'] = data_copy['macd_histogram'] < data_copy['macd_histogram_prev']
        
        return data_copy
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """Generate trading signals based on MACD analysis"""
        signals = []
        
        # Get the last few rows to check for recent signals
        recent_data = data.tail(3)
        
        for timestamp, row in recent_data.iterrows():
            # Skip if we don't have valid MACD values
            if pd.isna(row['macd_line']) or pd.isna(row['macd_signal']):
                continue
            
            signal_type = SignalType.HOLD
            confidence = 0.0
            metadata = {
                'macd_line': row['macd_line'],
                'macd_signal': row['macd_signal'],
                'macd_histogram': row['macd_histogram']
            }
            
            # Primary signals: MACD line crossovers
            if row['macd_bullish_crossover']:
                signal_type = SignalType.BUY
                confidence = self._calculate_crossover_confidence(row, 'bullish')
                metadata['signal_reason'] = 'macd_bullish_crossover'
            
            elif row['macd_bearish_crossover']:
                signal_type = SignalType.SELL
                confidence = self._calculate_crossover_confidence(row, 'bearish')
                metadata['signal_reason'] = 'macd_bearish_crossover'
            
            # Secondary signals: Zero line crossovers (trend confirmation)
            elif row['macd_zero_bullish']:
                signal_type = SignalType.BUY
                confidence = self._calculate_zero_crossover_confidence(row, 'bullish')
                metadata['signal_reason'] = 'macd_zero_bullish'
            
            elif row['macd_zero_bearish']:
                signal_type = SignalType.SELL
                confidence = self._calculate_zero_crossover_confidence(row, 'bearish')
                metadata['signal_reason'] = 'macd_zero_bearish'
            
            # Histogram divergence signals (early momentum detection)
            elif self._detect_histogram_signal(data, timestamp):
                histogram_signal = self._detect_histogram_signal(data, timestamp)
                if histogram_signal:
                    signal_type = histogram_signal['type']
                    confidence = histogram_signal['confidence']
                    metadata.update(histogram_signal['metadata'])
            
            # Only add signals with minimum confidence
            if signal_type != SignalType.HOLD and confidence >= 0.4:
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
    
    def _calculate_crossover_confidence(self, row: pd.Series, crossover_type: str) -> float:
        """Calculate confidence for MACD crossover signals"""
        confidence = 0.6  # Base confidence for crossovers
        
        # Consider the magnitude of the crossover
        macd_diff = abs(row['macd_line'] - row['macd_signal'])
        price = row['close']
        diff_percentage = macd_diff / price
        
        # Add confidence based on crossover strength
        confidence += min(diff_percentage * 1000, 0.2)
        
        # Consider position relative to zero line
        if crossover_type == 'bullish':
            if row['macd_line'] > 0:
                confidence += 0.1  # Bullish crossover above zero line
        else:  # bearish
            if row['macd_line'] < 0:
                confidence += 0.1  # Bearish crossover below zero line
        
        # Consider histogram momentum
        if crossover_type == 'bullish' and row['histogram_increasing']:
            confidence += 0.05
        elif crossover_type == 'bearish' and row['histogram_decreasing']:
            confidence += 0.05
        
        # Volume confirmation
        if 'volume' in row.index:
            volume_sma = row.get('volume_sma_10', row['volume'])
            if row['volume'] > volume_sma * 1.1:
                confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _calculate_zero_crossover_confidence(self, row: pd.Series, crossover_type: str) -> float:
        """Calculate confidence for zero line crossover signals"""
        confidence = 0.5  # Lower base confidence for zero crossovers
        
        # Consider the strength of the move
        macd_abs = abs(row['macd_line'])
        price = row['close']
        strength = macd_abs / price
        
        confidence += min(strength * 1000, 0.2)
        
        # Consider signal line position
        if crossover_type == 'bullish' and row['macd_signal'] > 0:
            confidence += 0.1
        elif crossover_type == 'bearish' and row['macd_signal'] < 0:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _detect_histogram_signal(self, data: pd.DataFrame, timestamp: pd.Timestamp) -> Dict:
        """Detect early signals from histogram divergence"""
        # Get recent data for divergence analysis
        recent_data = data.loc[:timestamp].tail(5)
        
        if len(recent_data) < 5:
            return None
        
        current_row = recent_data.iloc[-1]
        
        # Look for histogram reversal patterns
        histogram_values = recent_data['macd_histogram'].values
        
        # Check for bullish histogram reversal (bottom formation)
        if (len(histogram_values) >= 3 and 
            histogram_values[-1] > histogram_values[-2] and 
            histogram_values[-2] < histogram_values[-3] and
            histogram_values[-2] < 0):  # Bottom below zero
            
            return {
                'type': SignalType.BUY,
                'confidence': 0.4,
                'metadata': {
                    'signal_reason': 'histogram_bullish_reversal',
                    'histogram_bottom': histogram_values[-2]
                }
            }
        
        # Check for bearish histogram reversal (top formation)
        elif (len(histogram_values) >= 3 and 
              histogram_values[-1] < histogram_values[-2] and 
              histogram_values[-2] > histogram_values[-3] and
              histogram_values[-2] > 0):  # Top above zero
            
            return {
                'type': SignalType.SELL,
                'confidence': 0.4,
                'metadata': {
                    'signal_reason': 'histogram_bearish_reversal',
                    'histogram_top': histogram_values[-2]
                }
            }
        
        return None