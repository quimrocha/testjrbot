"""
RSI (Relative Strength Index) Strategy
Generates buy signals when RSI is oversold (< 30)
Generates sell signals when RSI is overbought (> 70)
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from .base_strategy import BaseStrategy, TradingSignal, SignalType


class RSIStrategy(BaseStrategy):
    """RSI Strategy for detecting overbought/oversold conditions"""
    
    def __init__(self, config: Dict):
        super().__init__("RSI Strategy", config)
        self.period = config.get('period', 14)
        self.oversold = config.get('oversold', 30)
        self.overbought = config.get('overbought', 70)
    
    def get_required_history(self) -> int:
        """Return minimum periods needed"""
        return self.period + 20
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI indicator"""
        data_copy = data.copy()
        
        # Calculate price changes
        delta = data_copy['close'].diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gain = gain.rolling(window=self.period, min_periods=self.period).mean()
        avg_loss = loss.rolling(window=self.period, min_periods=self.period).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        data_copy['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate RSI change for trend detection
        data_copy['rsi_prev'] = data_copy['rsi'].shift(1)
        data_copy['rsi_change'] = data_copy['rsi'] - data_copy['rsi_prev']
        
        # Detect oversold/overbought conditions
        data_copy['is_oversold'] = data_copy['rsi'] < self.oversold
        data_copy['is_overbought'] = data_copy['rsi'] > self.overbought
        
        # Detect RSI crossing thresholds
        data_copy['rsi_buy_signal'] = (
            (data_copy['rsi'] > self.oversold) & 
            (data_copy['rsi_prev'] <= self.oversold)
        )
        
        data_copy['rsi_sell_signal'] = (
            (data_copy['rsi'] < self.overbought) & 
            (data_copy['rsi_prev'] >= self.overbought)
        )
        
        return data_copy
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """Generate trading signals based on RSI levels"""
        signals = []
        
        # Get the last few rows to check for recent signals
        recent_data = data.tail(3)
        
        for timestamp, row in recent_data.iterrows():
            # Skip if we don't have valid RSI values
            if pd.isna(row['rsi']):
                continue
            
            signal_type = SignalType.HOLD
            confidence = 0.0
            metadata = {
                'rsi': row['rsi'],
                'rsi_change': row['rsi_change'],
                'oversold_threshold': self.oversold,
                'overbought_threshold': self.overbought
            }
            
            # Check for buy signal (RSI crossing above oversold)
            if row['rsi_buy_signal']:
                signal_type = SignalType.BUY
                confidence = self._calculate_buy_confidence(row)
                metadata['signal_reason'] = 'rsi_oversold_exit'
            
            # Check for sell signal (RSI crossing below overbought)
            elif row['rsi_sell_signal']:
                signal_type = SignalType.SELL
                confidence = self._calculate_sell_confidence(row)
                metadata['signal_reason'] = 'rsi_overbought_exit'
            
            # Alternative: Direct oversold/overbought signals (less reliable)
            elif row['is_oversold'] and row['rsi_change'] > 0:
                # RSI is oversold and starting to turn up
                signal_type = SignalType.BUY
                confidence = self._calculate_buy_confidence(row) * 0.7  # Lower confidence
                metadata['signal_reason'] = 'rsi_oversold_reversal'
            
            elif row['is_overbought'] and row['rsi_change'] < 0:
                # RSI is overbought and starting to turn down
                signal_type = SignalType.SELL
                confidence = self._calculate_sell_confidence(row) * 0.7  # Lower confidence
                metadata['signal_reason'] = 'rsi_overbought_reversal'
            
            # Only add signals with minimum confidence
            if signal_type != SignalType.HOLD and confidence >= 0.3:
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
    
    def _calculate_buy_confidence(self, row: pd.Series) -> float:
        """Calculate confidence for buy signals"""
        confidence = 0.5
        
        rsi = row['rsi']
        
        # Higher confidence for more oversold conditions
        if rsi <= 20:
            confidence += 0.3
        elif rsi <= 25:
            confidence += 0.2
        elif rsi <= self.oversold:
            confidence += 0.1
        
        # Consider RSI momentum
        if row['rsi_change'] > 0:
            confidence += min(row['rsi_change'] / 10, 0.2)
        
        # Volume confirmation
        if 'volume' in row.index:
            volume_sma = row.get('volume_sma_10', row['volume'])
            if row['volume'] > volume_sma * 1.2:  # 20% above average
                confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _calculate_sell_confidence(self, row: pd.Series) -> float:
        """Calculate confidence for sell signals"""
        confidence = 0.5
        
        rsi = row['rsi']
        
        # Higher confidence for more overbought conditions
        if rsi >= 80:
            confidence += 0.3
        elif rsi >= 75:
            confidence += 0.2
        elif rsi >= self.overbought:
            confidence += 0.1
        
        # Consider RSI momentum
        if row['rsi_change'] < 0:
            confidence += min(abs(row['rsi_change']) / 10, 0.2)
        
        # Volume confirmation
        if 'volume' in row.index:
            volume_sma = row.get('volume_sma_10', row['volume'])
            if row['volume'] > volume_sma * 1.2:  # 20% above average
                confidence += 0.1
        
        return min(confidence, 1.0)