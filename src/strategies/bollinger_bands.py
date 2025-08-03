"""
Bollinger Bands Strategy
Generates signals based on price position relative to Bollinger Bands
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from .base_strategy import BaseStrategy, TradingSignal, SignalType


class BollingerBandsStrategy(BaseStrategy):
    """Bollinger Bands Strategy for volatility-based trading"""
    
    def __init__(self, config: Dict):
        super().__init__("Bollinger Bands Strategy", config)
        self.period = config.get('period', 20)
        self.std_dev = config.get('std_dev', 2)
    
    def get_required_history(self) -> int:
        """Return minimum periods needed"""
        return self.period + 10
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands indicators"""
        data_copy = data.copy()
        
        # Calculate moving average (middle band)
        data_copy['bb_middle'] = data_copy['close'].rolling(
            window=self.period, min_periods=self.period
        ).mean()
        
        # Calculate standard deviation
        data_copy['bb_std'] = data_copy['close'].rolling(
            window=self.period, min_periods=self.period
        ).std()
        
        # Calculate upper and lower bands
        data_copy['bb_upper'] = data_copy['bb_middle'] + (self.std_dev * data_copy['bb_std'])
        data_copy['bb_lower'] = data_copy['bb_middle'] - (self.std_dev * data_copy['bb_std'])
        
        # Calculate band width (volatility measure)
        data_copy['bb_width'] = (data_copy['bb_upper'] - data_copy['bb_lower']) / data_copy['bb_middle']
        
        # Calculate %B (position within bands)
        data_copy['bb_percent'] = (
            (data_copy['close'] - data_copy['bb_lower']) / 
            (data_copy['bb_upper'] - data_copy['bb_lower'])
        )
        
        # Previous values for trend detection
        data_copy['bb_percent_prev'] = data_copy['bb_percent'].shift(1)
        data_copy['close_prev'] = data_copy['close'].shift(1)
        
        # Band squeeze detection (low volatility)
        bb_width_sma = data_copy['bb_width'].rolling(window=20).mean()
        data_copy['bb_squeeze'] = data_copy['bb_width'] < bb_width_sma * 0.8
        
        # Band breakout detection
        data_copy['bb_upper_breakout'] = (
            (data_copy['close'] > data_copy['bb_upper']) & 
            (data_copy['close_prev'] <= data_copy['bb_upper'])
        )
        
        data_copy['bb_lower_breakout'] = (
            (data_copy['close'] < data_copy['bb_lower']) & 
            (data_copy['close_prev'] >= data_copy['bb_lower'])
        )
        
        # Mean reversion signals
        data_copy['bb_oversold'] = data_copy['bb_percent'] < 0.2
        data_copy['bb_overbought'] = data_copy['bb_percent'] > 0.8
        
        # Band bounce signals
        data_copy['bb_bounce_up'] = (
            (data_copy['bb_percent'] > 0.2) & 
            (data_copy['bb_percent_prev'] <= 0.2)
        )
        
        data_copy['bb_bounce_down'] = (
            (data_copy['bb_percent'] < 0.8) & 
            (data_copy['bb_percent_prev'] >= 0.8)
        )
        
        return data_copy
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """Generate trading signals based on Bollinger Bands analysis"""
        signals = []
        
        # Get the last few rows to check for recent signals
        recent_data = data.tail(3)
        
        for timestamp, row in recent_data.iterrows():
            # Skip if we don't have valid band values
            if pd.isna(row['bb_upper']) or pd.isna(row['bb_lower']):
                continue
            
            signal_type = SignalType.HOLD
            confidence = 0.0
            metadata = {
                'bb_upper': row['bb_upper'],
                'bb_lower': row['bb_lower'],
                'bb_middle': row['bb_middle'],
                'bb_percent': row['bb_percent'],
                'bb_width': row['bb_width']
            }
            
            # Primary signals: Band breakouts (momentum strategy)
            if row['bb_upper_breakout']:
                signal_type = SignalType.BUY
                confidence = self._calculate_breakout_confidence(row, 'upper')
                metadata['signal_reason'] = 'upper_band_breakout'
            
            elif row['bb_lower_breakout']:
                signal_type = SignalType.SELL
                confidence = self._calculate_breakout_confidence(row, 'lower')
                metadata['signal_reason'] = 'lower_band_breakout'
            
            # Mean reversion signals (contrarian strategy)
            elif row['bb_bounce_up']:
                signal_type = SignalType.BUY
                confidence = self._calculate_bounce_confidence(row, 'up')
                metadata['signal_reason'] = 'lower_band_bounce'
            
            elif row['bb_bounce_down']:
                signal_type = SignalType.SELL
                confidence = self._calculate_bounce_confidence(row, 'down')
                metadata['signal_reason'] = 'upper_band_bounce'
            
            # Squeeze breakout signals (volatility expansion)
            elif self._detect_squeeze_breakout(data, timestamp):
                squeeze_signal = self._detect_squeeze_breakout(data, timestamp)
                if squeeze_signal:
                    signal_type = squeeze_signal['type']
                    confidence = squeeze_signal['confidence']
                    metadata.update(squeeze_signal['metadata'])
            
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
    
    def _calculate_breakout_confidence(self, row: pd.Series, breakout_type: str) -> float:
        """Calculate confidence for band breakout signals"""
        confidence = 0.5  # Base confidence for breakouts
        
        # Consider the magnitude of the breakout
        if breakout_type == 'upper':
            breakout_distance = (row['close'] - row['bb_upper']) / row['bb_upper']
        else:  # lower
            breakout_distance = (row['bb_lower'] - row['close']) / row['bb_lower']
        
        # Add confidence based on breakout strength
        confidence += min(breakout_distance * 10, 0.3)
        
        # Consider band width (volatility)
        if row['bb_width'] > 0.1:  # High volatility increases breakout reliability
            confidence += 0.1
        
        # Volume confirmation
        if 'volume' in row.index:
            volume_sma = row.get('volume_sma_10', row['volume'])
            if row['volume'] > volume_sma * 1.3:  # Strong volume
                confidence += 0.2
        
        # Consider position within bands
        if breakout_type == 'upper' and row['bb_percent'] > 1.0:
            confidence += 0.05
        elif breakout_type == 'lower' and row['bb_percent'] < 0.0:
            confidence += 0.05
        
        return min(confidence, 1.0)
    
    def _calculate_bounce_confidence(self, row: pd.Series, bounce_type: str) -> float:
        """Calculate confidence for band bounce signals (mean reversion)"""
        confidence = 0.4  # Lower base confidence for mean reversion
        
        # Consider how far price was from the band
        if bounce_type == 'up':
            band_distance = abs(row['bb_percent'] - 0.0)  # Distance from lower band
        else:  # down
            band_distance = abs(row['bb_percent'] - 1.0)  # Distance from upper band
        
        # Higher confidence for stronger oversold/overbought conditions
        confidence += min(band_distance * 2, 0.3)
        
        # Consider band width (low volatility favors mean reversion)
        if row['bb_width'] < 0.05:  # Low volatility
            confidence += 0.2
        
        # RSI confirmation (if available)
        if 'rsi' in row.index:
            if bounce_type == 'up' and row['rsi'] < 30:
                confidence += 0.15
            elif bounce_type == 'down' and row['rsi'] > 70:
                confidence += 0.15
        
        return min(confidence, 1.0)
    
    def _detect_squeeze_breakout(self, data: pd.DataFrame, timestamp: pd.Timestamp) -> Dict:
        """Detect breakout from Bollinger Band squeeze"""
        # Get recent data for squeeze analysis
        recent_data = data.loc[:timestamp].tail(10)
        
        if len(recent_data) < 5:
            return None
        
        current_row = recent_data.iloc[-1]
        
        # Check if we recently had a squeeze
        squeeze_count = recent_data['bb_squeeze'].sum()
        
        if squeeze_count < 3:  # Need at least 3 periods of squeeze in last 10
            return None
        
        # Check for volatility expansion
        current_width = current_row['bb_width']
        avg_width = recent_data['bb_width'].mean()
        
        if current_width > avg_width * 1.2:  # 20% increase in volatility
            # Determine direction based on price position and momentum
            price_momentum = recent_data['close'].pct_change().mean()
            
            if price_momentum > 0 and current_row['bb_percent'] > 0.6:
                return {
                    'type': SignalType.BUY,
                    'confidence': 0.6,
                    'metadata': {
                        'signal_reason': 'squeeze_breakout_bullish',
                        'price_momentum': price_momentum,
                        'volatility_expansion': current_width / avg_width
                    }
                }
            elif price_momentum < 0 and current_row['bb_percent'] < 0.4:
                return {
                    'type': SignalType.SELL,
                    'confidence': 0.6,
                    'metadata': {
                        'signal_reason': 'squeeze_breakout_bearish',
                        'price_momentum': price_momentum,
                        'volatility_expansion': current_width / avg_width
                    }
                }
        
        return None