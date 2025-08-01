"""
Strategy Manager
Manages and coordinates multiple trading strategies
"""

import pandas as pd
import yaml
from typing import Dict, List, Optional
from loguru import logger

from .base_strategy import BaseStrategy, TradingSignal, SignalType
from .sma_crossover import SMAcrossoverStrategy
from .rsi_strategy import RSIStrategy
from .macd_strategy import MACDStrategy
from .bollinger_bands import BollingerBandsStrategy


class StrategyManager:
    """Manages multiple trading strategies and their signals"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the strategy manager"""
        self.config = self._load_config(config_path)
        self.strategies: Dict[str, BaseStrategy] = {}
        self._initialize_strategies()
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def _initialize_strategies(self):
        """Initialize all configured strategies"""
        strategy_configs = self.config.get('strategies', {})
        
        # Initialize SMA Crossover Strategy
        if strategy_configs.get('sma_crossover', {}).get('enabled', False):
            self.strategies['sma_crossover'] = SMAcrossoverStrategy(
                strategy_configs['sma_crossover']
            )
            logger.info("Initialized SMA Crossover Strategy")
        
        # Initialize RSI Strategy
        if strategy_configs.get('rsi_strategy', {}).get('enabled', False):
            self.strategies['rsi_strategy'] = RSIStrategy(
                strategy_configs['rsi_strategy']
            )
            logger.info("Initialized RSI Strategy")
        
        # Initialize MACD Strategy
        if strategy_configs.get('macd_strategy', {}).get('enabled', False):
            self.strategies['macd_strategy'] = MACDStrategy(
                strategy_configs['macd_strategy']
            )
            logger.info("Initialized MACD Strategy")
        
        # Initialize Bollinger Bands Strategy
        if strategy_configs.get('bollinger_bands', {}).get('enabled', False):
            self.strategies['bollinger_bands'] = BollingerBandsStrategy(
                strategy_configs['bollinger_bands']
            )
            logger.info("Initialized Bollinger Bands Strategy")
        
        logger.info(f"Initialized {len(self.strategies)} strategies")
    
    def get_signals(self, symbol: str, data: pd.DataFrame) -> List[TradingSignal]:
        """Get trading signals from all enabled strategies"""
        all_signals = []
        
        for strategy_name, strategy in self.strategies.items():
            try:
                signals = strategy.execute(data, symbol)
                
                # Add strategy name to metadata
                for signal in signals:
                    signal.metadata['strategy'] = strategy_name
                
                all_signals.extend(signals)
                
                if signals:
                    logger.info(f"Strategy {strategy_name} generated {len(signals)} signals for {symbol}")
                
            except Exception as e:
                logger.error(f"Error executing strategy {strategy_name} for {symbol}: {e}")
        
        return all_signals
    
    def aggregate_signals(self, signals: List[TradingSignal]) -> Optional[TradingSignal]:
        """Aggregate multiple signals into a single consensus signal"""
        if not signals:
            return None
        
        # Group signals by type
        buy_signals = [s for s in signals if s.signal == SignalType.BUY]
        sell_signals = [s for s in signals if s.signal == SignalType.SELL]
        
        # If we have both buy and sell signals, check which is stronger
        if buy_signals and sell_signals:
            buy_confidence = sum(s.confidence for s in buy_signals) / len(buy_signals)
            sell_confidence = sum(s.confidence for s in sell_signals) / len(sell_signals)
            
            # Take the stronger signal if the difference is significant
            if abs(buy_confidence - sell_confidence) > 0.2:
                if buy_confidence > sell_confidence:
                    return self._create_consensus_signal(buy_signals, SignalType.BUY)
                else:
                    return self._create_consensus_signal(sell_signals, SignalType.SELL)
            else:
                # Conflicting signals with similar confidence - return None (HOLD)
                return None
        
        # Only buy signals
        elif buy_signals:
            return self._create_consensus_signal(buy_signals, SignalType.BUY)
        
        # Only sell signals
        elif sell_signals:
            return self._create_consensus_signal(sell_signals, SignalType.SELL)
        
        return None
    
    def _create_consensus_signal(self, signals: List[TradingSignal], signal_type: SignalType) -> TradingSignal:
        """Create a consensus signal from multiple signals of the same type"""
        # Calculate weighted average confidence
        total_confidence = sum(s.confidence for s in signals)
        avg_confidence = total_confidence / len(signals)
        
        # Use the most recent timestamp
        latest_timestamp = max(s.timestamp for s in signals)
        
        # Use the average price
        avg_price = sum(s.price for s in signals) / len(signals)
        
        # Aggregate metadata
        strategies = [s.metadata.get('strategy', 'unknown') for s in signals]
        reasons = [s.metadata.get('signal_reason', 'unknown') for s in signals]
        
        metadata = {
            'strategies': strategies,
            'signal_reasons': reasons,
            'signal_count': len(signals),
            'individual_confidences': [s.confidence for s in signals],
            'consensus_type': 'aggregated'
        }
        
        return TradingSignal(
            symbol=signals[0].symbol,
            signal=signal_type,
            price=avg_price,
            confidence=min(avg_confidence * (1 + len(signals) * 0.1), 1.0),  # Boost confidence for multiple signals
            timestamp=latest_timestamp,
            metadata=metadata
        )
    
    def get_strategy_performance(self, symbol: str, data: pd.DataFrame, 
                               start_date: Optional[pd.Timestamp] = None,
                               end_date: Optional[pd.Timestamp] = None) -> Dict:
        """Analyze performance of individual strategies"""
        performance = {}
        
        # Filter data by date range if provided
        if start_date or end_date:
            filtered_data = data.copy()
            if start_date:
                filtered_data = filtered_data[filtered_data.index >= start_date]
            if end_date:
                filtered_data = filtered_data[filtered_data.index <= end_date]
        else:
            filtered_data = data
        
        for strategy_name, strategy in self.strategies.items():
            try:
                signals = strategy.execute(filtered_data, symbol)
                
                performance[strategy_name] = {
                    'signal_count': len(signals),
                    'buy_signals': len([s for s in signals if s.signal == SignalType.BUY]),
                    'sell_signals': len([s for s in signals if s.signal == SignalType.SELL]),
                    'avg_confidence': sum(s.confidence for s in signals) / len(signals) if signals else 0,
                    'max_confidence': max(s.confidence for s in signals) if signals else 0,
                    'min_confidence': min(s.confidence for s in signals) if signals else 0,
                    'strategy_info': strategy.get_strategy_info()
                }
                
            except Exception as e:
                logger.error(f"Error analyzing strategy {strategy_name}: {e}")
                performance[strategy_name] = {'error': str(e)}
        
        return performance
    
    def get_enabled_strategies(self) -> Dict[str, BaseStrategy]:
        """Get all enabled strategies"""
        return self.strategies.copy()
    
    def enable_strategy(self, strategy_name: str):
        """Enable a specific strategy"""
        if strategy_name in self.strategies:
            self.strategies[strategy_name].enabled = True
            logger.info(f"Enabled strategy: {strategy_name}")
        else:
            logger.warning(f"Strategy not found: {strategy_name}")
    
    def disable_strategy(self, strategy_name: str):
        """Disable a specific strategy"""
        if strategy_name in self.strategies:
            self.strategies[strategy_name].enabled = False
            logger.info(f"Disabled strategy: {strategy_name}")
        else:
            logger.warning(f"Strategy not found: {strategy_name}")
    
    def get_strategy_status(self) -> Dict:
        """Get status of all strategies"""
        status = {}
        for name, strategy in self.strategies.items():
            status[name] = {
                'enabled': strategy.enabled,
                'name': strategy.name,
                'config': strategy.config,
                'required_history': strategy.get_required_history()
            }
        return status
    
    def update_strategy_config(self, strategy_name: str, new_config: Dict):
        """Update configuration for a specific strategy"""
        if strategy_name in self.strategies:
            # Recreate the strategy with new config
            strategy_class = type(self.strategies[strategy_name])
            self.strategies[strategy_name] = strategy_class(new_config)
            logger.info(f"Updated configuration for strategy: {strategy_name}")
        else:
            logger.warning(f"Strategy not found: {strategy_name}")
    
    def get_max_required_history(self) -> int:
        """Get the maximum history required by any enabled strategy"""
        if not self.strategies:
            return 100  # Default
        
        return max(
            strategy.get_required_history() 
            for strategy in self.strategies.values() 
            if strategy.enabled
        )