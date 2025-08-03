"""
Backtesting Engine
Comprehensive backtesting framework for trading strategies
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from loguru import logger
import sqlite3
import os

from ..strategies.base_strategy import TradingSignal, SignalType
from ..strategies.strategy_manager import StrategyManager
from ..data.market_data import MarketDataManager


@dataclass
class Trade:
    """Represents a completed trade"""
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    timestamp: pd.Timestamp
    commission: float
    signal_confidence: float
    strategy: str
    trade_id: str = None
    
    def __post_init__(self):
        if self.trade_id is None:
            self.trade_id = f"{self.symbol}_{self.side}_{self.timestamp.strftime('%Y%m%d_%H%M%S')}"


@dataclass
class Position:
    """Represents a current position"""
    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    timestamp: pd.Timestamp
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price
    
    @property
    def total_pnl(self) -> float:
        return self.unrealized_pnl + self.realized_pnl


@dataclass
class BacktestResult:
    """Comprehensive backtest results"""
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    trades: List[Trade]
    equity_curve: pd.DataFrame
    monthly_returns: pd.DataFrame
    statistics: Dict[str, Any]


class BacktestEngine:
    """Comprehensive backtesting engine"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the backtest engine"""
        self.strategy_manager = StrategyManager(config_path)
        self.data_manager = MarketDataManager(config_path)
        
        # Load config
        import yaml
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Backtest settings
        self.initial_capital = self.config['backtest']['initial_capital']
        self.commission = self.config['backtest']['commission']
        
        # Initialize state
        self.reset()
    
    def reset(self):
        """Reset backtest state"""
        self.cash = self.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_history: List[Tuple[pd.Timestamp, float]] = []
        self.current_timestamp = None
        
    def run_backtest(self, 
                    symbols: List[str],
                    start_date: str,
                    end_date: str,
                    timeframe: str = '1h',
                    initial_capital: Optional[float] = None) -> BacktestResult:
        """Run comprehensive backtest"""
        
        if initial_capital:
            self.initial_capital = initial_capital
            self.cash = initial_capital
        
        logger.info(f"Starting backtest for {symbols} from {start_date} to {end_date}")
        
        # Convert dates
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Get market data for all symbols
        all_data = {}
        for symbol in symbols:
            data = self.data_manager.get_market_data(
                symbol, timeframe, start_dt, end_dt
            )
            if data.empty:
                logger.warning(f"No data available for {symbol}")
                continue
            all_data[symbol] = data
        
        if not all_data:
            raise ValueError("No market data available for any symbol")
        
        # Get common time index
        common_index = self._get_common_time_index(all_data)
        
        # Run backtest simulation
        self._simulate_trading(all_data, common_index, symbols)
        
        # Calculate results
        result = self._calculate_results(start_dt, end_dt)
        
        logger.info(f"Backtest completed. Total return: {result.total_return:.2%}")
        
        return result
    
    def _get_common_time_index(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
        """Get common time index for all symbols"""
        indices = [data.index for data in data_dict.values()]
        
        # Find intersection of all indices
        common_index = indices[0]
        for idx in indices[1:]:
            common_index = common_index.intersection(idx)
        
        return common_index.sort_values()
    
    def _simulate_trading(self, all_data: Dict[str, pd.DataFrame], 
                         time_index: pd.DatetimeIndex, symbols: List[str]):
        """Simulate trading over the time period"""
        
        for timestamp in time_index:
            self.current_timestamp = timestamp
            
            # Update positions with current prices
            self._update_positions(all_data, timestamp)
            
            # Record equity
            total_equity = self._calculate_total_equity()
            self.equity_history.append((timestamp, total_equity))
            
            # Generate signals for all symbols
            for symbol in symbols:
                if symbol not in all_data:
                    continue
                
                # Get historical data up to current timestamp
                historical_data = all_data[symbol].loc[:timestamp]
                
                if len(historical_data) < self.strategy_manager.get_max_required_history():
                    continue
                
                # Get signals from strategy manager
                signals = self.strategy_manager.get_signals(symbol, historical_data)
                
                if signals:
                    # Aggregate signals into consensus
                    consensus_signal = self.strategy_manager.aggregate_signals(signals)
                    
                    if consensus_signal:
                        self._execute_signal(consensus_signal, timestamp)
    
    def _update_positions(self, all_data: Dict[str, pd.DataFrame], timestamp: pd.Timestamp):
        """Update position values with current market prices"""
        for symbol, position in self.positions.items():
            if symbol in all_data and timestamp in all_data[symbol].index:
                current_price = all_data[symbol].loc[timestamp, 'close']
                position.current_price = current_price
                position.unrealized_pnl = (current_price - position.avg_price) * position.quantity
    
    def _execute_signal(self, signal: TradingSignal, timestamp: pd.Timestamp):
        """Execute a trading signal"""
        symbol = signal.symbol
        price = signal.price
        
        # Calculate position size based on signal confidence and available capital
        position_size = self._calculate_position_size(signal)
        
        if signal.signal == SignalType.BUY:
            self._execute_buy(symbol, position_size, price, timestamp, signal)
        elif signal.signal == SignalType.SELL:
            self._execute_sell(symbol, position_size, price, timestamp, signal)
    
    def _calculate_position_size(self, signal: TradingSignal) -> float:
        """Calculate position size based on signal and risk management"""
        # Base allocation from config
        max_allocation = self.config['trading']['max_allocation_per_trade']
        
        # Adjust based on signal confidence
        confidence_multiplier = signal.confidence
        
        # Calculate available capital
        total_equity = self._calculate_total_equity()
        available_capital = total_equity * max_allocation * confidence_multiplier
        
        # Calculate quantity
        quantity = available_capital / signal.price
        
        # Apply minimum trade amount
        min_trade_amount = self.config['trading']['min_trade_amount']
        if available_capital < min_trade_amount:
            return 0
        
        return quantity
    
    def _execute_buy(self, symbol: str, quantity: float, price: float, 
                    timestamp: pd.Timestamp, signal: TradingSignal):
        """Execute buy order"""
        if quantity <= 0:
            return
        
        cost = quantity * price
        commission_cost = cost * self.commission
        total_cost = cost + commission_cost
        
        if total_cost > self.cash:
            # Insufficient funds - adjust quantity
            available_cash = self.cash
            max_quantity = (available_cash / (1 + self.commission)) / price
            if max_quantity < quantity * 0.1:  # Don't trade if less than 10% of intended size
                return
            quantity = max_quantity
            cost = quantity * price
            commission_cost = cost * self.commission
            total_cost = cost + commission_cost
        
        # Update cash
        self.cash -= total_cost
        
        # Update or create position
        if symbol in self.positions:
            position = self.positions[symbol]
            new_quantity = position.quantity + quantity
            new_avg_price = ((position.quantity * position.avg_price) + cost) / new_quantity
            position.quantity = new_quantity
            position.avg_price = new_avg_price
            position.current_price = price
            position.timestamp = timestamp
        else:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                avg_price=price,
                current_price=price,
                unrealized_pnl=0,
                realized_pnl=0,
                timestamp=timestamp
            )
        
        # Record trade
        trade = Trade(
            symbol=symbol,
            side='buy',
            quantity=quantity,
            price=price,
            timestamp=timestamp,
            commission=commission_cost,
            signal_confidence=signal.confidence,
            strategy=signal.metadata.get('strategy', 'unknown')
        )
        self.trades.append(trade)
        
        logger.debug(f"BUY {quantity:.6f} {symbol} at {price:.6f} (Commission: {commission_cost:.2f})")
    
    def _execute_sell(self, symbol: str, quantity: float, price: float, 
                     timestamp: pd.Timestamp, signal: TradingSignal):
        """Execute sell order"""
        if symbol not in self.positions or quantity <= 0:
            return
        
        position = self.positions[symbol]
        
        # Limit quantity to available position
        sell_quantity = min(quantity, position.quantity)
        
        if sell_quantity <= 0:
            return
        
        proceeds = sell_quantity * price
        commission_cost = proceeds * self.commission
        net_proceeds = proceeds - commission_cost
        
        # Calculate realized P&L
        cost_basis = sell_quantity * position.avg_price
        realized_pnl = net_proceeds - cost_basis
        
        # Update cash
        self.cash += net_proceeds
        
        # Update position
        position.quantity -= sell_quantity
        position.realized_pnl += realized_pnl
        position.current_price = price
        position.timestamp = timestamp
        
        # Remove position if fully closed
        if position.quantity <= 1e-8:  # Close to zero
            del self.positions[symbol]
        
        # Record trade
        trade = Trade(
            symbol=symbol,
            side='sell',
            quantity=sell_quantity,
            price=price,
            timestamp=timestamp,
            commission=commission_cost,
            signal_confidence=signal.confidence,
            strategy=signal.metadata.get('strategy', 'unknown')
        )
        self.trades.append(trade)
        
        logger.debug(f"SELL {sell_quantity:.6f} {symbol} at {price:.6f} (P&L: {realized_pnl:.2f})")
    
    def _calculate_total_equity(self) -> float:
        """Calculate total portfolio equity"""
        equity = self.cash
        
        for position in self.positions.values():
            equity += position.market_value
        
        return equity
    
    def _calculate_results(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> BacktestResult:
        """Calculate comprehensive backtest results"""
        
        # Basic metrics
        final_capital = self._calculate_total_equity()
        total_return = (final_capital - self.initial_capital) / self.initial_capital
        
        # Convert equity history to DataFrame
        equity_df = pd.DataFrame(self.equity_history, columns=['timestamp', 'equity'])
        equity_df.set_index('timestamp', inplace=True)
        
        # Calculate returns
        equity_df['returns'] = equity_df['equity'].pct_change()
        equity_df['cumulative_returns'] = (1 + equity_df['returns']).cumprod() - 1
        
        # Time-based metrics
        days = (end_date - start_date).days
        years = days / 365.25
        annualized_return = (final_capital / self.initial_capital) ** (1/years) - 1 if years > 0 else 0
        
        # Risk metrics
        returns = equity_df['returns'].dropna()
        
        if len(returns) > 1:
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            max_drawdown, max_dd_duration = self._calculate_max_drawdown(equity_df['equity'])
        else:
            sharpe_ratio = 0
            max_drawdown = 0
            max_dd_duration = 0
        
        # Trade analysis
        trade_analysis = self._analyze_trades()
        
        # Monthly returns
        monthly_returns = self._calculate_monthly_returns(equity_df)
        
        # Additional statistics
        statistics = self._calculate_additional_statistics(equity_df, trade_analysis)
        
        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_dd_duration,
            total_trades=trade_analysis['total_trades'],
            winning_trades=trade_analysis['winning_trades'],
            losing_trades=trade_analysis['losing_trades'],
            win_rate=trade_analysis['win_rate'],
            avg_win=trade_analysis['avg_win'],
            avg_loss=trade_analysis['avg_loss'],
            profit_factor=trade_analysis['profit_factor'],
            trades=self.trades,
            equity_curve=equity_df,
            monthly_returns=monthly_returns,
            statistics=statistics
        )
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if returns.std() == 0:
            return 0
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return np.sqrt(252) * excess_returns.mean() / returns.std()
    
    def _calculate_max_drawdown(self, equity: pd.Series) -> Tuple[float, int]:
        """Calculate maximum drawdown and duration"""
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max
        
        max_drawdown = drawdown.min()
        
        # Calculate max drawdown duration
        in_drawdown = drawdown < 0
        drawdown_periods = []
        current_period = 0
        
        for is_down in in_drawdown:
            if is_down:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        
        if current_period > 0:
            drawdown_periods.append(current_period)
        
        max_dd_duration = max(drawdown_periods) if drawdown_periods else 0
        
        return max_drawdown, max_dd_duration
    
    def _analyze_trades(self) -> Dict:
        """Analyze completed trades"""
        if not self.trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0
            }
        
        # Group trades by symbol to calculate P&L
        trade_pnl = []
        
        # For simplicity, calculate P&L on matched buy/sell pairs
        # This is a simplified approach - in reality would need more sophisticated matching
        
        # Calculate individual trade P&L (simplified)
        buys = [t for t in self.trades if t.side == 'buy']
        sells = [t for t in self.trades if t.side == 'sell']
        
        # Match trades (simplified FIFO)
        for sell in sells:
            for buy in buys:
                if buy.symbol == sell.symbol:
                    pnl = (sell.price - buy.price) * min(buy.quantity, sell.quantity) - buy.commission - sell.commission
                    trade_pnl.append(pnl)
                    break
        
        if not trade_pnl:
            return {
                'total_trades': len(self.trades),
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0
            }
        
        winning_trades = [pnl for pnl in trade_pnl if pnl > 0]
        losing_trades = [pnl for pnl in trade_pnl if pnl < 0]
        
        total_trades = len(trade_pnl)
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        
        total_wins = sum(winning_trades) if winning_trades else 0
        total_losses = abs(sum(losing_trades)) if losing_trades else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf') if total_wins > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': win_count,
            'losing_trades': loss_count,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }
    
    def _calculate_monthly_returns(self, equity_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate monthly returns"""
        monthly = equity_df['equity'].resample('M').last()
        monthly_returns = monthly.pct_change().dropna()
        
        return pd.DataFrame({
            'month': monthly_returns.index,
            'return': monthly_returns.values
        })
    
    def _calculate_additional_statistics(self, equity_df: pd.DataFrame, trade_analysis: Dict) -> Dict:
        """Calculate additional performance statistics"""
        returns = equity_df['returns'].dropna()
        
        statistics = {
            'total_return_pct': (equity_df['equity'].iloc[-1] / equity_df['equity'].iloc[0] - 1) * 100,
            'volatility': returns.std() * np.sqrt(252) if len(returns) > 1 else 0,
            'calmar_ratio': 0,  # Will calculate if max drawdown > 0
            'sortino_ratio': 0,
            'skewness': returns.skew() if len(returns) > 2 else 0,
            'kurtosis': returns.kurtosis() if len(returns) > 3 else 0,
            'var_95': returns.quantile(0.05) if len(returns) > 0 else 0,
            'cvar_95': returns[returns <= returns.quantile(0.05)].mean() if len(returns) > 0 else 0,
        }
        
        # Calmar ratio (annualized return / max drawdown)
        if 'max_drawdown' in locals() and abs(statistics.get('max_drawdown', 0)) > 0:
            statistics['calmar_ratio'] = abs(statistics['total_return_pct'] / 100) / abs(statistics.get('max_drawdown', 1))
        
        # Sortino ratio (excess return / downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 1:
            downside_std = downside_returns.std() * np.sqrt(252)
            statistics['sortino_ratio'] = (returns.mean() * 252) / downside_std
        
        return statistics