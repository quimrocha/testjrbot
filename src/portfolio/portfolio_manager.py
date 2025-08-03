"""
Portfolio Manager
Handles portfolio management, risk management, and position sizing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
import yaml
import sqlite3
import os

from ..strategies.base_strategy import TradingSignal, SignalType


@dataclass
class PortfolioPosition:
    """Represents a portfolio position"""
    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    weight: float
    last_updated: datetime
    
    @property
    def cost_basis(self) -> float:
        return self.quantity * self.avg_price


@dataclass
class RiskMetrics:
    """Risk management metrics"""
    portfolio_value: float
    daily_pnl: float
    daily_pnl_pct: float
    max_daily_loss_limit: float
    current_drawdown: float
    max_drawdown: float
    var_95: float  # Value at Risk 95%
    positions_at_risk: List[str]
    concentration_risk: Dict[str, float]


class PortfolioManager:
    """Manages portfolio positions and risk"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the portfolio manager"""
        self.config = self._load_config(config_path)
        self.db_path = self.config['database']['path']
        
        # Portfolio settings
        self.max_positions = self.config['trading']['max_positions']
        self.max_allocation_per_trade = self.config['trading']['max_allocation_per_trade']
        self.min_trade_amount = self.config['trading']['min_trade_amount']
        
        # Risk management settings
        self.stop_loss = self.config['risk']['stop_loss']
        self.take_profit = self.config['risk']['take_profit']
        self.max_daily_loss = self.config['risk']['max_daily_loss']
        self.position_sizing_method = self.config['risk']['position_sizing']
        
        # Initialize database
        self._init_database()
        
        # Portfolio state
        self.positions: Dict[str, PortfolioPosition] = {}
        self.cash_balance = 0.0
        self.portfolio_value = 0.0
        self.daily_start_value = 0.0
        
        # Load existing positions
        self._load_positions()
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def _init_database(self):
        """Initialize portfolio database tables"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # Positions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    quantity REAL NOT NULL,
                    avg_price REAL NOT NULL,
                    current_price REAL NOT NULL,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Portfolio history table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    portfolio_value REAL NOT NULL,
                    cash_balance REAL NOT NULL,
                    daily_pnl REAL,
                    daily_pnl_pct REAL,
                    num_positions INTEGER
                )
            """)
            
            # Risk events table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS risk_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    event_type TEXT NOT NULL,
                    symbol TEXT,
                    description TEXT,
                    severity TEXT,
                    action_taken TEXT
                )
            """)
            
            logger.info("Portfolio database initialized")
    
    def _load_positions(self):
        """Load existing positions from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT symbol, quantity, avg_price, current_price, last_updated
                    FROM positions
                """)
                
                for row in cursor.fetchall():
                    symbol, quantity, avg_price, current_price, last_updated = row
                    
                    position = PortfolioPosition(
                        symbol=symbol,
                        quantity=quantity,
                        avg_price=avg_price,
                        current_price=current_price,
                        market_value=quantity * current_price,
                        unrealized_pnl=(current_price - avg_price) * quantity,
                        unrealized_pnl_pct=((current_price - avg_price) / avg_price) if avg_price > 0 else 0,
                        weight=0.0,  # Will be calculated
                        last_updated=datetime.fromisoformat(last_updated)
                    )
                    
                    self.positions[symbol] = position
                
                logger.info(f"Loaded {len(self.positions)} positions from database")
                
        except Exception as e:
            logger.error(f"Error loading positions: {e}")
    
    def update_position_prices(self, price_data: Dict[str, float]):
        """Update current prices for all positions"""
        for symbol, position in self.positions.items():
            if symbol in price_data:
                old_price = position.current_price
                new_price = price_data[symbol]
                
                position.current_price = new_price
                position.market_value = position.quantity * new_price
                position.unrealized_pnl = (new_price - position.avg_price) * position.quantity
                position.unrealized_pnl_pct = ((new_price - position.avg_price) / position.avg_price) if position.avg_price > 0 else 0
                position.last_updated = datetime.now()
                
                # Update in database
                self._update_position_in_db(position)
                
                if abs(new_price - old_price) / old_price > 0.05:  # 5% price change
                    logger.info(f"Significant price change for {symbol}: {old_price:.6f} -> {new_price:.6f}")
        
        # Recalculate portfolio metrics
        self._calculate_portfolio_metrics()
    
    def _update_position_in_db(self, position: PortfolioPosition):
        """Update position in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE positions 
                    SET current_price = ?, last_updated = ?
                    WHERE symbol = ?
                """, (position.current_price, position.last_updated.isoformat(), position.symbol))
                
        except Exception as e:
            logger.error(f"Error updating position in database: {e}")
    
    def _calculate_portfolio_metrics(self):
        """Calculate portfolio-level metrics"""
        total_market_value = sum(pos.market_value for pos in self.positions.values())
        self.portfolio_value = self.cash_balance + total_market_value
        
        # Calculate position weights
        for position in self.positions.values():
            position.weight = position.market_value / self.portfolio_value if self.portfolio_value > 0 else 0
    
    def evaluate_signal(self, signal: TradingSignal, current_price: float) -> Tuple[bool, float, str]:
        """Evaluate if a signal should be executed and determine position size"""
        
        # Check if we can add more positions
        if signal.signal == SignalType.BUY and len(self.positions) >= self.max_positions:
            existing_position = self.positions.get(signal.symbol)
            if not existing_position:
                return False, 0, "Maximum positions reached"
        
        # Check daily loss limit
        risk_metrics = self.calculate_risk_metrics()
        if risk_metrics.daily_pnl < -abs(risk_metrics.max_daily_loss_limit):
            return False, 0, "Daily loss limit exceeded"
        
        # Calculate position size
        position_size = self._calculate_position_size(signal, current_price)
        
        if position_size <= 0:
            return False, 0, "Position size too small"
        
        # Check minimum trade amount
        trade_value = position_size * current_price
        if trade_value < self.min_trade_amount:
            return False, 0, "Trade value below minimum"
        
        # Additional risk checks for sell signals
        if signal.signal == SignalType.SELL:
            if signal.symbol not in self.positions:
                return False, 0, "No position to sell"
            
            available_quantity = self.positions[signal.symbol].quantity
            if position_size > available_quantity:
                position_size = available_quantity
        
        return True, position_size, "Signal approved"
    
    def _calculate_position_size(self, signal: TradingSignal, current_price: float) -> float:
        """Calculate position size based on risk management rules"""
        
        if self.position_sizing_method == "fixed":
            return self._fixed_position_sizing(signal, current_price)
        elif self.position_sizing_method == "volatility":
            return self._volatility_position_sizing(signal, current_price)
        elif self.position_sizing_method == "kelly":
            return self._kelly_position_sizing(signal, current_price)
        else:
            return self._fixed_position_sizing(signal, current_price)
    
    def _fixed_position_sizing(self, signal: TradingSignal, current_price: float) -> float:
        """Fixed percentage position sizing"""
        allocation = self.max_allocation_per_trade * signal.confidence
        available_capital = self.portfolio_value * allocation
        
        if signal.signal == SignalType.BUY:
            return available_capital / current_price
        else:  # SELL
            if signal.symbol in self.positions:
                return self.positions[signal.symbol].quantity * allocation
            return 0
    
    def _volatility_position_sizing(self, signal: TradingSignal, current_price: float) -> float:
        """Volatility-based position sizing"""
        # This would require historical volatility calculation
        # For now, use fixed sizing with volatility adjustment
        base_size = self._fixed_position_sizing(signal, current_price)
        
        # Reduce size for higher volatility (placeholder)
        volatility_adjustment = 1.0  # Would calculate actual volatility
        
        return base_size * volatility_adjustment
    
    def _kelly_position_sizing(self, signal: TradingSignal, current_price: float) -> float:
        """Kelly criterion position sizing"""
        # Simplified Kelly criterion implementation
        # f = (bp - q) / b
        # where f = fraction of capital to bet
        # b = odds (profit/loss ratio)
        # p = probability of win
        # q = probability of loss (1-p)
        
        win_prob = signal.confidence
        loss_prob = 1 - win_prob
        
        # Use historical profit/loss ratio (placeholder)
        profit_loss_ratio = 1.5  # Would calculate from historical data
        
        kelly_fraction = (profit_loss_ratio * win_prob - loss_prob) / profit_loss_ratio
        kelly_fraction = max(0, min(kelly_fraction, self.max_allocation_per_trade))  # Cap at max allocation
        
        available_capital = self.portfolio_value * kelly_fraction
        
        if signal.signal == SignalType.BUY:
            return available_capital / current_price
        else:  # SELL
            if signal.symbol in self.positions:
                return self.positions[signal.symbol].quantity * kelly_fraction
            return 0
    
    def execute_trade(self, signal: TradingSignal, quantity: float, execution_price: float) -> bool:
        """Execute a trade and update portfolio"""
        
        try:
            if signal.signal == SignalType.BUY:
                self._execute_buy(signal.symbol, quantity, execution_price)
            elif signal.signal == SignalType.SELL:
                self._execute_sell(signal.symbol, quantity, execution_price)
            
            # Update portfolio metrics
            self._calculate_portfolio_metrics()
            
            # Record portfolio state
            self._record_portfolio_state()
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False
    
    def _execute_buy(self, symbol: str, quantity: float, price: float):
        """Execute buy order"""
        cost = quantity * price
        commission = cost * 0.001  # 0.1% commission
        total_cost = cost + commission
        
        if total_cost > self.cash_balance:
            raise ValueError("Insufficient cash balance")
        
        self.cash_balance -= total_cost
        
        # Update or create position
        if symbol in self.positions:
            position = self.positions[symbol]
            new_quantity = position.quantity + quantity
            new_avg_price = ((position.quantity * position.avg_price) + cost) / new_quantity
            
            position.quantity = new_quantity
            position.avg_price = new_avg_price
            position.current_price = price
            position.market_value = new_quantity * price
            position.unrealized_pnl = (price - new_avg_price) * new_quantity
            position.unrealized_pnl_pct = ((price - new_avg_price) / new_avg_price) if new_avg_price > 0 else 0
            position.last_updated = datetime.now()
        else:
            self.positions[symbol] = PortfolioPosition(
                symbol=symbol,
                quantity=quantity,
                avg_price=price,
                current_price=price,
                market_value=quantity * price,
                unrealized_pnl=0,
                unrealized_pnl_pct=0,
                weight=0,
                last_updated=datetime.now()
            )
        
        # Update database
        self._save_position_to_db(self.positions[symbol])
        
        logger.info(f"BUY: {quantity:.6f} {symbol} at {price:.6f}")
    
    def _execute_sell(self, symbol: str, quantity: float, price: float):
        """Execute sell order"""
        if symbol not in self.positions:
            raise ValueError(f"No position in {symbol}")
        
        position = self.positions[symbol]
        if quantity > position.quantity:
            quantity = position.quantity
        
        proceeds = quantity * price
        commission = proceeds * 0.001  # 0.1% commission
        net_proceeds = proceeds - commission
        
        self.cash_balance += net_proceeds
        
        # Update position
        position.quantity -= quantity
        position.current_price = price
        position.market_value = position.quantity * price
        
        if position.quantity > 1e-8:  # Still have position
            position.unrealized_pnl = (price - position.avg_price) * position.quantity
            position.unrealized_pnl_pct = ((price - position.avg_price) / position.avg_price) if position.avg_price > 0 else 0
            position.last_updated = datetime.now()
            
            # Update database
            self._save_position_to_db(position)
        else:
            # Position closed
            self._remove_position_from_db(symbol)
            del self.positions[symbol]
        
        logger.info(f"SELL: {quantity:.6f} {symbol} at {price:.6f}")
    
    def _save_position_to_db(self, position: PortfolioPosition):
        """Save position to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO positions 
                    (symbol, quantity, avg_price, current_price, last_updated)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    position.symbol,
                    position.quantity,
                    position.avg_price,
                    position.current_price,
                    position.last_updated.isoformat()
                ))
                
        except Exception as e:
            logger.error(f"Error saving position to database: {e}")
    
    def _remove_position_from_db(self, symbol: str):
        """Remove position from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
                
        except Exception as e:
            logger.error(f"Error removing position from database: {e}")
    
    def _record_portfolio_state(self):
        """Record current portfolio state to database"""
        try:
            daily_pnl = self.portfolio_value - self.daily_start_value
            daily_pnl_pct = (daily_pnl / self.daily_start_value) if self.daily_start_value > 0 else 0
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO portfolio_history 
                    (portfolio_value, cash_balance, daily_pnl, daily_pnl_pct, num_positions)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    self.portfolio_value,
                    self.cash_balance,
                    daily_pnl,
                    daily_pnl_pct,
                    len(self.positions)
                ))
                
        except Exception as e:
            logger.error(f"Error recording portfolio state: {e}")
    
    def calculate_risk_metrics(self) -> RiskMetrics:
        """Calculate current risk metrics"""
        daily_pnl = self.portfolio_value - self.daily_start_value
        daily_pnl_pct = (daily_pnl / self.daily_start_value) if self.daily_start_value > 0 else 0
        
        max_daily_loss_limit = self.portfolio_value * self.max_daily_loss
        
        # Identify positions at risk (large unrealized losses)
        positions_at_risk = []
        for symbol, position in self.positions.items():
            if position.unrealized_pnl_pct < -self.stop_loss:
                positions_at_risk.append(symbol)
        
        # Calculate concentration risk
        concentration_risk = {}
        for symbol, position in self.positions.items():
            concentration_risk[symbol] = position.weight
        
        # Placeholder for VaR calculation
        var_95 = self.portfolio_value * 0.05  # Simplified 5% VaR
        
        return RiskMetrics(
            portfolio_value=self.portfolio_value,
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            max_daily_loss_limit=max_daily_loss_limit,
            current_drawdown=0,  # Would calculate from equity curve
            max_drawdown=0,      # Would calculate from historical data
            var_95=var_95,
            positions_at_risk=positions_at_risk,
            concentration_risk=concentration_risk
        )
    
    def check_stop_loss_take_profit(self) -> List[TradingSignal]:
        """Check for stop loss and take profit triggers"""
        signals = []
        
        for symbol, position in self.positions.items():
            # Check stop loss
            if position.unrealized_pnl_pct <= -self.stop_loss:
                signal = TradingSignal(
                    symbol=symbol,
                    signal=SignalType.SELL,
                    price=position.current_price,
                    confidence=1.0,
                    timestamp=pd.Timestamp.now(),
                    metadata={
                        'reason': 'stop_loss',
                        'pnl_pct': position.unrealized_pnl_pct,
                        'stop_loss_threshold': -self.stop_loss
                    }
                )
                signals.append(signal)
                
                # Log risk event
                self._log_risk_event("STOP_LOSS", symbol, 
                                   f"Stop loss triggered at {position.unrealized_pnl_pct:.2%}", 
                                   "HIGH", "Sell signal generated")
            
            # Check take profit
            elif position.unrealized_pnl_pct >= self.take_profit:
                signal = TradingSignal(
                    symbol=symbol,
                    signal=SignalType.SELL,
                    price=position.current_price,
                    confidence=0.8,
                    timestamp=pd.Timestamp.now(),
                    metadata={
                        'reason': 'take_profit',
                        'pnl_pct': position.unrealized_pnl_pct,
                        'take_profit_threshold': self.take_profit
                    }
                )
                signals.append(signal)
                
                logger.info(f"Take profit triggered for {symbol} at {position.unrealized_pnl_pct:.2%}")
        
        return signals
    
    def _log_risk_event(self, event_type: str, symbol: str, description: str, 
                       severity: str, action_taken: str):
        """Log risk event to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO risk_events 
                    (event_type, symbol, description, severity, action_taken)
                    VALUES (?, ?, ?, ?, ?)
                """, (event_type, symbol, description, severity, action_taken))
                
        except Exception as e:
            logger.error(f"Error logging risk event: {e}")
    
    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio summary"""
        self._calculate_portfolio_metrics()
        risk_metrics = self.calculate_risk_metrics()
        
        position_summaries = []
        for symbol, position in self.positions.items():
            position_summaries.append({
                'symbol': symbol,
                'quantity': position.quantity,
                'avg_price': position.avg_price,
                'current_price': position.current_price,
                'market_value': position.market_value,
                'unrealized_pnl': position.unrealized_pnl,
                'unrealized_pnl_pct': position.unrealized_pnl_pct,
                'weight': position.weight,
                'cost_basis': position.cost_basis
            })
        
        return {
            'cash_balance': self.cash_balance,
            'portfolio_value': self.portfolio_value,
            'num_positions': len(self.positions),
            'daily_pnl': risk_metrics.daily_pnl,
            'daily_pnl_pct': risk_metrics.daily_pnl_pct,
            'positions': position_summaries,
            'risk_metrics': {
                'max_daily_loss_limit': risk_metrics.max_daily_loss_limit,
                'positions_at_risk': risk_metrics.positions_at_risk,
                'concentration_risk': risk_metrics.concentration_risk,
                'var_95': risk_metrics.var_95
            }
        }
    
    def set_daily_start_value(self, value: float):
        """Set the portfolio value at start of day for P&L calculation"""
        self.daily_start_value = value
    
    def initialize_portfolio(self, initial_cash: float):
        """Initialize portfolio with starting cash"""
        self.cash_balance = initial_cash
        self.portfolio_value = initial_cash
        self.daily_start_value = initial_cash
        logger.info(f"Initialized portfolio with ${initial_cash:,.2f}")