"""
Cryptocurrency Trading Bot
Main orchestrator that coordinates all components
"""

import asyncio
import schedule
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from loguru import logger
import yaml
import os
from dotenv import load_dotenv

from ..data.market_data import MarketDataManager
from ..strategies.strategy_manager import StrategyManager
from ..portfolio.portfolio_manager import PortfolioManager
from ..backtesting.backtest_engine import BacktestEngine
from ..strategies.base_strategy import TradingSignal, SignalType


class CryptoTradingBot:
    """Main cryptocurrency trading bot"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the trading bot"""
        load_dotenv()  # Load environment variables
        
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize components
        self.data_manager = MarketDataManager(config_path)
        self.strategy_manager = StrategyManager(config_path)
        self.portfolio_manager = PortfolioManager(config_path)
        self.backtest_engine = BacktestEngine(config_path)
        
        # Bot state
        self.is_running = False
        self.is_live_trading = False
        self.symbols = self.config['data']['symbols']
        self.timeframes = self.config['data']['timeframes']
        self.update_interval = self.config['data']['update_interval']
        
        # Initialize portfolio
        initial_capital = self.config['backtest']['initial_capital']
        self.portfolio_manager.initialize_portfolio(initial_capital)
        
        # Setup logging
        self._setup_logging()
        
        logger.info("Crypto Trading Bot initialized")
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file"""
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_config = self.config.get('logging', {})
        log_file = log_config.get('file', 'logs/trader.log')
        log_level = log_config.get('level', 'INFO')
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Configure loguru
        logger.add(
            log_file,
            rotation=log_config.get('max_size', '10MB'),
            retention=log_config.get('backup_count', 5),
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}"
        )
    
    async def start_bot(self, live_trading: bool = False):
        """Start the trading bot"""
        self.is_running = True
        self.is_live_trading = live_trading
        
        if live_trading:
            logger.warning("Starting bot in LIVE TRADING mode")
        else:
            logger.info("Starting bot in PAPER TRADING mode")
        
        # Set daily start value for P&L tracking
        self.portfolio_manager.set_daily_start_value(self.portfolio_manager.portfolio_value)
        
        # Schedule data updates
        schedule.every(self.update_interval).seconds.do(self._update_market_data)
        schedule.every(1).minutes.do(self._check_signals)
        schedule.every(5).minutes.do(self._check_risk_management)
        schedule.every(1).hours.do(self._log_portfolio_status)
        
        logger.info("Trading bot started successfully")
        
        # Main bot loop
        while self.is_running:
            try:
                schedule.run_pending()
                await asyncio.sleep(1)
                
            except KeyboardInterrupt:
                logger.info("Received shutdown signal")
                break
            except Exception as e:
                logger.error(f"Error in main bot loop: {e}")
                await asyncio.sleep(5)
        
        await self.stop_bot()
    
    async def stop_bot(self):
        """Stop the trading bot"""
        self.is_running = False
        logger.info("Trading bot stopped")
    
    async def _update_market_data(self):
        """Update market data for all symbols"""
        try:
            await self.data_manager.update_all_symbols()
            
            # Update portfolio with latest prices
            latest_prices = {}
            for symbol in self.symbols:
                price = self.data_manager.get_latest_price(symbol)
                if price:
                    latest_prices[symbol] = price
            
            if latest_prices:
                self.portfolio_manager.update_position_prices(latest_prices)
            
        except Exception as e:
            logger.error(f"Error updating market data: {e}")
    
    async def _check_signals(self):
        """Check for trading signals from all strategies"""
        try:
            all_signals = []
            
            for symbol in self.symbols:
                # Get market data
                data = self.data_manager.get_market_data(symbol, '1h', limit=200)
                
                if data.empty:
                    continue
                
                # Get signals from strategy manager
                signals = self.strategy_manager.get_signals(symbol, data)
                all_signals.extend(signals)
            
            # Process signals
            if all_signals:
                await self._process_signals(all_signals)
                
        except Exception as e:
            logger.error(f"Error checking signals: {e}")
    
    async def _process_signals(self, signals: List[TradingSignal]):
        """Process and potentially execute trading signals"""
        try:
            # Group signals by symbol
            signals_by_symbol = {}
            for signal in signals:
                if signal.symbol not in signals_by_symbol:
                    signals_by_symbol[signal.symbol] = []
                signals_by_symbol[signal.symbol].append(signal)
            
            # Process each symbol's signals
            for symbol, symbol_signals in signals_by_symbol.items():
                # Aggregate signals into consensus
                consensus_signal = self.strategy_manager.aggregate_signals(symbol_signals)
                
                if consensus_signal:
                    await self._execute_signal(consensus_signal)
                    
        except Exception as e:
            logger.error(f"Error processing signals: {e}")
    
    async def _execute_signal(self, signal: TradingSignal):
        """Execute a trading signal"""
        try:
            # Get current price
            current_price = self.data_manager.get_latest_price(signal.symbol)
            if not current_price:
                logger.warning(f"Could not get current price for {signal.symbol}")
                return
            
            # Evaluate signal with portfolio manager
            can_execute, position_size, reason = self.portfolio_manager.evaluate_signal(signal, current_price)
            
            if not can_execute:
                logger.info(f"Signal not executed for {signal.symbol}: {reason}")
                return
            
            # Log signal
            logger.info(
                f"Executing {signal.signal.value} signal for {signal.symbol}: "
                f"Size={position_size:.6f}, Price={current_price:.6f}, "
                f"Confidence={signal.confidence:.2%}"
            )
            
            # Execute trade (paper trading or live)
            if self.is_live_trading:
                success = await self._execute_live_trade(signal, position_size, current_price)
            else:
                success = self._execute_paper_trade(signal, position_size, current_price)
            
            if success:
                logger.info(f"Trade executed successfully for {signal.symbol}")
            else:
                logger.error(f"Trade execution failed for {signal.symbol}")
                
        except Exception as e:
            logger.error(f"Error executing signal for {signal.symbol}: {e}")
    
    def _execute_paper_trade(self, signal: TradingSignal, position_size: float, price: float) -> bool:
        """Execute paper trade (simulation)"""
        try:
            return self.portfolio_manager.execute_trade(signal, position_size, price)
        except Exception as e:
            logger.error(f"Paper trade execution error: {e}")
            return False
    
    async def _execute_live_trade(self, signal: TradingSignal, position_size: float, price: float) -> bool:
        """Execute live trade on exchange"""
        try:
            # This would implement actual exchange API calls
            # For now, we'll just simulate
            logger.warning("Live trading not implemented - executing as paper trade")
            return self._execute_paper_trade(signal, position_size, price)
            
        except Exception as e:
            logger.error(f"Live trade execution error: {e}")
            return False
    
    async def _check_risk_management(self):
        """Check risk management rules and triggers"""
        try:
            # Check for stop loss and take profit triggers
            risk_signals = self.portfolio_manager.check_stop_loss_take_profit()
            
            if risk_signals:
                logger.info(f"Risk management triggered {len(risk_signals)} signals")
                await self._process_signals(risk_signals)
            
            # Get risk metrics
            risk_metrics = self.portfolio_manager.calculate_risk_metrics()
            
            # Check daily loss limit
            if risk_metrics.daily_pnl_pct < -self.config['risk']['max_daily_loss']:
                logger.warning(
                    f"Daily loss limit approaching: {risk_metrics.daily_pnl_pct:.2%} "
                    f"(limit: {-self.config['risk']['max_daily_loss']:.2%})"
                )
            
            # Log positions at risk
            if risk_metrics.positions_at_risk:
                logger.warning(f"Positions at risk: {risk_metrics.positions_at_risk}")
                
        except Exception as e:
            logger.error(f"Error in risk management check: {e}")
    
    def _log_portfolio_status(self):
        """Log current portfolio status"""
        try:
            summary = self.portfolio_manager.get_portfolio_summary()
            
            logger.info(
                f"Portfolio Status - Value: ${summary['portfolio_value']:,.2f}, "
                f"Cash: ${summary['cash_balance']:,.2f}, "
                f"Positions: {summary['num_positions']}, "
                f"Daily P&L: {summary['daily_pnl_pct']:.2%}"
            )
            
        except Exception as e:
            logger.error(f"Error logging portfolio status: {e}")
    
    async def run_backtest(self, start_date: str, end_date: str, symbols: Optional[List[str]] = None):
        """Run a backtest"""
        try:
            if symbols is None:
                symbols = self.symbols
            
            logger.info(f"Starting backtest for {symbols} from {start_date} to {end_date}")
            
            result = self.backtest_engine.run_backtest(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                timeframe='1h'
            )
            
            # Log results
            logger.info(f"Backtest Results:")
            logger.info(f"  Total Return: {result.total_return:.2%}")
            logger.info(f"  Annualized Return: {result.annualized_return:.2%}")
            logger.info(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
            logger.info(f"  Max Drawdown: {result.max_drawdown:.2%}")
            logger.info(f"  Total Trades: {result.total_trades}")
            logger.info(f"  Win Rate: {result.win_rate:.2%}")
            logger.info(f"  Profit Factor: {result.profit_factor:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return None
    
    def get_portfolio_status(self) -> Dict:
        """Get current portfolio status"""
        return self.portfolio_manager.get_portfolio_summary()
    
    def get_strategy_status(self) -> Dict:
        """Get strategy status"""
        return self.strategy_manager.get_strategy_status()
    
    def get_data_summary(self) -> Dict:
        """Get data availability summary"""
        return self.data_manager.get_data_summary()
    
    def enable_strategy(self, strategy_name: str):
        """Enable a trading strategy"""
        self.strategy_manager.enable_strategy(strategy_name)
    
    def disable_strategy(self, strategy_name: str):
        """Disable a trading strategy"""
        self.strategy_manager.disable_strategy(strategy_name)
    
    def update_strategy_config(self, strategy_name: str, new_config: Dict):
        """Update strategy configuration"""
        self.strategy_manager.update_strategy_config(strategy_name, new_config)
    
    def set_live_trading(self, enabled: bool):
        """Enable or disable live trading"""
        if enabled and not self.is_live_trading:
            logger.warning("Enabling live trading mode")
        elif not enabled and self.is_live_trading:
            logger.info("Disabling live trading mode")
        
        self.is_live_trading = enabled
    
    async def emergency_stop(self):
        """Emergency stop - close all positions and stop trading"""
        try:
            logger.warning("EMERGENCY STOP activated")
            
            # Generate sell signals for all positions
            emergency_signals = []
            for symbol, position in self.portfolio_manager.positions.items():
                signal = TradingSignal(
                    symbol=symbol,
                    signal=SignalType.SELL,
                    price=position.current_price,
                    confidence=1.0,
                    timestamp=pd.Timestamp.now(),
                    metadata={'reason': 'emergency_stop'}
                )
                emergency_signals.append(signal)
            
            # Execute all sell orders
            if emergency_signals:
                await self._process_signals(emergency_signals)
            
            # Stop the bot
            await self.stop_bot()
            
            logger.warning("Emergency stop completed")
            
        except Exception as e:
            logger.error(f"Error during emergency stop: {e}")


# Main execution function
async def main():
    """Main function to run the trading bot"""
    bot = CryptoTradingBot()
    
    try:
        # You can run a backtest first
        # result = await bot.run_backtest("2023-01-01", "2023-12-31")
        
        # Then start live trading (paper mode)
        await bot.start_bot(live_trading=False)
        
    except KeyboardInterrupt:
        logger.info("Bot shutdown requested")
    except Exception as e:
        logger.error(f"Error in main: {e}")
    finally:
        await bot.stop_bot()


if __name__ == "__main__":
    asyncio.run(main())