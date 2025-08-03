#!/usr/bin/env python3
"""
CryptoTrader Pro - Main Entry Point
Advanced Cryptocurrency Trading Bot with Multiple Strategies and Web Interface
"""

import asyncio
import sys
import argparse
from pathlib import Path
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.bot.crypto_trader import CryptoTradingBot
from src.web.app import TradingBotAPI


def setup_logging():
    """Setup logging configuration"""
    logger.remove()  # Remove default logger
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )


async def run_bot_only():
    """Run the trading bot without web interface"""
    bot = CryptoTradingBot()
    
    try:
        logger.info("Starting CryptoTrader Pro Bot")
        await bot.start_bot(live_trading=False)
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"Error running bot: {e}")
    finally:
        await bot.stop_bot()


async def run_backtest(start_date: str, end_date: str, symbols: list = None):
    """Run a backtest"""
    bot = CryptoTradingBot()
    
    try:
        logger.info(f"Running backtest from {start_date} to {end_date}")
        result = await bot.run_backtest(start_date, end_date, symbols)
        
        if result:
            print("\n" + "="*60)
            print("BACKTEST RESULTS")
            print("="*60)
            print(f"Period: {start_date} to {end_date}")
            print(f"Initial Capital: ${result.initial_capital:,.2f}")
            print(f"Final Capital: ${result.final_capital:,.2f}")
            print(f"Total Return: {result.total_return:.2%}")
            print(f"Annualized Return: {result.annualized_return:.2%}")
            print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
            print(f"Max Drawdown: {result.max_drawdown:.2%}")
            print(f"Max Drawdown Duration: {result.max_drawdown_duration} periods")
            print(f"Total Trades: {result.total_trades}")
            print(f"Winning Trades: {result.winning_trades}")
            print(f"Losing Trades: {result.losing_trades}")
            print(f"Win Rate: {result.win_rate:.2%}")
            print(f"Average Win: ${result.avg_win:.2f}")
            print(f"Average Loss: ${result.avg_loss:.2f}")
            print(f"Profit Factor: {result.profit_factor:.2f}")
            print("="*60)
        else:
            logger.error("Backtest failed to complete")
            
    except Exception as e:
        logger.error(f"Error running backtest: {e}")


def run_web_interface():
    """Run the web interface"""
    logger.info("Starting CryptoTrader Pro Web Interface")
    api = TradingBotAPI()
    api.run(host='0.0.0.0', port=8080, debug=False)


def main():
    """Main entry point"""
    setup_logging()
    
    parser = argparse.ArgumentParser(description='CryptoTrader Pro - Advanced Cryptocurrency Trading Bot')
    parser.add_argument('--mode', choices=['bot', 'web', 'backtest'], default='web',
                       help='Run mode: bot (console only), web (web interface), backtest (backtesting)')
    parser.add_argument('--start-date', type=str, help='Backtest start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='Backtest end date (YYYY-MM-DD)')
    parser.add_argument('--symbols', nargs='*', help='Symbols for backtesting (default: all configured)')
    
    args = parser.parse_args()
    
    logger.info("Starting CryptoTrader Pro")
    logger.info(f"Mode: {args.mode}")
    
    try:
        if args.mode == 'bot':
            asyncio.run(run_bot_only())
        elif args.mode == 'backtest':
            if not args.start_date or not args.end_date:
                logger.error("Backtest mode requires --start-date and --end-date")
                return 1
            asyncio.run(run_backtest(args.start_date, args.end_date, args.symbols))
        else:  # web mode
            run_web_interface()
            
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())