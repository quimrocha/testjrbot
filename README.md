# CryptoTrader Pro

**Advanced Cryptocurrency Trading Bot with Multiple Strategies, Backtesting, and Web Interface**

![CryptoTrader Pro](https://img.shields.io/badge/Version-1.0.0-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🚀 Features

### Trading Strategies
- **SMA Crossover Strategy**: Simple Moving Average crossover signals
- **RSI Strategy**: Relative Strength Index overbought/oversold detection
- **MACD Strategy**: Moving Average Convergence Divergence analysis
- **Bollinger Bands Strategy**: Volatility-based trading signals

### Core Capabilities
- ✅ **Multiple Trading Strategies** with individual enable/disable controls
- ✅ **Comprehensive Backtesting Engine** with detailed performance metrics
- ✅ **Risk Management System** with stop-loss, take-profit, and position sizing
- ✅ **Portfolio Management** with real-time P&L tracking
- ✅ **Web Dashboard** with live updates and interactive charts
- ✅ **Paper Trading Mode** for safe strategy testing
- ✅ **Live Trading Support** with exchange API integration
- ✅ **Historical Data Management** with SQLite storage
- ✅ **Real-time Market Data** fetching and processing

### Web Interface
- 📊 **Real-time Dashboard** with portfolio overview
- 📈 **Interactive Price Charts** using Plotly
- ⚙️ **Strategy Management** with toggle controls
- 🔙 **Backtesting Interface** with customizable parameters
- 🛡️ **Risk Monitoring** with live alerts
- 📱 **Responsive Design** for mobile and desktop

## 🏗️ Architecture

```
CryptoTrader Pro/
├── src/
│   ├── data/                 # Market data management
│   │   ├── market_data.py   # Data fetching and storage
│   │   └── __init__.py
│   ├── strategies/          # Trading strategies
│   │   ├── base_strategy.py # Strategy base class
│   │   ├── sma_crossover.py # SMA crossover strategy
│   │   ├── rsi_strategy.py  # RSI strategy
│   │   ├── macd_strategy.py # MACD strategy
│   │   ├── bollinger_bands.py # Bollinger Bands strategy
│   │   ├── strategy_manager.py # Strategy coordinator
│   │   └── __init__.py
│   ├── backtesting/         # Backtesting engine
│   │   ├── backtest_engine.py # Comprehensive backtesting
│   │   └── __init__.py
│   ├── portfolio/           # Portfolio management
│   │   ├── portfolio_manager.py # Portfolio and risk management
│   │   └── __init__.py
│   ├── bot/                 # Main trading bot
│   │   ├── crypto_trader.py # Main bot orchestrator
│   │   └── __init__.py
│   ├── web/                 # Web interface
│   │   ├── app.py          # Flask application
│   │   ├── templates/       # HTML templates
│   │   ├── static/         # CSS/JS assets
│   │   └── __init__.py
│   └── __init__.py
├── data/                    # Data storage
├── logs/                    # Log files
├── config.yaml             # Configuration file
├── requirements.txt        # Python dependencies
├── main.py                # Main entry point
└── README.md              # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/cryptotrader-pro.git
cd cryptotrader-pro
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure API keys** (copy and edit)
```bash
cp .env.example .env
# Edit .env with your exchange API keys
```

4. **Run the application**
```bash
# Start with web interface (recommended)
python main.py --mode web

# Or run bot only in console
python main.py --mode bot

# Or run a backtest
python main.py --mode backtest --start-date 2023-01-01 --end-date 2023-12-31
```

5. **Access the web interface**
Open your browser to `http://localhost:8080`

## ⚙️ Configuration

### config.yaml
The main configuration file controls all aspects of the bot:

```yaml
# Trading Configuration
trading:
  base_currency: "USDT"
  max_positions: 5
  max_allocation_per_trade: 0.2  # 20% per trade
  min_trade_amount: 10

# Risk Management
risk:
  stop_loss: 0.05      # 5% stop loss
  take_profit: 0.15    # 15% take profit
  max_daily_loss: 0.1  # 10% max daily loss
  position_sizing: "kelly"  # kelly, fixed, volatility

# Strategies (enable/disable)
strategies:
  sma_crossover:
    enabled: true
    short_window: 20
    long_window: 50
  
  rsi_strategy:
    enabled: true
    period: 14
    oversold: 30
    overbought: 70
```

### Environment Variables (.env)
```bash
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret
```

## 📊 Trading Strategies

### 1. SMA Crossover Strategy
- **Signal**: Buy when short SMA crosses above long SMA, sell when opposite
- **Parameters**: `short_window` (default: 20), `long_window` (default: 50)
- **Best for**: Trending markets

### 2. RSI Strategy
- **Signal**: Buy when RSI exits oversold region, sell when exits overbought
- **Parameters**: `period` (default: 14), `oversold` (30), `overbought` (70)
- **Best for**: Range-bound markets

### 3. MACD Strategy
- **Signal**: MACD line crossovers and histogram analysis
- **Parameters**: `fast_period` (12), `slow_period` (26), `signal_period` (9)
- **Best for**: Trend following and momentum detection

### 4. Bollinger Bands Strategy
- **Signal**: Band breakouts and mean reversion
- **Parameters**: `period` (20), `std_dev` (2)
- **Best for**: Volatility-based trading

## 🔙 Backtesting

### Command Line
```bash
python main.py --mode backtest --start-date 2023-01-01 --end-date 2023-12-31
```

### Web Interface
1. Navigate to the Backtesting panel
2. Select start and end dates
3. Click "Run Backtest"
4. View results in real-time

### Metrics Provided
- Total Return
- Annualized Return
- Sharpe Ratio
- Maximum Drawdown
- Win Rate
- Profit Factor
- Trade Statistics

## 🌐 Web Interface

### Dashboard Features
- **Portfolio Overview**: Real-time portfolio value, P&L, and positions
- **Interactive Charts**: Candlestick charts with technical indicators
- **Strategy Controls**: Enable/disable strategies with toggle switches
- **Risk Monitoring**: Live risk metrics and alerts
- **Backtesting**: Run and analyze historical performance

### Real-time Updates
The web interface uses WebSocket connections for real-time updates:
- Portfolio values update every 5 seconds
- Price charts refresh automatically
- Strategy signals appear instantly

## 🛡️ Risk Management

### Built-in Safety Features
- **Stop Loss**: Automatic position closure on losses
- **Take Profit**: Automatic profit-taking
- **Position Sizing**: Kelly criterion, fixed, or volatility-based
- **Daily Loss Limits**: Automatic trading halt on excessive losses
- **Emergency Stop**: One-click position closure

### Paper Trading
Always test strategies in paper trading mode before risking real money:
```python
# The bot defaults to paper trading mode
await bot.start_bot(live_trading=False)  # Safe mode
```

## 📈 Performance Monitoring

### Key Metrics
- Portfolio value and composition
- Daily, weekly, monthly returns
- Risk-adjusted returns (Sharpe ratio)
- Maximum drawdown analysis
- Strategy-specific performance

### Alerts and Notifications
- Stop loss triggers
- Take profit executions
- Daily loss limit warnings
- Strategy performance alerts

## 🔌 Exchange Integration

### Supported Exchanges
- **Binance**: Full support (spot trading)
- **Coinbase Pro**: Planned
- **Kraken**: Planned
- **Bitfinex**: Planned

### API Requirements
1. Create API keys on your chosen exchange
2. Enable spot trading permissions
3. Restrict to specific IP addresses (recommended)
4. Never share API keys or commit them to version control

## 🚀 Getting Started Guide

### 1. First Time Setup
```bash
# Install and configure
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
```

### 2. Paper Trading Test
```bash
# Start with web interface
python main.py --mode web
# Access http://localhost:8080
# Start bot in paper trading mode
```

### 3. Strategy Backtesting
```bash
# Test strategies on historical data
python main.py --mode backtest --start-date 2023-01-01 --end-date 2023-12-31
```

### 4. Live Trading (Advanced)
```bash
# Only after successful paper trading
# Enable live trading in web interface
# Monitor closely and start with small amounts
```

## 📊 Example Output

### Backtest Results
```
============================================================
BACKTEST RESULTS
============================================================
Period: 2023-01-01 to 2023-12-31
Initial Capital: $10,000.00
Final Capital: $12,450.00
Total Return: 24.50%
Annualized Return: 24.50%
Sharpe Ratio: 1.25
Max Drawdown: -8.30%
Max Drawdown Duration: 45 periods
Total Trades: 127
Winning Trades: 76
Losing Trades: 51
Win Rate: 59.84%
Average Win: $58.42
Average Loss: -$31.15
Profit Factor: 1.87
============================================================
```

## 🔧 Advanced Configuration

### Custom Strategies
Create your own trading strategies by extending the `BaseStrategy` class:

```python
from src.strategies.base_strategy import BaseStrategy, TradingSignal, SignalType

class MyCustomStrategy(BaseStrategy):
    def calculate_indicators(self, data):
        # Implement your indicators
        pass
    
    def generate_signals(self, data):
        # Implement your signal logic
        pass
```

### Database Customization
The bot uses SQLite by default, but can be configured for other databases:
```yaml
database:
  type: "postgresql"  # sqlite, postgresql, mysql
  host: "localhost"
  port: 5432
  name: "cryptotrader"
```

## 🐛 Troubleshooting

### Common Issues

1. **API Connection Errors**
   - Verify API keys are correct
   - Check network connectivity
   - Ensure exchange API permissions

2. **Data Issues**
   - Clear data directory for fresh start
   - Check symbol configuration
   - Verify timeframe settings

3. **Performance Issues**
   - Reduce number of symbols
   - Increase update intervals
   - Optimize strategy parameters

### Debug Mode
Enable debug logging:
```yaml
logging:
  level: "DEBUG"
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**IMPORTANT**: This software is for educational and research purposes only. Cryptocurrency trading involves substantial risk of loss. Never trade with money you cannot afford to lose. The authors are not responsible for any financial losses incurred through the use of this software.

### Risk Warnings
- Past performance does not guarantee future results
- Cryptocurrency markets are highly volatile
- Always start with paper trading
- Never invest more than you can afford to lose
- Consider consulting with financial advisors

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

### Development Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/cryptotrader-pro.git
cd cryptotrader-pro

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
pytest tests/
```

## 📞 Support

- **Documentation**: [Wiki](https://github.com/yourusername/cryptotrader-pro/wiki)
- **Issues**: [GitHub Issues](https://github.com/yourusername/cryptotrader-pro/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/cryptotrader-pro/discussions)

## 🎯 Roadmap

### Version 1.1 (Planned)
- [ ] Additional exchanges (Coinbase Pro, Kraken)
- [ ] More trading strategies (Ichimoku, Williams %R)
- [ ] Machine learning integration
- [ ] Advanced portfolio optimization

### Version 1.2 (Planned)
- [ ] Mobile app
- [ ] Cloud deployment options
- [ ] Advanced charting tools
- [ ] Social trading features

---

**Happy Trading! 🚀**

*Remember: Trade responsibly and never risk more than you can afford to lose.*