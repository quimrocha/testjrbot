"""
Flask Web Application for Crypto Trading Bot
Provides REST API and web interface for bot management
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import plotly.graph_objs as go
import plotly.utils
import pandas as pd
from loguru import logger

from ..bot.crypto_trader import CryptoTradingBot


class TradingBotAPI:
    """Flask API for the trading bot"""
    
    def __init__(self):
        self.app = Flask(__name__, template_folder='templates', static_folder='static')
        self.app.config['SECRET_KEY'] = 'your-secret-key-here'
        
        # Enable CORS for cross-origin requests
        CORS(self.app)
        
        # Initialize SocketIO for real-time updates
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Initialize the trading bot
        self.bot = CryptoTradingBot()
        
        # Setup routes
        self._setup_routes()
        self._setup_socketio_events()
        
        logger.info("Trading Bot API initialized")
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            """Main dashboard page"""
            return render_template('dashboard.html')
        
        @self.app.route('/api/status')
        def get_status():
            """Get bot status"""
            try:
                return jsonify({
                    'success': True,
                    'data': {
                        'is_running': self.bot.is_running,
                        'is_live_trading': self.bot.is_live_trading,
                        'symbols': self.bot.symbols,
                        'timestamp': datetime.now().isoformat()
                    }
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/portfolio')
        def get_portfolio():
            """Get portfolio status"""
            try:
                portfolio = self.bot.get_portfolio_status()
                return jsonify({
                    'success': True,
                    'data': portfolio
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/strategies')
        def get_strategies():
            """Get strategy status"""
            try:
                strategies = self.bot.get_strategy_status()
                return jsonify({
                    'success': True,
                    'data': strategies
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/strategies/<strategy_name>/toggle', methods=['POST'])
        def toggle_strategy(strategy_name):
            """Enable/disable a strategy"""
            try:
                data = request.get_json()
                enabled = data.get('enabled', False)
                
                if enabled:
                    self.bot.enable_strategy(strategy_name)
                else:
                    self.bot.disable_strategy(strategy_name)
                
                return jsonify({
                    'success': True,
                    'message': f"Strategy {strategy_name} {'enabled' if enabled else 'disabled'}"
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/bot/start', methods=['POST'])
        def start_bot():
            """Start the trading bot"""
            try:
                data = request.get_json() or {}
                live_trading = data.get('live_trading', False)
                
                # Start bot in background
                asyncio.create_task(self.bot.start_bot(live_trading))
                
                return jsonify({
                    'success': True,
                    'message': f"Bot started in {'live' if live_trading else 'paper'} trading mode"
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/bot/stop', methods=['POST'])
        def stop_bot():
            """Stop the trading bot"""
            try:
                asyncio.create_task(self.bot.stop_bot())
                return jsonify({
                    'success': True,
                    'message': "Bot stopped"
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/bot/emergency-stop', methods=['POST'])
        def emergency_stop():
            """Emergency stop the bot"""
            try:
                asyncio.create_task(self.bot.emergency_stop())
                return jsonify({
                    'success': True,
                    'message': "Emergency stop executed"
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/backtest', methods=['POST'])
        def run_backtest():
            """Run a backtest"""
            try:
                data = request.get_json()
                start_date = data.get('start_date')
                end_date = data.get('end_date')
                symbols = data.get('symbols')
                
                if not start_date or not end_date:
                    return jsonify({
                        'success': False,
                        'error': 'start_date and end_date are required'
                    }), 400
                
                # Run backtest in background and return task ID
                task = asyncio.create_task(
                    self.bot.run_backtest(start_date, end_date, symbols)
                )
                
                return jsonify({
                    'success': True,
                    'message': 'Backtest started',
                    'task_id': id(task)
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/data-summary')
        def get_data_summary():
            """Get data availability summary"""
            try:
                summary = self.bot.get_data_summary()
                return jsonify({
                    'success': True,
                    'data': summary
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/chart/<symbol>')
        def get_chart_data(symbol):
            """Get chart data for a symbol"""
            try:
                timeframe = request.args.get('timeframe', '1h')
                limit = int(request.args.get('limit', 100))
                
                # Get market data
                data = self.bot.data_manager.get_market_data(symbol, timeframe, limit=limit)
                
                if data.empty:
                    return jsonify({
                        'success': False,
                        'error': f'No data available for {symbol}'
                    }), 404
                
                # Create plotly chart
                fig = go.Figure(data=go.Candlestick(
                    x=data.index,
                    open=data['open'],
                    high=data['high'],
                    low=data['low'],
                    close=data['close'],
                    name=symbol
                ))
                
                fig.update_layout(
                    title=f'{symbol} Price Chart',
                    yaxis_title='Price',
                    xaxis_title='Time',
                    template='plotly_dark'
                )
                
                # Convert to JSON
                chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                
                return jsonify({
                    'success': True,
                    'data': {
                        'chart': chart_json,
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'latest_price': float(data['close'].iloc[-1]),
                        'latest_time': data.index[-1].isoformat()
                    }
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/performance')
        def get_performance():
            """Get portfolio performance metrics"""
            try:
                # This would calculate various performance metrics
                # For now, return placeholder data
                return jsonify({
                    'success': True,
                    'data': {
                        'total_return': 0.0,
                        'daily_return': 0.0,
                        'sharpe_ratio': 0.0,
                        'max_drawdown': 0.0,
                        'win_rate': 0.0,
                        'total_trades': 0
                    }
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
    
    def _setup_socketio_events(self):
        """Setup SocketIO events for real-time updates"""
        
        @self.socketio.on('connect')
        def handle_connect():
            logger.info('Client connected to WebSocket')
            emit('status', {'message': 'Connected to trading bot'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            logger.info('Client disconnected from WebSocket')
        
        @self.socketio.on('get_portfolio')
        def handle_get_portfolio():
            try:
                portfolio = self.bot.get_portfolio_status()
                emit('portfolio_update', portfolio)
            except Exception as e:
                emit('error', {'message': str(e)})
        
        @self.socketio.on('get_status')
        def handle_get_status():
            try:
                status = {
                    'is_running': self.bot.is_running,
                    'is_live_trading': self.bot.is_live_trading,
                    'symbols': self.bot.symbols,
                    'timestamp': datetime.now().isoformat()
                }
                emit('status_update', status)
            except Exception as e:
                emit('error', {'message': str(e)})
    
    def start_periodic_updates(self):
        """Start periodic updates to connected clients"""
        def update_clients():
            try:
                # Send portfolio updates
                portfolio = self.bot.get_portfolio_status()
                self.socketio.emit('portfolio_update', portfolio)
                
                # Send status updates
                status = {
                    'is_running': self.bot.is_running,
                    'is_live_trading': self.bot.is_live_trading,
                    'symbols': self.bot.symbols,
                    'timestamp': datetime.now().isoformat()
                }
                self.socketio.emit('status_update', status)
                
            except Exception as e:
                logger.error(f"Error in periodic update: {e}")
        
        # Schedule updates every 5 seconds
        self.socketio.start_background_task(target=self._periodic_update_task)
    
    def _periodic_update_task(self):
        """Background task for periodic updates"""
        while True:
            try:
                # Send portfolio updates
                portfolio = self.bot.get_portfolio_status()
                self.socketio.emit('portfolio_update', portfolio)
                
                # Send status updates
                status = {
                    'is_running': self.bot.is_running,
                    'is_live_trading': self.bot.is_live_trading,
                    'symbols': self.bot.symbols,
                    'timestamp': datetime.now().isoformat()
                }
                self.socketio.emit('status_update', status)
                
                self.socketio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in periodic update task: {e}")
                self.socketio.sleep(5)
    
    def run(self, host='0.0.0.0', port=8080, debug=True):
        """Run the Flask application"""
        logger.info(f"Starting web interface on http://{host}:{port}")
        
        # Start periodic updates
        self.start_periodic_updates()
        
        # Run the Flask app
        self.socketio.run(self.app, host=host, port=port, debug=debug)


# Create the API instance
api = TradingBotAPI()


if __name__ == '__main__':
    api.run()