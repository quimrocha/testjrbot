// CryptoTrader Pro Dashboard JavaScript

class TradingDashboard {
    constructor() {
        this.socket = io();
        this.isConnected = false;
        this.currentChart = null;
        
        this.init();
    }
    
    init() {
        this.setupSocketListeners();
        this.setupEventHandlers();
        this.setupInitialData();
        this.setupAutoRefresh();
    }
    
    setupSocketListeners() {
        this.socket.on('connect', () => {
            console.log('Connected to trading bot');
            this.isConnected = true;
            this.showNotification('Connected to trading bot', 'success');
            this.requestInitialData();
        });
        
        this.socket.on('disconnect', () => {
            console.log('Disconnected from trading bot');
            this.isConnected = false;
            this.showNotification('Disconnected from trading bot', 'error');
        });
        
        this.socket.on('portfolio_update', (data) => {
            this.updatePortfolioDisplay(data);
        });
        
        this.socket.on('status_update', (data) => {
            this.updateStatusDisplay(data);
        });
        
        this.socket.on('error', (data) => {
            this.showNotification(data.message, 'error');
        });
    }
    
    setupEventHandlers() {
        // Bot control buttons
        document.getElementById('start-bot').addEventListener('click', () => {
            this.startBot();
        });
        
        document.getElementById('stop-bot').addEventListener('click', () => {
            this.stopBot();
        });
        
        document.getElementById('emergency-stop').addEventListener('click', () => {
            this.emergencyStop();
        });
        
        // Symbol selection for chart
        document.getElementById('symbol-select').addEventListener('change', (e) => {
            this.loadChart(e.target.value);
        });
        
        // Backtest form
        document.getElementById('backtest-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.runBacktest();
        });
        
        // Live trading toggle
        document.getElementById('live-trading-toggle').addEventListener('change', (e) => {
            this.toggleLiveTrading(e.target.checked);
        });
    }
    
    setupInitialData() {
        // Set default dates for backtest
        const endDate = new Date();
        const startDate = new Date();
        startDate.setMonth(startDate.getMonth() - 3); // 3 months ago
        
        document.getElementById('start-date').value = startDate.toISOString().split('T')[0];
        document.getElementById('end-date').value = endDate.toISOString().split('T')[0];
        
        // Load initial data
        this.loadPortfolio();
        this.loadStrategies();
        this.loadChart('BTC/USDT');
    }
    
    setupAutoRefresh() {
        // Refresh data every 30 seconds
        setInterval(() => {
            if (this.isConnected) {
                this.loadPortfolio();
                this.loadStatus();
            }
        }, 30000);
    }
    
    requestInitialData() {
        this.socket.emit('get_portfolio');
        this.socket.emit('get_status');
    }
    
    async startBot() {
        try {
            const liveTrading = document.getElementById('live-trading-toggle').checked;
            
            if (liveTrading) {
                const confirmed = confirm('⚠️ WARNING: You are about to start LIVE TRADING. This will use real money. Are you sure?');
                if (!confirmed) return;
            }
            
            const response = await fetch('/api/bot/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ live_trading: liveTrading })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showNotification(data.message, 'success');
                this.loadStatus();
            } else {
                this.showNotification(data.error, 'error');
            }
        } catch (error) {
            this.showNotification('Error starting bot: ' + error.message, 'error');
        }
    }
    
    async stopBot() {
        try {
            const response = await fetch('/api/bot/stop', {
                method: 'POST'
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showNotification(data.message, 'success');
                this.loadStatus();
            } else {
                this.showNotification(data.error, 'error');
            }
        } catch (error) {
            this.showNotification('Error stopping bot: ' + error.message, 'error');
        }
    }
    
    async emergencyStop() {
        const confirmed = confirm('⚠️ EMERGENCY STOP: This will immediately close all positions and stop the bot. Continue?');
        if (!confirmed) return;
        
        try {
            const response = await fetch('/api/bot/emergency-stop', {
                method: 'POST'
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showNotification(data.message, 'warning');
                this.loadStatus();
                this.loadPortfolio();
            } else {
                this.showNotification(data.error, 'error');
            }
        } catch (error) {
            this.showNotification('Error during emergency stop: ' + error.message, 'error');
        }
    }
    
    async toggleLiveTrading(enabled) {
        // Update the trading mode display
        const modeElement = document.getElementById('trading-mode');
        modeElement.textContent = enabled ? 'Live' : 'Paper';
        modeElement.className = enabled ? 'badge bg-danger' : 'badge bg-info';
        
        if (enabled) {
            this.showNotification('⚠️ Live trading mode enabled', 'warning');
        } else {
            this.showNotification('Paper trading mode enabled', 'success');
        }
    }
    
    async loadPortfolio() {
        try {
            const response = await fetch('/api/portfolio');
            const data = await response.json();
            
            if (data.success) {
                this.updatePortfolioDisplay(data.data);
            }
        } catch (error) {
            console.error('Error loading portfolio:', error);
        }
    }
    
    async loadStatus() {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();
            
            if (data.success) {
                this.updateStatusDisplay(data.data);
            }
        } catch (error) {
            console.error('Error loading status:', error);
        }
    }
    
    async loadStrategies() {
        try {
            const response = await fetch('/api/strategies');
            const data = await response.json();
            
            if (data.success) {
                this.updateStrategiesDisplay(data.data);
            }
        } catch (error) {
            console.error('Error loading strategies:', error);
        }
    }
    
    async loadChart(symbol) {
        try {
            const response = await fetch(`/api/chart/${encodeURIComponent(symbol)}?timeframe=1h&limit=100`);
            const data = await response.json();
            
            if (data.success) {
                this.updateChart(data.data);
            } else {
                this.showNotification(`Error loading chart for ${symbol}: ${data.error}`, 'error');
            }
        } catch (error) {
            console.error('Error loading chart:', error);
            this.showNotification('Error loading chart: ' + error.message, 'error');
        }
    }
    
    updatePortfolioDisplay(portfolio) {
        // Update overview cards
        document.getElementById('portfolio-value').textContent = 
            this.formatCurrency(portfolio.portfolio_value || 0);
        
        document.getElementById('daily-pnl').textContent = 
            this.formatPercentage(portfolio.daily_pnl_pct || 0);
        
        document.getElementById('active-positions').textContent = 
            portfolio.num_positions || 0;
        
        document.getElementById('cash-balance').textContent = 
            this.formatCurrency(portfolio.cash_balance || 0);
        
        // Update daily P&L color
        const dailyPnlElement = document.getElementById('daily-pnl');
        const pnl = portfolio.daily_pnl_pct || 0;
        if (pnl > 0) {
            dailyPnlElement.className = 'text-success';
        } else if (pnl < 0) {
            dailyPnlElement.className = 'text-danger';
        } else {
            dailyPnlElement.className = 'text-info';
        }
        
        // Update positions table
        this.updatePositionsTable(portfolio.positions || []);
    }
    
    updatePositionsTable(positions) {
        const tbody = document.querySelector('#positions-table tbody');
        tbody.innerHTML = '';
        
        if (positions.length === 0) {
            tbody.innerHTML = '<tr><td colspan="8" class="text-center text-muted">No positions</td></tr>';
            return;
        }
        
        positions.forEach(position => {
            const row = document.createElement('tr');
            
            const pnlClass = position.unrealized_pnl >= 0 ? 'position-positive' : 'position-negative';
            
            row.innerHTML = `
                <td class="fw-bold">${position.symbol}</td>
                <td>${this.formatNumber(position.quantity, 6)}</td>
                <td>${this.formatCurrency(position.avg_price)}</td>
                <td>${this.formatCurrency(position.current_price)}</td>
                <td>${this.formatCurrency(position.market_value)}</td>
                <td class="${pnlClass}">${this.formatCurrency(position.unrealized_pnl)}</td>
                <td class="${pnlClass}">${this.formatPercentage(position.unrealized_pnl_pct)}</td>
                <td>${this.formatPercentage(position.weight)}</td>
            `;
            
            tbody.appendChild(row);
        });
    }
    
    updateStatusDisplay(status) {
        const statusElement = document.getElementById('bot-status');
        const modeElement = document.getElementById('trading-mode');
        
        // Update status
        if (status.is_running) {
            statusElement.textContent = 'Running';
            statusElement.className = 'badge bg-success';
        } else {
            statusElement.textContent = 'Stopped';
            statusElement.className = 'badge bg-secondary';
        }
        
        // Update mode
        modeElement.textContent = status.is_live_trading ? 'Live' : 'Paper';
        modeElement.className = status.is_live_trading ? 'badge bg-danger' : 'badge bg-info';
        
        // Update live trading toggle
        document.getElementById('live-trading-toggle').checked = status.is_live_trading;
    }
    
    updateStrategiesDisplay(strategies) {
        const container = document.getElementById('strategies-list');
        container.innerHTML = '';
        
        Object.entries(strategies).forEach(([name, strategy]) => {
            const strategyElement = document.createElement('div');
            strategyElement.className = 'strategy-item';
            
            strategyElement.innerHTML = `
                <div>
                    <div class="strategy-name">${strategy.name}</div>
                    <div class="strategy-status">${strategy.enabled ? 'Enabled' : 'Disabled'}</div>
                </div>
                <div class="form-check form-switch">
                    <input class="form-check-input" type="checkbox" 
                           id="strategy-${name}" ${strategy.enabled ? 'checked' : ''}
                           data-strategy="${name}">
                </div>
            `;
            
            // Add event listener for strategy toggle
            const toggle = strategyElement.querySelector('input');
            toggle.addEventListener('change', (e) => {
                this.toggleStrategy(name, e.target.checked);
            });
            
            container.appendChild(strategyElement);
        });
    }
    
    async toggleStrategy(strategyName, enabled) {
        try {
            const response = await fetch(`/api/strategies/${strategyName}/toggle`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ enabled })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showNotification(data.message, 'success');
            } else {
                this.showNotification(data.error, 'error');
                // Revert toggle on error
                document.querySelector(`[data-strategy="${strategyName}"]`).checked = !enabled;
            }
        } catch (error) {
            this.showNotification('Error toggling strategy: ' + error.message, 'error');
            // Revert toggle on error
            document.querySelector(`[data-strategy="${strategyName}"]`).checked = !enabled;
        }
    }
    
    updateChart(chartData) {
        try {
            const config = JSON.parse(chartData.chart);
            Plotly.newPlot('price-chart', config.data, config.layout, {
                responsive: true,
                displayModeBar: false
            });
        } catch (error) {
            console.error('Error updating chart:', error);
            document.getElementById('price-chart').innerHTML = 
                '<div class="text-center text-muted mt-5">Error loading chart</div>';
        }
    }
    
    async runBacktest() {
        try {
            const startDate = document.getElementById('start-date').value;
            const endDate = document.getElementById('end-date').value;
            
            if (!startDate || !endDate) {
                this.showNotification('Please select start and end dates', 'error');
                return;
            }
            
            const submitButton = document.querySelector('#backtest-form button[type="submit"]');
            submitButton.disabled = true;
            submitButton.innerHTML = '<span class="spinner"></span> Running...';
            
            const response = await fetch('/api/backtest', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    start_date: startDate,
                    end_date: endDate
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showNotification('Backtest started successfully', 'success');
                // You could implement polling for backtest results here
            } else {
                this.showNotification(data.error, 'error');
            }
        } catch (error) {
            this.showNotification('Error running backtest: ' + error.message, 'error');
        } finally {
            const submitButton = document.querySelector('#backtest-form button[type="submit"]');
            submitButton.disabled = false;
            submitButton.innerHTML = '<i class="fas fa-play me-1"></i>Run Backtest';
        }
    }
    
    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `alert alert-${type === 'error' ? 'danger' : type} notification`;
        notification.innerHTML = `
            ${message}
            <button type="button" class="btn-close" aria-label="Close"></button>
        `;
        
        document.body.appendChild(notification);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 5000);
        
        // Remove on click
        notification.querySelector('.btn-close').addEventListener('click', () => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        });
    }
    
    formatCurrency(value) {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD',
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        }).format(value || 0);
    }
    
    formatPercentage(value) {
        return new Intl.NumberFormat('en-US', {
            style: 'percent',
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        }).format((value || 0) / 100);
    }
    
    formatNumber(value, decimals = 2) {
        return new Intl.NumberFormat('en-US', {
            minimumFractionDigits: decimals,
            maximumFractionDigits: decimals
        }).format(value || 0);
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new TradingDashboard();
});