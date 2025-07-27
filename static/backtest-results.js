/**
 * Backtest Results Component
 * Handles display of backtest results including equity curves, metrics, and trade history
 */

class BacktestResults {
    constructor() {
        this.currentResults = null;
        this.equityChart = null;
        this.isLoading = false;
        
        // Chart configuration
        this.chartColors = {
            primary: '#2563eb',
            success: '#10b981', 
            danger: '#ef4444',
            warning: '#f59e0b',
            grid: '#e5e7eb'
        };
        
        this.init();
    }
    
    /**
     * Initialize the component
     */
    init() {
        this.setupEventListeners();
        this.setupRefreshTimer();
    }
    
    /**
     * Set up event listeners
     */
    setupEventListeners() {
        // Refresh button
        const refreshBtn = document.getElementById('refreshBacktestBtn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.refresh());
        }
        
        // Filter controls
        const strategyFilter = document.getElementById('backtestStrategyFilter');
        if (strategyFilter) {
            strategyFilter.addEventListener('change', () => this.applyFilters());
        }
        
        const sharpeFilter = document.getElementById('backtestSharpeFilter');
        if (sharpeFilter) {
            sharpeFilter.addEventListener('input', () => this.applyFilters());
        }
        
        // Chart timeframe selector
        const timeframeSelector = document.getElementById('backtestTimeframe');
        if (timeframeSelector) {
            timeframeSelector.addEventListener('change', (e) => {
                this.updateChartTimeframe(e.target.value);
            });
        }
    }
    
    /**
     * Set up automatic refresh timer
     */
    setupRefreshTimer() {
        // Refresh every 30 seconds when dashboard is visible
        setInterval(() => {
            if (!document.hidden && !this.isLoading) {
                this.refresh();
            }
        }, 30000);
    }
    
    /**
     * Fetch backtest results from API
     */
    async fetchBacktestResults() {
        if (!window.dashboard?.settings?.apiKey) {
            console.warn('No API key available for backtest results');
            return;
        }
        
        this.showLoadingState();
        
        try {
            const response = await fetch('/dashboard/backtest/results', {
                headers: {
                    'Authorization': `Bearer ${window.dashboard.settings.apiKey}`,
                    'Content-Type': 'application/json'
                }
            });
            
            if (response.ok) {
                const data = await response.json();
                this.currentResults = data.backtest_results;
                this.displayResults(this.currentResults);
            } else if (response.status === 401) {
                window.dashboard.showNotification('error', 'Authentication Error', 'Please refresh the page');
            } else {
                throw new Error(`Server error: ${response.status}`);
            }
            
        } catch (error) {
            console.error('Failed to fetch backtest results:', error);
            window.dashboard.showNotification('error', 'Failed to Load', 'Could not fetch backtest results');
        } finally {
            this.hideLoadingState();
        }
    }
    
    /**
     * Fetch backtest results with filters
     */
    async fetchBacktestResultsWithFilters(filters = {}) {
        if (!window.dashboard?.settings?.apiKey) {
            console.warn('No API key available for backtest results');
            return;
        }
        
        this.showLoadingState();
        
        try {
            const params = new URLSearchParams();
            Object.entries(filters).forEach(([key, value]) => {
                if (value !== null && value !== undefined && value !== '') {
                    params.append(key, value);
                }
            });
            
            const url = `/dashboard/backtest/results${params.toString() ? '?' + params.toString() : ''}`;
            
            const response = await fetch(url, {
                headers: {
                    'Authorization': `Bearer ${window.dashboard.settings.apiKey}`,
                    'Content-Type': 'application/json'
                }
            });
            
            if (response.ok) {
                const data = await response.json();
                this.currentResults = data.backtest_results;
                this.displayResults(this.currentResults);
            } else {
                throw new Error(`Server error: ${response.status}`);
            }
            
        } catch (error) {
            console.error('Failed to fetch filtered backtest results:', error);
            window.dashboard.showNotification('error', 'Failed to Load', 'Could not fetch backtest results');
        } finally {
            this.hideLoadingState();
        }
    }
    
    /**
     * Display backtest results in the UI
     */
    displayResults(results) {
        if (!results) {
            this.displayNoResults();
            return;
        }
        
        this.displayMetrics(results);
        this.updateEquityChart(results.equity_curve || []);
        this.displayTradeHistory(results.trade_history || []);
        this.displayPerformanceStats(results.performance_metrics || {});
    }
    
    /**
     * Display performance metrics
     */
    displayMetrics(results) {
        const metricsContainer = document.getElementById('backtest-metrics');
        if (!metricsContainer) return;
        
        const totalReturn = this.formatPercentage(results.total_return || 0);
        const sharpeRatio = this.formatNumber(results.sharpe_ratio || 0);
        const maxDrawdown = this.formatPercentage(results.max_drawdown || 0);
        const winRate = this.formatPercentage(results.win_rate || 0);
        const totalTrades = results.total_trades || 0;
        const avgTradeReturn = window.dashboard.formatCurrency(results.avg_profit_per_trade || 0);
        
        metricsContainer.innerHTML = `
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Strategy</div>
                    <div class="metric-value">${results.strategy_name || 'Unknown'}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Total Return</div>
                    <div class="metric-value ${results.total_return >= 0 ? 'positive' : 'negative'}">${totalReturn}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Sharpe Ratio</div>
                    <div class="metric-value">${sharpeRatio}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Max Drawdown</div>
                    <div class="metric-value negative">${maxDrawdown}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Win Rate</div>
                    <div class="metric-value">${winRate}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Total Trades</div>
                    <div class="metric-value">${totalTrades}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Avg Trade P&L</div>
                    <div class="metric-value ${results.avg_profit_per_trade >= 0 ? 'positive' : 'negative'}">${avgTradeReturn}</div>
                </div>
            </div>
        `;
    }
    
    /**
     * Update equity curve chart
     */
    updateEquityChart(equityData) {
        const chartContainer = document.getElementById('backtest-equity-chart');
        if (!chartContainer) return;
        
        // Prepare chart data
        const labels = equityData.map(point => this.formatDate(point.timestamp));
        const values = equityData.map(point => point.value);
        
        const chartConfig = {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Portfolio Value',
                    data: values,
                    borderColor: this.chartColors.primary,
                    backgroundColor: this.chartColors.primary + '10',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Equity Curve'
                    },
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Time'
                        },
                        grid: {
                            color: this.chartColors.grid
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Portfolio Value ($)'
                        },
                        grid: {
                            color: this.chartColors.grid
                        },
                        ticks: {
                            callback: function(value) {
                                return window.dashboard.formatCurrency(value);
                            }
                        }
                    }
                },
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                elements: {
                    point: {
                        radius: 0,
                        hoverRadius: 5
                    }
                }
            }
        };
        
        // Create or update chart
        if (this.equityChart) {
            this.equityChart.data = chartConfig.data;
            this.equityChart.update();
        } else {
            const canvas = chartContainer.querySelector('canvas') || this.createChartCanvas(chartContainer);
            this.equityChart = new Chart(canvas, chartConfig);
        }
    }
    
    /**
     * Create canvas for chart
     */
    createChartCanvas(container) {
        container.innerHTML = '<canvas id="equityChartCanvas" width="800" height="400"></canvas>';
        return container.querySelector('canvas');
    }
    
    /**
     * Display trade history
     */
    displayTradeHistory(trades) {
        const historyContainer = document.getElementById('backtest-trade-history');
        if (!historyContainer) return;
        
        if (trades.length === 0) {
            historyContainer.innerHTML = '<div class="no-data">No trades to display</div>';
            return;
        }
        
        const tradesHtml = trades.slice(0, 100).map(trade => {
            const pnlClass = (trade.pnl || 0) >= 0 ? 'positive' : 'negative';
            const time = this.formatDate(trade.timestamp);
            const pnl = window.dashboard.formatCurrency(trade.pnl || 0);
            
            return `
                <tr class="trade-row">
                    <td>${time}</td>
                    <td>${trade.symbol || 'N/A'}</td>
                    <td><span class="trade-side ${trade.side?.toLowerCase()}">${trade.side || 'N/A'}</span></td>
                    <td>${this.formatNumber(trade.quantity || 0)}</td>
                    <td>${window.dashboard.formatCurrency(trade.price || 0)}</td>
                    <td class="pnl ${pnlClass}">${pnl}</td>
                    <td>${this.formatNumber(trade.duration_hours || 0)}h</td>
                </tr>
            `;
        }).join('');
        
        historyContainer.innerHTML = `
            <div class="table-container">
                <table class="trade-history-table">
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Symbol</th>
                            <th>Side</th>
                            <th>Quantity</th>
                            <th>Price</th>
                            <th>P&L</th>
                            <th>Duration</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${tradesHtml}
                    </tbody>
                </table>
            </div>
            ${trades.length > 100 ? '<div class="table-footer">Showing first 100 trades</div>' : ''}
        `;
    }
    
    /**
     * Display additional performance statistics
     */
    displayPerformanceStats(stats) {
        const statsContainer = document.getElementById('backtest-performance-stats');
        if (!statsContainer) return;
        
        const statsHtml = Object.entries(stats).map(([key, value]) => {
            const label = this.formatStatLabel(key);
            const formattedValue = this.formatStatValue(key, value);
            
            return `
                <div class="stat-item">
                    <span class="stat-label">${label}:</span>
                    <span class="stat-value">${formattedValue}</span>
                </div>
            `;
        }).join('');
        
        if (statsHtml) {
            statsContainer.innerHTML = `
                <h4>Additional Performance Metrics</h4>
                <div class="stats-grid">
                    ${statsHtml}
                </div>
            `;
        }
    }
    
    /**
     * Display no results message
     */
    displayNoResults() {
        const containers = [
            'backtest-metrics',
            'backtest-equity-chart', 
            'backtest-trade-history',
            'backtest-performance-stats'
        ];
        
        containers.forEach(id => {
            const container = document.getElementById(id);
            if (container) {
                container.innerHTML = '<div class="no-data">No backtest results available</div>';
            }
        });
        
        if (this.equityChart) {
            this.equityChart.destroy();
            this.equityChart = null;
        }
    }
    
    /**
     * Show loading state
     */
    showLoadingState() {
        this.isLoading = true;
        
        const containers = [
            'backtest-metrics',
            'backtest-equity-chart',
            'backtest-trade-history'
        ];
        
        containers.forEach(id => {
            const container = document.getElementById(id);
            if (container) {
                container.innerHTML = '<div class="loading">Loading backtest results...</div>';
            }
        });
    }
    
    /**
     * Hide loading state
     */
    hideLoadingState() {
        this.isLoading = false;
    }
    
    /**
     * Apply current filters
     */
    applyFilters() {
        const strategyFilter = document.getElementById('backtestStrategyFilter');
        const sharpeFilter = document.getElementById('backtestSharpeFilter');
        
        const filters = {};
        
        if (strategyFilter?.value) {
            filters.strategy_name = strategyFilter.value;
        }
        
        if (sharpeFilter?.value) {
            filters.min_sharpe_ratio = parseFloat(sharpeFilter.value);
        }
        
        this.fetchBacktestResultsWithFilters(filters);
    }
    
    /**
     * Update chart timeframe
     */
    updateChartTimeframe(timeframe) {
        // This would fetch different time periods of data
        console.log(`Updating chart timeframe to: ${timeframe}`);
        // Implementation would depend on API support for different timeframes
    }
    
    /**
     * Refresh backtest results
     */
    async refresh() {
        await this.fetchBacktestResults();
    }
    
    /**
     * Cleanup resources
     */
    destroy() {
        if (this.equityChart) {
            this.equityChart.destroy();
            this.equityChart = null;
        }
    }
    
    // Utility methods
    
    /**
     * Format percentage values
     */
    formatPercentage(value) {
        return `${(value * 100).toFixed(2)}%`;
    }
    
    /**
     * Format numeric values
     */
    formatNumber(value) {
        if (typeof value !== 'number') return '0';
        return value % 1 === 0 ? value.toString() : value.toFixed(2);
    }
    
    /**
     * Format date strings
     */
    formatDate(dateString) {
        try {
            const date = new Date(dateString);
            return date.toLocaleDateString('en-US', {
                month: 'short',
                day: 'numeric',
                year: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
            });
        } catch (error) {
            return dateString;
        }
    }
    
    /**
     * Format statistic labels
     */
    formatStatLabel(key) {
        return key.split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }
    
    /**
     * Format statistic values
     */
    formatStatValue(key, value) {
        if (typeof value === 'number') {
            if (key.includes('ratio') || key.includes('factor')) {
                return this.formatNumber(value);
            } else if (key.includes('pct') || key.includes('rate')) {
                return this.formatPercentage(value);
            } else {
                return this.formatNumber(value);
            }
        }
        return value.toString();
    }
}

// Export for use in tests
if (typeof module !== 'undefined' && module.exports) {
    module.exports = BacktestResults;
}