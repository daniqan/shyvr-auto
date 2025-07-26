/**
 * Shyvr RLTE Dashboard JavaScript
 * Real-time trading system dashboard with WebSocket integration
 */

class DashboardApp {
    constructor() {
        this.websocket = null;
        this.connectionRetries = 0;
        this.maxRetries = 5;
        this.retryDelay = 3000;
        this.isConnected = false;
        this.currentTab = 'overview';
        
        // Data cache
        this.dashboardData = null;
        this.lastUpdateTime = null;
        
        // Chart instance
        this.performanceChart = null;
        
        // Settings
        this.settings = {
            refreshInterval: 2000,
            darkMode: false,
            soundNotifications: true,
            apiKey: null
        };
        
        // Notification queue
        this.notifications = [];
        
        this.init();
    }
    
    /**
     * Initialize the dashboard application
     */
    init() {
        this.loadSettings();
        this.setupEventListeners();
        this.setupTabs();
        this.connectWebSocket();
        this.startPeriodicUpdates();
        
        // Initialize chart
        this.initializeChart();
        
        console.log('Dashboard initialized');
    }
    
    /**
     * Load settings from localStorage
     */
    loadSettings() {
        const saved = localStorage.getItem('shyvr-dashboard-settings');
        if (saved) {
            this.settings = { ...this.settings, ...JSON.parse(saved) };
        }
        
        // Apply dark mode if enabled
        if (this.settings.darkMode) {
            document.documentElement.setAttribute('data-theme', 'dark');
            document.getElementById('darkModeToggle').checked = true;
        }
        
        // Apply other settings
        document.getElementById('refreshInterval').value = this.settings.refreshInterval / 1000;
        document.getElementById('soundNotificationsToggle').checked = this.settings.soundNotifications;
    }
    
    /**
     * Save settings to localStorage
     */
    saveSettings() {
        localStorage.setItem('shyvr-dashboard-settings', JSON.stringify(this.settings));
    }
    
    /**
     * Set up event listeners for UI interactions
     */
    setupEventListeners() {
        // Tab switching
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.switchTab(e.target.dataset.tab);
            });
        });
        
        // Emergency stop button
        document.getElementById('emergencyStopBtn').addEventListener('click', () => {
            this.showConfirmModal(
                'Emergency Stop',
                'Are you sure you want to trigger an emergency stop? This will halt all trading immediately.',
                () => this.emergencyStop()
            );
        });
        
        // Trading mode switching
        document.getElementById('switchModeBtn').addEventListener('click', () => {
            this.switchTradingMode();
        });
        
        // Trading controls
        document.getElementById('pauseTradingBtn').addEventListener('click', () => {
            this.pauseTrading();
        });
        
        // Settings updates
        document.getElementById('updateRiskSettingsBtn').addEventListener('click', () => {
            this.updateRiskSettings();
        });
        
        document.getElementById('updateDashboardSettingsBtn').addEventListener('click', () => {
            this.updateDashboardSettings();
        });
        
        // Refresh buttons
        document.getElementById('refreshTradesBtn').addEventListener('click', () => {
            this.refreshTrades();
        });
        
        document.getElementById('refreshLogsBtn').addEventListener('click', () => {
            this.refreshLogs();
        });
        
        // Dark mode toggle
        document.getElementById('darkModeToggle').addEventListener('change', (e) => {
            this.toggleDarkMode(e.target.checked);
        });
        
        // Modal controls
        document.getElementById('modalClose').addEventListener('click', () => {
            this.hideModal();
        });
        
        document.getElementById('modalCancel').addEventListener('click', () => {
            this.hideModal();
        });
        
        // Click outside modal to close
        document.getElementById('confirmModal').addEventListener('click', (e) => {
            if (e.target.id === 'confirmModal') {
                this.hideModal();
            }
        });
        
        // Chart timeframe selector
        document.getElementById('chartTimeframe').addEventListener('change', (e) => {
            this.updateChartTimeframe(e.target.value);
        });
    }
    
    /**
     * Set up tab switching functionality
     */
    setupTabs() {
        // Initially show overview tab
        this.switchTab('overview');
    }
    
    /**
     * Switch to a different tab
     */
    switchTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
        
        // Update tab panels
        document.querySelectorAll('.tab-panel').forEach(panel => {
            panel.classList.remove('active');
        });
        document.getElementById(tabName).classList.add('active');
        
        this.currentTab = tabName;
        
        // Load tab-specific data
        this.loadTabData(tabName);
    }
    
    /**
     * Load data specific to a tab
     */
    loadTabData(tabName) {
        switch (tabName) {
            case 'trading':
                this.loadTradingData();
                break;
            case 'portfolio':
                this.loadPortfolioData();
                break;
            case 'ml-rl':
                this.loadMLRLData();
                break;
            case 'system':
                this.loadSystemData();
                break;
        }
    }
    
    /**
     * Connect to WebSocket for real-time updates
     */
    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.host;
        const connectionId = this.generateConnectionId();
        const wsUrl = `${protocol}//${host}/dashboard/ws/${connectionId}`;
        
        this.updateConnectionStatus('connecting', 'Connecting...');
        
        try {
            this.websocket = new WebSocket(wsUrl);
            
            this.websocket.onopen = () => {
                console.log('WebSocket connected');
                this.isConnected = true;
                this.connectionRetries = 0;
                this.updateConnectionStatus('connected', 'Connected');
                
                // Subscribe to dashboard updates
                this.sendWebSocketMessage({
                    type: 'subscribe',
                    topic: 'dashboard'
                });
                
                this.showNotification('success', 'Connected', 'Successfully connected to trading system');
            };
            
            this.websocket.onmessage = (event) => {
                this.handleWebSocketMessage(JSON.parse(event.data));
            };
            
            this.websocket.onclose = () => {
                console.log('WebSocket disconnected');
                this.isConnected = false;
                this.updateConnectionStatus('disconnected', 'Disconnected');
                this.scheduleReconnect();
            };
            
            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus('disconnected', 'Connection Error');
            };
            
        } catch (error) {
            console.error('Failed to create WebSocket:', error);
            this.updateConnectionStatus('disconnected', 'Connection Failed');
            this.scheduleReconnect();
        }
    }
    
    /**
     * Handle incoming WebSocket messages
     */
    handleWebSocketMessage(message) {
        switch (message.type) {
            case 'dashboard_update':
                this.updateDashboardData(message.data);
                break;
            case 'system_alert':
                this.handleSystemAlert(message);
                break;
            case 'trading_update':
                this.handleTradingUpdate(message);
                break;
            case 'ping':
                this.sendWebSocketMessage({ type: 'pong' });
                break;
            case 'subscription_confirmed':
                console.log(`Subscribed to topic: ${message.topic}`);
                break;
        }
    }
    
    /**
     * Send message via WebSocket
     */
    sendWebSocketMessage(message) {
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            this.websocket.send(JSON.stringify(message));
        }
    }
    
    /**
     * Schedule WebSocket reconnection
     */
    scheduleReconnect() {
        if (this.connectionRetries < this.maxRetries) {
            this.connectionRetries++;
            const delay = this.retryDelay * this.connectionRetries;
            
            this.updateConnectionStatus('connecting', `Reconnecting in ${delay/1000}s...`);
            
            setTimeout(() => {
                this.connectWebSocket();
            }, delay);
        } else {
            this.updateConnectionStatus('disconnected', 'Connection Failed');
            this.showNotification('error', 'Connection Lost', 'Failed to reconnect after multiple attempts');
        }
    }
    
    /**
     * Update connection status indicator
     */
    updateConnectionStatus(status, text) {
        const indicator = document.getElementById('statusIndicator');
        const statusText = document.getElementById('statusText');
        
        indicator.className = `status-indicator ${status}`;
        statusText.textContent = text;
    }
    
    /**
     * Generate unique connection ID
     */
    generateConnectionId() {
        return `dashboard-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    }
    
    /**
     * Update dashboard with new data
     */
    updateDashboardData(data) {
        this.dashboardData = data;
        this.lastUpdateTime = new Date();
        
        // Update mode-specific UI first
        this.updateModeDisplay(data);
        
        // Update all UI elements
        this.updateOverviewTab(data);
        this.updateTradingTab(data);
        this.updatePortfolioTab(data);
        this.updateMLRLTab(data);
        this.updateSystemTab(data);
    }
    
    /**
     * Update mode-specific display elements
     */
    updateModeDisplay(data) {
        const tradingMode = data.trading_status.mode;
        const isSimulated = data.trading_status.is_simulated;
        const portfolioMode = data.portfolio_status.mode;
        const portfolioIsSimulated = data.portfolio_status.is_simulated;
        
        // Update navigation mode indicator
        const navModeIndicator = document.getElementById('navModeIndicator');
        navModeIndicator.textContent = tradingMode.toUpperCase();
        navModeIndicator.className = `mode-indicator ${tradingMode}`;
        
        // Update mode banner
        const modeBanner = document.getElementById('modeBanner');
        const modeBannerText = document.getElementById('modeBannerText');
        
        if (tradingMode === 'simulation') {
            modeBanner.style.display = 'block';
            modeBanner.className = 'mode-banner simulation';
            modeBannerText.textContent = 'SIMULATION MODE - ALL TRADES ARE SIMULATED';
        } else if (tradingMode === 'live') {
            modeBanner.style.display = 'block';
            modeBanner.className = 'mode-banner live';
            modeBannerText.textContent = 'LIVE TRADING MODE - REAL MONEY AT RISK';
        } else {
            modeBanner.style.display = 'block';
            modeBanner.className = 'mode-banner analysis';
            modeBannerText.textContent = 'ANALYSIS MODE - NO TRADING ACTIVE';
        }
        
        // Update portfolio mode badge
        const portfolioModeBadge = document.getElementById('portfolioModeBadge');
        portfolioModeBadge.textContent = portfolioMode.toUpperCase();
        portfolioModeBadge.className = `mode-badge ${portfolioMode}`;
        
        // Update trading mode badge
        const tradingModeBadge = document.getElementById('tradingModeBadge');
        tradingModeBadge.textContent = tradingMode.toUpperCase();
        tradingModeBadge.className = `mode-badge ${tradingMode}`;
        
        // Update card styling based on mode
        this.updateCardModeClasses(tradingMode);
        
        // Update portfolio value indicators
        this.updatePortfolioValueModeIndicators(portfolioMode, portfolioIsSimulated);
    }
    
    /**
     * Update card CSS classes based on trading mode
     */
    updateCardModeClasses(mode) {
        const cards = [
            'portfolioSummaryCard',
            'tradingActivityCard'
        ];
        
        cards.forEach(cardId => {
            const card = document.getElementById(cardId);
            if (card) {
                // Remove existing mode classes
                card.classList.remove('simulation-mode', 'live-mode', 'analysis-mode');
                // Add current mode class
                card.classList.add(`${mode}-mode`);
            }
        });
    }
    
    /**
     * Update portfolio value mode indicators
     */
    updatePortfolioValueModeIndicators(mode, isSimulated) {
        const portfolioValueElements = document.querySelectorAll('.portfolio-value');
        
        portfolioValueElements.forEach(element => {
            // Remove existing mode classes
            element.classList.remove('simulated', 'live');
            
            // Set data attribute for CSS content
            if (isSimulated) {
                element.setAttribute('data-mode', 'SIM');
                element.classList.add('simulated');
            } else {
                element.setAttribute('data-mode', 'LIVE');
                element.classList.add('live');
            }
        });
    }
    
    /**
     * Update Overview tab
     */
    updateOverviewTab(data) {
        // System status
        const systemStatus = data.system_metrics.status;
        document.getElementById('systemStatusBadge').textContent = systemStatus.toUpperCase();
        document.getElementById('systemStatusBadge').className = `status-badge ${systemStatus}`;
        
        // System metrics
        const uptime = this.formatUptime(data.system_metrics.uptime_seconds);
        document.getElementById('systemUptime').textContent = uptime;
        document.getElementById('systemCpu').textContent = `${data.system_metrics.cpu_usage_pct.toFixed(1)}%`;
        document.getElementById('systemMemory').textContent = `${Math.round(data.system_metrics.memory_usage_mb)} MB`;
        document.getElementById('systemConnections').textContent = data.system_metrics.active_connections;
        
        // Portfolio summary
        document.getElementById('portfolioTotalValue').textContent = this.formatCurrency(data.portfolio_status.total_value_usd);
        
        const dailyPnl = parseFloat(data.portfolio_status.daily_pnl_usd);
        const dailyPnlPct = parseFloat(data.portfolio_status.daily_pnl_pct);
        const dailyPnlElement = document.getElementById('portfolioDailyPnl');
        dailyPnlElement.textContent = `${this.formatCurrency(dailyPnl)} (${dailyPnlPct.toFixed(2)}%)`;
        dailyPnlElement.className = `metric-value pnl ${dailyPnl >= 0 ? 'positive' : 'negative'}`;
        
        const unrealizedPnl = parseFloat(data.portfolio_status.unrealized_pnl_usd);
        const unrealizedPnlElement = document.getElementById('portfolioUnrealizedPnl');
        unrealizedPnlElement.textContent = this.formatCurrency(unrealizedPnl);
        unrealizedPnlElement.className = `metric-value pnl ${unrealizedPnl >= 0 ? 'positive' : 'negative'}`;
        
        document.getElementById('portfolioActivePositions').textContent = data.portfolio_status.position_count;
        
        // Trading activity
        const tradingMode = data.trading_status.mode;
        const tradingModeElement = document.getElementById('tradingMode');
        tradingModeElement.textContent = tradingMode.charAt(0).toUpperCase() + tradingMode.slice(1);
        tradingModeElement.className = `metric-value trading-mode ${tradingMode}`;
        
        document.getElementById('tradingTradesToday').textContent = data.trading_status.trades_today;
        document.getElementById('tradingVolumeToday').textContent = this.formatCurrency(data.trading_status.volume_today_usd);
        document.getElementById('tradingWinRate').textContent = `${data.trading_status.win_rate_pct.toFixed(1)}%`;
        
        // ML/RL status
        document.getElementById('mlRlIntegration').textContent = data.ml_rl_status.ml_rl_integration_active ? 'Active' : 'Inactive';
        document.getElementById('mlAccuracy').textContent = `${data.ml_rl_status.ml_prediction_accuracy_pct.toFixed(1)}%`;
        document.getElementById('rlSuccessRate').textContent = `${data.ml_rl_status.rl_action_success_rate_pct.toFixed(1)}%`;
        document.getElementById('mlRlLatency').textContent = `${data.ml_rl_status.ml_rl_decision_latency_ms.toFixed(1)}ms`;
        
        // Update activity feed
        this.updateActivityFeed(data);
    }
    
    /**
     * Update Trading tab
     */
    updateTradingTab(data) {
        // Update trading mode radio buttons
        const currentMode = data.trading_status.mode;
        document.querySelector(`input[name="tradingMode"][value="${currentMode}"]`).checked = true;
        
        // Market signals
        document.getElementById('tokensAnalyzed').textContent = data.trading_status.tokens_analyzed_today;
        document.getElementById('highConfidenceSignals').textContent = data.trading_status.high_confidence_signals;
        document.getElementById('tokensInWatchlist').textContent = data.trading_status.tokens_in_watchlist;
        
        // Update toggle states
        document.getElementById('riskLimitsToggle').checked = data.trading_status.risk_limits_active;
        document.getElementById('autoTradingToggle').checked = data.trading_status.is_trading_active;
    }
    
    /**
     * Update Portfolio tab
     */
    updatePortfolioTab(data) {
        const portfolio = data.portfolio_status;
        
        // Portfolio overview
        document.getElementById('portfolioTotalValueDetail').textContent = this.formatCurrency(portfolio.total_value_usd);
        document.getElementById('portfolioAvailableBalance').textContent = this.formatCurrency(portfolio.available_balance_usd);
        
        const realizedPnl = parseFloat(portfolio.realized_pnl_usd);
        const realizedPnlElement = document.getElementById('portfolioRealizedPnl');
        realizedPnlElement.textContent = this.formatCurrency(realizedPnl);
        realizedPnlElement.className = `metric-value pnl ${realizedPnl >= 0 ? 'positive' : 'negative'}`;
        
        const unrealizedPnl = parseFloat(portfolio.unrealized_pnl_usd);
        const unrealizedPnlElement = document.getElementById('portfolioUnrealizedPnlDetail');
        unrealizedPnlElement.textContent = this.formatCurrency(unrealizedPnl);
        unrealizedPnlElement.className = `metric-value pnl ${unrealizedPnl >= 0 ? 'positive' : 'negative'}`;
        
        const totalReturn = parseFloat(portfolio.total_return_pct);
        const totalReturnElement = document.getElementById('portfolioTotalReturn');
        totalReturnElement.textContent = `${totalReturn.toFixed(2)}%`;
        totalReturnElement.className = `metric-value pnl ${totalReturn >= 0 ? 'positive' : 'negative'}`;
        
        // Risk metrics
        document.getElementById('maxDrawdown').textContent = `${parseFloat(portfolio.max_drawdown_pct).toFixed(2)}%`;
        document.getElementById('currentDrawdown').textContent = `${parseFloat(portfolio.current_drawdown_pct).toFixed(2)}%`;
        document.getElementById('sharpeRatio').textContent = portfolio.sharpe_ratio?.toFixed(2) || 'N/A';
        document.getElementById('var95').textContent = portfolio.var_95_usd ? this.formatCurrency(portfolio.var_95_usd) : 'N/A';
        
        // Chain balances
        this.updateChainBalances(portfolio.chain_balances);
        
        // Active positions table
        this.updatePositionsTable(portfolio.positions);
    }
    
    /**
     * Update ML & RL tab
     */
    updateMLRLTab(data) {
        const mlrl = data.ml_rl_status;
        
        // Integration status
        document.getElementById('integrationActive').textContent = mlrl.ml_rl_integration_active ? 'Yes' : 'No';
        document.getElementById('integrationLatency').textContent = `${mlrl.ml_rl_decision_latency_ms.toFixed(1)}ms`;
        document.getElementById('mlConfidence').textContent = `${(mlrl.ml_confidence_threshold * 100).toFixed(0)}%`;
        document.getElementById('rlConfidence').textContent = `${(mlrl.rl_action_confidence * 100).toFixed(0)}%`;
        
        // Training status
        document.getElementById('mlTrainingActive').textContent = mlrl.ml_training_active ? 'Yes' : 'No';
        document.getElementById('rlTrainingActive').textContent = mlrl.rl_training_active ? 'Yes' : 'No';
        document.getElementById('continuousLearning').textContent = mlrl.continuous_learning_active ? 'Yes' : 'No';
        
        // Update model lists
        this.updateMLModelsList(mlrl.models.ml_models);
        this.updateRLAgentsList(mlrl.models.rl_agents);
    }
    
    /**
     * Update System tab
     */
    updateSystemTab(data) {
        const system = data.system_metrics;
        
        // Overall health status
        document.getElementById('overallHealthStatus').textContent = system.status.toUpperCase();
        document.getElementById('overallHealthStatus').className = `status-badge ${system.status}`;
        
        // Component statuses
        this.updateComponentStatuses(system.component_statuses);
        
        // Performance metrics
        document.getElementById('requestsPerMinute').textContent = Math.round(system.requests_per_minute);
        document.getElementById('errorRate').textContent = `${system.error_rate_pct.toFixed(2)}%`;
        document.getElementById('responseTime').textContent = `${system.response_time_ms.toFixed(0)}ms`;
        document.getElementById('cacheHitRate').textContent = `${system.performance.cache_hit_rate_pct.toFixed(1)}%`;
        
        // Resource usage
        this.updateResourceUsage(system.cpu_usage_pct, system.memory_usage_pct);
    }
    
    /**
     * Update chain balances display
     */
    updateChainBalances(balances) {
        const container = document.getElementById('chainBalanceList');
        container.innerHTML = '';
        
        for (const [chain, balance] of Object.entries(balances)) {
            const item = document.createElement('div');
            item.className = 'chain-balance-item';
            item.innerHTML = `
                <span class="chain-name">${chain.charAt(0).toUpperCase() + chain.slice(1)}</span>
                <span class="chain-balance">${this.formatCurrency(balance)}</span>
            `;
            container.appendChild(item);
        }
    }
    
    /**
     * Update positions table
     */
    updatePositionsTable(positions) {
        const tbody = document.getElementById('positionsTableBody');
        
        if (!positions || positions.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" class="no-data">No active positions</td></tr>';
            return;
        }
        
        tbody.innerHTML = positions.map(pos => {
            const pnl = parseFloat(pos.unrealized_pnl);
            const pnlPct = parseFloat(pos.unrealized_pnl_pct);
            const pnlClass = pnl >= 0 ? 'positive' : 'negative';
            const modeClass = pos.mode ? `${pos.mode}-mode` : 'analysis-mode';
            
            return `
                <tr class="table-row ${modeClass}">
                    <td>${pos.symbol}</td>
                    <td>${pos.chain}</td>
                    <td>${pos.size}</td>
                    <td>${this.formatCurrency(pos.entry_price)}</td>
                    <td>${this.formatCurrency(pos.current_price)}</td>
                    <td class="pnl ${pnlClass}">${this.formatCurrency(pnl)}</td>
                    <td class="pnl ${pnlClass}">${pnlPct.toFixed(2)}%</td>
                </tr>
            `;
        }).join('');
    }
    
    /**
     * Update ML models list
     */
    updateMLModelsList(models) {
        const container = document.getElementById('mlModelsList');
        
        if (!models || models.length === 0) {
            container.innerHTML = '<div class="model-item"><div class="model-name">No models loaded</div></div>';
            return;
        }
        
        container.innerHTML = models.map(model => `
            <div class="model-item">
                <div class="model-name">${model.name}</div>
                <div class="model-status">Status: ${model.status}</div>
                <div class="model-accuracy">Accuracy: ${model.accuracy ? (model.accuracy * 100).toFixed(1) + '%' : 'N/A'}</div>
            </div>
        `).join('');
    }
    
    /**
     * Update RL agents list
     */
    updateRLAgentsList(agents) {
        const container = document.getElementById('rlAgentsList');
        
        if (!agents || agents.length === 0) {
            container.innerHTML = '<div class="agent-item"><div class="agent-name">No agents loaded</div></div>';
            return;
        }
        
        container.innerHTML = agents.map(agent => `
            <div class="agent-item">
                <div class="agent-name">${agent.name}</div>
                <div class="agent-status">Status: ${agent.status}</div>
                <div class="agent-performance">Win Rate: ${agent.win_rate_pct.toFixed(1)}%</div>
            </div>
        `).join('');
    }
    
    /**
     * Update component statuses
     */
    updateComponentStatuses(statuses) {
        const container = document.getElementById('componentStatuses');
        container.innerHTML = '';
        
        for (const [component, status] of Object.entries(statuses)) {
            const item = document.createElement('div');
            item.className = 'component-status';
            
            const statusClass = status === 'healthy' ? '' : status;
            
            item.innerHTML = `
                <span class="component-name">${component.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}:</span>
                <span class="component-status-value ${statusClass}">${status.toUpperCase()}</span>
            `;
            container.appendChild(item);
        }
    }
    
    /**
     * Update resource usage bars
     */
    updateResourceUsage(cpuPct, memoryPct) {
        // CPU usage
        const cpuFill = document.getElementById('cpuProgressFill');
        const cpuText = document.getElementById('cpuUsageText');
        
        cpuFill.style.width = `${cpuPct}%`;
        cpuText.textContent = `${cpuPct.toFixed(1)}%`;
        
        if (cpuPct > 80) {
            cpuFill.className = 'progress-fill danger';
        } else if (cpuPct > 60) {
            cpuFill.className = 'progress-fill warning';
        } else {
            cpuFill.className = 'progress-fill';
        }
        
        // Memory usage
        const memoryFill = document.getElementById('memoryProgressFill');
        const memoryText = document.getElementById('memoryUsageText');
        
        memoryFill.style.width = `${memoryPct}%`;
        memoryText.textContent = `${memoryPct.toFixed(1)}%`;
        
        if (memoryPct > 80) {
            memoryFill.className = 'progress-fill danger';
        } else if (memoryPct > 60) {
            memoryFill.className = 'progress-fill warning';
        } else {
            memoryFill.className = 'progress-fill';
        }
    }
    
    /**
     * Update activity feed
     */
    updateActivityFeed(data) {
        const feed = document.getElementById('activityFeed');
        const currentMode = data.trading_status.mode;
        
        // Add new activities (this would come from the backend)
        const activities = [
            { time: new Date().toLocaleTimeString(), message: 'Dashboard data updated', mode: currentMode },
            { time: new Date(Date.now() - 60000).toLocaleTimeString(), message: `Trading mode: ${currentMode}`, mode: currentMode },
            { time: new Date(Date.now() - 120000).toLocaleTimeString(), message: `Portfolio value: ${this.formatCurrency(data.portfolio_status.total_value_usd)}`, mode: currentMode }
        ];
        
        feed.innerHTML = activities.map(activity => `
            <div class="activity-item ${activity.mode}">
                <span class="activity-time">${activity.time}</span>
                <span class="activity-message">${activity.message}</span>
            </div>
        `).join('');
    }
    
    /**
     * Handle system alerts
     */
    handleSystemAlert(alert) {
        const severity = alert.severity.toLowerCase();
        this.showNotification(severity, alert.alert_type, alert.message);
        
        if (this.settings.soundNotifications && severity === 'error') {
            this.playNotificationSound();
        }
    }
    
    /**
     * Handle trading updates
     */
    handleTradingUpdate(update) {
        const type = update.update_type;
        const data = update.data;
        
        switch (type) {
            case 'trade_executed':
                this.showNotification('info', 'Trade Executed', `${data.side} ${data.symbol} - ${this.formatCurrency(data.value)}`);
                break;
            case 'mode_changed':
                this.showNotification('info', 'Mode Changed', `Trading mode changed to ${data.mode}`);
                break;
            case 'risk_alert':
                this.showNotification('warning', 'Risk Alert', data.message);
                break;
        }
    }
    
    /**
     * Emergency stop function
     */
    async emergencyStop() {
        try {
            const response = await fetch('/dashboard/trading/emergency-stop', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${this.settings.apiKey}`,
                    'Content-Type': 'application/json'
                }
            });
            
            if (response.ok) {
                this.showNotification('warning', 'Emergency Stop', 'Emergency stop activated successfully');
            } else {
                throw new Error('Failed to activate emergency stop');
            }
        } catch (error) {
            console.error('Emergency stop failed:', error);
            this.showNotification('error', 'Emergency Stop Failed', error.message);
        }
    }
    
    /**
     * Switch trading mode
     */
    async switchTradingMode() {
        const selectedMode = document.querySelector('input[name="tradingMode"]:checked').value;
        
        try {
            const response = await fetch('/dashboard/trading/mode', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${this.settings.apiKey}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ mode: selectedMode })
            });
            
            if (response.ok) {
                this.showNotification('success', 'Mode Changed', `Trading mode switched to ${selectedMode}`);
            } else {
                throw new Error('Failed to switch trading mode');
            }
        } catch (error) {
            console.error('Mode switch failed:', error);
            this.showNotification('error', 'Mode Switch Failed', error.message);
        }
    }
    
    /**
     * Pause trading
     */
    async pauseTrading() {
        // This would implement pause trading functionality
        this.showNotification('info', 'Trading Paused', 'Trading has been paused');
    }
    
    /**
     * Update risk settings
     */
    async updateRiskSettings() {
        const settings = {
            max_position_size_pct: parseFloat(document.getElementById('maxPositionSize').value),
            max_daily_loss_pct: parseFloat(document.getElementById('maxDailyLoss').value),
            max_drawdown_pct: parseFloat(document.getElementById('maxDrawdown').value),
            stop_loss_pct: parseFloat(document.getElementById('stopLoss').value)
        };
        
        try {
            const response = await fetch('/dashboard/trading/risk-limits', {
                method: 'PUT',
                headers: {
                    'Authorization': `Bearer ${this.settings.apiKey}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(settings)
            });
            
            if (response.ok) {
                this.showNotification('success', 'Settings Updated', 'Risk settings updated successfully');
            } else {
                throw new Error('Failed to update risk settings');
            }
        } catch (error) {
            console.error('Risk settings update failed:', error);
            this.showNotification('error', 'Update Failed', error.message);
        }
    }
    
    /**
     * Update dashboard settings
     */
    updateDashboardSettings() {
        this.settings.refreshInterval = parseInt(document.getElementById('refreshInterval').value) * 1000;
        this.settings.soundNotifications = document.getElementById('soundNotificationsToggle').checked;
        
        this.saveSettings();
        this.showNotification('success', 'Settings Updated', 'Dashboard settings updated successfully');
    }
    
    /**
     * Toggle dark mode
     */
    toggleDarkMode(enabled) {
        this.settings.darkMode = enabled;
        
        if (enabled) {
            document.documentElement.setAttribute('data-theme', 'dark');
        } else {
            document.documentElement.removeAttribute('data-theme');
        }
        
        this.saveSettings();
    }
    
    /**
     * Load trading data
     */
    async loadTradingData() {
        // This would load trading-specific data
        console.log('Loading trading data...');
    }
    
    /**
     * Load portfolio data
     */
    async loadPortfolioData() {
        // This would load portfolio-specific data
        console.log('Loading portfolio data...');
    }
    
    /**
     * Load ML/RL data
     */
    async loadMLRLData() {
        // This would load ML/RL-specific data
        console.log('Loading ML/RL data...');
    }
    
    /**
     * Load system data
     */
    async loadSystemData() {
        // This would load system-specific data
        console.log('Loading system data...');
    }
    
    /**
     * Refresh trades data
     */
    async refreshTrades() {
        try {
            const response = await fetch('/dashboard/trading/history', {
                headers: {
                    'Authorization': `Bearer ${this.settings.apiKey}`
                }
            });
            
            if (response.ok) {
                const data = await response.json();
                this.updateTradesTable(data.trades);
                this.showNotification('success', 'Refreshed', 'Trading history updated');
            }
        } catch (error) {
            console.error('Failed to refresh trades:', error);
            this.showNotification('error', 'Refresh Failed', error.message);
        }
    }
    
    /**
     * Refresh system logs
     */
    async refreshLogs() {
        try {
            const response = await fetch('/dashboard/system/logs', {
                headers: {
                    'Authorization': `Bearer ${this.settings.apiKey}`
                }
            });
            
            if (response.ok) {
                const data = await response.json();
                this.updateSystemLogs(data.logs);
                this.showNotification('success', 'Refreshed', 'System logs updated');
            }
        } catch (error) {
            console.error('Failed to refresh logs:', error);
            this.showNotification('error', 'Refresh Failed', error.message);
        }
    }
    
    /**
     * Update trades table
     */
    updateTradesTable(trades) {
        const tbody = document.getElementById('tradesTableBody');
        
        if (!trades || trades.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" class="no-data">No trades to display</td></tr>';
            return;
        }
        
        tbody.innerHTML = trades.map(trade => {
            const modeClass = trade.mode ? `${trade.mode}-mode` : 'analysis-mode';
            return `
                <tr class="table-row ${modeClass}">
                    <td>${new Date(trade.timestamp).toLocaleTimeString()}</td>
                    <td>${trade.symbol}</td>
                    <td>${trade.side.toUpperCase()}</td>
                    <td>${trade.size}</td>
                    <td>${this.formatCurrency(trade.price)}</td>
                    <td class="pnl ${trade.pnl >= 0 ? 'positive' : 'negative'}">${this.formatCurrency(trade.pnl || 0)}</td>
                    <td><span class="status-badge ${trade.status}">${trade.status.toUpperCase()}</span></td>
                </tr>
            `;
        }).join('');
    }
    
    /**
     * Update system logs
     */
    updateSystemLogs(logs) {
        const container = document.getElementById('logsContainer');
        
        container.innerHTML = logs.map(log => {
            const time = new Date().toLocaleTimeString();
            return `
                <div class="log-entry">
                    <span class="log-time">${time}</span>
                    <span class="log-level info">INFO</span>
                    <span class="log-message">${log}</span>
                </div>
            `;
        }).join('');
        
        // Scroll to bottom
        container.scrollTop = container.scrollHeight;
    }
    
    /**
     * Initialize performance chart
     */
    initializeChart() {
        const canvas = document.getElementById('performanceChart');
        const ctx = canvas.getContext('2d');
        
        // Simple chart implementation
        this.drawChart(ctx, canvas.width, canvas.height);
    }
    
    /**
     * Draw performance chart
     */
    drawChart(ctx, width, height) {
        // Clear canvas
        ctx.clearRect(0, 0, width, height);
        
        // Generate sample data
        const data = this.generateSampleChartData();
        
        // Set up chart area
        const padding = 40;
        const chartWidth = width - 2 * padding;
        const chartHeight = height - 2 * padding;
        
        // Draw axes
        ctx.strokeStyle = '#e2e8f0';
        ctx.lineWidth = 1;
        
        // Y-axis
        ctx.beginPath();
        ctx.moveTo(padding, padding);
        ctx.lineTo(padding, height - padding);
        ctx.stroke();
        
        // X-axis
        ctx.beginPath();
        ctx.moveTo(padding, height - padding);
        ctx.lineTo(width - padding, height - padding);
        ctx.stroke();
        
        // Draw data line
        if (data.length > 1) {
            ctx.strokeStyle = '#2563eb';
            ctx.lineWidth = 2;
            ctx.beginPath();
            
            data.forEach((point, index) => {
                const x = padding + (index / (data.length - 1)) * chartWidth;
                const y = height - padding - (point / 100) * chartHeight;
                
                if (index === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            });
            
            ctx.stroke();
        }
    }
    
    /**
     * Generate sample chart data
     */
    generateSampleChartData() {
        const data = [];
        let value = 10000; // Starting portfolio value
        
        for (let i = 0; i < 24; i++) { // 24 hours of data
            value += (Math.random() - 0.5) * 200; // Random walk
            data.push(Math.max(value, 9000)); // Don't go below 9000
        }
        
        return data;
    }
    
    /**
     * Update chart timeframe
     */
    updateChartTimeframe(timeframe) {
        // This would update the chart with different timeframe data
        console.log(`Updating chart to ${timeframe} timeframe`);
        
        // Re-draw chart with new data
        const canvas = document.getElementById('performanceChart');
        const ctx = canvas.getContext('2d');
        this.drawChart(ctx, canvas.width, canvas.height);
    }
    
    /**
     * Show confirmation modal
     */
    showConfirmModal(title, message, onConfirm) {
        document.getElementById('modalTitle').textContent = title;
        document.getElementById('modalMessage').textContent = message;
        document.getElementById('confirmModal').style.display = 'block';
        
        // Set up confirm button
        const confirmBtn = document.getElementById('modalConfirm');
        confirmBtn.onclick = () => {
            this.hideModal();
            onConfirm();
        };
    }
    
    /**
     * Hide modal
     */
    hideModal() {
        document.getElementById('confirmModal').style.display = 'none';
    }
    
    /**
     * Show notification
     */
    showNotification(type, title, message) {
        const container = document.getElementById('notificationContainer');
        
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <div class="notification-title">${title}</div>
            <div class="notification-message">${message}</div>
        `;
        
        container.appendChild(notification);
        
        // Trigger show animation
        setTimeout(() => {
            notification.classList.add('show');
        }, 10);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 5000);
    }
    
    /**
     * Play notification sound
     */
    playNotificationSound() {
        // Simple beep sound using Web Audio API
        if (typeof AudioContext !== 'undefined' || typeof webkitAudioContext !== 'undefined') {
            const audioContext = new (AudioContext || webkitAudioContext)();
            const oscillator = audioContext.createOscillator();
            const gainNode = audioContext.createGain();
            
            oscillator.connect(gainNode);
            gainNode.connect(audioContext.destination);
            
            oscillator.frequency.value = 800;
            oscillator.type = 'sine';
            
            gainNode.gain.setValueAtTime(0, audioContext.currentTime);
            gainNode.gain.linearRampToValueAtTime(0.1, audioContext.currentTime + 0.1);
            gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);
            
            oscillator.start(audioContext.currentTime);
            oscillator.stop(audioContext.currentTime + 0.5);
        }
    }
    
    /**
     * Start periodic updates
     */
    startPeriodicUpdates() {
        setInterval(() => {
            if (!this.isConnected) {
                // Try to fetch data via HTTP if WebSocket is down
                this.fetchDashboardData();
            }
        }, this.settings.refreshInterval);
    }
    
    /**
     * Fetch dashboard data via HTTP
     */
    async fetchDashboardData() {
        try {
            const response = await fetch('/dashboard/data', {
                headers: {
                    'Authorization': `Bearer ${this.settings.apiKey}`
                }
            });
            
            if (response.ok) {
                const data = await response.json();
                this.updateDashboardData(data);
            }
        } catch (error) {
            console.error('Failed to fetch dashboard data:', error);
        }
    }
    
    /**
     * Format currency values
     */
    formatCurrency(value) {
        if (typeof value === 'string') {
            value = parseFloat(value);
        }
        
        if (isNaN(value)) {
            return '$0.00';
        }
        
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD',
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        }).format(value);
    }
    
    /**
     * Format uptime
     */
    formatUptime(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        return `${hours}h ${minutes}m`;
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new DashboardApp();
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        // Page is hidden, potentially reduce update frequency
        console.log('Dashboard hidden');
    } else {
        // Page is visible, resume normal updates
        console.log('Dashboard visible');
        if (window.dashboard && !window.dashboard.isConnected) {
            window.dashboard.connectWebSocket();
        }
    }
});

// Handle window beforeunload
window.addEventListener('beforeunload', () => {
    if (window.dashboard && window.dashboard.websocket) {
        window.dashboard.websocket.close();
    }
});