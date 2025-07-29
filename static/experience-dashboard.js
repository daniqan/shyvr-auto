/**
 * Shyvr RLTE Experience Dashboard JavaScript
 * Real-time RL experience monitoring and visualization
 */

class ExperienceDashboard {
    constructor(dashboardApp) {
        this.dashboard = dashboardApp;
        this.logger = console;
        
        // Experience data cache
        this.experienceData = [];
        this.experienceStats = {};
        this.performanceMetrics = {};
        
        // Chart instances
        this.rewardChart = null;
        this.performanceChart = null;
        this.sessionChart = null;
        
        // UI state
        this.currentTimeframe = '24h';
        this.selectedSession = null;
        this.isAutoRefresh = true;
        this.refreshInterval = null;
        
        // Chart configurations
        this.chartConfig = {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    borderColor: '#444',
                    borderWidth: 1
                }
            },
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'minute',
                        displayFormats: {
                            minute: 'HH:mm',
                            hour: 'HH:mm'
                        }
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#666'
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#666'
                    }
                }
            }
        };
        
        this.logger.info('Experience dashboard initialized');
    }
    
    /**
     * Initialize experience dashboard functionality
     */
    async initialize() {
        try {
            // Check if we have Chart.js available
            if (typeof Chart === 'undefined') {
                this.logger.warn('Chart.js not available, loading from CDN');
                await this.loadChartJS();
            }
            
            // Set up experience monitoring
            this.setupExperienceMonitoring();
            
            // Initialize charts
            this.initializeCharts();
            
            // Set up event listeners
            this.setupEventListeners();
            
            // Start auto-refresh if enabled
            if (this.isAutoRefresh) {
                this.startAutoRefresh();
            }
            
            // Load initial data
            await this.refreshAllData();
            
            this.logger.info('Experience dashboard fully initialized');
            
        } catch (error) {
            this.logger.error('Failed to initialize experience dashboard:', error);
            this.showError('Failed to initialize experience dashboard');
        }
    }
    
    /**
     * Load Chart.js from CDN if not available
     */
    async loadChartJS() {
        return new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = 'https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.js';
            script.onload = () => resolve();
            script.onerror = () => reject(new Error('Failed to load Chart.js'));
            document.head.appendChild(script);
        });
    }
    
    /**
     * Set up experience monitoring integration
     */
    setupExperienceMonitoring() {
        // Subscribe to WebSocket updates for experience data
        if (this.dashboard.websocket) {
            this.dashboard.websocket.addEventListener('message', (event) => {
                const message = JSON.parse(event.data);
                this.handleWebSocketMessage(message);
            });
            
            // Subscribe to experience updates
            this.dashboard.sendWebSocketMessage({
                type: 'subscribe',
                topic: 'experiences'
            });
        }
        
        // Add experience tab to ML-RL section
        this.addExperienceTabToMLRL();
    }
    
    /**
     * Add experience monitoring tab to ML-RL section
     */
    addExperienceTabToMLRL() {
        const mlrlTab = document.getElementById('ml-rl');
        if (!mlrlTab) return;
        
        // Find the dashboard grid in ML-RL tab
        const dashboardGrid = mlrlTab.querySelector('.dashboard-grid');
        if (!dashboardGrid) return;
        
        // Create experience monitoring card
        const experienceCard = document.createElement('div');
        experienceCard.className = 'card experience-monitoring-card';
        experienceCard.innerHTML = `
            <div class="card-header">
                <h3>Experience Monitoring</h3>
                <div class="experience-controls">
                    <select id="experienceTimeframe" class="timeframe-selector">
                        <option value="1h">Last Hour</option>
                        <option value="4h">Last 4 Hours</option>
                        <option value="24h" selected>Last 24 Hours</option>
                        <option value="7d">Last 7 Days</option>
                    </select>
                    <button id="experienceRefresh" class="refresh-btn" title="Refresh Data">🔄</button>
                    <button id="experienceAutoRefresh" class="auto-refresh-btn active" title="Auto Refresh">⏱️</button>
                </div>
            </div>
            <div class="card-content">
                <div class="experience-metrics">
                    <div class="metric-item">
                        <span class="metric-label">Total Experiences</span>
                        <span class="metric-value" id="totalExperiences">0</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Average Reward</span>
                        <span class="metric-value" id="averageReward">0.00</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Success Rate</span>
                        <span class="metric-value" id="successRate">0%</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Active Sessions</span>
                        <span class="metric-value" id="activeSessions">0</span>
                    </div>
                </div>
                <div class="experience-chart-container">
                    <canvas id="experienceRewardChart" width="400" height="200"></canvas>
                </div>
            </div>
        `;
        
        // Add performance analysis card
        const performanceCard = document.createElement('div');
        performanceCard.className = 'card experience-performance-card';
        performanceCard.innerHTML = `
            <div class="card-header">
                <h3>Experience Performance</h3>
                <div class="performance-controls">
                    <button id="showPerformanceDetails" class="details-btn">Details</button>
                </div>
            </div>
            <div class="card-content">
                <div class="performance-metrics">
                    <div class="metric-row">
                        <span class="metric-label">Cumulative Reward:</span>
                        <span class="metric-value" id="cumulativeReward">0.00</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Win Rate:</span>
                        <span class="metric-value" id="winRate">0%</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Avg Episode Length:</span>
                        <span class="metric-value" id="avgEpisodeLength">0</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Training Efficiency:</span>
                        <span class="metric-value" id="trainingEfficiency">0%</span>
                    </div>
                </div>
                <div class="performance-chart-container">
                    <canvas id="experiencePerformanceChart" width="400" height="200"></canvas>
                </div>
            </div>
        `;
        
        // Add session analysis card
        const sessionCard = document.createElement('div');
        sessionCard.className = 'card experience-session-card';
        sessionCard.innerHTML = `
            <div class="card-header">
                <h3>Training Sessions</h3>
                <div class="session-controls">
                    <select id="sessionSelector" class="session-selector">
                        <option value="">All Sessions</option>
                    </select>
                    <button id="analyzeSession" class="analyze-btn">Analyze</button>
                </div>
            </div>
            <div class="card-content">
                <div class="session-list" id="sessionList">
                    <div class="session-item loading">Loading sessions...</div>
                </div>
                <div class="session-chart-container" style="display: none;">
                    <canvas id="sessionAnalysisChart" width="400" height="200"></canvas>
                </div>
            </div>
        `;
        
        // Insert cards into the grid
        dashboardGrid.appendChild(experienceCard);
        dashboardGrid.appendChild(performanceCard);
        dashboardGrid.appendChild(sessionCard);
        
        this.logger.info('Experience monitoring UI added to ML-RL tab');
    }
    
    /**
     * Initialize all charts
     */
    initializeCharts() {
        try {
            this.initializeRewardChart();
            this.initializePerformanceChart();
            this.initializeSessionChart();
            
            this.logger.info('Experience charts initialized');
        } catch (error) {
            this.logger.error('Failed to initialize charts:', error);
        }
    }
    
    /**
     * Initialize reward tracking chart
     */
    initializeRewardChart() {
        const canvas = document.getElementById('experienceRewardChart');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        
        this.rewardChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Average Reward',
                    data: [],
                    borderColor: '#4CAF50',
                    backgroundColor: 'rgba(76, 175, 80, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }, {
                    label: 'Cumulative Reward',
                    data: [],
                    borderColor: '#2196F3',
                    backgroundColor: 'rgba(33, 150, 243, 0.1)',
                    borderWidth: 2,
                    fill: false,
                    tension: 0.4
                }]
            },
            options: {
                ...this.chartConfig,
                scales: {
                    ...this.chartConfig.scales,
                    y: {
                        ...this.chartConfig.scales.y,
                        title: {
                            display: true,
                            text: 'Reward Value'
                        }
                    }
                }
            }
        });
    }
    
    /**
     * Initialize performance metrics chart
     */
    initializePerformanceChart() {
        const canvas = document.getElementById('experiencePerformanceChart');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        
        this.performanceChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Win Rate %',
                    data: [],
                    borderColor: '#FF9800',
                    backgroundColor: 'rgba(255, 152, 0, 0.1)',
                    borderWidth: 2,
                    yAxisID: 'y'
                }, {
                    label: 'Training Efficiency %',
                    data: [],
                    borderColor: '#9C27B0',
                    backgroundColor: 'rgba(156, 39, 176, 0.1)',
                    borderWidth: 2,
                    yAxisID: 'y'
                }, {
                    label: 'Episode Length',
                    data: [],
                    borderColor: '#607D8B',
                    backgroundColor: 'rgba(96, 125, 139, 0.1)',
                    borderWidth: 2,
                    yAxisID: 'y1'
                }]
            },
            options: {
                ...this.chartConfig,
                scales: {
                    ...this.chartConfig.scales,
                    y: {
                        ...this.chartConfig.scales.y,
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Percentage (%)'
                        },
                        min: 0,
                        max: 100
                    },
                    y1: {
                        ...this.chartConfig.scales.y,
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Episode Length'
                        },
                        grid: {
                            drawOnChartArea: false
                        }
                    }
                }
            }
        });
    }
    
    /**
     * Initialize session analysis chart
     */
    initializeSessionChart() {
        const canvas = document.getElementById('sessionAnalysisChart');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        
        this.sessionChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Total Experiences',
                    data: [],
                    backgroundColor: 'rgba(33, 150, 243, 0.6)',
                    borderColor: '#2196F3',
                    borderWidth: 1
                }, {
                    label: 'Avg Reward',
                    data: [],
                    backgroundColor: 'rgba(76, 175, 80, 0.6)',
                    borderColor: '#4CAF50',
                    borderWidth: 1,
                    yAxisID: 'y1'
                }]
            },
            options: {
                ...this.chartConfig,
                scales: {
                    ...this.chartConfig.scales,
                    y: {
                        ...this.chartConfig.scales.y,
                        title: {
                            display: true,
                            text: 'Experience Count'
                        }
                    },
                    y1: {
                        ...this.chartConfig.scales.y,
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Average Reward'
                        },
                        grid: {
                            drawOnChartArea: false
                        }
                    }
                }
            }
        });
    }
    
    /**
     * Set up event listeners for experience dashboard
     */
    setupEventListeners() {
        // Timeframe selector
        const timeframeSelector = document.getElementById('experienceTimeframe');
        if (timeframeSelector) {
            timeframeSelector.addEventListener('change', (e) => {
                this.currentTimeframe = e.target.value;
                this.refreshAllData();
            });
        }
        
        // Refresh button
        const refreshBtn = document.getElementById('experienceRefresh');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => {
                this.refreshAllData();
            });
        }
        
        // Auto-refresh toggle
        const autoRefreshBtn = document.getElementById('experienceAutoRefresh');
        if (autoRefreshBtn) {
            autoRefreshBtn.addEventListener('click', () => {
                this.toggleAutoRefresh();
            });
        }
        
        // Performance details button
        const detailsBtn = document.getElementById('showPerformanceDetails');
        if (detailsBtn) {
            detailsBtn.addEventListener('click', () => {
                this.showPerformanceDetails();
            });
        }
        
        // Session selector
        const sessionSelector = document.getElementById('sessionSelector');
        if (sessionSelector) {
            sessionSelector.addEventListener('change', (e) => {
                this.selectedSession = e.target.value || null;
                this.updateSessionAnalysis();
            });
        }
        
        // Analyze session button
        const analyzeBtn = document.getElementById('analyzeSession');
        if (analyzeBtn) {
            analyzeBtn.addEventListener('click', () => {
                this.analyzeSelectedSession();
            });
        }
    }
    
    /**
     * Handle WebSocket messages for experience updates
     */
    handleWebSocketMessage(message) {
        switch (message.type) {
            case 'experience_update':
                this.handleExperienceUpdate(message.data);
                break;
            case 'experience_stats':
                this.handleExperienceStats(message.data);
                break;
            case 'experience_performance':
                this.handlePerformanceUpdate(message.data);
                break;
            case 'training_session_update':
                this.handleSessionUpdate(message.data);
                break;
        }
    }
    
    /**
     * Handle real-time experience updates
     */
    handleExperienceUpdate(data) {
        // Add new experience to cache
        this.experienceData.push(data);
        
        // Keep only recent data to prevent memory issues
        if (this.experienceData.length > 1000) {
            this.experienceData = this.experienceData.slice(-500);
        }
        
        // Update charts with new data point
        this.updateRewardChart();
        
        // Update metrics
        this.updateExperienceMetrics();
        
        this.logger.debug('Experience update received:', data);
    }
    
    /**
     * Handle experience statistics updates
     */
    handleExperienceStats(stats) {
        this.experienceStats = stats;
        this.updateExperienceMetrics();
        this.updateSessionList();
        
        this.logger.debug('Experience stats updated:', stats);
    }
    
    /**
     * Handle performance metrics updates
     */
    handlePerformanceUpdate(performance) {
        this.performanceMetrics = performance;
        this.updatePerformanceMetrics();
        this.updatePerformanceChart();
        
        this.logger.debug('Performance metrics updated:', performance);
    }
    
    /**
     * Handle training session updates
     */
    handleSessionUpdate(sessionData) {
        this.updateSessionList();
        this.updateSessionChart();
        
        this.logger.debug('Session update received:', sessionData);
    }
    
    /**
     * Refresh all experience data
     */
    async refreshAllData() {
        try {
            // Show loading indicators
            this.showLoadingState();
            
            // Fetch all data in parallel
            const [recentExperiences, stats, performance] = await Promise.all([
                this.fetchRecentExperiences(),
                this.fetchExperienceStats(),
                this.fetchPerformanceMetrics()
            ]);
            
            // Update data
            if (recentExperiences) {
                this.experienceData = recentExperiences.experiences || [];
                this.updateRewardChart();
            }
            
            if (stats) {
                this.experienceStats = stats;
                this.updateExperienceMetrics();
                this.updateSessionList();
            }
            
            if (performance) {
                this.performanceMetrics = performance;
                this.updatePerformanceMetrics();
                this.updatePerformanceChart();
                this.updateSessionChart();
            }
            
            // Hide loading indicators
            this.hideLoadingState();
            
            this.logger.info('Experience data refreshed successfully');
            
        } catch (error) {
            this.logger.error('Failed to refresh experience data:', error);
            this.showError('Failed to refresh experience data: ' + error.message);
            this.hideLoadingState();
        }
    }
    
    /**
     * Fetch recent experiences from API
     */
    async fetchRecentExperiences() {
        const params = new URLSearchParams({
            limit: '100',
            hours_back: this.getHoursFromTimeframe(this.currentTimeframe).toString()
        });
        
        const response = await fetch(`/api/v1/experiences/recent?${params}`, {
            headers: {
                'Authorization': `Bearer ${this.dashboard.settings.apiKey}`,
                'Content-Type': 'application/json'
            }
        });
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status} ${response.statusText}`);
        }
        
        return await response.json();
    }
    
    /**
     * Fetch experience statistics from API
     */
    async fetchExperienceStats() {
        const params = new URLSearchParams({
            hours_back: this.getHoursFromTimeframe(this.currentTimeframe).toString()
        });
        
        const response = await fetch(`/api/v1/experiences/stats?${params}`, {
            headers: {
                'Authorization': `Bearer ${this.dashboard.settings.apiKey}`,
                'Content-Type': 'application/json'
            }
        });
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status} ${response.statusText}`);
        }
        
        return await response.json();
    }
    
    /**
     * Fetch performance metrics from API
     */
    async fetchPerformanceMetrics() {
        const params = new URLSearchParams({
            hours_back: this.getHoursFromTimeframe(this.currentTimeframe).toString()
        });
        
        const response = await fetch(`/api/v1/experiences/performance?${params}`, {
            headers: {
                'Authorization': `Bearer ${this.dashboard.settings.apiKey}`,
                'Content-Type': 'application/json'
            }
        });
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status} ${response.statusText}`);
        }
        
        return await response.json();
    }
    
    /**
     * Update reward chart with current data
     */
    updateRewardChart() {
        if (!this.rewardChart || !this.experienceData.length) return;
        
        try {
            // Process data for chart
            const timeSeriesData = this.processExperienceTimeSeriesData(this.experienceData);
            
            this.rewardChart.data.labels = timeSeriesData.labels;
            this.rewardChart.data.datasets[0].data = timeSeriesData.averageRewards;
            this.rewardChart.data.datasets[1].data = timeSeriesData.cumulativeRewards;
            
            this.rewardChart.update('none');
            
        } catch (error) {
            this.logger.error('Failed to update reward chart:', error);
        }
    }
    
    /**
     * Update performance chart with current data
     */
    updatePerformanceChart() {
        if (!this.performanceChart || !this.performanceMetrics.time_series) return;
        
        try {
            const timeSeries = this.performanceMetrics.time_series;
            
            this.performanceChart.data.labels = timeSeries.timestamps || [];
            this.performanceChart.data.datasets[0].data = timeSeries.win_rates || [];
            this.performanceChart.data.datasets[1].data = timeSeries.training_efficiency || [];
            this.performanceChart.data.datasets[2].data = timeSeries.episode_lengths || [];
            
            this.performanceChart.update('none');
            
        } catch (error) {
            this.logger.error('Failed to update performance chart:', error);
        }
    }
    
    /**
     * Update session analysis chart
     */
    updateSessionChart() {
        if (!this.sessionChart || !this.performanceMetrics.session_analysis) return;
        
        try {
            const analysis = this.performanceMetrics.session_analysis;
            
            this.sessionChart.data.labels = analysis.session_names || [];
            this.sessionChart.data.datasets[0].data = analysis.experience_counts || [];
            this.sessionChart.data.datasets[1].data = analysis.average_rewards || [];
            
            this.sessionChart.update('none');
            
            // Show chart container
            const container = document.querySelector('.session-chart-container');
            if (container) {
                container.style.display = 'block';
            }
            
        } catch (error) {
            this.logger.error('Failed to update session chart:', error);
        }
    }
    
    /**
     * Update experience metrics display
     */
    updateExperienceMetrics() {
        try {
            const totalElement = document.getElementById('totalExperiences');
            const avgRewardElement = document.getElementById('averageReward');
            const successRateElement = document.getElementById('successRate');
            const activeSessionsElement = document.getElementById('activeSessions');
            
            if (totalElement && this.experienceStats.total_experiences !== undefined) {
                totalElement.textContent = this.formatNumber(this.experienceStats.total_experiences);
            }
            
            if (avgRewardElement && this.experienceStats.average_reward !== undefined) {
                avgRewardElement.textContent = this.formatNumber(this.experienceStats.average_reward, 2);
            }
            
            if (successRateElement && this.experienceStats.success_rate !== undefined) {
                successRateElement.textContent = `${(this.experienceStats.success_rate * 100).toFixed(1)}%`;
            }
            
            if (activeSessionsElement && this.experienceStats.active_sessions !== undefined) {
                activeSessionsElement.textContent = this.formatNumber(this.experienceStats.active_sessions);
            }
            
        } catch (error) {
            this.logger.error('Failed to update experience metrics:', error);
        }
    }
    
    /**
     * Update performance metrics display
     */
    updatePerformanceMetrics() {
        try {
            const overall = this.performanceMetrics.overall_performance || {};
            
            const cumulativeElement = document.getElementById('cumulativeReward');
            const winRateElement = document.getElementById('winRate');
            const episodeLengthElement = document.getElementById('avgEpisodeLength');
            const efficiencyElement = document.getElementById('trainingEfficiency');
            
            if (cumulativeElement && overall.cumulative_reward !== undefined) {
                cumulativeElement.textContent = this.formatNumber(overall.cumulative_reward, 2);
            }
            
            if (winRateElement && overall.win_rate !== undefined) {
                winRateElement.textContent = `${(overall.win_rate * 100).toFixed(1)}%`;
            }
            
            if (episodeLengthElement && overall.average_episode_length !== undefined) {
                episodeLengthElement.textContent = this.formatNumber(overall.average_episode_length, 1);
            }
            
            if (efficiencyElement && overall.training_efficiency !== undefined) {
                efficiencyElement.textContent = `${(overall.training_efficiency * 100).toFixed(1)}%`;
            }
            
        } catch (error) {
            this.logger.error('Failed to update performance metrics:', error);
        }
    }
    
    /**
     * Update session list display
     */
    updateSessionList() {
        const sessionList = document.getElementById('sessionList');
        const sessionSelector = document.getElementById('sessionSelector');
        
        if (!sessionList || !this.experienceStats.sessions) return;
        
        try {
            const sessions = this.experienceStats.sessions;
            
            // Update session list
            sessionList.innerHTML = sessions.map(session => `
                <div class="session-item" data-session-id="${session.session_id}">
                    <div class="session-header">
                        <span class="session-name">${session.session_name}</span>
                        <span class="session-status ${session.status}">${session.status.toUpperCase()}</span>
                    </div>
                    <div class="session-metrics">
                        <span class="session-metric">
                            <label>Experiences:</label>
                            <value>${session.experience_count}</value>
                        </span>
                        <span class="session-metric">
                            <label>Avg Reward:</label>
                            <value>${this.formatNumber(session.average_reward, 2)}</value>
                        </span>
                        <span class="session-metric">
                            <label>Duration:</label>
                            <value>${this.formatDuration(session.duration_minutes)}</value>
                        </span>
                    </div>
                </div>
            `).join('');
            
            // Update session selector
            if (sessionSelector) {
                const currentValue = sessionSelector.value;
                sessionSelector.innerHTML = '<option value="">All Sessions</option>' +
                    sessions.map(session => `
                        <option value="${session.session_id}">${session.session_name}</option>
                    `).join('');
                
                // Restore previous selection if it still exists
                if (currentValue && sessions.find(s => s.session_id === currentValue)) {
                    sessionSelector.value = currentValue;
                }
            }
            
        } catch (error) {
            this.logger.error('Failed to update session list:', error);
        }
    }
    
    /**
     * Process experience data for time series visualization
     */
    processExperienceTimeSeriesData(experiences) {
        const timeSeriesData = {
            labels: [],
            averageRewards: [],
            cumulativeRewards: []
        };
        
        if (!experiences.length) return timeSeriesData;
        
        // Group experiences by time intervals
        const intervalMinutes = this.getIntervalFromTimeframe(this.currentTimeframe);
        const intervals = new Map();
        
        experiences.forEach(exp => {
            const timestamp = new Date(exp.created_at);
            const intervalKey = Math.floor(timestamp.getTime() / (intervalMinutes * 60 * 1000));
            
            if (!intervals.has(intervalKey)) {
                intervals.set(intervalKey, []);
            }
            intervals.get(intervalKey).push(exp);
        });
        
        // Process intervals
        let cumulativeReward = 0;
        const sortedIntervals = Array.from(intervals.entries()).sort((a, b) => a[0] - b[0]);
        
        sortedIntervals.forEach(([intervalKey, intervalExperiences]) => {
            const timestamp = new Date(intervalKey * intervalMinutes * 60 * 1000);
            const avgReward = intervalExperiences.reduce((sum, exp) => sum + exp.reward, 0) / intervalExperiences.length;
            cumulativeReward += avgReward * intervalExperiences.length;
            
            timeSeriesData.labels.push(timestamp);
            timeSeriesData.averageRewards.push(avgReward);
            timeSeriesData.cumulativeRewards.push(cumulativeReward);
        });
        
        return timeSeriesData;
    }
    
    /**
     * Show performance details modal/popup
     */
    showPerformanceDetails() {
        if (!this.performanceMetrics.detailed_analysis) {
            this.dashboard.showNotification('info', 'Performance Details', 'No detailed performance data available');
            return;
        }
        
        const details = this.performanceMetrics.detailed_analysis;
        const detailsHtml = `
            <div class="performance-details">
                <h4>Detailed Performance Analysis</h4>
                <div class="detail-section">
                    <h5>Reward Distribution</h5>
                    <p>Mean: ${this.formatNumber(details.reward_stats?.mean || 0, 4)}</p>
                    <p>Std Dev: ${this.formatNumber(details.reward_stats?.std || 0, 4)}</p>
                    <p>Min: ${this.formatNumber(details.reward_stats?.min || 0, 4)}</p>
                    <p>Max: ${this.formatNumber(details.reward_stats?.max || 0, 4)}</p>
                </div>
                <div class="detail-section">
                    <h5>Action Distribution</h5>
                    ${Object.entries(details.action_distribution || {}).map(([action, count]) => 
                        `<p>Action ${action}: ${count} (${((count / details.total_actions) * 100).toFixed(1)}%)</p>`
                    ).join('')}
                </div>
                <div class="detail-section">
                    <h5>Learning Progress</h5>
                    <p>Learning Rate: ${this.formatNumber(details.learning_rate || 0, 6)}</p>
                    <p>Exploration Rate: ${this.formatNumber(details.exploration_rate || 0, 4)}</p>
                    <p>Model Confidence: ${((details.model_confidence || 0) * 100).toFixed(1)}%</p>
                </div>
            </div>
        `;
        
        // Create and show modal (you can integrate with existing modal system)
        this.showModal('Performance Details', detailsHtml);
    }
    
    /**
     * Analyze selected training session
     */
    async analyzeSelectedSession() {
        if (!this.selectedSession) {
            this.dashboard.showNotification('warning', 'Session Analysis', 'Please select a session to analyze');
            return;
        }
        
        try {
            // Fetch detailed session data
            const response = await fetch(`/api/v1/experiences/search?session_id=${this.selectedSession}`, {
                headers: {
                    'Authorization': `Bearer ${this.dashboard.settings.apiKey}`,
                    'Content-Type': 'application/json'
                }
            });
            
            if (!response.ok) {
                throw new Error(`Failed to fetch session data: ${response.statusText}`);
            }
            
            const sessionData = await response.json();
            
            // Display session analysis
            this.showSessionAnalysis(sessionData);
            
        } catch (error) {
            this.logger.error('Failed to analyze session:', error);
            this.dashboard.showNotification('error', 'Session Analysis', 'Failed to analyze session: ' + error.message);
        }
    }
    
    /**
     * Show session analysis results
     */
    showSessionAnalysis(sessionData) {
        const experiences = sessionData.experiences || [];
        
        if (!experiences.length) {
            this.dashboard.showNotification('info', 'Session Analysis', 'No experiences found for selected session');
            return;
        }
        
        // Calculate session metrics
        const totalReward = experiences.reduce((sum, exp) => sum + exp.reward, 0);
        const avgReward = totalReward / experiences.length;
        const successfulExperiences = experiences.filter(exp => exp.reward > 0).length;
        const successRate = (successfulExperiences / experiences.length) * 100;
        
        const analysisHtml = `
            <div class="session-analysis">
                <h4>Session Analysis: ${sessionData.session_name || 'Unknown Session'}</h4>
                <div class="analysis-metrics">
                    <div class="metric-item">
                        <label>Total Experiences:</label>
                        <value>${experiences.length}</value>
                    </div>
                    <div class="metric-item">
                        <label>Total Reward:</label>
                        <value>${this.formatNumber(totalReward, 2)}</value>
                    </div>
                    <div class="metric-item">
                        <label>Average Reward:</label>
                        <value>${this.formatNumber(avgReward, 4)}</value>
                    </div>
                    <div class="metric-item">
                        <label>Success Rate:</label>
                        <value>${successRate.toFixed(1)}%</value>
                    </div>
                </div>
                <div class="analysis-charts">
                    <p>Detailed charts and analysis would be displayed here in a full implementation.</p>
                </div>
            </div>
        `;
        
        this.showModal('Session Analysis', analysisHtml);
    }
    
    /**
     * Toggle auto-refresh functionality
     */
    toggleAutoRefresh() {
        this.isAutoRefresh = !this.isAutoRefresh;
        
        const autoRefreshBtn = document.getElementById('experienceAutoRefresh');
        if (autoRefreshBtn) {
            autoRefreshBtn.classList.toggle('active', this.isAutoRefresh);
        }
        
        if (this.isAutoRefresh) {
            this.startAutoRefresh();
        } else {
            this.stopAutoRefresh();
        }
        
        this.dashboard.showNotification('info', 'Auto Refresh', 
            `Auto refresh ${this.isAutoRefresh ? 'enabled' : 'disabled'}`);
    }
    
    /**
     * Start auto-refresh timer
     */
    startAutoRefresh() {
        this.stopAutoRefresh(); // Clear any existing timer
        
        this.refreshInterval = setInterval(() => {
            this.refreshAllData();
        }, 30000); // Refresh every 30 seconds
        
        this.logger.info('Auto-refresh started');
    }
    
    /**
     * Stop auto-refresh timer
     */
    stopAutoRefresh() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
            this.refreshInterval = null;
        }
        
        this.logger.info('Auto-refresh stopped');
    }
    
    /**
     * Show loading state on UI elements
     */
    showLoadingState() {
        const elements = [
            'totalExperiences',
            'averageReward',
            'successRate',
            'activeSessions',
            'cumulativeReward',
            'winRate',
            'avgEpisodeLength',
            'trainingEfficiency'
        ];
        
        elements.forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = '...';
                element.classList.add('loading');
            }
        });
    }
    
    /**
     * Hide loading state from UI elements
     */
    hideLoadingState() {
        const loadingElements = document.querySelectorAll('.loading');
        loadingElements.forEach(element => {
            element.classList.remove('loading');
        });
    }
    
    /**
     * Show error message to user
     */
    showError(message) {
        if (this.dashboard && this.dashboard.showNotification) {
            this.dashboard.showNotification('error', 'Experience Dashboard Error', message);
        } else {
            console.error('Experience Dashboard Error:', message);
        }
    }
    
    /**
     * Show modal dialog (integrate with existing modal system if available)
     */
    showModal(title, content) {
        // This would integrate with the existing modal system
        // For now, we'll use a simple alert as fallback
        if (this.dashboard && this.dashboard.showConfirmModal) {
            this.dashboard.showConfirmModal(title, content, () => {});
        } else {
            alert(`${title}\n\n${content.replace(/<[^>]*>/g, '')}`);
        }
    }
    
    /**
     * Utility: Convert timeframe to hours
     */
    getHoursFromTimeframe(timeframe) {
        const timeframes = {
            '1h': 1,
            '4h': 4,
            '24h': 24,
            '7d': 168
        };
        return timeframes[timeframe] || 24;
    }
    
    /**
     * Utility: Get chart interval from timeframe
     */
    getIntervalFromTimeframe(timeframe) {
        const intervals = {
            '1h': 5,    // 5 minute intervals
            '4h': 15,   // 15 minute intervals
            '24h': 60,  // 1 hour intervals
            '7d': 360   // 6 hour intervals
        };
        return intervals[timeframe] || 60;
    }
    
    /**
     * Utility: Format numbers for display
     */
    formatNumber(value, decimals = 0) {
        if (typeof value !== 'number' || isNaN(value)) {
            return '0';
        }
        
        return value.toLocaleString('en-US', {
            minimumFractionDigits: decimals,
            maximumFractionDigits: decimals
        });
    }
    
    /**
     * Utility: Format duration in minutes to human readable
     */
    formatDuration(minutes) {
        if (!minutes || minutes < 1) {
            return '< 1min';
        }
        
        if (minutes < 60) {
            return `${Math.round(minutes)}min`;
        }
        
        const hours = Math.floor(minutes / 60);
        const remainingMinutes = Math.round(minutes % 60);
        
        if (hours < 24) {
            return remainingMinutes > 0 ? `${hours}h ${remainingMinutes}min` : `${hours}h`;
        }
        
        const days = Math.floor(hours / 24);
        const remainingHours = hours % 24;
        
        return remainingHours > 0 ? `${days}d ${remainingHours}h` : `${days}d`;
    }
    
    /**
     * Cleanup resources when dashboard is destroyed
     */
    cleanup() {
        this.stopAutoRefresh();
        
        // Destroy charts
        if (this.rewardChart) {
            this.rewardChart.destroy();
            this.rewardChart = null;
        }
        
        if (this.performanceChart) {
            this.performanceChart.destroy();
            this.performanceChart = null;
        }
        
        if (this.sessionChart) {
            this.sessionChart.destroy();
            this.sessionChart = null;
        }
        
        this.logger.info('Experience dashboard cleaned up');
    }
}

// Export for use in main dashboard
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ExperienceDashboard;
} else if (typeof window !== 'undefined') {
    window.ExperienceDashboard = ExperienceDashboard;
}