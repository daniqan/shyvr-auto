/**
 * Tests for backtest results frontend components
 */

// Mock Chart.js
global.Chart = {
    Chart: jest.fn().mockImplementation(() => ({
        destroy: jest.fn(),
        update: jest.fn(),
        data: { datasets: [] }
    })),
    registerables: []
};

// Mock dashboard app
global.dashboard = {
    formatCurrency: jest.fn((value) => `$${value.toFixed(2)}`),
    showNotification: jest.fn(),
    settings: { apiKey: 'test-key' }
};

// Mock fetch
global.fetch = jest.fn();

describe('BacktestResults', () => {
    let backtestResults;
    let container;

    beforeEach(() => {
        // Set up DOM
        document.body.innerHTML = `
            <div id="backtest-results-container">
                <div id="backtest-metrics"></div>
                <div id="backtest-equity-chart"></div>
                <div id="backtest-trade-history"></div>
                <div id="backtest-performance-stats"></div>
            </div>
        `;
        
        container = document.getElementById('backtest-results-container');
        
        // Reset mocks
        global.fetch.mockClear();
        global.dashboard.showNotification.mockClear();
        global.Chart.Chart.mockClear();
        
        // Import BacktestResults after DOM setup
        const BacktestResults = require('../../static/backtest-results');
        backtestResults = new BacktestResults();
    });

    afterEach(() => {
        document.body.innerHTML = '';
    });

    describe('Constructor', () => {
        test('should initialize with default values', () => {
            expect(backtestResults.currentResults).toBeNull();
            expect(backtestResults.equityChart).toBeNull();
            expect(backtestResults.isLoading).toBe(false);
        });

        test('should set up event listeners', () => {
            // Mock addEventListener
            const mockAddEventListener = jest.fn();
            document.getElementById = jest.fn(() => ({
                addEventListener: mockAddEventListener
            }));

            new (require('../../static/backtest-results'))();
            
            expect(mockAddEventListener).toHaveBeenCalled();
        });
    });

    describe('fetchBacktestResults', () => {
        const sampleResults = {
            backtest_results: {
                strategy_name: 'LSTM_DQN_Strategy',
                total_return: 0.125,
                sharpe_ratio: 1.42,
                max_drawdown: -0.08,
                equity_curve: [
                    { timestamp: '2024-01-01T00:00:00Z', value: 10000 },
                    { timestamp: '2024-01-31T23:59:59Z', value: 11250 }
                ],
                trade_history: [
                    {
                        timestamp: '2024-01-02T10:30:00Z',
                        symbol: 'SOL/USDC',
                        side: 'BUY',
                        pnl: 45.20
                    }
                ]
            }
        };

        test('should fetch results successfully', async () => {
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve(sampleResults)
            });

            await backtestResults.fetchBacktestResults();

            expect(global.fetch).toHaveBeenCalledWith('/dashboard/backtest/results', {
                headers: {
                    'Authorization': 'Bearer test-key',
                    'Content-Type': 'application/json'
                }
            });
            expect(backtestResults.currentResults).toEqual(sampleResults.backtest_results);
        });

        test('should handle fetch error', async () => {
            global.fetch.mockRejectedValueOnce(new Error('Network error'));

            await backtestResults.fetchBacktestResults();

            expect(global.dashboard.showNotification).toHaveBeenCalledWith(
                'error',
                'Failed to Load',
                'Could not fetch backtest results'
            );
        });

        test('should handle no results', async () => {
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ backtest_results: null })
            });

            await backtestResults.fetchBacktestResults();

            expect(backtestResults.currentResults).toBeNull();
        });

        test('should fetch with filters', async () => {
            const filters = {
                strategy_name: 'LSTM_DQN_Strategy',
                min_sharpe_ratio: 1.0
            };

            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve(sampleResults)
            });

            await backtestResults.fetchBacktestResultsWithFilters(filters);

            expect(global.fetch).toHaveBeenCalledWith(
                '/dashboard/backtest/results?strategy_name=LSTM_DQN_Strategy&min_sharpe_ratio=1',
                expect.any(Object)
            );
        });
    });

    describe('displayResults', () => {
        const sampleResults = {
            strategy_name: 'LSTM_DQN_Strategy',
            total_return: 0.125,
            sharpe_ratio: 1.42,
            max_drawdown: -0.08,
            win_rate: 0.65,
            total_trades: 156,
            equity_curve: [
                { timestamp: '2024-01-01T00:00:00Z', value: 10000 },
                { timestamp: '2024-01-31T23:59:59Z', value: 11250 }
            ],
            trade_history: [
                {
                    timestamp: '2024-01-02T10:30:00Z',
                    symbol: 'SOL/USDC',
                    side: 'BUY',
                    pnl: 45.20
                }
            ]
        };

        test('should display performance metrics', () => {
            backtestResults.displayResults(sampleResults);

            const metricsContainer = document.getElementById('backtest-metrics');
            expect(metricsContainer.innerHTML).toContain('12.50%'); // total return
            expect(metricsContainer.innerHTML).toContain('1.42'); // sharpe ratio
            expect(metricsContainer.innerHTML).toContain('-8.00%'); // max drawdown
            expect(metricsContainer.innerHTML).toContain('65.0%'); // win rate
        });

        test('should create equity curve chart', () => {
            backtestResults.displayResults(sampleResults);

            expect(global.Chart.Chart).toHaveBeenCalled();
            const chartCall = global.Chart.Chart.mock.calls[0];
            expect(chartCall[1].type).toBe('line');
            expect(chartCall[1].data.datasets[0].label).toBe('Portfolio Value');
        });

        test('should display trade history', () => {
            backtestResults.displayResults(sampleResults);

            const tradeHistoryContainer = document.getElementById('backtest-trade-history');
            expect(tradeHistoryContainer.innerHTML).toContain('SOL/USDC');
            expect(tradeHistoryContainer.innerHTML).toContain('BUY');
            expect(tradeHistoryContainer.innerHTML).toContain('$45.20');
        });

        test('should handle empty results', () => {
            backtestResults.displayResults(null);

            const metricsContainer = document.getElementById('backtest-metrics');
            expect(metricsContainer.innerHTML).toContain('No backtest results available');
        });
    });

    describe('updateEquityChart', () => {
        const equityData = [
            { timestamp: '2024-01-01T00:00:00Z', value: 10000 },
            { timestamp: '2024-01-05T00:00:00Z', value: 10150 },
            { timestamp: '2024-01-10T00:00:00Z', value: 10320 }
        ];

        test('should create new chart if none exists', () => {
            backtestResults.updateEquityChart(equityData);

            expect(global.Chart.Chart).toHaveBeenCalled();
            const chartConfig = global.Chart.Chart.mock.calls[0][1];
            expect(chartConfig.data.labels).toHaveLength(3);
            expect(chartConfig.data.datasets[0].data).toEqual([10000, 10150, 10320]);
        });

        test('should update existing chart', () => {
            // Create initial chart
            const mockChart = {
                destroy: jest.fn(),
                update: jest.fn(),
                data: { labels: [], datasets: [{ data: [] }] }
            };
            backtestResults.equityChart = mockChart;

            backtestResults.updateEquityChart(equityData);

            expect(mockChart.data.labels).toHaveLength(3);
            expect(mockChart.data.datasets[0].data).toEqual([10000, 10150, 10320]);
            expect(mockChart.update).toHaveBeenCalled();
        });
    });

    describe('formatters', () => {
        test('should format percentage correctly', () => {
            expect(backtestResults.formatPercentage(0.125)).toBe('12.50%');
            expect(backtestResults.formatPercentage(-0.08)).toBe('-8.00%');
            expect(backtestResults.formatPercentage(0)).toBe('0.00%');
        });

        test('should format numbers correctly', () => {
            expect(backtestResults.formatNumber(1.42567)).toBe('1.43');
            expect(backtestResults.formatNumber(156)).toBe('156');
            expect(backtestResults.formatNumber(0)).toBe('0');
        });

        test('should format dates correctly', () => {
            const date = '2024-01-02T10:30:00Z';
            const formatted = backtestResults.formatDate(date);
            expect(formatted).toMatch(/Jan 2, 2024/); // Locale dependent
        });
    });

    describe('showLoadingState', () => {
        test('should show loading indicators', () => {
            backtestResults.showLoadingState();

            expect(backtestResults.isLoading).toBe(true);
            
            const metricsContainer = document.getElementById('backtest-metrics');
            expect(metricsContainer.innerHTML).toContain('Loading');
        });
    });

    describe('hideLoadingState', () => {
        test('should hide loading indicators', () => {
            backtestResults.isLoading = true;
            backtestResults.hideLoadingState();

            expect(backtestResults.isLoading).toBe(false);
        });
    });

    describe('refresh', () => {
        test('should refresh backtest results', async () => {
            const spy = jest.spyOn(backtestResults, 'fetchBacktestResults')
                .mockResolvedValue();

            await backtestResults.refresh();

            expect(spy).toHaveBeenCalled();
        });
    });

    describe('destroy', () => {
        test('should cleanup chart and resources', () => {
            const mockChart = { destroy: jest.fn() };
            backtestResults.equityChart = mockChart;

            backtestResults.destroy();

            expect(mockChart.destroy).toHaveBeenCalled();
            expect(backtestResults.equityChart).toBeNull();
        });
    });
});

describe('BacktestResultsIntegration', () => {
    test('should integrate with dashboard app', () => {
        // Mock dashboard methods that BacktestResults depends on
        global.dashboard.formatCurrency = jest.fn((val) => `$${val}`);
        global.dashboard.showNotification = jest.fn();
        
        const BacktestResults = require('../../static/backtest-results');
        const instance = new BacktestResults();
        
        // Should be able to use dashboard methods
        expect(typeof global.dashboard.formatCurrency).toBe('function');
        expect(typeof global.dashboard.showNotification).toBe('function');
    });
});