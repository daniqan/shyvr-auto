# Dashboard Troubleshooting Guide

## Port Conflicts

### Issue: "Address already in use" on port 8080

**Quick Solutions:**

1. **Use a different port:**
   ```bash
   PORT=8081 python main.py
   ```

2. **Kill existing process:**
   ```bash
   # Find process using port 8080
   lsof -i :8080
   
   # Kill the process (replace PID with actual process ID)
   kill [PID]
   ```

3. **Kill by process name:**
   ```bash
   pkill -f "python main.py"
   ```

### Configuration

The dashboard port can be configured in multiple ways:

1. **Environment variable:**
   ```bash
   export PORT=8081
   python main.py
   ```

2. **Config file (config/config.yaml):**
   ```yaml
   app:
     host: "0.0.0.0"
     port: 8081
   ```

3. **Local environment file (.env.local):**
   ```
   PORT=8081
   HOST=localhost
   ```

## Component Initialization Issues

### ModeManager requires config and portfolio

The dashboard now creates mock configurations automatically:
- Mock Portfolio with $10,000 USDC initial balance
- ModeManagerConfig with safe defaults
- Graceful fallback for missing components

### Missing Components

If components fail to initialize, the dashboard will:
- Log warnings for missing components
- Continue with mock data
- Provide basic functionality for testing

## Testing Dashboard Startup

1. **Run the test script:**
   ```bash
   python test_dashboard_local.py
   ```

2. **Manual testing:**
   ```bash
   python main.py
   ```

3. **Check health endpoint:**
   ```bash
   curl http://localhost:8080/health
   ```

## Common Error Solutions

### "SystemHealthMonitor.__init__() missing arguments"
- Fixed: Dashboard service now handles missing component dependencies gracefully

### "PortfolioManager.__init__() missing config"
- Fixed: Mock configurations created automatically

### "cannot import name 'RLConfig'"
- Expected: RL components may not be fully initialized in development
- Dashboard continues with mock data

## Development Mode

For local development, the dashboard will:
- Use mock data when real components aren't available
- Provide warnings instead of failing
- Allow testing of dashboard UI and basic functionality

## Getting Help

If you encounter other issues:
1. Check the logs for specific error messages
2. Verify all dependencies are installed: `pip install -r requirements.txt`
3. Ensure PostgreSQL is running if database features are needed
4. Check the GitHub Actions CI for similar issues