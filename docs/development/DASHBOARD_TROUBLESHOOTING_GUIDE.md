# Dashboard Troubleshooting Guide

*Comprehensive troubleshooting guide for the Enhanced Dashboard with XAI integration*

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

## XAI Dashboard Troubleshooting

### XAI Explanation Endpoints

The Enhanced Dashboard includes 4 dedicated XAI endpoints for explainable AI features:

#### XAI Endpoints Overview
```
/xai/explanations          - Recent XAI explanations with pagination
/xai/explanations/{id}     - Specific explanation details  
/xai/feature-importance    - Feature importance summary
/xai/cache-stats          - XAI caching performance metrics
```

### Common XAI Issues

#### Issue: "XAI explanations not loading"

**Quick Solutions:**

1. **Check XAI system initialization:**
   ```bash
   # Verify XAI components are properly loaded
   curl http://localhost:8080/xai/cache-stats
   ```

2. **Verify explainer factory:**
   ```bash
   # Check if explainers are available
   python -c "from src.xai.factory import ExplainerFactory; print(ExplainerFactory().get_available_explainers())"
   ```

3. **Check trading integration:**
   ```bash
   # Verify XAI trading integration
   python -c "from src.xai.trading_integration import TradingIntegration; print('XAI integration available')"
   ```

#### Issue: "Feature importance data missing"

**Possible Causes:**
- No recent trading decisions with XAI explanations
- XAI caching issues
- Missing XAI model integration

**Solutions:**

1. **Generate test explanations:**
   ```python
   # Run in Python console to generate test data
   from src.xai.factory import ExplainerFactory
   factory = ExplainerFactory()
   # This will populate cache with test explanations
   ```

2. **Clear XAI cache:**
   ```bash
   # Clear explanation cache if stale
   curl -X POST http://localhost:8080/xai/cache-stats  # Check cache status first
   ```

#### Issue: "XAI endpoint timeouts"

**Performance Optimization:**

1. **Check explanation generation performance:**
   ```bash
   # Monitor XAI performance
   curl http://localhost:8080/xai/cache-stats
   ```

2. **Verify caching is working:**
   - Default TTL: 5 minutes for explanations
   - Check cache hit rates in `/xai/cache-stats`

3. **Optimize explanation queries:**
   ```bash
   # Use pagination for large explanation sets
   curl "http://localhost:8080/xai/explanations?limit=10"
   ```

### Dashboard API Performance

#### Expected Performance Metrics
```
Dashboard API Targets:
├── /xai/explanations .......... <100ms response time
├── /xai/feature-importance ..... <50ms response time  
├── /xai/cache-stats ............ <25ms response time
└── All endpoints ............... 5x faster than 500ms target
```

#### Performance Monitoring

1. **Monitor API response times:**
   ```bash
   # Test XAI endpoint performance
   time curl http://localhost:8080/xai/explanations
   time curl http://localhost:8080/xai/feature-importance
   ```

2. **Check system health with XAI:**
   ```bash
   curl http://localhost:8080/health
   # Should include XAI system status
   ```

### XAI Integration Diagnostics

#### Verify Complete XAI Integration

1. **Check all 3 explainer types:**
   ```python
   from src.xai.factory import ExplainerFactory
   factory = ExplainerFactory()
   
   # Should show: ['lime', 'permutation', 'gradient']
   print(factory.get_available_explainers())
   ```

2. **Test trading mode integration:**
   ```python
   # Verify XAI works with trading modes
   from src.xai.trading_integration import TradingIntegration
   integration = TradingIntegration(ExplainerFactory())
   print("XAI trading integration ready")
   ```

3. **Check explanation data models:**
   ```python
   from src.xai.data_models import ExplanationData
   # Verify data structures are properly defined
   print(ExplanationData.__annotations__)
   ```

### Authentication & Authorization

#### XAI Endpoint Security

All XAI endpoints require proper authentication:

```bash
# Test with authentication headers
curl -H "Authorization: Bearer YOUR_TOKEN" \
     http://localhost:8080/xai/explanations
```

**Required Permissions:**
- `/xai/explanations`: `require_read` permission
- `/xai/feature-importance`: `require_read` permission  
- `/xai/cache-stats`: `require_read` permission

### Development Mode XAI

For local development with XAI features:

1. **Mock XAI data available**: Dashboard provides sample explanations
2. **Graceful degradation**: Missing XAI components won't break dashboard
3. **Test endpoints work**: All XAI endpoints functional with mock data

### Troubleshooting Checklist

#### XAI System Health Check

- [ ] XAI explainer factory initializes without errors
- [ ] All 3 explainer types (LIME, Permutation, Gradient) are available
- [ ] Trading integration component loads successfully
- [ ] XAI cache system is functioning (check `/xai/cache-stats`)
- [ ] All 4 XAI endpoints respond within performance targets
- [ ] Authentication works for XAI endpoints
- [ ] Feature importance data is populated and accessible

#### Dashboard Integration Health Check

- [ ] Dashboard loads without XAI-related errors
- [ ] XAI explanation data displays properly in UI
- [ ] Feature importance visualizations render correctly
- [ ] Real-time explanation updates work during trading
- [ ] Explanation history is searchable and filterable
- [ ] Performance correlation analysis displays properly

## Getting Help

If you encounter other issues:
1. Check the logs for specific error messages
2. Verify all dependencies are installed: `pip install -r requirements.txt`
3. Ensure PostgreSQL is running if database features are needed
4. **For XAI issues**: Check XAI system logs and cache statistics
5. **For Dashboard API issues**: Monitor response times and authentication
6. Check the GitHub Actions CI for similar issues