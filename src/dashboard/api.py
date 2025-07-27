"""
Dashboard API Routes
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, status, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, validator
import structlog
import csv
import io

from .auth import User, require_read, require_write, require_admin, require_trading
from .service import dashboard_service
from .websocket_manager import websocket_manager
from .base import DashboardData, DashboardError
from .activity_integration import dashboard_activity
from ..logging.activity_logger import (
    activity_logger, ActivityCategory, ActivityAction, ActivitySeverity,
    performance_tracker
)

logger = structlog.get_logger()


# Pydantic models for API requests/responses
class TradingModeRequest(BaseModel):
    mode: str
    
    @validator('mode')
    def validate_mode(cls, v):
        if v not in ['analysis', 'simulation', 'live']:
            raise ValueError('Mode must be one of: analysis, simulation, live')
        return v


class RiskLimitsRequest(BaseModel):
    max_position_size_usd: Optional[float] = None
    max_daily_loss_usd: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None


class SystemResponse(BaseModel):
    success: bool
    message: str
    timestamp: datetime


class DashboardAPI:
    """Dashboard API handler"""
    
    def __init__(self):
        self.router = APIRouter(prefix="/dashboard", tags=["Dashboard"])
        self._setup_routes()
    
    def _setup_routes(self):
        """Set up API routes"""
        
        # Dashboard data endpoints
        self.router.add_api_route(
            "/data",
            self.get_dashboard_data,
            methods=["GET"],
            dependencies=[Depends(require_read)]
        )
        
        self.router.add_api_route(
            "/system/metrics",
            self.get_system_metrics,
            methods=["GET"],
            dependencies=[Depends(require_read)]
        )
        
        self.router.add_api_route(
            "/portfolio/status",
            self.get_portfolio_status,
            methods=["GET"],
            dependencies=[Depends(require_read)]
        )
        
        self.router.add_api_route(
            "/trading/status",
            self.get_trading_status,
            methods=["GET"],
            dependencies=[Depends(require_read)]
        )
        
        self.router.add_api_route(
            "/trading/history",
            self.get_trading_history,
            methods=["GET"],
            dependencies=[Depends(require_read)]
        )
        
        self.router.add_api_route(
            "/ml-rl/status",
            self.get_ml_rl_status,
            methods=["GET"],
            dependencies=[Depends(require_read)]
        )
        
        self.router.add_api_route(
            "/backtest/results",
            self.get_backtest_results,
            methods=["GET"],
            dependencies=[Depends(require_read)]
        )
        
        # Control endpoints
        self.router.add_api_route(
            "/trading/mode",
            self.switch_trading_mode,
            methods=["POST"],
            dependencies=[Depends(require_trading)]
        )
        
        self.router.add_api_route(
            "/trading/emergency-stop",
            self.emergency_stop,
            methods=["POST"],
            dependencies=[Depends(require_trading)]
        )
        
        self.router.add_api_route(
            "/trading/risk-limits",
            self.update_risk_limits,
            methods=["PUT"],
            dependencies=[Depends(require_admin)]
        )
        
        # System management endpoints
        self.router.add_api_route(
            "/system/logs",
            self.get_system_logs,
            methods=["GET"],
            dependencies=[Depends(require_admin)]
        )
        
        self.router.add_api_route(
            "/system/health",
            self.get_system_health,
            methods=["GET"],
            dependencies=[Depends(require_read)]
        )
        
        # Activity log endpoints
        self.router.add_api_route(
            "/activity/logs",
            self.get_activity_logs,
            methods=["GET"],
            dependencies=[Depends(require_read)]
        )
        
        self.router.add_api_route(
            "/activity/stats",
            self.get_activity_stats,
            methods=["GET"],
            dependencies=[Depends(require_read)]
        )
        
        self.router.add_api_route(
            "/activity/export",
            self.export_activity_logs,
            methods=["GET"],
            dependencies=[Depends(require_admin)]
        )
        
        # WebSocket endpoint
        self.router.add_websocket_route(
            "/ws/{connection_id}",
            self.websocket_endpoint
        )
    
    async def get_dashboard_data(self, user: User = Depends(require_read)) -> Dict[str, Any]:
        """Get complete dashboard data"""
        start_time = asyncio.get_event_loop().time()
        
        # Log API call
        await activity_logger.log_api_call(
            api_name="dashboard_api",
            endpoint="/dashboard/data",
            method="GET",
            status_code=200,  # Will be updated if there's an error
            response_time_ms=0,  # Will be updated later
            success=True,  # Will be updated if there's an error
            user_id=int(user.user_id.split('-')[-1], 16) % 10000,
            metadata={
                "username": user.username,
                "user_permissions": list(user.permissions)
            }
        )
        
        async with performance_tracker(
            source="dashboard_api",
            operation="get_dashboard_data",
            category=ActivityCategory.API,
            metadata={"user": user.username, "endpoint": "/dashboard/data"}
        ) as tracker:
            try:
                # Log API request
                await activity_logger.log_activity(
                    category=ActivityCategory.API,
                    action=ActivityAction.READ,
                    source="dashboard_api",
                    event_type="api_request",
                    title=f"Dashboard data requested by {user.username}",
                    severity=ActivitySeverity.INFO,
                    user_id=int(user.user_id.split('-')[-1], 16) % 10000,
                    api_endpoint="/dashboard/data",
                    http_method="GET",
                    metadata={
                        "username": user.username,
                        "user_permissions": list(user.permissions)
                    }
                )
                
                data = await dashboard_service.get_dashboard_data()
                
                # Record request metrics
                response_time = asyncio.get_event_loop().time() - start_time
                dashboard_service.record_request(response_time)
                
                # Log successful API response
                await activity_logger.log_activity(
                    category=ActivityCategory.API,
                    action=ActivityAction.SUCCESS,
                    source="dashboard_api",
                    event_type="api_response_success",
                    title=f"Dashboard data successfully served to {user.username}",
                    severity=ActivitySeverity.INFO,
                    user_id=int(user.user_id.split('-')[-1], 16) % 10000,
                    api_endpoint="/dashboard/data",
                    http_method="GET",
                    http_status=200,
                    response_time_ms=int(response_time * 1000),
                    metadata={
                        "username": user.username,
                        "response_time_seconds": response_time,
                        "data_elements": ["system_metrics", "portfolio_status", "trading_status", "ml_rl_status"]
                    }
                )
                
                return self._serialize_dashboard_data(data)
                
            except Exception as e:
                # Record error
                response_time = asyncio.get_event_loop().time() - start_time
                dashboard_service.record_request(response_time, error=True)
                
                # Log API error
                await activity_logger.log_error(
                    category=ActivityCategory.API,
                    source="dashboard_api",
                    event_type="api_request_failed",
                    title=f"Failed to serve dashboard data to {user.username}",
                    error_message=str(e),
                    exception=e,
                    severity=ActivitySeverity.ERROR,
                    user_id=int(user.user_id.split('-')[-1], 16) % 10000,
                    api_endpoint="/dashboard/data",
                    http_method="GET",
                    http_status=500,
                    response_time_ms=int(response_time * 1000),
                    metadata={
                        "username": user.username,
                        "response_time_seconds": response_time
                    }
                )
                
                logger.error("Failed to get dashboard data", error=str(e), user=user.username)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to retrieve dashboard data"
                )
    
    async def get_system_metrics(self, user: User = Depends(require_read)) -> Dict[str, Any]:
        """Get system metrics"""
        try:
            data = await dashboard_service.get_dashboard_data()
            return {
                "system_metrics": self._serialize_system_metrics(data.system_metrics),
                "timestamp": data.timestamp.isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to get system metrics", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve system metrics"
            )
    
    async def get_portfolio_status(self, user: User = Depends(require_read)) -> Dict[str, Any]:
        """Get portfolio status"""
        try:
            data = await dashboard_service.get_dashboard_data()
            return {
                "portfolio_status": self._serialize_portfolio_status(data.portfolio_status),
                "timestamp": data.timestamp.isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to get portfolio status", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve portfolio status"
            )
    
    async def get_trading_status(self, user: User = Depends(require_read)) -> Dict[str, Any]:
        """Get trading status"""
        try:
            data = await dashboard_service.get_dashboard_data()
            return {
                "trading_status": self._serialize_trading_status(data.trading_status),
                "timestamp": data.timestamp.isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to get trading status", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve trading status"
            )
    
    async def get_trading_history(self, limit: int = 100, 
                                  user: User = Depends(require_read)) -> Dict[str, Any]:
        """Get trading history"""
        try:
            history = await dashboard_service.get_trading_history(limit)
            return {
                "trades": [self._serialize_trade(trade) for trade in history],
                "count": len(history),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to get trading history", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve trading history"
            )
    
    async def get_ml_rl_status(self, user: User = Depends(require_read)) -> Dict[str, Any]:
        """Get ML/RL status"""
        try:
            data = await dashboard_service.get_dashboard_data()
            return {
                "ml_rl_status": self._serialize_ml_rl_status(data.ml_rl_status),
                "timestamp": data.timestamp.isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to get ML/RL status", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve ML/RL status"
            )
    
    async def get_backtest_results(self, 
                                   strategy_name: Optional[str] = Query(None, description="Filter by strategy name"),
                                   start_date: Optional[str] = Query(None, description="Filter by start date (YYYY-MM-DD)"),
                                   end_date: Optional[str] = Query(None, description="Filter by end date (YYYY-MM-DD)"),
                                   min_sharpe_ratio: Optional[float] = Query(None, description="Filter by minimum Sharpe ratio"),
                                   user: User = Depends(require_read)) -> Dict[str, Any]:
        """Get backtest results from analysis mode"""
        try:
            # Log API request
            await activity_logger.log_activity(
                category=ActivityCategory.API,
                action=ActivityAction.READ,
                source="dashboard_api",
                event_type="backtest_results_request",
                title=f"Backtest results requested by {user.username}",
                severity=ActivitySeverity.INFO,
                user_id=int(user.user_id.split('-')[-1], 16) % 10000,
                api_endpoint="/dashboard/backtest/results",
                http_method="GET",
                metadata={
                    "username": user.username,
                    "strategy_name": strategy_name,
                    "start_date": start_date,
                    "end_date": end_date,
                    "min_sharpe_ratio": min_sharpe_ratio
                }
            )
            
            # Get backtest results from service
            backtest_results = await dashboard_service.get_backtest_results(
                strategy_name=strategy_name,
                start_date=start_date,
                end_date=end_date,
                min_sharpe_ratio=min_sharpe_ratio
            )
            
            if backtest_results is None:
                return {
                    "backtest_results": None,
                    "message": "No backtest results available",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Log successful response
            await activity_logger.log_activity(
                category=ActivityCategory.API,
                action=ActivityAction.SUCCESS,
                source="dashboard_api",
                event_type="backtest_results_success",
                title=f"Backtest results successfully served to {user.username}",
                severity=ActivitySeverity.INFO,
                user_id=int(user.user_id.split('-')[-1], 16) % 10000,
                api_endpoint="/dashboard/backtest/results",
                http_method="GET",
                http_status=200,
                metadata={
                    "username": user.username,
                    "strategy_name": backtest_results.get("strategy_name"),
                    "total_return": backtest_results.get("total_return"),
                    "sharpe_ratio": backtest_results.get("sharpe_ratio"),
                    "total_trades": backtest_results.get("total_trades")
                }
            )
            
            return {
                "backtest_results": backtest_results,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            # Log API error
            await activity_logger.log_error(
                category=ActivityCategory.API,
                source="dashboard_api",
                event_type="backtest_results_failed",
                title=f"Failed to serve backtest results to {user.username}",
                error_message=str(e),
                exception=e,
                user_id=int(user.user_id.split('-')[-1], 16) % 10000,
                api_endpoint="/dashboard/backtest/results",
                http_method="GET",
                http_status=500,
                metadata={"username": user.username}
            )
            
            logger.error("Failed to get backtest results", error=str(e), user=user.username)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve backtest results"
            )
    
    async def switch_trading_mode(self, request: TradingModeRequest,
                                  user: User = Depends(require_trading)) -> SystemResponse:
        """Switch trading mode"""
        try:
            success = await dashboard_service.switch_trading_mode(request.mode, user.user_id)
            
            if success:
                return SystemResponse(
                    success=True,
                    message=f"Trading mode switched to {request.mode}",
                    timestamp=datetime.utcnow()
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to switch trading mode"
                )
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Failed to switch trading mode", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal error switching trading mode"
            )
    
    async def emergency_stop(self, user: User = Depends(require_trading)) -> SystemResponse:
        """Activate emergency stop"""
        try:
            success = await dashboard_service.emergency_stop(user.user_id)
            
            if success:
                return SystemResponse(
                    success=True,
                    message="Emergency stop activated",
                    timestamp=datetime.utcnow()
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to activate emergency stop"
                )
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Failed to activate emergency stop", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal error activating emergency stop"
            )
    
    async def update_risk_limits(self, request: RiskLimitsRequest,
                                 user: User = Depends(require_admin)) -> SystemResponse:
        """Update risk management limits"""
        try:
            # Convert to dict, excluding None values
            limits = {k: v for k, v in request.dict().items() if v is not None}
            
            success = await dashboard_service.update_risk_limits(limits, user.user_id)
            
            if success:
                return SystemResponse(
                    success=True,
                    message="Risk limits updated successfully",
                    timestamp=datetime.utcnow()
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to update risk limits"
                )
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Failed to update risk limits", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal error updating risk limits"
            )
    
    async def get_system_logs(self, limit: int = 100,
                              user: User = Depends(require_admin)) -> Dict[str, Any]:
        """Get system logs"""
        try:
            logs = await dashboard_service.get_system_logs(limit)
            return {
                "logs": logs,
                "count": len(logs),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to get system logs", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve system logs"
            )
    
    async def get_system_health(self, user: User = Depends(require_read)) -> Dict[str, Any]:
        """Get system health summary"""
        try:
            data = await dashboard_service.get_dashboard_data()
            
            # Determine overall health
            overall_status = "healthy"
            if data.system_metrics.error_rate_pct > 5:
                overall_status = "warning"
            if data.system_metrics.status.value != "healthy":
                overall_status = "error"
            
            return {
                "overall_status": overall_status,
                "uptime_seconds": data.system_metrics.uptime_seconds,
                "active_connections": data.system_metrics.active_connections,
                "error_rate_pct": data.system_metrics.error_rate_pct,
                "component_statuses": {
                    "database": data.system_metrics.database_status.value,
                    "ml_models": data.system_metrics.ml_models_status.value,
                    "rl_agent": data.system_metrics.rl_agent_status.value,
                    "dex_connections": data.system_metrics.dex_connections_status.value
                },
                "timestamp": data.timestamp.isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to get system health", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve system health"
            )
    
    async def get_activity_logs(
        self,
        limit: int = Query(100, ge=1, le=1000, description="Number of logs to return"),
        offset: int = Query(0, ge=0, description="Offset for pagination"),
        category: Optional[str] = Query(None, description="Filter by category"),
        severity: Optional[str] = Query(None, description="Filter by severity"),
        user_id: Optional[int] = Query(None, description="Filter by user ID"),
        hours_back: int = Query(24, ge=1, le=168, description="Hours to look back"),
        search: Optional[str] = Query(None, description="Search in title and description"),
        source: Optional[str] = Query(None, description="Filter by source component"),
        user: User = Depends(require_read)
    ) -> Dict[str, Any]:
        """Get activity logs with filtering and pagination"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Log API call
            await activity_logger.log_api_call(
                api_name="dashboard_api",
                endpoint="/dashboard/activity/logs",
                method="GET",
                status_code=200,
                response_time_ms=0,
                success=True,
                user_id=int(user.user_id.split('-')[-1], 16) % 10000,
                metadata={
                    "username": user.username,
                    "limit": limit,
                    "offset": offset,
                    "category": category,
                    "severity": severity,
                    "hours_back": hours_back
                }
            )
            
            # Get activity logs from database
            async with performance_tracker(
                source="dashboard_api",
                operation="get_activity_logs",
                category=ActivityCategory.API,
                metadata={"user": user.username, "endpoint": "/dashboard/activity/logs"}
            ) as tracker:
                
                # Build query filters
                filters = {}
                if category:
                    filters['category'] = category
                if severity:
                    filters['severity'] = severity
                if user_id:
                    filters['user_id'] = user_id
                if source:
                    filters['source'] = source
                
                # Get logs from activity integration
                logs = await dashboard_activity.get_recent_activity(
                    limit=limit + offset,  # Get extra to handle offset
                    category=category,
                    severity=severity,
                    user_id=user_id,
                    hours_back=hours_back
                )
                
                # Apply search filter if provided
                if search:
                    search_lower = search.lower()
                    logs = [
                        log for log in logs 
                        if search_lower in log.get('title', '').lower() 
                        or search_lower in log.get('description', '').lower()
                    ]
                
                # Apply pagination
                total_count = len(logs)
                paginated_logs = logs[offset:offset + limit]
                
                # Calculate response time
                response_time = asyncio.get_event_loop().time() - start_time
                dashboard_service.record_request(response_time)
                
                # Log successful response
                await activity_logger.log_activity(
                    category=ActivityCategory.API,
                    action=ActivityAction.SUCCESS,
                    source="dashboard_api",
                    event_type="activity_logs_retrieved",
                    title=f"Activity logs retrieved by {user.username}",
                    severity=ActivitySeverity.INFO,
                    user_id=int(user.user_id.split('-')[-1], 16) % 10000,
                    api_endpoint="/dashboard/activity/logs",
                    http_method="GET",
                    http_status=200,
                    response_time_ms=int(response_time * 1000),
                    metadata={
                        "username": user.username,
                        "logs_returned": len(paginated_logs),
                        "total_available": total_count,
                        "filters_applied": filters
                    }
                )
                
                return {
                    "logs": paginated_logs,
                    "pagination": {
                        "limit": limit,
                        "offset": offset,
                        "total": total_count,
                        "has_more": offset + limit < total_count
                    },
                    "filters": filters,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            # Record error
            response_time = asyncio.get_event_loop().time() - start_time
            dashboard_service.record_request(response_time, error=True)
            
            # Log API error
            await activity_logger.log_error(
                category=ActivityCategory.API,
                source="dashboard_api",
                event_type="activity_logs_failed",
                title=f"Failed to retrieve activity logs for {user.username}",
                error_message=str(e),
                exception=e,
                severity=ActivitySeverity.ERROR,
                user_id=int(user.user_id.split('-')[-1], 16) % 10000,
                api_endpoint="/dashboard/activity/logs",
                http_method="GET",
                http_status=500,
                response_time_ms=int(response_time * 1000),
                metadata={"username": user.username}
            )
            
            logger.error("Failed to get activity logs", error=str(e), user=user.username)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve activity logs"
            )
    
    async def get_activity_stats(
        self,
        hours_back: int = Query(24, ge=1, le=168, description="Hours to analyze"),
        user: User = Depends(require_read)
    ) -> Dict[str, Any]:
        """Get activity statistics and metrics"""
        try:
            # Get error summary
            error_stats = await dashboard_activity.get_error_summary(hours_back)
            
            # Get performance metrics
            performance_stats = await dashboard_activity.get_performance_metrics(hours_back)
            
            # Get user activity stats if user_id is available
            user_stats = None
            if hasattr(user, 'user_id') and user.user_id:
                try:
                    user_numeric_id = int(user.user_id.split('-')[-1], 16) % 10000
                    user_stats = await dashboard_activity.get_user_activity_stats(
                        user_numeric_id, 
                        days_back=max(1, hours_back // 24)
                    )
                except (ValueError, AttributeError):
                    user_stats = None
            
            return {
                "time_range": {
                    "hours_back": hours_back,
                    "start_time": (datetime.utcnow() - timedelta(hours=hours_back)).isoformat(),
                    "end_time": datetime.utcnow().isoformat()
                },
                "error_summary": error_stats,
                "performance_metrics": performance_stats,
                "user_stats": user_stats,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to get activity stats", error=str(e), user=user.username)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve activity statistics"
            )
    
    async def export_activity_logs(
        self,
        format: str = Query("csv", regex="^(csv|json)$", description="Export format"),
        hours_back: int = Query(24, ge=1, le=168, description="Hours to export"),
        category: Optional[str] = Query(None, description="Filter by category"),
        severity: Optional[str] = Query(None, description="Filter by severity"),
        user: User = Depends(require_admin)
    ) -> StreamingResponse:
        """Export activity logs in CSV or JSON format"""
        try:
            # Log export request
            await activity_logger.log_activity(
                category=ActivityCategory.API,
                action=ActivityAction.READ,
                source="dashboard_api",
                event_type="activity_export_request",
                title=f"Activity log export requested by {user.username}",
                severity=ActivitySeverity.INFO,
                user_id=int(user.user_id.split('-')[-1], 16) % 10000,
                api_endpoint="/dashboard/activity/export",
                http_method="GET",
                metadata={
                    "username": user.username,
                    "format": format,
                    "hours_back": hours_back,
                    "category": category,
                    "severity": severity
                }
            )
            
            # Get activity logs (no limit for export)
            logs = await dashboard_activity.get_recent_activity(
                limit=10000,  # Large limit for export
                category=category,
                severity=severity,
                hours_back=hours_back
            )
            
            if format == "csv":
                # Create CSV export
                output = io.StringIO()
                
                if logs:
                    fieldnames = logs[0].keys()
                    writer = csv.DictWriter(output, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(logs)
                
                content = output.getvalue()
                output.close()
                
                # Create streaming response
                def generate():
                    yield content
                
                filename = f"activity_logs_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
                return StreamingResponse(
                    generate(),
                    media_type="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={filename}"}
                )
            
            else:  # JSON format
                content = json.dumps({
                    "exported_at": datetime.utcnow().isoformat(),
                    "export_parameters": {
                        "hours_back": hours_back,
                        "category": category,
                        "severity": severity
                    },
                    "total_records": len(logs),
                    "logs": logs
                }, indent=2, default=str)
                
                def generate():
                    yield content
                
                filename = f"activity_logs_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
                return StreamingResponse(
                    generate(),
                    media_type="application/json",
                    headers={"Content-Disposition": f"attachment; filename={filename}"}
                )
                
        except Exception as e:
            logger.error("Failed to export activity logs", error=str(e), user=user.username)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to export activity logs"
            )
    
    async def websocket_endpoint(self, websocket: WebSocket, connection_id: str):
        """WebSocket endpoint for real-time updates"""
        try:
            # Connect to WebSocket manager
            await websocket_manager.connect(websocket, connection_id)
            
            logger.info("WebSocket client connected", connection_id=connection_id)
            
            while True:
                try:
                    # Receive messages from client
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    # Handle message
                    await websocket_manager.handle_message(connection_id, message)
                    
                except WebSocketDisconnect:
                    break
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON received", connection_id=connection_id)
                except Exception as e:
                    logger.error("WebSocket message error", error=str(e), connection_id=connection_id)
        
        except Exception as e:
            logger.error("WebSocket connection error", error=str(e), connection_id=connection_id)
        
        finally:
            # Disconnect from WebSocket manager
            await websocket_manager.disconnect(connection_id)
            logger.info("WebSocket client disconnected", connection_id=connection_id)
    
    def _serialize_dashboard_data(self, data: DashboardData) -> Dict[str, Any]:
        """Serialize complete dashboard data"""
        return {
            "system_metrics": self._serialize_system_metrics(data.system_metrics),
            "portfolio_status": self._serialize_portfolio_status(data.portfolio_status),
            "trading_status": self._serialize_trading_status(data.trading_status),
            "ml_rl_status": self._serialize_ml_rl_status(data.ml_rl_status),
            "alerts": {
                "active_alerts": data.active_alerts,
                "warnings_count": data.warnings_count,
                "errors_count": data.errors_count
            },
            "recent_activity": {
                "logs": data.recent_logs,
                "notifications": data.recent_notifications
            },
            "timestamp": data.timestamp.isoformat()
        }
    
    def _serialize_system_metrics(self, metrics) -> Dict[str, Any]:
        """Serialize system metrics"""
        return {
            "status": metrics.status.value,
            "uptime_seconds": metrics.uptime_seconds,
            "cpu_usage_pct": metrics.cpu_usage_pct,
            "memory_usage_mb": metrics.memory_usage_mb,
            "memory_usage_pct": metrics.memory_usage_pct,
            "active_connections": metrics.active_connections,
            "requests_per_minute": metrics.requests_per_minute,
            "error_rate_pct": metrics.error_rate_pct,
            "response_time_ms": metrics.response_time_ms,
            "component_statuses": {
                "database": metrics.database_status.value,
                "ml_models": metrics.ml_models_status.value,
                "rl_agent": metrics.rl_agent_status.value,
                "dex_connections": metrics.dex_connections_status.value
            },
            "performance": {
                "total_requests": metrics.total_requests,
                "total_errors": metrics.total_errors,
                "cache_hit_rate_pct": metrics.cache_hit_rate_pct
            },
            "last_updated": metrics.last_updated.isoformat()
        }
    
    def _serialize_portfolio_status(self, status) -> Dict[str, Any]:
        """Serialize portfolio status"""
        return {
            "total_value_usd": str(status.total_value_usd),
            "available_balance_usd": str(status.available_balance_usd),
            "margin_used_usd": str(status.margin_used_usd),
            "unrealized_pnl_usd": str(status.unrealized_pnl_usd),
            "realized_pnl_usd": str(status.realized_pnl_usd),
            "daily_pnl_usd": str(status.daily_pnl_usd),
            "daily_pnl_pct": str(status.daily_pnl_pct),
            "total_return_pct": str(status.total_return_pct),
            "risk_metrics": {
                "max_drawdown_pct": str(status.max_drawdown_pct),
                "current_drawdown_pct": str(status.current_drawdown_pct),
                "sharpe_ratio": status.sharpe_ratio,
                "volatility_pct": status.volatility_pct,
                "var_95_usd": str(status.var_95_usd) if status.var_95_usd else None
            },
            "positions": {
                "active_count": status.position_count,
                "positions": [self._serialize_position(pos) for pos in status.active_positions]
            },
            "chain_balances": {k: str(v) for k, v in status.chain_balances.items()},
            "last_updated": status.last_updated.isoformat(),
            # Mode context for clear distinction
            "mode": status.mode.value,
            "is_simulated": status.is_simulated
        }
    
    def _serialize_trading_status(self, status) -> Dict[str, Any]:
        """Serialize trading status"""
        return {
            "mode": status.mode.value,
            "is_trading_active": status.is_trading_active,
            "last_trade_time": status.last_trade_time.isoformat() if status.last_trade_time else None,
            "daily_stats": {
                "trades_today": status.trades_today,
                "volume_today_usd": str(status.volume_today_usd),
                "tokens_analyzed_today": status.tokens_analyzed_today
            },
            "mode_status": {
                "analysis_running": status.analysis_running,
                "simulation_running": status.simulation_running,
                "live_trading_enabled": status.live_trading_enabled
            },
            "safety": {
                "emergency_stop_active": status.emergency_stop_active,
                "risk_limits_active": status.risk_limits_active
            },
            "performance": {
                "win_rate_pct": status.win_rate_pct,
                "avg_trade_duration_hours": status.avg_trade_duration_hours,
                "avg_profit_per_trade_usd": str(status.avg_profit_per_trade_usd)
            },
            "signals": {
                "tokens_in_watchlist": status.tokens_in_watchlist,
                "high_confidence_signals": status.high_confidence_signals
            },
            "last_updated": status.last_updated.isoformat(),
            # Mode context for clear distinction
            "is_simulated": status.is_simulated
        }
    
    def _serialize_ml_rl_status(self, status) -> Dict[str, Any]:
        """Serialize ML/RL status"""
        return {
            "models": {
                "ml_models": [self._serialize_ml_model(model) for model in status.ml_models],
                "rl_agents": [self._serialize_rl_agent(agent) for agent in status.rl_agents]
            },
            "integration": {
                "ml_rl_integration_active": status.ml_rl_integration_active,
                "ml_rl_decision_latency_ms": status.ml_rl_decision_latency_ms,
                "ml_confidence_threshold": status.ml_confidence_threshold,
                "rl_action_confidence": status.rl_action_confidence
            },
            "training": {
                "ml_training_active": status.ml_training_active,
                "rl_training_active": status.rl_training_active,
                "continuous_learning_active": status.continuous_learning_active
            },
            "performance": {
                "ml_prediction_accuracy_pct": status.ml_prediction_accuracy_pct,
                "rl_action_success_rate_pct": status.rl_action_success_rate_pct,
                "ensemble_agreement_pct": status.ensemble_agreement_pct
            },
            "last_updated": status.last_updated.isoformat()
        }
    
    def _serialize_position(self, position) -> Dict[str, Any]:
        """Serialize position data"""
        return {
            "symbol": position.symbol,
            "chain": position.chain,
            "side": position.side,
            "size": str(position.size),
            "entry_price": str(position.entry_price),
            "current_price": str(position.current_price),
            "unrealized_pnl": str(position.unrealized_pnl),
            "unrealized_pnl_pct": str(position.unrealized_pnl_pct),
            "leverage": position.leverage,
            "entry_time": position.entry_time.isoformat(),
            "last_updated": position.last_updated.isoformat(),
            # Mode context for clear distinction
            "mode": position.mode.value,
            "is_simulated": position.is_simulated
        }
    
    def _serialize_trade(self, trade) -> Dict[str, Any]:
        """Serialize trade data"""
        return {
            "id": trade.id,
            "symbol": trade.symbol,
            "chain": trade.chain,
            "dex": trade.dex,
            "side": trade.side,
            "size": str(trade.size),
            "price": str(trade.price),
            "value_usd": str(trade.value_usd),
            "fee": str(trade.fee),
            "slippage_pct": str(trade.slippage_pct),
            "execution_time_ms": trade.execution_time_ms,
            "status": trade.status,
            "timestamp": trade.timestamp.isoformat(),
            # Mode context for clear distinction
            "mode": trade.mode.value,
            "is_simulated": trade.is_simulated
        }
    
    def _serialize_ml_model(self, model) -> Dict[str, Any]:
        """Serialize ML model data"""
        return {
            "name": model.name,
            "type": model.type,
            "status": model.status.value,
            "accuracy": model.accuracy,
            "last_training_time": model.last_training_time.isoformat() if model.last_training_time else None,
            "predictions_today": model.predictions_today,
            "avg_prediction_time_ms": model.avg_prediction_time_ms,
            "model_size_mb": model.model_size_mb,
            "version": model.version
        }
    
    def _serialize_rl_agent(self, agent) -> Dict[str, Any]:
        """Serialize RL agent data"""
        return {
            "name": agent.name,
            "algorithm": agent.algorithm,
            "status": agent.status.value,
            "episode": agent.episode,
            "epsilon": agent.epsilon,
            "avg_reward": agent.avg_reward,
            "win_rate_pct": agent.win_rate_pct,
            "experience_buffer_size": agent.experience_buffer_size,
            "last_training_time": agent.last_training_time.isoformat() if agent.last_training_time else None,
            "actions_today": agent.actions_today,
            "avg_decision_time_ms": agent.avg_decision_time_ms
        }


# Create global API instance
dashboard_api = DashboardAPI()