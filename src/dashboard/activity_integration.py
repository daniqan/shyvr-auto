"""
Dashboard Activity Integration
Integration layer between activity logging and dashboard components
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from decimal import Decimal

from ..activity_logging.activity_logger import (
    activity_logger,
    ActivityCategory,
    ActivityAction,
    ActivitySeverity,
    TradingMode,
    ChainType,
    performance_tracker,
    log_dashboard_action,
    log_system_event
)
from ..utils.database import get_database_pool


class DashboardActivityIntegration:
    """Integration between dashboard and activity logging"""
    
    def __init__(self):
        self._pool = None
    
    async def start(self):
        """Initialize the integration"""
        self._pool = await get_database_pool()
        await activity_logger.start()
    
    async def stop(self):
        """Stop the integration"""
        await activity_logger.stop()
    
    # =============================================================================
    # DASHBOARD ACTIVITY LOGGING
    # =============================================================================
    
    async def log_dashboard_access(
        self,
        user_id: int,
        component: str,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> str:
        """Log dashboard component access"""
        return await log_dashboard_action(
            user_id=user_id,
            component=component,
            action="access",
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent
        )
    
    async def log_dashboard_action(
        self,
        user_id: int,
        component: str,
        action: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """Log specific dashboard action"""
        return await activity_logger.log_user_action(
            user_id=user_id,
            action=ActivityAction.EXECUTE,
            component=component,
            title=f"Dashboard action: {action}",
            session_id=session_id,
            dashboard_action=action,
            metadata=metadata,
            **kwargs
        )
    
    async def log_dashboard_error(
        self,
        component: str,
        error_type: str,
        error_message: str,
        user_id: Optional[int] = None,
        session_id: Optional[str] = None,
        stack_trace: Optional[str] = None,
        **kwargs
    ) -> str:
        """Log dashboard error"""
        return await activity_logger.log_error(
            category=ActivityCategory.DASHBOARD,
            source=f"dashboard_{component}",
            event_type=f"dashboard_error_{error_type}",
            title=f"Dashboard error in {component}",
            error_message=error_message,
            user_id=user_id,
            session_id=session_id,
            stack_trace=stack_trace,
            dashboard_component=component,
            **kwargs
        )
    
    # =============================================================================
    # TRADING ACTIVITY LOGGING
    # =============================================================================
    
    async def log_trading_mode_switch(
        self,
        user_id: int,
        from_mode: str,
        to_mode: str,
        session_id: Optional[str] = None
    ) -> str:
        """Log trading mode switch"""
        return await activity_logger.log_trading_activity(
            action=ActivityAction.UPDATE,
            title=f"Trading mode switched from {from_mode} to {to_mode}",
            trading_mode=TradingMode(to_mode),
            user_id=user_id,
            session_id=session_id,
            metadata={
                "from_mode": from_mode,
                "to_mode": to_mode,
                "switched_by": user_id
            }
        )
    
    async def log_emergency_stop(
        self,
        user_id: int,
        session_id: Optional[str] = None,
        reason: Optional[str] = None
    ) -> str:
        """Log emergency stop activation"""
        return await activity_logger.log_activity(
            category=ActivityCategory.TRADING,
            action=ActivityAction.STOP,
            source="dashboard_control",
            event_type="emergency_stop",
            title="Emergency stop activated",
            severity=ActivitySeverity.ALERT,
            user_id=user_id,
            session_id=session_id,
            metadata={"reason": reason, "activated_by": user_id}
        )
    
    async def log_risk_limits_update(
        self,
        user_id: int,
        old_limits: Dict[str, Any],
        new_limits: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> str:
        """Log risk limits update"""
        return await activity_logger.log_activity(
            category=ActivityCategory.CONFIGURATION,
            action=ActivityAction.UPDATE,
            source="dashboard_settings",
            event_type="risk_limits_update",
            title="Risk limits updated",
            severity=ActivitySeverity.NOTICE,
            user_id=user_id,
            session_id=session_id,
            metadata={
                "old_limits": old_limits,
                "new_limits": new_limits,
                "updated_by": user_id
            }
        )
    
    # =============================================================================
    # PERFORMANCE TRACKING
    # =============================================================================
    
    def track_dashboard_performance(
        self,
        component: str,
        operation: str,
        user_id: Optional[int] = None,
        session_id: Optional[str] = None
    ) -> performance_tracker:
        """Context manager for tracking dashboard operation performance"""
        return performance_tracker(
            source=f"dashboard_{component}",
            operation=operation,
            category=ActivityCategory.DASHBOARD,
            user_id=user_id,
            session_id=session_id,
            dashboard_component=component,
            dashboard_action=operation
        )
    
    async def log_api_performance(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        response_time_ms: int,
        user_id: Optional[int] = None
    ) -> str:
        """Log dashboard API performance"""
        success = 200 <= status_code < 400
        return await activity_logger.log_api_call(
            api_name="dashboard_api",
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            response_time_ms=response_time_ms,
            success=success,
            user_id=user_id
        )
    
    # =============================================================================
    # ACTIVITY QUERIES FOR DASHBOARD
    # =============================================================================
    
    async def get_recent_activity(
        self,
        limit: int = 100,
        category: Optional[str] = None,
        severity: Optional[str] = None,
        user_id: Optional[int] = None,
        hours_back: int = 24
    ) -> List[Dict[str, Any]]:
        """Get recent activity for dashboard display"""
        
        query = """
            SELECT 
                activity_id,
                created_at,
                category,
                action,
                severity,
                source,
                event_type,
                title,
                description,
                user_id,
                trading_mode,
                execution_time_ms,
                dashboard_component,
                dashboard_action,
                metadata
            FROM activity_logs 
            WHERE created_at > NOW() - INTERVAL '%s hours'
        """
        
        params = [hours_back]
        conditions = []
        
        if category:
            conditions.append("AND category = $%d")
            params.append(category)
        
        if severity:
            conditions.append("AND severity = $%d")
            params.append(severity)
        
        if user_id:
            conditions.append("AND user_id = $%d")
            params.append(user_id)
        
        query += " " + " ".join(conditions)
        query += " ORDER BY created_at DESC LIMIT $%d" % (len(params) + 1)
        params.append(limit)
        
        # Update parameter placeholders
        for i, condition in enumerate(conditions):
            query = query.replace("$%d" % (i + 2), "$%d" % (i + 2))
        
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            
            return [
                {
                    "activity_id": str(row["activity_id"]),
                    "created_at": row["created_at"].isoformat(),
                    "category": row["category"],
                    "action": row["action"],
                    "severity": row["severity"],
                    "source": row["source"],
                    "event_type": row["event_type"],
                    "title": row["title"],
                    "description": row["description"],
                    "user_id": row["user_id"],
                    "trading_mode": row["trading_mode"],
                    "execution_time_ms": row["execution_time_ms"],
                    "dashboard_component": row["dashboard_component"],
                    "dashboard_action": row["dashboard_action"],
                    "metadata": row["metadata"]
                }
                for row in rows
            ]
    
    async def get_error_summary(
        self,
        hours_back: int = 24
    ) -> List[Dict[str, Any]]:
        """Get error summary for dashboard monitoring"""
        
        query = """
            SELECT 
                category,
                source,
                error_code,
                COUNT(*) as error_count,
                MAX(created_at) as last_occurrence,
                array_agg(DISTINCT severity) as severities,
                array_agg(DISTINCT event_type) as event_types
            FROM activity_logs 
            WHERE severity IN ('error', 'critical', 'alert', 'emergency')
            AND created_at > NOW() - INTERVAL '%s hours'
            GROUP BY category, source, error_code
            ORDER BY error_count DESC, last_occurrence DESC
            LIMIT 50
        """
        
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, hours_back)
            
            return [
                {
                    "category": row["category"],
                    "source": row["source"],
                    "error_code": row["error_code"],
                    "error_count": row["error_count"],
                    "last_occurrence": row["last_occurrence"].isoformat(),
                    "severities": row["severities"],
                    "event_types": row["event_types"]
                }
                for row in rows
            ]
    
    async def get_user_activity_stats(
        self,
        user_id: int,
        days_back: int = 7
    ) -> Dict[str, Any]:
        """Get user activity statistics"""
        
        query = """
            SELECT 
                COUNT(*) as total_activities,
                COUNT(DISTINCT DATE(created_at)) as active_days,
                COUNT(DISTINCT dashboard_component) as components_used,
                COUNT(*) FILTER (WHERE severity IN ('error', 'critical')) as errors,
                MAX(created_at) as last_activity,
                array_agg(DISTINCT category) as categories,
                AVG(execution_time_ms) as avg_response_time
            FROM activity_logs 
            WHERE user_id = $1 
            AND created_at > NOW() - INTERVAL '%s days'
        """
        
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, user_id, days_back)
            
            if row:
                return {
                    "total_activities": row["total_activities"],
                    "active_days": row["active_days"],
                    "components_used": row["components_used"],
                    "errors_encountered": row["errors"],
                    "last_activity": row["last_activity"].isoformat() if row["last_activity"] else None,
                    "categories_used": row["categories"] or [],
                    "avg_response_time_ms": float(row["avg_response_time"]) if row["avg_response_time"] else None
                }
            else:
                return {
                    "total_activities": 0,
                    "active_days": 0,
                    "components_used": 0,
                    "errors_encountered": 0,
                    "last_activity": None,
                    "categories_used": [],
                    "avg_response_time_ms": None
                }
    
    async def get_performance_metrics(
        self,
        hours_back: int = 1
    ) -> List[Dict[str, Any]]:
        """Get performance metrics for system monitoring"""
        
        query = """
            SELECT 
                source,
                category,
                COUNT(*) as total_operations,
                AVG(execution_time_ms) as avg_execution_time,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY execution_time_ms) as p95_execution_time,
                AVG(response_time_ms) as avg_response_time,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms) as p95_response_time,
                COUNT(*) FILTER (WHERE severity IN ('error', 'critical')) as error_count,
                ROUND(
                    COUNT(*) FILTER (WHERE severity IN ('error', 'critical')) * 100.0 / COUNT(*), 
                    2
                ) as error_rate_pct
            FROM activity_logs 
            WHERE created_at > NOW() - INTERVAL '%s hours'
            AND (execution_time_ms IS NOT NULL OR response_time_ms IS NOT NULL)
            GROUP BY source, category
            HAVING COUNT(*) >= 5  -- Only include sources with meaningful data
            ORDER BY total_operations DESC
            LIMIT 50
        """
        
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, hours_back)
            
            return [
                {
                    "source": row["source"],
                    "category": row["category"],
                    "total_operations": row["total_operations"],
                    "avg_execution_time_ms": float(row["avg_execution_time"]) if row["avg_execution_time"] else None,
                    "p95_execution_time_ms": float(row["p95_execution_time"]) if row["p95_execution_time"] else None,
                    "avg_response_time_ms": float(row["avg_response_time"]) if row["avg_response_time"] else None,
                    "p95_response_time_ms": float(row["p95_response_time"]) if row["p95_response_time"] else None,
                    "error_count": row["error_count"],
                    "error_rate_pct": float(row["error_rate_pct"]) if row["error_rate_pct"] else 0.0
                }
                for row in rows
            ]


# Global instance
dashboard_activity = DashboardActivityIntegration()


# Convenience functions for common dashboard operations
async def log_dashboard_startup():
    """Log dashboard startup"""
    return await log_system_event(
        event_type="dashboard_startup",
        title="Dashboard service started",
        severity=ActivitySeverity.INFO,
        source="dashboard_service"
    )


async def log_dashboard_shutdown():
    """Log dashboard shutdown"""
    return await log_system_event(
        event_type="dashboard_shutdown",
        title="Dashboard service stopped",
        severity=ActivitySeverity.INFO,
        source="dashboard_service"
    )


async def log_websocket_connection(user_id: Optional[int] = None, session_id: Optional[str] = None):
    """Log WebSocket connection"""
    return await activity_logger.log_activity(
        category=ActivityCategory.DASHBOARD,
        action=ActivityAction.ACCESS,
        source="websocket_manager",
        event_type="websocket_connect",
        title="WebSocket connection established",
        user_id=user_id,
        session_id=session_id
    )


async def log_websocket_disconnection(user_id: Optional[int] = None, session_id: Optional[str] = None):
    """Log WebSocket disconnection"""
    return await activity_logger.log_activity(
        category=ActivityCategory.DASHBOARD,
        action=ActivityAction.STOP,
        source="websocket_manager",
        event_type="websocket_disconnect",
        title="WebSocket connection closed",
        user_id=user_id,
        session_id=session_id
    )