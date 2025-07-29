#!/usr/bin/env python3
"""
Experience Data Growth and Usage Pattern Analysis

Analyzes experience storage growth patterns, usage trends, and provides
insights for capacity planning and optimization.
"""

import asyncio
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import structlog

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.database import get_database_connection
from src.utils.config import get_config

logger = structlog.get_logger()


class ExperienceGrowthAnalyzer:
    """
    Analyzer for experience storage growth patterns and usage trends.
    
    Provides insights into data growth rates, storage efficiency,
    and usage patterns for capacity planning.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the growth analyzer."""
        self.config = config
        self.analysis_results: Dict[str, Any] = {}
        
    async def analyze_growth_patterns(self, days_back: int = 30) -> Dict[str, Any]:
        """Analyze experience data growth patterns over specified period."""
        try:
            logger.info("Starting experience growth pattern analysis", days_back=days_back)
            
            # Get time range for analysis
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days_back)
            
            async with get_database_connection() as conn:
                # Get daily growth statistics
                daily_stats = await self._get_daily_growth_stats(conn, start_time, end_time)
                
                # Get session-level growth patterns
                session_patterns = await self._get_session_growth_patterns(conn, start_time, end_time)
                
                # Get storage efficiency metrics
                storage_efficiency = await self._get_storage_efficiency_metrics(conn)
                
                # Get usage pattern analysis
                usage_patterns = await self._get_usage_patterns(conn, start_time, end_time)
                
                # Perform trend analysis
                trend_analysis = self._analyze_trends(daily_stats)
                
                # Generate capacity planning recommendations
                capacity_planning = self._generate_capacity_recommendations(
                    daily_stats, trend_analysis, storage_efficiency
                )
                
                results = {
                    "analysis_period": {
                        "start_time": start_time.isoformat(),
                        "end_time": end_time.isoformat(),
                        "days_analyzed": days_back
                    },
                    "daily_growth_stats": daily_stats,
                    "session_patterns": session_patterns,
                    "storage_efficiency": storage_efficiency,
                    "usage_patterns": usage_patterns,
                    "trend_analysis": trend_analysis,
                    "capacity_planning": capacity_planning,
                    "analysis_timestamp": datetime.now().isoformat()
                }
                
                self.analysis_results = results
                return results
                
        except Exception as e:
            logger.error("Failed to analyze growth patterns", error=str(e))
            raise
    
    async def _get_daily_growth_stats(self, conn, start_time: datetime, 
                                    end_time: datetime) -> List[Dict[str, Any]]:
        """Get daily experience growth statistics."""
        query = """
        WITH daily_stats AS (
            SELECT 
                DATE(created_at) as date,
                COUNT(*) as experiences_added,
                COUNT(DISTINCT session_id) as active_sessions,
                AVG(priority) as avg_priority,
                SUM(CASE WHEN done THEN 1 ELSE 0 END) as terminal_experiences,
                AVG(reward) as avg_reward,
                MAX(step_number) as max_step_number
            FROM rl_experiences 
            WHERE created_at >= $1 AND created_at < $2
            GROUP BY DATE(created_at)
            ORDER BY date
        ),
        cumulative_stats AS (
            SELECT 
                date,
                experiences_added,
                SUM(experiences_added) OVER (ORDER BY date) as cumulative_experiences,
                active_sessions,
                avg_priority,
                terminal_experiences,
                avg_reward,
                max_step_number
            FROM daily_stats
        )
        SELECT * FROM cumulative_stats
        """
        
        rows = await conn.fetch(query, start_time, end_time)
        return [dict(row) for row in rows]
    
    async def _get_session_growth_patterns(self, conn, start_time: datetime,
                                         end_time: datetime) -> List[Dict[str, Any]]:
        """Get growth patterns by training session."""
        query = """
        WITH session_stats AS (
            SELECT 
                re.session_id,
                ts.session_name,
                ts.trading_mode,
                COUNT(re.id) as total_experiences,
                MIN(re.created_at) as first_experience,
                MAX(re.created_at) as last_experience,
                EXTRACT(EPOCH FROM (MAX(re.created_at) - MIN(re.created_at)))/3600 as session_duration_hours,
                COUNT(re.id) / GREATEST(EXTRACT(EPOCH FROM (MAX(re.created_at) - MIN(re.created_at)))/3600, 0.01) as experiences_per_hour,
                AVG(re.reward) as avg_reward,
                STDDEV(re.reward) as reward_stddev,
                MAX(re.step_number) as max_step_number,
                COUNT(DISTINCT DATE(re.created_at)) as active_days
            FROM rl_experiences re
            JOIN rl_training_sessions ts ON re.session_id = ts.session_id
            WHERE re.created_at >= $1 AND re.created_at < $2
            GROUP BY re.session_id, ts.session_name, ts.trading_mode
            HAVING COUNT(re.id) > 10  -- Only sessions with meaningful data
        )
        SELECT 
            session_id,
            session_name,
            trading_mode,
            total_experiences,
            first_experience,
            last_experience,
            session_duration_hours,
            experiences_per_hour,
            avg_reward,
            reward_stddev,
            max_step_number,
            active_days,
            CASE 
                WHEN session_duration_hours > 0 
                THEN total_experiences / session_duration_hours 
                ELSE 0 
            END as growth_rate
        FROM session_stats
        ORDER BY total_experiences DESC
        """
        
        rows = await conn.fetch(query, start_time, end_time)
        return [dict(row) for row in rows]
    
    async def _get_storage_efficiency_metrics(self, conn) -> Dict[str, Any]:
        """Get storage efficiency and optimization metrics."""
        queries = {
            "total_storage": """
                SELECT 
                    COUNT(*) as total_experiences,
                    COUNT(DISTINCT session_id) as unique_sessions,
                    pg_size_pretty(pg_total_relation_size('rl_experiences')) as table_size,
                    pg_total_relation_size('rl_experiences') as table_size_bytes
                FROM rl_experiences
            """,
            
            "index_usage": """
                SELECT 
                    schemaname,
                    tablename,
                    indexname,
                    idx_scan as index_scans,
                    idx_tup_read as tuples_read,
                    idx_tup_fetch as tuples_fetched
                FROM pg_stat_user_indexes 
                WHERE tablename = 'rl_experiences'
                ORDER BY idx_scan DESC
            """,
            
            "duplicate_analysis": """
                WITH state_hashes AS (
                    SELECT 
                        session_id,
                        md5(state::text) as state_hash,
                        action,
                        COUNT(*) as occurrence_count
                    FROM rl_experiences 
                    GROUP BY session_id, md5(state::text), action
                    HAVING COUNT(*) > 1
                )
                SELECT 
                    COUNT(*) as duplicate_state_action_pairs,
                    SUM(occurrence_count - 1) as redundant_experiences,
                    AVG(occurrence_count) as avg_duplicates_per_state
                FROM state_hashes
            """,
            
            "priority_distribution": """
                SELECT 
                    CASE 
                        WHEN priority < 0.1 THEN 'very_low'
                        WHEN priority < 0.5 THEN 'low'
                        WHEN priority < 1.0 THEN 'medium'
                        WHEN priority < 2.0 THEN 'high'
                        ELSE 'very_high'
                    END as priority_bucket,
                    COUNT(*) as count,
                    COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() as percentage
                FROM rl_experiences
                GROUP BY 1
                ORDER BY 
                    CASE 
                        WHEN priority < 0.1 THEN 1
                        WHEN priority < 0.5 THEN 2
                        WHEN priority < 1.0 THEN 3
                        WHEN priority < 2.0 THEN 4
                        ELSE 5
                    END
            """
        }
        
        results = {}
        for metric_name, query in queries.items():
            try:
                rows = await conn.fetch(query)
                results[metric_name] = [dict(row) for row in rows]
            except Exception as e:
                logger.error(f"Failed to get {metric_name} metrics", error=str(e))
                results[metric_name] = []
        
        return results
    
    async def _get_usage_patterns(self, conn, start_time: datetime,
                                end_time: datetime) -> Dict[str, Any]:
        """Analyze experience usage patterns and access frequency."""
        queries = {
            "hourly_patterns": """
                SELECT 
                    EXTRACT(HOUR FROM created_at) as hour,
                    COUNT(*) as experiences_created,
                    AVG(priority) as avg_priority
                FROM rl_experiences
                WHERE created_at >= $1 AND created_at < $2
                GROUP BY EXTRACT(HOUR FROM created_at)
                ORDER BY hour
            """,
            
            "daily_patterns": """
                SELECT 
                    EXTRACT(DOW FROM created_at) as day_of_week,
                    COUNT(*) as experiences_created,
                    COUNT(DISTINCT session_id) as active_sessions
                FROM rl_experiences
                WHERE created_at >= $1 AND created_at < $2
                GROUP BY EXTRACT(DOW FROM created_at)
                ORDER BY day_of_week
            """,
            
            "reward_patterns": """
                WITH reward_buckets AS (
                    SELECT 
                        CASE 
                            WHEN reward < -10 THEN 'very_negative'
                            WHEN reward < -1 THEN 'negative'
                            WHEN reward < 1 THEN 'neutral'
                            WHEN reward < 10 THEN 'positive'
                            ELSE 'very_positive'
                        END as reward_bucket,
                        COUNT(*) as count
                    FROM rl_experiences
                    WHERE created_at >= $1 AND created_at < $2
                    GROUP BY 1
                )
                SELECT 
                    reward_bucket,
                    count,
                    count * 100.0 / SUM(count) OVER() as percentage
                FROM reward_buckets
                ORDER BY 
                    CASE reward_bucket
                        WHEN 'very_negative' THEN 1
                        WHEN 'negative' THEN 2
                        WHEN 'neutral' THEN 3
                        WHEN 'positive' THEN 4
                        WHEN 'very_positive' THEN 5
                    END
            """,
            
            "episode_completion_patterns": """
                SELECT 
                    done,
                    COUNT(*) as count,
                    AVG(reward) as avg_reward,
                    AVG(priority) as avg_priority
                FROM rl_experiences
                WHERE created_at >= $1 AND created_at < $2
                GROUP BY done
            """
        }
        
        results = {}
        for pattern_name, query in queries.items():
            try:
                rows = await conn.fetch(query, start_time, end_time)
                results[pattern_name] = [dict(row) for row in rows]
            except Exception as e:
                logger.error(f"Failed to get {pattern_name} patterns", error=str(e))
                results[pattern_name] = []
        
        return results
    
    def _analyze_trends(self, daily_stats: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze growth trends and patterns."""
        if len(daily_stats) < 7:
            return {"error": "Insufficient data for trend analysis (need at least 7 days)"}
        
        try:
            # Extract time series data
            days = list(range(len(daily_stats)))
            experiences = [stat['experiences_added'] for stat in daily_stats]
            cumulative = [stat['cumulative_experiences'] for stat in daily_stats]
            sessions = [stat['active_sessions'] for stat in daily_stats]
            
            # Calculate growth rate
            growth_rates = []
            for i in range(1, len(cumulative)):
                if cumulative[i-1] > 0:
                    rate = (cumulative[i] - cumulative[i-1]) / cumulative[i-1] * 100
                    growth_rates.append(rate)
            
            # Linear regression for trend
            if len(experiences) > 1:
                coeffs = np.polyfit(days, experiences, 1)
                trend_slope = coeffs[0]
                trend_direction = "increasing" if trend_slope > 0 else "decreasing" if trend_slope < 0 else "stable"
            else:
                trend_slope = 0
                trend_direction = "unknown"
            
            # Calculate moving averages
            window_size = min(7, len(experiences))
            moving_avg = []
            for i in range(window_size - 1, len(experiences)):
                avg = sum(experiences[i-window_size+1:i+1]) / window_size
                moving_avg.append(avg)
            
            # Volatility analysis
            volatility = np.std(experiences) if len(experiences) > 1 else 0
            
            # Peak analysis
            max_daily = max(experiences) if experiences else 0
            min_daily = min(experiences) if experiences else 0
            max_daily_index = experiences.index(max_daily) if experiences else 0
            
            return {
                "trend_direction": trend_direction,
                "trend_slope": float(trend_slope),
                "average_daily_growth": float(np.mean(experiences)) if experiences else 0,
                "growth_volatility": float(volatility),
                "max_daily_experiences": max_daily,
                "min_daily_experiences": min_daily,
                "peak_day_index": max_daily_index,
                "moving_average_7day": moving_avg[-1] if moving_avg else 0,
                "total_growth_rate": float(np.mean(growth_rates)) if growth_rates else 0,
                "growth_acceleration": float(np.mean(growth_rates[-3:])) - float(np.mean(growth_rates[:3])) if len(growth_rates) >= 6 else 0
            }
            
        except Exception as e:
            logger.error("Failed to analyze trends", error=str(e))
            return {"error": f"Trend analysis failed: {str(e)}"}
    
    def _generate_capacity_recommendations(self, daily_stats: List[Dict[str, Any]],
                                         trend_analysis: Dict[str, Any],
                                         storage_efficiency: Dict[str, Any]) -> Dict[str, Any]:
        """Generate capacity planning recommendations."""
        try:
            recommendations = {
                "immediate_actions": [],
                "short_term_planning": [],
                "long_term_strategy": [],
                "optimization_opportunities": []
            }
            
            # Current storage analysis
            if storage_efficiency.get("total_storage"):
                total_exp = storage_efficiency["total_storage"][0].get("total_experiences", 0)
                table_size_bytes = storage_efficiency["total_storage"][0].get("table_size_bytes", 0)
                
                if total_exp > 0:
                    bytes_per_experience = table_size_bytes / total_exp
                    
                    # Storage efficiency recommendations
                    if bytes_per_experience > 10000:  # More than 10KB per experience
                        recommendations["optimization_opportunities"].append({
                            "issue": "high_storage_per_experience",
                            "description": f"Each experience uses {bytes_per_experience:.0f} bytes on average",
                            "recommendation": "Consider state compression or more efficient serialization"
                        })
            
            # Growth trend recommendations
            avg_daily_growth = trend_analysis.get("average_daily_growth", 0)
            trend_direction = trend_analysis.get("trend_direction", "unknown")
            
            if trend_direction == "increasing" and avg_daily_growth > 1000:
                recommendations["short_term_planning"].append({
                    "issue": "rapid_growth",
                    "description": f"Average daily growth of {avg_daily_growth:.0f} experiences",
                    "recommendation": "Plan for increased storage capacity and consider data lifecycle policies"
                })
            
            # Duplicate analysis recommendations
            if storage_efficiency.get("duplicate_analysis"):
                dup_data = storage_efficiency["duplicate_analysis"]
                if dup_data and dup_data[0].get("redundant_experiences", 0) > 0:
                    redundant = dup_data[0]["redundant_experiences"]
                    recommendations["optimization_opportunities"].append({
                        "issue": "duplicate_experiences",
                        "description": f"{redundant} redundant experiences found",
                        "recommendation": "Implement deduplication logic or state hashing"
                    })
            
            # Index usage recommendations
            if storage_efficiency.get("index_usage"):
                unused_indexes = [idx for idx in storage_efficiency["index_usage"] 
                                if idx.get("index_scans", 0) == 0]
                if unused_indexes:
                    recommendations["optimization_opportunities"].append({
                        "issue": "unused_indexes", 
                        "description": f"{len(unused_indexes)} indexes are not being used",
                        "recommendation": "Consider dropping unused indexes to improve write performance"
                    })
            
            # Priority distribution recommendations
            if storage_efficiency.get("priority_distribution"):
                priority_dist = storage_efficiency["priority_distribution"]
                very_low_pct = next((p["percentage"] for p in priority_dist 
                                   if p["priority_bucket"] == "very_low"), 0)
                if very_low_pct > 50:
                    recommendations["optimization_opportunities"].append({
                        "issue": "low_priority_experiences",
                        "description": f"{very_low_pct:.1f}% of experiences have very low priority",
                        "recommendation": "Consider more aggressive cleanup of low-priority experiences"
                    })
            
            # Capacity projections
            if avg_daily_growth > 0:
                days_to_million = (1000000 - total_exp) / avg_daily_growth if total_exp < 1000000 else 0
                if days_to_million > 0 and days_to_million < 90:
                    recommendations["short_term_planning"].append({
                        "issue": "approaching_capacity_milestone",
                        "description": f"Will reach 1M experiences in approximately {days_to_million:.0f} days",
                        "recommendation": "Prepare for increased resource requirements"
                    })
            
            return recommendations
            
        except Exception as e:
            logger.error("Failed to generate capacity recommendations", error=str(e))
            return {"error": f"Capacity planning failed: {str(e)}"}
    
    async def save_analysis_results(self, output_file: Optional[str] = None) -> str:
        """Save analysis results to file."""
        if not self.analysis_results:
            raise ValueError("No analysis results to save")
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"experience_growth_analysis_{timestamp}.json"
        
        output_path = Path(output_file)
        
        with open(output_path, 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)
        
        logger.info("Analysis results saved", output_file=str(output_path))
        return str(output_path)
    
    def print_summary(self) -> None:
        """Print a summary of the analysis results."""
        if not self.analysis_results:
            print("No analysis results available")
            return
        
        print("\n" + "="*80)
        print("EXPERIENCE DATA GROWTH ANALYSIS SUMMARY")
        print("="*80)
        
        # Analysis period
        period = self.analysis_results.get("analysis_period", {})
        print(f"\nAnalysis Period: {period.get('days_analyzed', 0)} days")
        print(f"From: {period.get('start_time', 'N/A')}")
        print(f"To: {period.get('end_time', 'N/A')}")
        
        # Trend analysis
        trends = self.analysis_results.get("trend_analysis", {})
        print(f"\nGrowth Trends:")
        print(f"  Direction: {trends.get('trend_direction', 'unknown').upper()}")
        print(f"  Average Daily Growth: {trends.get('average_daily_growth', 0):.0f} experiences")
        print(f"  Growth Volatility: {trends.get('growth_volatility', 0):.2f}")
        print(f"  Total Growth Rate: {trends.get('total_growth_rate', 0):.2f}% per day")
        
        # Storage efficiency
        storage = self.analysis_results.get("storage_efficiency", {})
        if storage.get("total_storage"):
            total_info = storage["total_storage"][0]
            print(f"\nStorage Statistics:")
            print(f"  Total Experiences: {total_info.get('total_experiences', 0):,}")
            print(f"  Unique Sessions: {total_info.get('unique_sessions', 0)}")
            print(f"  Table Size: {total_info.get('table_size', 'N/A')}")
        
        # Recommendations
        recs = self.analysis_results.get("capacity_planning", {})
        immediate = recs.get("immediate_actions", [])
        optimizations = recs.get("optimization_opportunities", [])
        
        if immediate or optimizations:
            print(f"\nKey Recommendations:")
            for action in immediate[:3]:  # Show top 3
                print(f"  • {action.get('description', 'N/A')}")
            for opt in optimizations[:3]:  # Show top 3
                print(f"  • {opt.get('description', 'N/A')}")
        
        print("\n" + "="*80)


async def main():
    """Main entry point for experience growth analysis."""
    try:
        logger.info("Starting experience data growth analysis")
        
        # Load configuration  
        config = get_config()
        
        # Create analyzer
        analyzer = ExperienceGrowthAnalyzer(config)
        
        # Parse command line arguments
        days_back = 30
        if len(sys.argv) > 1:
            try:
                days_back = int(sys.argv[1])
            except ValueError:
                logger.warning("Invalid days argument, using default", default=30)
        
        # Run analysis
        results = await analyzer.analyze_growth_patterns(days_back)
        
        # Print summary
        analyzer.print_summary()
        
        # Save results
        output_file = await analyzer.save_analysis_results()
        print(f"\nDetailed results saved to: {output_file}")
        
    except Exception as e:
        logger.error("Experience growth analysis failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())