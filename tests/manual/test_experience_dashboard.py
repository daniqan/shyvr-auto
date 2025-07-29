#!/usr/bin/env python3
"""
Test script for Experience Dashboard Frontend
Tests the frontend with simulated experience data
"""

import asyncio
import json
import random
import time
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Mock experience data
mock_experiences = []
mock_sessions = [
    {
        "session_id": "test-session-1",
        "session_name": "Test Trading Session 1",
        "status": "running",
        "experience_count": 0,
        "average_reward": 0.0,
        "duration_minutes": 0
    },
    {
        "session_id": "test-session-2", 
        "session_name": "Test Trading Session 2",
        "status": "completed",
        "experience_count": 150,
        "average_reward": 0.25,
        "duration_minutes": 45
    }
]

def generate_mock_experience():
    """Generate a mock experience"""
    now = datetime.now(timezone.utc)
    return {
        "id": len(mock_experiences) + 1,
        "session_id": random.choice(["test-session-1", "test-session-2"]),
        "step_number": len(mock_experiences),
        "state": [random.random() for _ in range(10)],
        "action": random.randint(0, 3),
        "reward": random.uniform(-1.0, 2.0),
        "next_state": [random.random() for _ in range(10)],
        "done": random.random() < 0.1,
        "priority": random.uniform(0.1, 1.0),
        "created_at": now.isoformat(),
        "metadata": {
            "trading_pair": random.choice(["BTC/USD", "ETH/USD", "SOL/USD"]),
            "market_condition": random.choice(["bullish", "bearish", "sideways"])
        }
    }

# Generate initial mock data
for _ in range(100):
    mock_experiences.append(generate_mock_experience())

@app.get("/")
async def serve_dashboard():
    """Serve the main dashboard HTML"""
    with open("static/index.html", "r") as f:
        content = f.read()
    return HTMLResponse(content=content)

@app.get("/api/auth/key")
async def get_api_key():
    """Mock API key endpoint"""
    return {"api_key": "test-api-key-12345"}

@app.get("/api/v1/experiences/recent")
async def get_recent_experiences(limit: int = 100, offset: int = 0, hours_back: int = 24):
    """Mock recent experiences endpoint"""
    # Simulate filtering by hours_back
    recent_experiences = mock_experiences[-limit:]
    
    return {
        "experiences": recent_experiences,
        "total_count": len(mock_experiences),
        "has_more": len(mock_experiences) > limit + offset,
        "pagination": {
            "limit": limit,
            "offset": offset,
            "total": len(mock_experiences)
        }
    }

@app.get("/api/v1/experiences/stats")
async def get_experience_stats(hours_back: int = 24):
    """Mock experience statistics endpoint"""
    total_experiences = len(mock_experiences)
    if total_experiences == 0:
        return {
            "total_experiences": 0,
            "average_reward": 0.0,
            "success_rate": 0.0,
            "active_sessions": 0,
            "sessions": []
        }
    
    average_reward = sum(exp["reward"] for exp in mock_experiences) / total_experiences
    successful_experiences = len([exp for exp in mock_experiences if exp["reward"] > 0])
    success_rate = successful_experiences / total_experiences
    
    # Update session statistics
    for session in mock_sessions:
        session_experiences = [exp for exp in mock_experiences if exp["session_id"] == session["session_id"]]
        session["experience_count"] = len(session_experiences)
        if session_experiences:
            session["average_reward"] = sum(exp["reward"] for exp in session_experiences) / len(session_experiences)
            session["duration_minutes"] = len(session_experiences) * 2  # Mock duration
    
    return {
        "total_experiences": total_experiences,
        "average_reward": average_reward,
        "success_rate": success_rate,
        "active_sessions": len([s for s in mock_sessions if s["status"] == "running"]),
        "sessions": mock_sessions
    }

@app.get("/api/v1/experiences/performance")
async def get_experience_performance(hours_back: int = 24):
    """Mock experience performance endpoint"""
    if not mock_experiences:
        return {
            "overall_performance": {
                "cumulative_reward": 0.0,
                "win_rate": 0.0,
                "average_episode_length": 0.0,
                "training_efficiency": 0.0
            },
            "time_series": {
                "timestamps": [],
                "win_rates": [],
                "training_efficiency": [],
                "episode_lengths": []
            },
            "session_analysis": {
                "session_names": [],
                "experience_counts": [],
                "average_rewards": []
            },
            "detailed_analysis": {
                "reward_stats": {
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0
                },
                "action_distribution": {},
                "total_actions": 0,
                "learning_rate": 0.001,
                "exploration_rate": 0.1,
                "model_confidence": 0.5
            }
        }
    
    # Calculate overall performance
    cumulative_reward = sum(exp["reward"] for exp in mock_experiences)
    wins = len([exp for exp in mock_experiences if exp["reward"] > 0])
    win_rate = wins / len(mock_experiences)
    
    # Generate time series data (mock)
    now = datetime.now(timezone.utc)
    timestamps = []
    win_rates = []
    training_efficiency = []
    episode_lengths = []
    
    for i in range(24):  # 24 hour periods
        hour_ago = now.timestamp() - (i * 3600)
        timestamps.insert(0, datetime.fromtimestamp(hour_ago, timezone.utc).isoformat())
        win_rates.insert(0, random.uniform(0.3, 0.8) * 100)
        training_efficiency.insert(0, random.uniform(0.6, 0.95) * 100)
        episode_lengths.insert(0, random.uniform(10, 50))
    
    # Session analysis
    session_names = [s["session_name"] for s in mock_sessions]
    session_counts = [s["experience_count"] for s in mock_sessions]
    session_rewards = [s["average_reward"] for s in mock_sessions]
    
    # Action distribution
    action_dist = {}
    for exp in mock_experiences:
        action = str(exp["action"])
        action_dist[action] = action_dist.get(action, 0) + 1
    
    rewards = [exp["reward"] for exp in mock_experiences]
    reward_mean = sum(rewards) / len(rewards)
    reward_std = (sum((r - reward_mean) ** 2 for r in rewards) / len(rewards)) ** 0.5
    
    return {
        "overall_performance": {
            "cumulative_reward": cumulative_reward,
            "win_rate": win_rate,
            "average_episode_length": 25.5,
            "training_efficiency": 0.85
        },
        "time_series": {
            "timestamps": timestamps,
            "win_rates": win_rates,
            "training_efficiency": training_efficiency,
            "episode_lengths": episode_lengths
        },
        "session_analysis": {
            "session_names": session_names,
            "experience_counts": session_counts,
            "average_rewards": session_rewards
        },
        "detailed_analysis": {
            "reward_stats": {
                "mean": reward_mean,
                "std": reward_std,
                "min": min(rewards),
                "max": max(rewards)
            },
            "action_distribution": action_dist,
            "total_actions": len(mock_experiences),
            "learning_rate": 0.001,
            "exploration_rate": 0.1,
            "model_confidence": 0.75
        }
    }

@app.get("/api/v1/experiences/search")
async def search_experiences(session_id: str = None, limit: int = 100):
    """Mock experience search endpoint"""
    if session_id:
        filtered_experiences = [exp for exp in mock_experiences if exp["session_id"] == session_id]
        session_name = next((s["session_name"] for s in mock_sessions if s["session_id"] == session_id), "Unknown Session")
    else:
        filtered_experiences = mock_experiences
        session_name = "All Sessions"
    
    return {
        "experiences": filtered_experiences[:limit],
        "session_name": session_name,
        "total_count": len(filtered_experiences),
        "filters": {"session_id": session_id} if session_id else {}
    }

@app.get("/dashboard/data")
async def get_dashboard_data():
    """Mock dashboard data endpoint"""
    return {
        "system_metrics": {
            "status": "healthy",
            "uptime_seconds": 3600,
            "cpu_usage_pct": 25.5,
            "memory_usage_mb": 512,
            "active_connections": 5,
            "requests_per_minute": 45,
            "error_rate_pct": 0.1,
            "response_time_ms": 85,
            "performance": {"cache_hit_rate_pct": 95.2},
            "component_statuses": {
                "database": "healthy",
                "trading_engine": "healthy",
                "ml_service": "healthy",
                "risk_manager": "healthy"
            }
        },
        "trading_status": {
            "mode": "simulation",
            "is_simulated": True,
            "is_trading_active": True,
            "risk_limits_active": True,
            "trades_today": 12,
            "volume_today_usd": 25000.50,
            "win_rate_pct": 68.5,
            "tokens_analyzed_today": 150,
            "high_confidence_signals": 8,
            "tokens_in_watchlist": 25
        },
        "portfolio_status": {
            "mode": "simulation",
            "is_simulated": True,
            "total_value_usd": 10500.25,
            "available_balance_usd": 2500.75,
            "daily_pnl_usd": 150.50,
            "daily_pnl_pct": 1.45,
            "unrealized_pnl_usd": 75.25,
            "realized_pnl_usd": 275.80,
            "total_return_pct": 5.25,
            "max_drawdown_pct": -2.15,
            "current_drawdown_pct": -0.85,
            "sharpe_ratio": 1.85,
            "var_95_usd": -125.50,
            "position_count": 5,
            "chain_balances": {
                "ethereum": 5250.30,
                "solana": 3150.45,
                "polygon": 2099.50
            },
            "positions": []
        },
        "ml_rl_status": {
            "ml_rl_integration_active": True,
            "ml_prediction_accuracy_pct": 72.5,
            "rl_action_success_rate_pct": 68.2,
            "ml_rl_decision_latency_ms": 45.8,
            "ml_confidence_threshold": 0.75,
            "rl_action_confidence": 0.82,
            "ml_training_active": True,
            "rl_training_active": True,
            "continuous_learning_active": True,
            "models": {
                "ml_models": [
                    {"name": "Price Predictor", "status": "active", "accuracy": 0.725},
                    {"name": "Signal Classifier", "status": "active", "accuracy": 0.689}
                ],
                "rl_agents": [
                    {"name": "Trading Agent", "status": "active", "win_rate_pct": 68.2},
                    {"name": "Risk Agent", "status": "active", "win_rate_pct": 75.5}
                ]
            }
        },
        "recent_activity": {
            "logs": [
                "System started successfully",
                "Trading mode set to simulation", 
                "ML models loaded and initialized",
                "Portfolio balance updated"
            ]
        }
    }

# Background task to generate new experiences
async def generate_experiences_background():
    """Background task to generate new experiences periodically"""
    while True:
        await asyncio.sleep(random.uniform(1, 5))  # Random interval between 1-5 seconds
        new_experience = generate_mock_experience()
        mock_experiences.append(new_experience)
        
        # Keep only last 500 experiences to prevent memory issues
        if len(mock_experiences) > 500:
            mock_experiences.pop(0)
        
        print(f"Generated new experience: Reward={new_experience['reward']:.3f}, Action={new_experience['action']}")

@app.on_event("startup")
async def startup_event():
    """Start background tasks"""
    asyncio.create_task(generate_experiences_background())
    print("Test experience dashboard server started")
    print("Navigate to http://localhost:8000 to view the dashboard")
    print("The ML & RL tab contains the experience monitoring interface")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")