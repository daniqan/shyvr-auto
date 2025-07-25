#!/usr/bin/env python3
"""
Test script to verify critical import fixes for DEX and RL Agent modules.
This follows TDD - create failing tests first, then fix to make them pass.
"""

def test_dex_imports():
    """Test that all DEX module classes can be imported correctly."""
    try:
        from src.dex import JupiterDEXClient
        print("✅ JupiterDEXClient import successful")
    except ImportError as e:
        print(f"❌ JupiterDEXClient import failed: {e}")
        return False
    
    try:
        from src.dex import UniswapV3Client
        print("✅ UniswapV3Client import successful")
    except ImportError as e:
        print(f"❌ UniswapV3Client import failed: {e}")
        return False
        
    try:
        from src.dex import HyperliquidDEXClient
        print("✅ HyperliquidDEXClient import successful")
    except ImportError as e:
        print(f"❌ HyperliquidDEXClient import failed: {e}")
        return False
    
    return True

def test_rl_agent_imports():
    """Test that all RL Agent module classes can be imported correctly."""
    try:
        from src.rl_agent import TradingEnvironment
        print("✅ TradingEnvironment import successful")
    except ImportError as e:
        print(f"❌ TradingEnvironment import failed: {e}")
        return False
    
    try:
        from src.rl_agent import ExperienceReplayBuffer
        print("✅ ExperienceReplayBuffer import successful")
    except ImportError as e:
        print(f"❌ ExperienceReplayBuffer import failed: {e}")
        return False
        
    try:
        from src.rl_agent import PrioritizedExperienceReplayBuffer
        print("✅ PrioritizedExperienceReplayBuffer import successful")
    except ImportError as e:
        print(f"❌ PrioritizedExperienceReplayBuffer import failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Testing critical imports...")
    print("\n=== DEX Module Imports ===")
    dex_success = test_dex_imports()
    
    print("\n=== RL Agent Module Imports ===")
    rl_success = test_rl_agent_imports()
    
    print(f"\n=== Summary ===")
    print(f"DEX imports: {'✅ PASS' if dex_success else '❌ FAIL'}")
    print(f"RL Agent imports: {'✅ PASS' if rl_success else '❌ FAIL'}")
    
    if dex_success and rl_success:
        print("🎉 All critical imports working!")
        exit(0)
    else:
        print("🚫 Import fixes needed")
        exit(1)