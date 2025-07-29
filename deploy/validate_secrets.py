#!/usr/bin/env python3
"""
Secret validation utility for Shyvr RLTE
Validates that all required secrets are accessible from GCP Secret Manager
"""
import os
import sys
from google.cloud import secretmanager
from typing import Dict, List, Optional

def get_secret_value(project_id: str, secret_name: str, version: str = "latest") -> Optional[str]:
    """Retrieve secret value from Secret Manager"""
    try:
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/{secret_name}/versions/{version}"
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        print(f"❌ Error accessing {secret_name}: {e}")
        return None

def validate_secrets(project_id: str) -> Dict[str, bool]:
    """Validate all required secrets"""
    
    # Define secret categories for validation
    required_secrets = [
        "TELEGRAM_TOKEN", "WEBHOOK_SECRET", "DB_PASSWORD", "DATABASE_URL"
    ]
    
    optional_secrets = [
        "HELIUS_API_KEY", "BIRDEYE_API_KEY", "ETHERSCAN_API_KEY",
        "XAI_API_KEY", "OPENAI_API_KEY", "LUNARCRUSH_API_KEY",
        "COINGECKO_API_KEY", "X_BEARER_TOKEN", "X_API_KEY"
    ]
    
    trading_secrets = [
        "SOLANA_PRIVATE_KEY", "ETHEREUM_PRIVATE_KEY", "HYPERLIQUID_PRIVATE_KEY"
    ]
    
    results = {}
    
    print(f"🔍 Validating secrets in project: {project_id}")
    print()
    
    # Check required secrets
    print("📋 Required Secrets:")
    for secret in required_secrets:
        value = get_secret_value(project_id, secret)
        if value:
            print(f"  ✅ {secret}: Available")
            results[secret] = True
        else:
            print(f"  ❌ {secret}: Missing or inaccessible")
            results[secret] = False
    
    print()
    
    # Check optional secrets
    print("📋 Optional Secrets:")
    for secret in optional_secrets:
        value = get_secret_value(project_id, secret)
        if value:
            print(f"  ✅ {secret}: Available")
            results[secret] = True
        else:
            print(f"  ⚠️  {secret}: Not configured")
            results[secret] = False
    
    print()
    
    # Check trading secrets (with warning)
    print("📋 Trading Secrets (Live Mode Only):")
    for secret in trading_secrets:
        value = get_secret_value(project_id, secret)
        if value:
            print(f"  ⚠️  {secret}: Available (LIVE TRADING ENABLED)")
            results[secret] = True
        else:
            print(f"  ℹ️  {secret}: Not configured (Live trading disabled)")
            results[secret] = False
    
    return results

def main():
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "shvyr-ai-bots")
    
    if len(sys.argv) > 1:
        project_id = sys.argv[1]
    
    print(f"🔐 Shyvr RLTE Secret Validation")
    print(f"Project: {project_id}")
    print("=" * 50)
    
    results = validate_secrets(project_id)
    
    # Summary
    total_secrets = len(results)
    available_secrets = sum(results.values())
    
    print()
    print("=" * 50)
    print(f"📊 Summary: {available_secrets}/{total_secrets} secrets available")
    
    required_available = sum(1 for k, v in results.items() 
                           if k in ["TELEGRAM_TOKEN", "WEBHOOK_SECRET", "DB_PASSWORD", "DATABASE_URL"] 
                           and v)
    
    if required_available == 4:
        print("✅ All required secrets are available - System can start")
        return 0
    else:
        print(f"❌ Missing required secrets - {4-required_available} required secrets unavailable")
        return 1

if __name__ == "__main__":
    exit(main())