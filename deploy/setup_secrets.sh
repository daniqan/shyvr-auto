#!/bin/bash
# Comprehensive Secret Manager setup for Shyvr RLTE
# Handles all API keys and sensitive configuration systematically

set -e

# Configuration
PROJECT_ID="shvyr-ai-bots"
REGION="us-central1"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔐 Setting up Secret Manager for Shyvr RLTE${NC}"
echo -e "${PURPLE}🤖 Comprehensive API key and sensitive configuration management${NC}"

# Enable Secret Manager API
echo -e "${YELLOW}📦 Enabling Secret Manager API...${NC}"
gcloud services enable secretmanager.googleapis.com --project=$PROJECT_ID

# Core System Secrets (Required)
declare -A REQUIRED_SECRETS=(
    ["TELEGRAM_TOKEN"]="Bot token from @BotFather on Telegram"
    ["WEBHOOK_SECRET"]="Random string for webhook authentication"
    ["DB_PASSWORD"]="PostgreSQL database password"
    ["DATABASE_URL"]="Complete PostgreSQL connection string"
)

# Blockchain & RPC Secrets
declare -A BLOCKCHAIN_SECRETS=(
    ["HELIUS_API_KEY"]="Solana RPC provider API key"
    ["BIRDEYE_API_KEY"]="DeFi data aggregator API key"
    ["ETHERSCAN_API_KEY"]="Ethereum blockchain explorer API key"
    ["ALCHEMY_API_KEY"]="General Ethereum/Polygon RPC provider"
    ["ETHEREUM_API_KEY"]="Ethereum-specific RPC API key"
    ["BASE_API_KEY"]="Base chain RPC API key"
    ["SOLANA_RPC_URL"]="Custom Solana RPC endpoint URL"
    ["ETHEREUM_RPC_URL"]="Custom Ethereum RPC endpoint URL"
)

# Trading & Wallet Secrets (Critical - Live Trading)
declare -A TRADING_SECRETS=(
    ["SOLANA_PRIVATE_KEY"]="Solana wallet private key (base58)"
    ["ETHEREUM_PRIVATE_KEY"]="Ethereum wallet private key (hex)"
    ["HYPERLIQUID_PRIVATE_KEY"]="Hyperliquid exchange private key"
    ["HYPERLIQUID_API_KEY"]="Hyperliquid exchange API key"
    ["WALLET_PRIVATE_KEY"]="Primary wallet private key"
)

# AI & ML API Secrets
declare -A AI_SECRETS=(
    ["XAI_API_KEY"]="xAI/Grok API key for analysis"
    ["OPENAI_API_KEY"]="OpenAI GPT API key"
    ["AGENT_API_KEY"]="Custom AI agent API key"
)

# Market Data & Analytics Secrets
declare -A MARKET_SECRETS=(
    ["LUNARCRUSH_API_KEY"]="Social sentiment data API key"
    ["COINGECKO_API_KEY"]="CoinGecko market data API key"
    ["COINGECKO_PRO_API_KEY"]="CoinGecko Pro API key"
    ["GLASSNODE_API_KEY"]="On-chain analytics API key"
    ["MESSARI_API_KEY"]="Crypto research data API key"
    ["JUPITER_API_KEY"]="Jupiter DEX aggregator API key"
)

# Social Media & External APIs
declare -A SOCIAL_SECRETS=(
    ["X_BEARER_TOKEN"]="X (Twitter) API bearer token"
    ["X_API_KEY"]="X (Twitter) API key"
    ["X_API_SECRET"]="X (Twitter) API secret"
)

# Function to create or update a secret
create_or_update_secret() {
    local secret_name=$1
    local description=$2
    local is_required=${3:-false}
    
    echo -e "${BLUE}Processing secret: $secret_name${NC}"
    
    # Check if secret already exists
    if gcloud secrets describe "$secret_name" --project=$PROJECT_ID --quiet 2>/dev/null; then
        echo -e "${YELLOW}  ⚠ Secret $secret_name already exists${NC}"
        
        # Prompt for update
        read -p "  Update existing secret? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${BLUE}  📝 Enter new value for $secret_name:${NC}"
            echo -e "${YELLOW}  Description: $description${NC}"
            read -s secret_value
            
            if [[ -n "$secret_value" ]]; then
                echo "$secret_value" | gcloud secrets versions add "$secret_name" --data-file=- --project=$PROJECT_ID
                echo -e "${GREEN}  ✓ Updated secret: $secret_name${NC}"
            else
                echo -e "${YELLOW}  ⚠ Skipped empty value for: $secret_name${NC}"
            fi
        else
            echo -e "${YELLOW}  ⚠ Skipped update for: $secret_name${NC}"
        fi
    else
        echo -e "${GREEN}  + Creating new secret: $secret_name${NC}"
        echo -e "${YELLOW}  Description: $description${NC}"
        
        if [[ "$is_required" == "true" ]]; then
            echo -e "${RED}  ⚠ REQUIRED SECRET - Must provide value${NC}"
        fi
        
        echo -e "${BLUE}  📝 Enter value for $secret_name (press Enter to skip):${NC}"
        read -s secret_value
        
        if [[ -n "$secret_value" ]]; then
            # Create the secret
            gcloud secrets create "$secret_name" \
                --project=$PROJECT_ID \
                --data-file=<(echo -n "$secret_value") \
                --quiet
            
            echo -e "${GREEN}  ✓ Created secret: $secret_name${NC}"
        else
            if [[ "$is_required" == "true" ]]; then
                echo -e "${RED}  ❌ ERROR: Required secret $secret_name cannot be empty${NC}"
                return 1
            else
                echo -e "${YELLOW}  ⚠ Skipped optional secret: $secret_name${NC}"
            fi
        fi
    fi
}

# Function to setup secrets by category
setup_secret_category() {
    local category_name=$1
    local -n secrets_ref=$2
    local is_required=${3:-false}
    
    echo -e "${PURPLE}📋 Setting up $category_name secrets...${NC}"
    
    for secret_name in "${!secrets_ref[@]}"; do
        description="${secrets_ref[$secret_name]}"
        create_or_update_secret "$secret_name" "$description" "$is_required"
    done
    
    echo -e "${GREEN}✅ Completed $category_name secrets setup${NC}"
    echo
}

# Main setup process
echo -e "${YELLOW}🚀 Starting comprehensive secret setup...${NC}"
echo

# Setup required secrets first
setup_secret_category "Core System (Required)" REQUIRED_SECRETS true

# Setup optional secret categories
setup_secret_category "Blockchain & RPC" BLOCKCHAIN_SECRETS false
setup_secret_category "AI & ML APIs" AI_SECRETS false
setup_secret_category "Market Data & Analytics" MARKET_SECRETS false
setup_secret_category "Social Media APIs" SOCIAL_SECRETS false

# Warning for trading secrets
echo -e "${RED}⚠️  WARNING: Trading secrets contain sensitive wallet private keys${NC}"
echo -e "${RED}   These should only be set up for production live trading mode${NC}"
echo -e "${RED}   Ensure proper security measures are in place${NC}"
echo
read -p "Setup trading/wallet secrets? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    setup_secret_category "Trading & Wallet (CRITICAL)" TRADING_SECRETS false
else
    echo -e "${YELLOW}⚠ Skipped trading secrets setup${NC}"
fi

# List all created secrets
echo -e "${BLUE}📋 Listing all secrets in project...${NC}"
gcloud secrets list --project=$PROJECT_ID --format="table(name:label='Secret Name',createTime:label='Created')"

# Generate IAM permissions for Cloud Run
echo -e "${YELLOW}🔒 Setting up IAM permissions for Cloud Run service...${NC}"
SERVICE_ACCOUNT="shyvr-rlte@$PROJECT_ID.iam.gserviceaccount.com"

echo -e "${BLUE}  📝 Granting Secret Manager access to service account: $SERVICE_ACCOUNT${NC}"

# Grant access to all secrets for the service account
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/secretmanager.secretAccessor" \
    --quiet

echo -e "${GREEN}✅ IAM permissions configured${NC}"

# Generate secret validation script
echo -e "${YELLOW}🧪 Generating secret validation utilities...${NC}"

cat > "validate_secrets.py" << 'EOF'
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
EOF

chmod +x validate_secrets.py

echo -e "${GREEN}✅ Created validate_secrets.py utility${NC}"

# Generate secret rotation script
cat > "rotate_secrets.py" << 'EOF'
#!/usr/bin/env python3
"""
Secret rotation utility for Shyvr RLTE
Handles rotating API keys and other secrets with proper backup
"""
import os
import sys
import json
import datetime
from google.cloud import secretmanager
from typing import Dict, List, Optional

class SecretRotator:
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.client = secretmanager.SecretManagerServiceClient()
    
    def backup_secret(self, secret_name: str) -> Optional[str]:
        """Create backup of current secret version"""
        try:
            name = f"projects/{self.project_id}/secrets/{secret_name}/versions/latest"
            response = self.client.access_secret_version(request={"name": name})
            current_value = response.payload.data.decode("UTF-8")
            
            # Create backup with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{secret_name}_backup_{timestamp}"
            
            # Store backup
            self.client.create_secret(
                request={
                    "parent": f"projects/{self.project_id}",
                    "secret_id": backup_name,
                    "secret": {"replication": {"automatic": {}}},
                }
            )
            
            self.client.add_secret_version(
                request={
                    "parent": f"projects/{self.project_id}/secrets/{backup_name}",
                    "payload": {"data": current_value.encode("UTF-8")},
                }
            )
            
            print(f"✅ Backed up {secret_name} as {backup_name}")
            return backup_name
            
        except Exception as e:
            print(f"❌ Failed to backup {secret_name}: {e}")
            return None
    
    def rotate_secret(self, secret_name: str, new_value: str) -> bool:
        """Rotate a secret with backup"""
        try:
            # Create backup first
            backup_name = self.backup_secret(secret_name)
            if not backup_name:
                print(f"❌ Cannot rotate {secret_name} - backup failed")
                return False
            
            # Add new version
            name = f"projects/{self.project_id}/secrets/{secret_name}"
            self.client.add_secret_version(
                request={
                    "parent": name,
                    "payload": {"data": new_value.encode("UTF-8")},
                }
            )
            
            print(f"✅ Rotated {secret_name} successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to rotate {secret_name}: {e}")
            return False
    
    def list_backups(self) -> List[str]:
        """List all backup secrets"""
        try:
            parent = f"projects/{self.project_id}"
            secrets = self.client.list_secrets(request={"parent": parent})
            
            backups = []
            for secret in secrets:
                secret_name = secret.name.split("/")[-1]
                if "_backup_" in secret_name:
                    backups.append(secret_name)
            
            return backups
            
        except Exception as e:
            print(f"❌ Failed to list backups: {e}")
            return []

def main():
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "shvyr-ai-bots")
    
    if len(sys.argv) < 2:
        print("Usage: python rotate_secrets.py <command> [args]")
        print("Commands:")
        print("  list-backups                    - List all backup secrets")
        print("  rotate <secret_name>            - Rotate a secret (prompts for new value)")
        print("  backup <secret_name>            - Create backup of a secret")
        return 1
    
    command = sys.argv[1]
    rotator = SecretRotator(project_id)
    
    if command == "list-backups":
        backups = rotator.list_backups()
        print(f"📋 Found {len(backups)} backup secrets:")
        for backup in backups:
            print(f"  • {backup}")
    
    elif command == "rotate" and len(sys.argv) == 3:
        secret_name = sys.argv[2]
        print(f"🔄 Rotating secret: {secret_name}")
        print("Enter new value:")
        new_value = input().strip()
        
        if new_value:
            rotator.rotate_secret(secret_name, new_value)
        else:
            print("❌ Empty value provided, rotation cancelled")
    
    elif command == "backup" and len(sys.argv) == 3:
        secret_name = sys.argv[2]
        rotator.backup_secret(secret_name)
    
    else:
        print("❌ Invalid command or arguments")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
EOF

chmod +x rotate_secrets.py

echo -e "${GREEN}✅ Created rotate_secrets.py utility${NC}"

echo -e "${GREEN}🎉 Secret Manager setup completed!${NC}"
echo
echo -e "${BLUE}📋 Next steps:${NC}"
echo -e "  1. Run validation: python validate_secrets.py"
echo -e "  2. Update deployment scripts with secret integration"
echo -e "  3. Test secret retrieval in development environment"
echo -e "  4. Configure monitoring for secret access patterns"
echo
echo -e "${YELLOW}💡 Utilities created:${NC}"
echo -e "  • validate_secrets.py - Validate secret accessibility"
echo -e "  • rotate_secrets.py   - Handle secret rotation with backup"
echo
echo -e "${PURPLE}🔐 Security Reminders:${NC}"
echo -e "  • Never commit actual secret values to version control"
echo -e "  • Regularly rotate API keys and access tokens"
echo -e "  • Monitor secret access patterns for anomalies"
echo -e "  • Use least-privilege IAM permissions"
echo -e "  • Keep trading secrets separate from development secrets"