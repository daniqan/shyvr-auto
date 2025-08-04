#!/bin/bash
# Comprehensive Secret Manager setup for Shyvr RLTE
# Handles all API keys and sensitive configuration systematically
# Enhanced with deploy-utils.sh integration and production-ready features

set -euo pipefail

# Source deployment utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/deploy-utils.sh"

# Configuration with defaults from deploy-utils.sh
PROJECT_ID="${DEFAULT_PROJECT_ID}"
REGION="${DEFAULT_REGION}"
DRY_RUN=false
VALIDATE_ONLY=false

# Environment file configuration
ENV_FILE=""
ENV_TEMP_FILE=""

# Track created secrets for summary
CREATED_SECRETS=()
UPDATED_SECRETS=()
SKIPPED_SECRETS=()

# Function to load environment variables from .env file
load_env_file() {
    local env_file=$1
    
    if [[ ! -f "$env_file" ]]; then
        util_log_warning "Environment file not found: $env_file"
        return 1
    fi
    
    util_log_info "Loading environment variables from: $env_file"
    
    # Create a temporary file to store processed env vars
    ENV_TEMP_FILE=$(mktemp)
    
    # Read and parse .env file, ignoring comments and empty lines
    local count=0
    while IFS= read -r line || [[ -n "$line" ]]; do
        # Skip comments and empty lines
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${line// }" ]] && continue
        
        # Parse KEY=VALUE pairs using basic pattern matching
        if [[ "$line" == *"="* ]]; then
            local key="${line%%=*}"
            local value="${line#*=}"
            
            # Clean up key (remove leading/trailing whitespace)
            key=$(echo "$key" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
            
            # Clean up value (remove quotes if present)
            value=$(echo "$value" | sed "s/^['\"]//;s/['\"]$//")
            
            # Only process valid variable names
            if [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
                echo "$key=$value" >> "$ENV_TEMP_FILE"
                ((count++))
            fi
        fi
    done < "$env_file"
    
    util_log_success "Loaded $count environment variables"
}

# Function to get value from environment file or return empty
get_env_value() {
    local key=$1
    
    if [[ -n "$ENV_TEMP_FILE" && -f "$ENV_TEMP_FILE" ]]; then
        local line
        line=$(grep "^$key=" "$ENV_TEMP_FILE" 2>/dev/null)
        if [[ -n "$line" ]]; then
            echo "${line#*=}"
            return
        fi
    fi
    echo ""
}

# Cleanup function
cleanup_env_temp() {
    if [[ -n "$ENV_TEMP_FILE" && -f "$ENV_TEMP_FILE" ]]; then
        rm -f "$ENV_TEMP_FILE"
    fi
}

# Set up cleanup trap
trap cleanup_env_temp EXIT

# Show usage information
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Comprehensive Secret Manager setup for Shyvr RLTE production deployment.

OPTIONS:
    --env-file FILE       Load secrets from environment file
    --project-id PROJECT  GCP project ID (default: $PROJECT_ID)
    --region REGION       GCP region (default: $REGION)
    --dry-run            Show what would be done without executing
    --validate-only      Only validate existing secrets
    --help, -h           Show this help message

EXAMPLES:
    # Interactive setup
    $0
    
    # Load from environment file
    $0 --env-file .env.production
    
    # Dry run to see what would be created
    $0 --env-file .env.production --dry-run
    
    # Validate existing secrets
    $0 --validate-only

EOF
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --env-file)
                ENV_FILE="$2"
                shift 2
                ;;
            --project-id)
                PROJECT_ID="$2"
                shift 2
                ;;
            --region)
                REGION="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --validate-only)
                VALIDATE_ONLY=true
                shift
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            *)
                util_log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

# Main initialization
initialize_setup() {
    util_log_header "🔐 Setting up Secret Manager for Shyvr RLTE"
    util_log_info "Comprehensive API key and sensitive configuration management"
    
    # Validate prerequisites
    if ! check_required_tools; then
        util_log_error "Prerequisites check failed"
        exit 1
    fi
    
    if ! check_gcp_auth "$PROJECT_ID"; then
        util_log_error "GCP authentication check failed"
        exit 1
    fi
    
    # Set project context
    gcloud config set project "$PROJECT_ID" --quiet
    util_log_success "Using project: $PROJECT_ID"
}

# Enable required APIs
enable_secret_manager_api() {
    util_log_info "Enabling Secret Manager API"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        util_log_info "[DRY RUN] Would enable secretmanager.googleapis.com"
        return 0
    fi
    
    if gcloud services enable secretmanager.googleapis.com --project="$PROJECT_ID" --quiet; then
        util_log_success "Secret Manager API enabled"
    else
        util_log_warning "Secret Manager API may already be enabled"
    fi
}

# Core System Secrets (Required)
REQUIRED_SECRETS=(
    "TELEGRAM_TOKEN|Bot token from @BotFather on Telegram"
    "WEBHOOK_SECRET|Random string for webhook authentication"
    "DB_PASSWORD|PostgreSQL database password"
    "DATABASE_URL|Complete PostgreSQL connection string"
)

# Blockchain & RPC Secrets
BLOCKCHAIN_SECRETS=(
    "HELIUS_API_KEY|Solana RPC provider API key"
    "BIRDEYE_API_KEY|DeFi data aggregator API key"
    "ETHERSCAN_API_KEY|Ethereum blockchain explorer API key"
    "ALCHEMY_API_KEY|General Ethereum/Polygon RPC provider"
    "ETHEREUM_API_KEY|Ethereum-specific RPC API key"
    "BASE_API_KEY|Base chain RPC API key"
    "SOLANA_RPC_URL|Custom Solana RPC endpoint URL"
    "ETHEREUM_RPC_URL|Custom Ethereum RPC endpoint URL"
)

# Trading & Wallet Secrets (Critical - Live Trading)
TRADING_SECRETS=(
    "SOLANA_PRIVATE_KEY|Solana wallet private key (base58)"
    "ETHEREUM_PRIVATE_KEY|Ethereum wallet private key (hex)"
    "HYPERLIQUID_PRIVATE_KEY|Hyperliquid exchange private key"
    "HYPERLIQUID_API_KEY|Hyperliquid exchange API key"
    "WALLET_PRIVATE_KEY|Primary wallet private key"
)

# AI & ML API Secrets
AI_SECRETS=(
    "XAI_API_KEY|xAI/Grok API key for analysis"
    "OPENAI_API_KEY|OpenAI GPT API key"
    "AGENT_API_KEY|Custom AI agent API key"
)

# Market Data & Analytics Secrets
MARKET_SECRETS=(
    "LUNARCRUSH_API_KEY|Social sentiment data API key"
    "COINGECKO_API_KEY|CoinGecko market data API key"
    "COINGECKO_PRO_API_KEY|CoinGecko Pro API key"
    "GLASSNODE_API_KEY|On-chain analytics API key"
    "MESSARI_API_KEY|Crypto research data API key"
    "JUPITER_API_KEY|Jupiter DEX aggregator API key"
)

# Social Media & External APIs
SOCIAL_SECRETS=(
    "X_BEARER_TOKEN|X (Twitter) API bearer token"
    "X_API_KEY|X (Twitter) API key"
    "X_API_SECRET|X (Twitter) API secret"
)

# Function to validate existing secrets
validate_existing_secrets() {
    util_log_info "Validating existing secrets"
    
    local all_secrets=()
    all_secrets+=("${REQUIRED_SECRETS[@]}")
    all_secrets+=("${BLOCKCHAIN_SECRETS[@]}")
    all_secrets+=("${AI_SECRETS[@]}")
    all_secrets+=("${MARKET_SECRETS[@]}")
    all_secrets+=("${SOCIAL_SECRETS[@]}")
    
    local validation_passed=true
    local total_secrets=0
    local available_secrets=0
    
    for secret_entry in "${all_secrets[@]}"; do
        local secret_name="${secret_entry%%|*}"
        ((total_secrets++))
        
        if validate_secret "$secret_name" "$PROJECT_ID"; then
            util_log_success "✓ $secret_name: Available"
            ((available_secrets++))
        else
            util_log_warning "⚠ $secret_name: Missing or inaccessible"
            validation_passed=false
        fi
    done
    
    util_log_info "Validation summary: $available_secrets/$total_secrets secrets available"
    
    if [[ "$validation_passed" == "true" ]]; then
        util_log_success "All secrets validation passed"
        return 0
    else
        util_log_warning "Some secrets are missing or inaccessible"
        return 1
    fi
}

# Function to create or update a secret
create_or_update_secret() {
    local secret_name=$1
    local description=$2
    local is_required=${3:-false}
    local secret_value=""
    
    util_log_info "Processing secret: $secret_name"
    
    # First, try to get value from environment file
    local env_value
    env_value=$(get_env_value "$secret_name")
    
    # Handle specific mappings for common secrets
    if [[ -z "$env_value" ]]; then
        case "$secret_name" in
            "WEBHOOK_SECRET")
                env_value=$(get_env_value "TELEGRAM_WEBHOOK_SECRET")
                ;;
            "AGENT_API_KEY")
                # Try multiple possible keys for agent API
                env_value=$(get_env_value "AGENT_API_KEY")
                [[ -z "$env_value" ]] && env_value=$(get_env_value "OPENAI_API_KEY")
                [[ -z "$env_value" ]] && env_value=$(get_env_value "XAI_API_KEY")
                ;;
            "WALLET_PRIVATE_KEY")
                # Try different wallet private keys
                env_value=$(get_env_value "SOLANA_PRIVATE_KEY")
                [[ -z "$env_value" ]] && env_value=$(get_env_value "ETH_PRIVATE_KEY")
                [[ -z "$env_value" ]] && env_value=$(get_env_value "ETHEREUM_PRIVATE_KEY")
                ;;
            "SOLANA_RPC_URL")
                env_value=$(get_env_value "SOLANA_RPC_URL")
                ;;
            "ETHEREUM_RPC_URL")
                env_value=$(get_env_value "ETH_RPC_URL")
                ;;
        esac
    fi
    
    # Check if secret already exists
    if gcloud secrets describe "$secret_name" --project="$PROJECT_ID" --quiet 2>/dev/null; then
        util_log_warning "Secret $secret_name already exists"
        
        if [[ -n "$env_value" && "$env_value" != "your_"* ]]; then
            util_log_info "Using value from environment file"
            secret_value="$env_value"
        else
            if [[ "$DRY_RUN" == "true" ]]; then
                util_log_info "[DRY RUN] Would prompt to update existing secret"
                return 0
            fi
            
            # Prompt for update
            read -p "  Update existing secret? (y/n): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                util_log_info "Enter new value for $secret_name:"
                util_log_info "Description: $description"
                read -s secret_value
            else
                util_log_warning "Skipped update for: $secret_name"
                SKIPPED_SECRETS+=("$secret_name")
                return 0
            fi
        fi
        
        if [[ -n "$secret_value" ]]; then
            if [[ "$DRY_RUN" == "true" ]]; then
                util_log_info "[DRY RUN] Would update secret: $secret_name"
            else
                echo "$secret_value" | gcloud secrets versions add "$secret_name" --data-file=- --project="$PROJECT_ID" --quiet
                util_log_success "Updated secret: $secret_name"
                UPDATED_SECRETS+=("$secret_name")
            fi
        else
            util_log_warning "Skipped empty value for: $secret_name"
            SKIPPED_SECRETS+=("$secret_name")
        fi
    else
        util_log_info "Creating new secret: $secret_name"
        util_log_info "Description: $description"
        
        if [[ "$is_required" == "true" ]]; then
            util_log_warning "REQUIRED SECRET - Must provide value"
        fi
        
        # Use environment value if available and not a placeholder
        if [[ -n "$env_value" && "$env_value" != "your"* ]]; then
            util_log_info "Using value from environment file"
            secret_value="$env_value"
        else
            if [[ -n "$env_value" ]]; then
                util_log_warning "Found placeholder value in env file: $env_value"
            fi
            
            if [[ "$DRY_RUN" == "true" ]]; then
                util_log_info "[DRY RUN] Would prompt for secret value"
                secret_value="placeholder-value"
            else
                util_log_info "Enter value for $secret_name (press Enter to skip):"
                read -s secret_value
            fi
        fi
        
        if [[ -n "$secret_value" ]]; then
            if [[ "$DRY_RUN" == "true" ]]; then
                util_log_info "[DRY RUN] Would create secret: $secret_name"
                CREATED_SECRETS+=("$secret_name")
            else
                # Create the secret
                if gcloud secrets create "$secret_name" \
                    --project="$PROJECT_ID" \
                    --data-file=<(echo -n "$secret_value") \
                    --quiet; then
                    util_log_success "Created secret: $secret_name"
                    CREATED_SECRETS+=("$secret_name")
                else
                    util_log_error "Failed to create secret: $secret_name"
                    return 1
                fi
            fi
        else
            if [[ "$is_required" == "true" ]]; then
                util_log_error "Required secret $secret_name cannot be empty"
                return 1
            else
                util_log_warning "Skipped optional secret: $secret_name"
                SKIPPED_SECRETS+=("$secret_name")
            fi
        fi
    fi
}

# Function to setup secrets by category
setup_secret_category() {
    local category_name=$1
    local secrets_array_name=$2
    local is_required=${3:-false}
    
    util_log_header "Setting up $category_name secrets"
    
    # Use eval to reference the array by name
    eval "local secrets=(\"\${${secrets_array_name}[@]}\")"
    
    for secret_entry in "${secrets[@]}"; do
        local secret_name="${secret_entry%%|*}"
        local description="${secret_entry##*|}"
        create_or_update_secret "$secret_name" "$description" "$is_required"
    done
    
    util_log_success "Completed $category_name secrets setup"
}

# List all created secrets
list_secrets() {
    util_log_info "Listing all secrets in project"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        util_log_info "[DRY RUN] Would list all secrets in project"
        return 0
    fi
    
    gcloud secrets list --project="$PROJECT_ID" --format="table(name:label='Secret Name',createTime:label='Created')"
}

# Generate IAM permissions for Cloud Run
setup_iam_permissions() {
    util_log_info "Setting up IAM permissions for Cloud Run service"
    
    local service_account="shyvr-rlte@$PROJECT_ID.iam.gserviceaccount.com"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        util_log_info "[DRY RUN] Would grant Secret Manager access to service account: $service_account"
        return 0
    fi
    
    util_log_info "Granting Secret Manager access to service account: $service_account"
    
    # Grant access to all secrets for the service account
    if gcloud projects add-iam-policy-binding "$PROJECT_ID" \
        --member="serviceAccount:$service_account" \
        --role="roles/secretmanager.secretAccessor" \
        --quiet; then
        util_log_success "IAM permissions configured"
    else
        util_log_warning "IAM permissions may already be configured"
    fi
}

# Main setup process
main_setup_process() {
    util_log_info "Starting comprehensive secret setup"
    
    # Setup required secrets first
    setup_secret_category "Core System (Required)" "REQUIRED_SECRETS" true
    
    # Setup optional secret categories
    setup_secret_category "Blockchain & RPC" "BLOCKCHAIN_SECRETS" false
    setup_secret_category "AI & ML APIs" "AI_SECRETS" false
    setup_secret_category "Market Data & Analytics" "MARKET_SECRETS" false
    setup_secret_category "Social Media APIs" "SOCIAL_SECRETS" false
    
    # Warning for trading secrets
    if [[ "$DRY_RUN" == "true" ]]; then
        util_log_info "[DRY RUN] Would prompt for trading/wallet secrets setup"
        setup_secret_category "Trading & Wallet (CRITICAL)" "TRADING_SECRETS" false
    else
        util_log_warning "Trading secrets contain sensitive wallet private keys"
        util_log_warning "These should only be set up for production live trading mode"
        util_log_warning "Ensure proper security measures are in place"
        echo
        read -p "Setup trading/wallet secrets? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            setup_secret_category "Trading & Wallet (CRITICAL)" "TRADING_SECRETS" false
        else
            util_log_warning "Skipped trading secrets setup"
        fi
    fi
}

# Generate setup summary
generate_setup_summary() {
    util_log_header "Secret Manager Setup Summary"
    
    echo "==========================================="
    echo "Project: $PROJECT_ID"
    echo "Region: $REGION"
    echo "Mode: $([ "$DRY_RUN" = "true" ] && echo "DRY RUN" || echo "LIVE")"
    echo "Setup Time: $(date)"
    echo ""
    echo "RESULTS:"
    echo "✓ Created: ${#CREATED_SECRETS[@]} secrets"
    echo "↻ Updated: ${#UPDATED_SECRETS[@]} secrets"
    echo "⚠ Skipped: ${#SKIPPED_SECRETS[@]} secrets"
    echo ""
    
    if [[ ${#CREATED_SECRETS[@]} -gt 0 ]]; then
        echo "Created secrets:"
        printf "  - %s\n" "${CREATED_SECRETS[@]}"
        echo ""
    fi
    
    if [[ ${#UPDATED_SECRETS[@]} -gt 0 ]]; then
        echo "Updated secrets:"
        printf "  - %s\n" "${UPDATED_SECRETS[@]}"
        echo ""
    fi
    
    echo "Next Steps:"
    echo "1. Run validation: python validate_secrets.py"
    echo "2. Update deployment scripts with secret integration"
    echo "3. Test secret retrieval in development environment"
    echo "4. Configure monitoring for secret access patterns"
    echo "==========================================="
}

# Generate validation and rotation utilities
generate_utilities() {
    util_log_info "Generating secret validation and rotation utilities"
    
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
    util_log_success "Created validate_secrets.py utility"
    
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
    util_log_success "Created rotate_secrets.py utility"
}

# Main execution function
main() {
    parse_arguments "$@"
    
    # Load environment file if specified
    if [[ -n "$ENV_FILE" ]]; then
        if load_env_file "$ENV_FILE"; then
            util_log_success "Environment file loaded - will use values when available"
            util_log_info "Secrets not found in env file will prompt for manual input"
        else
            util_log_error "Failed to load environment file: $ENV_FILE"
            exit 1
        fi
    fi
    
    # Initialize setup
    initialize_setup
    
    # Validate only mode
    if [[ "$VALIDATE_ONLY" == "true" ]]; then
        if validate_existing_secrets; then
            util_log_success "All secrets validation passed"
            exit 0
        else
            util_log_error "Secrets validation failed"
            exit 1
        fi
    fi
    
    # Enable APIs
    enable_secret_manager_api
    
    # Run main setup process
    main_setup_process
    
    # List secrets and setup permissions
    list_secrets
    setup_iam_permissions
    
    # Generate utilities (only in live mode)
    if [[ "$DRY_RUN" != "true" ]]; then
        generate_utilities
    fi
    
    # Generate summary
    generate_setup_summary
    
    util_log_success "Secret Manager setup completed!"
}

# Execute main function if script is called directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi