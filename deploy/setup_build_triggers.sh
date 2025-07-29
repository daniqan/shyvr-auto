#!/bin/bash

# Cloud Build Triggers Setup for Shyvr RLTE
# Creates automated deployment triggers for different branches and environments

set -e

# Configuration
PROJECT_ID="shvyr-ai-bots"
REPO_NAME="shyvrai-rlte"
REPO_OWNER="shyvr-ai"  # Update with actual GitHub username/org
REGION="us-central1"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔧 Setting up Cloud Build Triggers for Shyvr RLTE${NC}"
echo -e "${PURPLE}AI-augmented cryptocurrency trading system CI/CD pipeline${NC}"
echo ""

# Function to check if gcloud is authenticated and project is set
check_gcloud_setup() {
    echo -e "${YELLOW}🔐 Checking gcloud authentication...${NC}"
    
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        echo -e "${RED}❌ Error: Not authenticated with gcloud${NC}"
        echo -e "${YELLOW}Please run: gcloud auth login${NC}"
        exit 1
    fi
    
    CURRENT_PROJECT=$(gcloud config get-value project 2>/dev/null)
    if [[ "$CURRENT_PROJECT" != "$PROJECT_ID" ]]; then
        echo -e "${YELLOW}⚠️  Setting project to $PROJECT_ID${NC}"
        gcloud config set project "$PROJECT_ID"
    fi
    
    echo -e "${GREEN}✅ Authentication verified${NC}"
}

# Function to enable required APIs
enable_apis() {
    echo -e "${YELLOW}🔌 Enabling required Google Cloud APIs...${NC}"
    
    gcloud services enable cloudbuild.googleapis.com
    gcloud services enable containerregistry.googleapis.com
    gcloud services enable artifactregistry.googleapis.com
    gcloud services enable run.googleapis.com
    gcloud services enable secretmanager.googleapis.com
    gcloud services enable sourcerepo.googleapis.com
    
    echo -e "${GREEN}✅ APIs enabled${NC}"
}

# Function to create Cloud Storage buckets for build artifacts and cache
create_storage_buckets() {
    echo -e "${YELLOW}🪣 Creating Cloud Storage buckets for build artifacts and cache...${NC}"
    
    # Create build cache bucket
    if ! gsutil ls gs://$PROJECT_ID-build-cache &>/dev/null; then
        gsutil mb -l $REGION gs://$PROJECT_ID-build-cache
        echo -e "${GREEN}✅ Build cache bucket created${NC}"
    else
        echo -e "${BLUE}ℹ️  Build cache bucket already exists${NC}"
    fi
    
    # Create build artifacts bucket
    if ! gsutil ls gs://$PROJECT_ID-build-artifacts &>/dev/null; then
        gsutil mb -l $REGION gs://$PROJECT_ID-build-artifacts
        echo -e "${GREEN}✅ Build artifacts bucket created${NC}"
    else
        echo -e "${BLUE}ℹ️  Build artifacts bucket already exists${NC}"
    fi
    
    # Set lifecycle rules for cache bucket (delete after 30 days)
    cat > /tmp/cache-lifecycle.json << EOF
{
  "rule": [
    {
      "action": {"type": "Delete"},
      "condition": {"age": 30}
    }
  ]
}
EOF
    
    gsutil lifecycle set /tmp/cache-lifecycle.json gs://$PROJECT_ID-build-cache
    rm /tmp/cache-lifecycle.json
    
    echo -e "${GREEN}✅ Storage buckets configured${NC}"
}

# Function to create Artifact Registry repository
create_artifact_registry() {
    echo -e "${YELLOW}📦 Setting up Artifact Registry...${NC}"
    
    # Check if repository exists
    if ! gcloud artifacts repositories describe shyvr-ai-prod --location=$REGION &>/dev/null; then
        gcloud artifacts repositories create shyvr-ai-prod \
            --repository-format=docker \
            --location=$REGION \
            --description="Shyvr RLTE Trading System Docker Images"
        echo -e "${GREEN}✅ Artifact Registry repository created${NC}"
    else
        echo -e "${BLUE}ℹ️  Artifact Registry repository already exists${NC}"
    fi
}

# Function to connect GitHub repository to Cloud Build
connect_github_repo() {
    echo -e "${YELLOW}🔗 Connecting GitHub repository to Cloud Build...${NC}"
    
    # Check if repository is already connected
    if ! gcloud source repos list --filter="name~$REPO_NAME" --format="value(name)" | grep -q "$REPO_NAME"; then
        echo -e "${BLUE}ℹ️  Please connect your GitHub repository manually:${NC}"
        echo -e "${YELLOW}   1. Go to: https://console.cloud.google.com/cloud-build/triggers${NC}"
        echo -e "${YELLOW}   2. Click 'Connect Repository'${NC}"
        echo -e "${YELLOW}   3. Select GitHub and authenticate${NC}"
        echo -e "${YELLOW}   4. Select your repository: $REPO_OWNER/$REPO_NAME${NC}"
        echo -e "${YELLOW}   5. Click 'Connect'${NC}"
        echo ""
        read -p "Press Enter after connecting the repository..."
    else
        echo -e "${GREEN}✅ Repository already connected${NC}"
    fi
}

# Function to create main branch trigger (production deployment)
create_main_trigger() {
    echo -e "${YELLOW}🚀 Creating main branch trigger (production)...${NC}"
    
    gcloud builds triggers create github \
        --repo-name="$REPO_NAME" \
        --repo-owner="$REPO_OWNER" \
        --branch-pattern="^main$" \
        --build-config="cloudbuild.yaml" \
        --name="shyvr-rlte-main-trigger" \
        --description="Production deployment trigger for Shyvr RLTE main branch" \
        --include-logs-with-status \
        --substitutions="_SERVICE_NAME=shyvr-rlte,_REPOSITORY=shyvr-ai-prod,_REGION=$REGION,_CLOUD_SQL_INSTANCE=shyvr-rlte-db" \
        --quiet || {
            echo -e "${YELLOW}⚠️  Main trigger may already exist${NC}"
        }
    
    echo -e "${GREEN}✅ Main branch trigger configured${NC}"
}

# Function to create develop branch trigger (staging deployment)
create_develop_trigger() {
    echo -e "${YELLOW}🧪 Creating develop branch trigger (staging)...${NC}"
    
    gcloud builds triggers create github \
        --repo-name="$REPO_NAME" \
        --repo-owner="$REPO_OWNER" \
        --branch-pattern="^develop$" \
        --build-config="cloudbuild.yaml" \
        --name="shyvr-rlte-develop-trigger" \
        --description="Staging deployment trigger for Shyvr RLTE develop branch" \
        --include-logs-with-status \
        --substitutions="_SERVICE_NAME=shyvr-rlte-staging,_REPOSITORY=shyvr-ai-prod,_REGION=$REGION,_CLOUD_SQL_INSTANCE=shyvr-rlte-db" \
        --quiet || {
            echo -e "${YELLOW}⚠️  Develop trigger may already exist${NC}"
        }
    
    echo -e "${GREEN}✅ Develop branch trigger configured${NC}"
}

# Function to create feature branch trigger (testing only)
create_feature_trigger() {
    echo -e "${YELLOW}🔬 Creating feature branch trigger (testing only)...${NC}"
    
    # Create a simplified cloudbuild file for feature branches (testing only)
    cat > /tmp/cloudbuild-feature.yaml << 'EOF'
# Simplified Cloud Build for feature branches - testing only
steps:
# Git setup
- name: 'gcr.io/cloud-builders/git'
  id: 'setup-git'
  entrypoint: 'bash'
  args:
  - '-c'
  - |git config --global --add safe.directory /workspace

# Install dependencies and run tests
- name: 'python:3.12-slim'
  id: 'test-feature'
  entrypoint: 'bash'
  args:
  - '-c'
  - |
    set -e
    apt-get update && apt-get install -y build-essential curl git pkg-config libpq-dev
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    export PYTHONPATH="/workspace/src"
    export ENVIRONMENT="test"
    
    echo "🧪 Running tests for feature branch..."
    uv sync --frozen --dev
    uv run pytest tests/ --tb=short -x
    echo "✅ Feature branch tests passed!"

options:
  machineType: 'E2_STANDARD_2'
  diskSizeGb: 32
timeout: '600s'
EOF
    
    # Upload feature branch build config
    gsutil cp /tmp/cloudbuild-feature.yaml gs://$PROJECT_ID-build-cache/cloudbuild-feature.yaml
    rm /tmp/cloudbuild-feature.yaml
    
    gcloud builds triggers create github \
        --repo-name="$REPO_NAME" \
        --repo-owner="$REPO_OWNER" \
        --branch-pattern="^feature/.*" \
        --build-config="gs://$PROJECT_ID-build-cache/cloudbuild-feature.yaml" \
        --name="shyvr-rlte-feature-trigger" \
        --description="Testing trigger for Shyvr RLTE feature branches" \
        --include-logs-with-status \
        --quiet || {
            echo -e "${YELLOW}⚠️  Feature trigger may already exist${NC}"
        }
    
    echo -e "${GREEN}✅ Feature branch trigger configured${NC}"
}

# Function to create pull request trigger
create_pr_trigger() {
    echo -e "${YELLOW}🔀 Creating pull request trigger (testing)...${NC}"
    
    gcloud builds triggers create github \
        --repo-name="$REPO_NAME" \
        --repo-owner="$REPO_OWNER" \
        --pull-request-pattern="^main$" \
        --build-config="gs://$PROJECT_ID-build-cache/cloudbuild-feature.yaml" \
        --name="shyvr-rlte-pr-trigger" \
        --description="Testing trigger for Shyvr RLTE pull requests to main" \
        --include-logs-with-status \
        --comment-control="COMMENTS_ENABLED" \
        --quiet || {
            echo -e "${YELLOW}⚠️  PR trigger may already exist${NC}"
        }
    
    echo -e "${GREEN}✅ Pull request trigger configured${NC}"
}

# Function to set up IAM permissions for Cloud Build
setup_iam_permissions() {
    echo -e "${YELLOW}🔐 Setting up IAM permissions for Cloud Build...${NC}"
    
    # Get Cloud Build service account
    BUILD_SA=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")@cloudbuild.gserviceaccount.com
    
    # Grant necessary permissions
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:$BUILD_SA" \
        --role="roles/run.admin" \
        --quiet
    
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:$BUILD_SA" \
        --role="roles/cloudsql.client" \
        --quiet
    
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:$BUILD_SA" \
        --role="roles/secretmanager.secretAccessor" \
        --quiet
    
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:$BUILD_SA" \
        --role="roles/storage.admin" \
        --quiet
    
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:$BUILD_SA" \
        --role="roles/artifactregistry.writer" \
        --quiet
    
    echo -e "${GREEN}✅ IAM permissions configured${NC}"
}

# Function to test triggers
test_triggers() {
    echo -e "${YELLOW}🧪 Testing trigger configuration...${NC}"
    
    # List all triggers
    echo -e "${BLUE}📋 Configured triggers:${NC}"
    gcloud builds triggers list --format="table(name,github.name,github.branch,github.pullRequest.branch,status)"
    
    echo -e "${GREEN}✅ Trigger configuration complete${NC}"
}

# Function to display summary
display_summary() {
    echo ""
    echo -e "${PURPLE}🎉 Cloud Build Triggers Setup Complete!${NC}"
    echo -e "${GREEN}===============================================${NC}"
    echo -e "${BLUE}Project ID:          $PROJECT_ID${NC}"
    echo -e "${BLUE}Repository:          $REPO_OWNER/$REPO_NAME${NC}"
    echo -e "${BLUE}Region:              $REGION${NC}"
    echo ""
    echo -e "${YELLOW}🚀 Configured Triggers:${NC}"
    echo -e "${GREEN}   ✓ main branch → Production deployment (shyvr-rlte)${NC}"
    echo -e "${GREEN}   ✓ develop branch → Staging deployment (shyvr-rlte-staging)${NC}"
    echo -e "${GREEN}   ✓ feature/* branches → Testing only${NC}"
    echo -e "${GREEN}   ✓ Pull requests → Testing only${NC}"
    echo ""
    echo -e "${YELLOW}📦 Infrastructure Created:${NC}"
    echo -e "   ✓ Artifact Registry repository (shyvr-ai-prod)"
    echo -e "   ✓ Build cache storage bucket"
    echo -e "   ✓ Build artifacts storage bucket"
    echo -e "   ✓ IAM permissions for Cloud Build"
    echo ""
    echo -e "${BLUE}💡 Next Steps:${NC}"
    echo -e "   1. Commit and push your cloudbuild.yaml file"
    echo -e "   2. Create a pull request to test the PR trigger"
    echo -e "   3. Merge to develop to test staging deployment"
    echo -e "   4. Merge to main for production deployment"
    echo ""
    echo -e "${BLUE}📊 Monitor builds at:${NC}"
    echo -e "   https://console.cloud.google.com/cloud-build/builds"
    echo ""
    echo -e "${GREEN}🚀 Your CI/CD pipeline is ready!${NC}"
}

# Function to show help
show_help() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  setup         Complete trigger setup (default)"
    echo "  connect       Connect GitHub repository only"
    echo "  triggers      Create triggers only"
    echo "  test          Test trigger configuration"
    echo "  cleanup       Delete all triggers (development only)"
    echo "  help          Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  PROJECT_ID    Google Cloud Project ID (default: shvyr-ai-bots)"
    echo "  REPO_NAME     Repository name (default: shyvrai-rlte)"
    echo "  REPO_OWNER    Repository owner/organization"
    echo "  REGION        Cloud Build region (default: us-central1)"
}

# Function to cleanup triggers (development only)
cleanup_triggers() {
    echo -e "${RED}⚠️  WARNING: This will delete all Cloud Build triggers!${NC}"
    read -p "Are you sure you want to delete all triggers for $REPO_NAME? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}🗑️  Deleting Cloud Build triggers...${NC}"
        
        # Delete all triggers for this repo
        for trigger in $(gcloud builds triggers list --filter="github.name=$REPO_NAME" --format="value(name)"); do
            gcloud builds triggers delete "$trigger" --quiet
            echo -e "${GREEN}✅ Deleted trigger: $trigger${NC}"
        done
        
        echo -e "${GREEN}✅ All triggers deleted${NC}"
    else
        echo -e "${BLUE}Cleanup cancelled${NC}"
    fi
}

# Main execution
main() {
    case "${1:-setup}" in
        "setup")
            check_gcloud_setup
            enable_apis
            create_storage_buckets
            create_artifact_registry
            connect_github_repo
            setup_iam_permissions
            create_main_trigger
            create_develop_trigger
            create_feature_trigger
            create_pr_trigger
            test_triggers
            display_summary
            ;;
        "connect")
            check_gcloud_setup
            connect_github_repo
            ;;
        "triggers")
            check_gcloud_setup
            create_main_trigger
            create_develop_trigger
            create_feature_trigger
            create_pr_trigger
            test_triggers
            ;;
        "test")
            test_triggers
            ;;
        "cleanup")
            cleanup_triggers
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            echo -e "${RED}Error: Unknown option '$1'${NC}"
            show_help
            exit 1
            ;;
    esac
}

# Check if required variables are set
if [[ -z "$REPO_OWNER" ]]; then
    echo -e "${RED}❌ Error: REPO_OWNER environment variable not set${NC}"
    echo -e "${YELLOW}Please set: export REPO_OWNER=your-github-username${NC}"
    exit 1
fi

# Run main function with all arguments
main "$@"