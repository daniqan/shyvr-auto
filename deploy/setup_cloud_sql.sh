#!/bin/bash

# Cloud SQL Setup for Shyvr RLTE
# Creates and configures Cloud SQL PostgreSQL instance for production deployment

set -e

# Configuration
PROJECT_ID="shvyr-ai-bots"
INSTANCE_NAME="shyvr-rlte-db"
DATABASE_NAME="shyvr_rlte"
DATABASE_USER="rlte_user"
REGION="us-central1"
ZONE="us-central1-a"
TIER="db-f1-micro"  # Can be upgraded to db-g1-small or db-n1-standard-1 for production
VERSION="POSTGRES_14"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${BLUE}🗄️  Cloud SQL Setup for Shyvr RLTE${NC}"
echo -e "${PURPLE}AI-augmented cryptocurrency trading database infrastructure${NC}"
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
    
    gcloud services enable sqladmin.googleapis.com
    gcloud services enable sql-component.googleapis.com
    gcloud services enable cloudresourcemanager.googleapis.com
    
    echo -e "${GREEN}✅ APIs enabled${NC}"
}

# Function to check if Cloud SQL instance exists
instance_exists() {
    gcloud sql instances describe "$INSTANCE_NAME" --quiet 2>/dev/null
    return $?
}

# Function to generate secure password
generate_password() {
    # Generate a secure password with special characters
    python3 -c "import secrets, string; chars = string.ascii_letters + string.digits + '!@#$%^&*'; print(''.join(secrets.choice(chars) for _ in range(32)))"
}

# Function to create Cloud SQL instance
create_instance() {
    echo -e "${YELLOW}🏗️  Creating Cloud SQL instance: $INSTANCE_NAME${NC}"
    
    # Generate secure password for postgres user
    POSTGRES_PASSWORD=$(generate_password)
    
    # Create the instance
    gcloud sql instances create "$INSTANCE_NAME" \
        --database-version="$VERSION" \
        --tier="$TIER" \
        --region="$REGION" \
        --availability-type=zonal \
        --storage-type=SSD \
        --storage-size=10GB \
        --storage-auto-increase \
        --maintenance-window-day=SUN \
        --maintenance-window-hour=3 \
        --maintenance-release-channel=production \
        --backup-start-time=04:00 \
        --enable-bin-log \
        --deletion-protection \
        --root-password="$POSTGRES_PASSWORD" \
        --quiet
    
    echo -e "${GREEN}✅ Cloud SQL instance created${NC}"
    
    # Store password in Secret Manager
    echo -e "${YELLOW}🔐 Storing postgres password in Secret Manager...${NC}"
    echo -n "$POSTGRES_PASSWORD" | gcloud secrets create postgres-password --data-file=-
    echo -e "${GREEN}✅ Password stored in Secret Manager as 'postgres-password'${NC}"
}

# Function to create database and user
setup_database() {
    echo -e "${YELLOW}📊 Setting up database and user...${NC}"
    
    # Create the application database
    gcloud sql databases create "$DATABASE_NAME" --instance="$INSTANCE_NAME"
    echo -e "${GREEN}✅ Database '$DATABASE_NAME' created${NC}"
    
    # Generate password for application user
    APP_PASSWORD=$(generate_password)
    
    # Create application user
    gcloud sql users create "$DATABASE_USER" \
        --instance="$INSTANCE_NAME" \
        --password="$APP_PASSWORD"
    
    echo -e "${GREEN}✅ User '$DATABASE_USER' created${NC}"
    
    # Store application password in Secret Manager
    echo -e "${YELLOW}🔐 Storing application password in Secret Manager...${NC}"
    echo -n "$APP_PASSWORD" | gcloud secrets create DB_PASSWORD --data-file=-
    echo -e "${GREEN}✅ Password stored in Secret Manager as 'DB_PASSWORD'${NC}"
}

# Function to configure database connection
configure_connection() {
    echo -e "${YELLOW}🔗 Configuring database connection...${NC}"
    
    # Get the connection name
    CONNECTION_NAME=$(gcloud sql instances describe "$INSTANCE_NAME" --format="value(connectionName)")
    
    # Create DATABASE_URL secret
    DATABASE_URL="postgresql://$DATABASE_USER:placeholder@/$DATABASE_NAME?host=/cloudsql/$CONNECTION_NAME"
    echo -n "$DATABASE_URL" | gcloud secrets create DATABASE_URL --data-file=-
    
    echo -e "${GREEN}✅ Connection configuration stored${NC}"
    echo -e "${BLUE}📝 Connection name: $CONNECTION_NAME${NC}"
}

# Function to run database migrations
run_migrations() {
    echo -e "${YELLOW}🔄 Running database schema initialization...${NC}"
    
    # Get the database password
    DB_PASSWORD=$(gcloud secrets versions access latest --secret="DB_PASSWORD")
    CONNECTION_NAME=$(gcloud sql instances describe "$INSTANCE_NAME" --format="value(connectionName)")
    
    # Set environment variables for database connection
    export DB_HOST="/cloudsql/$CONNECTION_NAME"
    export DB_PORT="5432"
    export DB_NAME="$DATABASE_NAME"
    export DB_USER="$DATABASE_USER"
    export DB_PASSWORD="$DB_PASSWORD"
    
    # Get script directory and project root
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
    
    echo -e "${BLUE}📁 Project root: $PROJECT_ROOT${NC}"
    
    # Check if we have Cloud SQL Proxy
    if ! command -v cloud_sql_proxy &> /dev/null; then
        echo -e "${YELLOW}⬇️  Installing Cloud SQL Auth Proxy...${NC}"
        curl -o cloud_sql_proxy https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.8.0/cloud-sql-proxy.linux.amd64
        chmod +x cloud_sql_proxy
        sudo mv cloud_sql_proxy /usr/local/bin/
    fi
    
    # Start Cloud SQL Proxy in background for migration
    echo -e "${YELLOW}🔌 Starting Cloud SQL Auth Proxy...${NC}"
    cloud_sql_proxy "$CONNECTION_NAME" --port=5433 &
    PROXY_PID=$!
    
    # Wait for proxy to be ready
    sleep 10
    
    # Update environment for proxy connection
    export DB_HOST="localhost"
    export DB_PORT="5433"
    
    # Run database initialization
    cd "$PROJECT_ROOT"
    if command -v uv &> /dev/null; then
        echo -e "${BLUE}🐍 Using uv to run database initialization...${NC}"
        uv run python database/init_database.py --verbose
    else
        echo -e "${BLUE}🐍 Using python to run database initialization...${NC}"
        python database/init_database.py --verbose
    fi
    
    # Stop Cloud SQL Proxy
    kill $PROXY_PID 2>/dev/null || true
    
    echo -e "${GREEN}✅ Database schema initialized${NC}"
}

# Function to test database connectivity
test_connectivity() {
    echo -e "${YELLOW}🔍 Testing database connectivity...${NC}"
    
    # Get connection details
    DB_PASSWORD=$(gcloud secrets versions access latest --secret="DB_PASSWORD")
    CONNECTION_NAME=$(gcloud sql instances describe "$INSTANCE_NAME" --format="value(connectionName)")
    
    # Test connection using gcloud sql connect
    echo -e "${BLUE}📡 Testing connection to Cloud SQL instance...${NC}"
    
    # Create a simple test query
    echo "SELECT version();" > /tmp/test_query.sql
    
    # Test the connection
    if gcloud sql connect "$INSTANCE_NAME" --user="$DATABASE_USER" --database="$DATABASE_NAME" < /tmp/test_query.sql; then
        echo -e "${GREEN}✅ Database connectivity test passed${NC}"
    else
        echo -e "${RED}❌ Database connectivity test failed${NC}"
        rm -f /tmp/test_query.sql
        return 1
    fi
    
    rm -f /tmp/test_query.sql
}

# Function to display summary
display_summary() {
    CONNECTION_NAME=$(gcloud sql instances describe "$INSTANCE_NAME" --format="value(connectionName)")
    IP_ADDRESS=$(gcloud sql instances describe "$INSTANCE_NAME" --format="value(ipAddresses[0].ipAddress)")
    
    echo ""
    echo -e "${PURPLE}🎉 Cloud SQL Setup Complete!${NC}"
    echo -e "${GREEN}===============================================${NC}"
    echo -e "${BLUE}Instance Name:       $INSTANCE_NAME${NC}"
    echo -e "${BLUE}Database Name:       $DATABASE_NAME${NC}"
    echo -e "${BLUE}Database User:       $DATABASE_USER${NC}"
    echo -e "${BLUE}Connection Name:     $CONNECTION_NAME${NC}"
    echo -e "${BLUE}IP Address:          $IP_ADDRESS${NC}"
    echo -e "${BLUE}Region:              $REGION${NC}"
    echo ""
    echo -e "${YELLOW}🔐 Secrets Created:${NC}"
    echo -e "   postgres-password (root user password)"
    echo -e "   DB_PASSWORD (application user password)"
    echo -e "   DATABASE_URL (full connection string)"
    echo ""
    echo -e "${YELLOW}🔗 Connection Methods:${NC}"
    echo -e "   Cloud SQL Proxy: cloud_sql_proxy $CONNECTION_NAME"
    echo -e "   Direct Connect:  gcloud sql connect $INSTANCE_NAME --user=$DATABASE_USER"
    echo ""
    echo -e "${BLUE}💡 Next Steps:${NC}"
    echo -e "   1. Update your deployment scripts to use Cloud SQL"
    echo -e "   2. Configure Cloud SQL Auth Proxy in your application"
    echo -e "   3. Test the database connection from your application"
    echo -e "   4. Monitor database performance and scale as needed"
    echo ""
    echo -e "${GREEN}🚀 Your production database is ready!${NC}"
}

# Function to show help
show_help() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  setup         Complete Cloud SQL setup (default)"
    echo "  create        Create instance only"
    echo "  migrate       Run migrations only"
    echo "  test          Test connectivity only"
    echo "  cleanup       Delete instance (development only)"
    echo "  help          Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  PROJECT_ID    Google Cloud Project ID (default: shvyr-ai-bots)"
    echo "  INSTANCE_NAME Cloud SQL instance name (default: shyvr-rlte-db)"
    echo "  DATABASE_NAME Database name (default: shyvr_rlte)"
    echo "  DATABASE_USER Database user (default: rlte_user)"
    echo "  REGION        Cloud SQL region (default: us-central1)"
    echo "  TIER          Instance tier (default: db-f1-micro)"
}

# Function to cleanup (development only)
cleanup_instance() {
    echo -e "${RED}⚠️  WARNING: This will delete the Cloud SQL instance!${NC}"
    read -p "Are you sure you want to delete $INSTANCE_NAME? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}🗑️  Deleting Cloud SQL instance...${NC}"
        
        # Remove deletion protection first
        gcloud sql instances patch "$INSTANCE_NAME" --no-deletion-protection --quiet
        
        # Delete the instance
        gcloud sql instances delete "$INSTANCE_NAME" --quiet
        
        echo -e "${GREEN}✅ Instance deleted${NC}"
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
            
            if instance_exists; then
                echo -e "${YELLOW}⚠️  Instance $INSTANCE_NAME already exists${NC}"
                echo -e "${BLUE}Skipping instance creation...${NC}"
            else
                create_instance
                setup_database
                configure_connection
            fi
            
            run_migrations
            test_connectivity
            display_summary
            ;;
        "create")
            check_gcloud_setup
            enable_apis
            create_instance
            setup_database
            configure_connection
            display_summary
            ;;
        "migrate")
            run_migrations
            ;;
        "test")
            test_connectivity
            ;;
        "cleanup")
            cleanup_instance
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

# Run main function with all arguments
main "$@"