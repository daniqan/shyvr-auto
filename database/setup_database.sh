#!/bin/bash
# Database setup script for RLTE Activity Logging
# This script helps set up the PostgreSQL database for the RLTE system

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${GREEN}RLTE Database Setup${NC}"
echo "========================="

# Check if PostgreSQL is running
if ! pg_isready -q; then
    echo -e "${RED}Error: PostgreSQL is not running${NC}"
    echo "Please start PostgreSQL and try again"
    exit 1
fi

# Check if Python environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo -e "${YELLOW}Warning: Virtual environment not detected${NC}"
    echo "Make sure you have the required dependencies installed"
fi

# Check for required environment variables
required_vars=("DB_PASSWORD")
for var in "${required_vars[@]}"; do
    if [[ -z "${!var}" ]]; then
        echo -e "${RED}Error: Environment variable $var is not set${NC}"
        echo "Please set all required environment variables and try again"
        exit 1
    fi
done

# Function to run database initialization
init_database() {
    echo -e "${GREEN}Initializing database...${NC}"
    cd "$PROJECT_ROOT"
    python database/init_database.py "$@"
}

# Function to test database connection
test_connection() {
    echo -e "${GREEN}Testing database connection...${NC}"
    cd "$PROJECT_ROOT"
    python database/init_database.py --test-connection
}

# Function to reset database (development only)
reset_database() {
    echo -e "${YELLOW}Warning: This will completely reset the database!${NC}"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${GREEN}Resetting database...${NC}"
        cd "$PROJECT_ROOT"
        python database/init_database.py --reset
    else
        echo "Reset cancelled"
    fi
}

# Function to run migrations only
migrate_only() {
    echo -e "${GREEN}Running migrations...${NC}"
    cd "$PROJECT_ROOT"
    python database/init_database.py --migrate-only
}

# Function to show help
show_help() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  init          Initialize database (default)"
    echo "  test          Test database connection"
    echo "  migrate       Run migrations only"
    echo "  reset         Reset database (development only)"
    echo "  help          Show this help message"
    echo ""
    echo "Environment variables required:"
    echo "  DB_PASSWORD   Database password"
    echo ""
    echo "Optional environment variables:"
    echo "  DB_HOST       Database host (default: localhost)"
    echo "  DB_PORT       Database port (default: 5432)"
    echo "  DB_NAME       Database name (default: shyvr_rlte)"
    echo "  DB_USER       Database user (default: rlte_user)"
}

# Parse command line arguments
case "${1:-init}" in
    "init")
        init_database
        ;;
    "test")
        test_connection
        ;;
    "migrate")
        migrate_only
        ;;
    "reset")
        reset_database
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

echo -e "${GREEN}Done!${NC}"