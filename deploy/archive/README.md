# Archived Deployment Scripts

This directory contains deployment scripts that have been archived during the deployment system refactoring. These scripts are kept for reference but are no longer part of the main deployment workflow.

## Archived Scripts

### `deploy_latest.sh`
- **Original Purpose**: Legacy production deployment script with comprehensive secret management
- **Why Archived**: Superseded by `automated_deployment_pipeline.sh` which provides better error handling, rollback capabilities, and pipeline structure
- **Functionality Now In**: `automated_deployment_pipeline.sh` + `deploy.sh`

### `deploy_and_configure.sh`
- **Original Purpose**: Simple deployment orchestrator that ran deployment and set up Telegram webhook
- **Why Archived**: Basic functionality moved to the unified `deploy.sh` runner
- **Functionality Now In**: `deploy.sh` (unified deployment interface)

### `final_deployment_package.sh`
- **Original Purpose**: Complex Phase 8.2 deployment package with comprehensive validation and sign-off procedures
- **Why Archived**: Overly complex for regular use; key functionality integrated into main pipeline
- **Functionality Now In**: `automated_deployment_pipeline.sh` (comprehensive pipeline) + validation scripts

## Migration Guide

If you were using any of these archived scripts, here's how to migrate:

### From `deploy_latest.sh`
```bash
# Old way
./deploy/deploy_latest.sh

# New way
./deploy/deploy.sh production
```

### From `deploy_and_configure.sh`
```bash
# Old way
./deploy/deploy_and_configure.sh

# New way - same functionality with better error handling
./deploy/deploy.sh staging
# or
./deploy/deploy.sh production
```

### From `final_deployment_package.sh`
```bash
# Old way
./deploy/final_deployment_package.sh

# New way - comprehensive pipeline with better modularity
./deploy/deploy.sh production
```

## Restore Instructions

If you need to temporarily restore any of these scripts for debugging or comparison:

1. Copy the script from `deploy/archive/` to `deploy/`
2. Make it executable: `chmod +x deploy/script_name.sh`
3. Update any path references if needed

## Cleanup

These archived scripts can be safely deleted after:
- Confirming the new deployment system works as expected
- All team members are familiar with the new workflow
- Any custom modifications have been ported to the new system

---

**Archive Date**: $(date)  
**Archived By**: Deployment System Refactoring  
**New System Version**: 2.0