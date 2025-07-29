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