#!/usr/bin/env python3
"""
Migration script: Extract inline MCP server configs to standalone files.

This script scans agent configs in configs/agents/ and configs/templates/
for inline MCP server definitions, extracts them to configs/mcp_servers/,
and updates the agent configs to use references.

Usage:
    python scripts/migrate_mcp_servers.py [--dry-run] [--backup-dir BACKUP_DIR]

Options:
    --dry-run       Show what would be done without making changes
    --backup-dir    Directory for config backups (default: ./configs/backup)
"""

import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def load_yaml(path: Path) -> Optional[Dict[str, Any]]:
    """Load a YAML file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"  Error loading {path}: {e}")
        return None


def save_yaml(path: Path, data: Dict[str, Any]) -> bool:
    """Save data to a YAML file."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(
                data,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False
            )
        return True
    except Exception as e:
        print(f"  Error saving {path}: {e}")
        return False


def backup_file(source: Path, backup_dir: Path) -> Optional[Path]:
    """Create a backup of a file."""
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{source.stem}_{timestamp}{source.suffix}"
    backup_path = backup_dir / backup_name

    try:
        shutil.copy2(source, backup_path)
        return backup_path
    except Exception as e:
        print(f"  Error backing up {source}: {e}")
        return None


def extract_mcp_server_to_definition(server: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an inline MCP server config to a standalone definition format."""
    definition = {
        'name': server.get('name'),
        'version': '1.0.0',
        'description': server.get('description'),
        'transport': server.get('transport', 'stdio'),
    }

    # Add optional fields
    if server.get('command'):
        definition['command'] = server['command']
    if server.get('url'):
        definition['url'] = server['url']
    if server.get('args'):
        definition['args'] = server['args']
    if server.get('env'):
        definition['env'] = server['env']

    definition['stateful'] = server.get('stateful', False)

    if server.get('selected_tools') is not None:
        definition['selected_tools'] = server['selected_tools']

    definition['tags'] = []

    return definition


def servers_are_equivalent(server1: Dict[str, Any], server2: Dict[str, Any]) -> bool:
    """Check if two server definitions are equivalent (ignoring metadata)."""
    # Compare key fields
    key_fields = ['transport', 'command', 'url', 'args', 'env', 'stateful']
    for field in key_fields:
        v1 = server1.get(field)
        v2 = server2.get(field)
        if v1 != v2:
            return False
    return True


def create_reference(server_name: str, selected_tools: Optional[List[str]] = None) -> Dict[str, Any]:
    """Create an MCP server reference."""
    ref = {'ref': server_name}
    if selected_tools is not None:
        ref['selected_tools'] = selected_tools
    return ref


def scan_configs(configs_dir: Path) -> List[Tuple[Path, Dict[str, Any]]]:
    """Scan a directory for YAML configs with inline MCP servers."""
    configs_with_mcp = []

    for yaml_file in configs_dir.glob("*.yaml"):
        config = load_yaml(yaml_file)
        if config and config.get('mcp_servers'):
            # Check if any are inline configs (not references)
            inline_servers = []
            for server in config['mcp_servers']:
                if isinstance(server, dict) and 'name' in server and 'ref' not in server:
                    inline_servers.append(server)

            if inline_servers:
                configs_with_mcp.append((yaml_file, config))

    return configs_with_mcp


def migrate(dry_run: bool = False, backup_dir: Optional[Path] = None) -> bool:
    """
    Run the migration.

    Args:
        dry_run: If True, only show what would be done
        backup_dir: Directory for backups

    Returns:
        True if successful, False otherwise
    """
    project_root = get_project_root()
    agents_dir = project_root / "configs" / "agents"
    templates_dir = project_root / "configs" / "templates"
    mcp_servers_dir = project_root / "configs" / "mcp_servers"

    if backup_dir is None:
        backup_dir = project_root / "configs" / "backup"

    print("=" * 60)
    print("MCP Server Configuration Migration")
    print("=" * 60)

    if dry_run:
        print("\n[DRY RUN MODE - No changes will be made]\n")

    # Find all configs with inline MCP servers
    print(f"\nScanning for configs with inline MCP servers...")

    all_configs = []

    if agents_dir.exists():
        print(f"  Scanning: {agents_dir}")
        all_configs.extend(scan_configs(agents_dir))

    if templates_dir.exists():
        print(f"  Scanning: {templates_dir}")
        all_configs.extend(scan_configs(templates_dir))

    if not all_configs:
        print("\nNo configs with inline MCP servers found. Nothing to migrate.")
        return True

    print(f"\nFound {len(all_configs)} config(s) with inline MCP servers:")
    for config_path, config in all_configs:
        inline_count = sum(1 for s in config['mcp_servers'] if isinstance(s, dict) and 'name' in s and 'ref' not in s)
        print(f"  - {config_path.name}: {inline_count} inline server(s)")

    # Collect all unique MCP server definitions
    mcp_servers: Dict[str, Dict[str, Any]] = {}
    conflicts: List[str] = []

    print("\nExtracting MCP server definitions...")

    for config_path, config in all_configs:
        for server in config.get('mcp_servers', []):
            if isinstance(server, dict) and 'name' in server and 'ref' not in server:
                server_name = server['name'].strip().lower().replace(' ', '_')
                definition = extract_mcp_server_to_definition(server)

                if server_name in mcp_servers:
                    # Check if they're equivalent
                    if not servers_are_equivalent(mcp_servers[server_name], definition):
                        conflicts.append(f"Server '{server_name}' has conflicting definitions")
                        print(f"  WARNING: Conflicting definition for '{server_name}' in {config_path.name}")
                    else:
                        print(f"  Duplicate (equivalent): {server_name}")
                else:
                    mcp_servers[server_name] = definition
                    print(f"  Extracted: {server_name}")

    if conflicts:
        print(f"\nWARNING: {len(conflicts)} conflict(s) found:")
        for conflict in conflicts:
            print(f"  - {conflict}")
        print("\nThe first definition encountered will be used.")

    # Create MCP server definition files
    print(f"\n{'Would create' if dry_run else 'Creating'} MCP server definition files...")

    if not dry_run:
        mcp_servers_dir.mkdir(parents=True, exist_ok=True)

    for server_name, definition in mcp_servers.items():
        server_file = mcp_servers_dir / f"{server_name}.yaml"

        if server_file.exists():
            print(f"  Skipping (already exists): {server_file.name}")
            continue

        print(f"  {'Would create' if dry_run else 'Creating'}: {server_file.name}")

        if not dry_run:
            if not save_yaml(server_file, definition):
                return False

    # Update agent configs to use references
    print(f"\n{'Would update' if dry_run else 'Updating'} agent configs to use references...")

    for config_path, config in all_configs:
        print(f"\n  Processing: {config_path.name}")

        # Create backup
        if not dry_run:
            backup_path = backup_file(config_path, backup_dir)
            if backup_path:
                print(f"    Backed up to: {backup_path.name}")
            else:
                print(f"    WARNING: Failed to create backup!")
        else:
            print(f"    Would backup to: {backup_dir}")

        # Convert inline servers to references
        new_mcp_servers = []
        for server in config.get('mcp_servers', []):
            if isinstance(server, dict) and 'name' in server and 'ref' not in server:
                server_name = server['name'].strip().lower().replace(' ', '_')
                selected_tools = server.get('selected_tools')
                ref = create_reference(server_name, selected_tools)
                new_mcp_servers.append(ref)
                print(f"    Converting '{server_name}' to reference")
            elif isinstance(server, dict) and 'ref' in server:
                # Already a reference, keep as-is
                new_mcp_servers.append(server)
                print(f"    Keeping existing reference: {server.get('ref')}")
            else:
                # Unknown format, keep as-is
                new_mcp_servers.append(server)
                print(f"    Keeping unknown format: {server}")

        # Update config
        config['mcp_servers'] = new_mcp_servers

        if not dry_run:
            if not save_yaml(config_path, config):
                return False
            print(f"    Updated: {config_path.name}")
        else:
            print(f"    Would update: {config_path.name}")

    # Summary
    print("\n" + "=" * 60)
    print("Migration Summary")
    print("=" * 60)
    print(f"  MCP server definitions extracted: {len(mcp_servers)}")
    print(f"  Agent configs updated: {len(all_configs)}")
    print(f"  Conflicts detected: {len(conflicts)}")

    if dry_run:
        print("\n[DRY RUN COMPLETE - Run without --dry-run to apply changes]")
    else:
        print("\n[MIGRATION COMPLETE]")
        print(f"\nMCP server configs saved to: {mcp_servers_dir}")
        print(f"Backups saved to: {backup_dir}")

    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate inline MCP server configs to standalone files"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )
    parser.add_argument(
        '--backup-dir',
        type=Path,
        default=None,
        help='Directory for config backups (default: ./configs/backup)'
    )

    args = parser.parse_args()

    success = migrate(
        dry_run=args.dry_run,
        backup_dir=args.backup_dir
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
