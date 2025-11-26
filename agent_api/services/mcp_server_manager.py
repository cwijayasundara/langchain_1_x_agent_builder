"""
MCP Server configuration manager.
Handles loading, saving, and validating MCP server definitions from configs/mcp_servers/.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import ValidationError

from agent_api.models.config_schema import (
    MCPServerConfig,
    MCPServerDefinition,
    MCPServerReference,
)

logger = logging.getLogger(__name__)


class MCPServerError(Exception):
    """Exception raised for MCP server configuration errors."""
    pass


class MCPServerManager:
    """Manages MCP server configurations stored in configs/mcp_servers/."""

    def __init__(self, mcp_servers_dir: str = "./configs/mcp_servers"):
        """
        Initialize the MCP server manager.

        Args:
            mcp_servers_dir: Directory for storing MCP server configurations
        """
        self.mcp_servers_dir = Path(mcp_servers_dir)
        self.mcp_servers_dir.mkdir(parents=True, exist_ok=True)

    def load_server(self, server_name: str) -> MCPServerDefinition:
        """
        Load an MCP server definition by name.

        Args:
            server_name: Name of the MCP server (without .yaml extension)

        Returns:
            MCPServerDefinition object

        Raises:
            MCPServerError: If server not found or invalid
        """
        # Normalize name
        server_name = server_name.strip().lower()
        config_path = self.mcp_servers_dir / f"{server_name}.yaml"

        if not config_path.exists():
            raise MCPServerError(f"MCP server not found: '{server_name}'")

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)

            if not config_data:
                raise MCPServerError(f"MCP server config is empty: '{server_name}'")

            # Parse and validate using Pydantic
            server = MCPServerDefinition(**config_data)
            return server

        except yaml.YAMLError as e:
            raise MCPServerError(f"Invalid YAML syntax in '{server_name}': {str(e)}")
        except ValidationError as e:
            raise MCPServerError(f"MCP server config validation failed for '{server_name}': {str(e)}")
        except Exception as e:
            raise MCPServerError(f"Error loading MCP server '{server_name}': {str(e)}")

    def save_server(self, server: MCPServerDefinition, overwrite: bool = False) -> str:
        """
        Save an MCP server definition to a YAML file.

        Args:
            server: MCPServerDefinition object to save
            overwrite: Whether to overwrite existing file

        Returns:
            Path to the saved configuration file

        Raises:
            MCPServerError: If server already exists and overwrite=False
        """
        config_path = self.mcp_servers_dir / f"{server.name}.yaml"

        if config_path.exists() and not overwrite:
            raise MCPServerError(f"MCP server '{server.name}' already exists. Use overwrite=True to replace.")

        # Convert to dictionary and save as YAML
        config_dict = server.model_dump(exclude_none=True, by_alias=True)

        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(
                config_dict,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False
            )

        logger.info(f"Saved MCP server config: {config_path}")
        return str(config_path)

    def list_servers(self) -> List[Dict[str, Any]]:
        """
        List all available MCP server configurations.

        Returns:
            List of server metadata dictionaries
        """
        servers = []

        for config_file in self.mcp_servers_dir.glob("*.yaml"):
            try:
                server = self.load_server(config_file.stem)
                stat = config_file.stat()

                servers.append({
                    "name": server.name,
                    "description": server.description,
                    "transport": server.transport,
                    "url": server.url,
                    "command": server.command,
                    "stateful": server.stateful,
                    "version": server.version,
                    "tags": server.tags,
                    "selected_tools": server.selected_tools,
                    "config_path": str(config_file),
                    "created_at": datetime.fromtimestamp(stat.st_ctime),
                    "updated_at": datetime.fromtimestamp(stat.st_mtime),
                })
            except MCPServerError as e:
                logger.warning(f"Skipping invalid MCP server config '{config_file.name}': {str(e)}")
                continue

        return servers

    def delete_server(self, server_name: str) -> bool:
        """
        Delete an MCP server configuration.

        Args:
            server_name: Name of the server to delete

        Returns:
            True if deleted, False if not found
        """
        server_name = server_name.strip().lower()
        config_path = self.mcp_servers_dir / f"{server_name}.yaml"

        if config_path.exists():
            config_path.unlink()
            logger.info(f"Deleted MCP server config: {server_name}")
            return True
        return False

    def server_exists(self, server_name: str) -> bool:
        """
        Check if an MCP server configuration exists.

        Args:
            server_name: Name of the server

        Returns:
            True if exists, False otherwise
        """
        server_name = server_name.strip().lower()
        config_path = self.mcp_servers_dir / f"{server_name}.yaml"
        return config_path.exists()

    def validate_server(self, server_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate an MCP server configuration without saving.

        Args:
            server_data: Server configuration data to validate

        Returns:
            Dictionary with validation results:
                - valid: bool
                - errors: List[Dict] (if invalid)
                - warnings: List[str]
        """
        try:
            server = MCPServerDefinition(**server_data)

            warnings = []

            # Validate transport-specific requirements
            if server.transport == "stdio":
                if not server.command:
                    warnings.append("'command' should be specified for stdio transport")
                elif server.command.endswith('.py'):
                    # Check if Python file exists (relative to project root)
                    from pathlib import Path
                    project_root = Path(__file__).parent.parent.parent
                    full_path = project_root / server.command
                    if not full_path.exists():
                        warnings.append(f"Command file not found: {server.command}")

            elif server.transport in ["http", "sse", "streamable_http"]:
                if not server.url:
                    warnings.append(f"'url' should be specified for {server.transport} transport")
                elif not server.url.startswith(("http://", "https://")):
                    warnings.append("URL should start with http:// or https://")

            return {
                "valid": True,
                "errors": [],
                "warnings": warnings
            }

        except ValidationError as e:
            errors = []
            for error in e.errors():
                errors.append({
                    "field": ".".join(str(loc) for loc in error["loc"]),
                    "message": error["msg"],
                    "type": error["type"]
                })

            return {
                "valid": False,
                "errors": errors,
                "warnings": []
            }

    def resolve_reference(
        self,
        ref: MCPServerReference,
        override_tools: bool = True
    ) -> MCPServerConfig:
        """
        Resolve an MCP server reference to a full MCPServerConfig.

        Args:
            ref: MCPServerReference with server name and optional tool override
            override_tools: Whether to apply the reference's selected_tools override

        Returns:
            MCPServerConfig with full server configuration

        Raises:
            MCPServerError: If referenced server not found
        """
        # Load the server definition
        server_def = self.load_server(ref.ref)

        # Determine which tools to use
        selected_tools = server_def.selected_tools  # Default from server definition
        if override_tools and ref.selected_tools is not None:
            # Per-agent override takes precedence
            selected_tools = ref.selected_tools
            logger.debug(f"Applied per-agent tool override for server '{ref.ref}': {selected_tools}")

        # Convert to MCPServerConfig (inline format)
        return MCPServerConfig(
            name=server_def.name,
            description=server_def.description,
            transport=server_def.transport,
            command=server_def.command,
            url=server_def.url,
            args=server_def.args,
            env=server_def.env,
            stateful=server_def.stateful,
            selected_tools=selected_tools,
        )

    def resolve_all_references(
        self,
        mcp_servers: List[Union[MCPServerConfig, MCPServerReference]]
    ) -> List[MCPServerConfig]:
        """
        Resolve all MCP server references in a list.

        Inline MCPServerConfig entries are kept as-is.
        MCPServerReference entries are resolved to MCPServerConfig.

        Args:
            mcp_servers: List of inline configs or references

        Returns:
            List of MCPServerConfig objects (all resolved)

        Raises:
            MCPServerError: If any reference cannot be resolved
        """
        resolved = []

        for server in mcp_servers:
            if isinstance(server, MCPServerReference):
                # Resolve reference
                resolved_config = self.resolve_reference(server)
                resolved.append(resolved_config)
                logger.info(f"Resolved MCP server reference: '{server.ref}'")
            elif isinstance(server, MCPServerConfig):
                # Keep inline config as-is
                resolved.append(server)
            elif isinstance(server, dict):
                # Handle dict format (from YAML parsing)
                if 'ref' in server:
                    # It's a reference
                    ref = MCPServerReference(**server)
                    resolved_config = self.resolve_reference(ref)
                    resolved.append(resolved_config)
                    logger.info(f"Resolved MCP server reference: '{server['ref']}'")
                else:
                    # It's an inline config
                    inline_config = MCPServerConfig(**server)
                    resolved.append(inline_config)
            else:
                raise MCPServerError(f"Unknown MCP server format: {type(server)}")

        return resolved

    def validate_references(
        self,
        mcp_servers: List[Union[MCPServerConfig, MCPServerReference, Dict[str, Any]]]
    ) -> List[str]:
        """
        Validate that all MCP server references exist.

        Args:
            mcp_servers: List of inline configs or references

        Returns:
            List of error messages (empty if all valid)
        """
        errors = []

        for server in mcp_servers:
            if isinstance(server, MCPServerReference):
                if not self.server_exists(server.ref):
                    errors.append(f"MCP server reference not found: '{server.ref}'")
            elif isinstance(server, dict) and 'ref' in server:
                ref_name = server['ref']
                if not self.server_exists(ref_name):
                    errors.append(f"MCP server reference not found: '{ref_name}'")

        return errors

    def export_server_to_dict(self, server: MCPServerDefinition) -> Dict[str, Any]:
        """
        Export server configuration to a dictionary.

        Args:
            server: MCPServerDefinition object

        Returns:
            Configuration as dictionary
        """
        return server.model_dump(exclude_none=True, by_alias=True)

    def get_server_info(self, server_name: str) -> Dict[str, Any]:
        """
        Get detailed information about an MCP server.

        Args:
            server_name: Name of the server

        Returns:
            Dictionary with server info including metadata

        Raises:
            MCPServerError: If server not found
        """
        server = self.load_server(server_name)
        config_path = self.mcp_servers_dir / f"{server_name}.yaml"
        stat = config_path.stat()

        return {
            "name": server.name,
            "description": server.description,
            "transport": server.transport,
            "url": server.url,
            "command": server.command,
            "args": server.args,
            "env": server.env,
            "stateful": server.stateful,
            "selected_tools": server.selected_tools,
            "version": server.version,
            "tags": server.tags,
            "config_path": str(config_path),
            "created_at": datetime.fromtimestamp(stat.st_ctime),
            "updated_at": datetime.fromtimestamp(stat.st_mtime),
        }
