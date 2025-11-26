"""
Configuration manager for loading, validating, and saving agent configurations.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import ValidationError
from langchain_core.tools import BaseTool

from agent_api.models.config_schema import AgentConfig
from agent_api.services.prompt_helper import PromptHelper

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Exception raised for configuration errors."""
    pass


class ConfigManager:
    """Manages agent configurations."""

    def __init__(self, configs_dir: str = "./configs/agents"):
        """
        Initialize the configuration manager.

        Args:
            configs_dir: Directory for storing agent configurations
        """
        self.configs_dir = Path(configs_dir)
        self.configs_dir.mkdir(parents=True, exist_ok=True)

    def load_config(self, config_path: str) -> AgentConfig:
        """
        Load and validate an agent configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration file

        Returns:
            Validated AgentConfig object

        Raises:
            ConfigurationError: If config is invalid or file not found
        """
        try:
            path = Path(config_path)
            if not path.exists():
                raise ConfigurationError(f"Configuration file not found: {config_path}")

            with open(path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)

            if not config_data:
                raise ConfigurationError("Configuration file is empty")

            # Parse and validate using Pydantic
            config = AgentConfig(**config_data)
            return config

        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML syntax: {str(e)}")
        except ValidationError as e:
            raise ConfigurationError(f"Configuration validation failed: {str(e)}")
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration: {str(e)}")

    def load_config_from_dict(self, config_data: Dict[str, Any]) -> AgentConfig:
        """
        Load and validate an agent configuration from a dictionary.

        Args:
            config_data: Configuration data as dictionary

        Returns:
            Validated AgentConfig object

        Raises:
            ConfigurationError: If config is invalid
        """
        try:
            config = AgentConfig(**config_data)
            return config
        except ValidationError as e:
            raise ConfigurationError(f"Configuration validation failed: {str(e)}")

    def save_config(self, config: AgentConfig, config_path: Optional[str] = None) -> str:
        """
        Save an agent configuration to a YAML file.

        Args:
            config: AgentConfig object to save
            config_path: Optional specific path. If not provided, uses default location

        Returns:
            Path to the saved configuration file
        """
        if config_path is None:
            # Generate filename from agent name
            filename = f"{config.name}.yaml"
            config_path = self.configs_dir / filename
        else:
            config_path = Path(config_path)

        # Ensure directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dictionary and save as YAML
        config_dict = config.model_dump(exclude_none=True, by_alias=True)

        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(
                config_dict,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False
            )

        return str(config_path)

    def list_configs(self) -> List[Dict[str, Any]]:
        """
        List all available agent configurations.

        Returns:
            List of configuration metadata
        """
        configs = []

        for config_file in self.configs_dir.glob("*.yaml"):
            try:
                config = self.load_config(str(config_file))
                stat = config_file.stat()

                configs.append({
                    "agent_id": config.name,
                    "name": config.name,
                    "version": config.version,
                    "description": config.description,
                    "tags": config.tags,
                    "config_path": str(config_file),
                    "created_at": datetime.fromtimestamp(stat.st_ctime),
                    "updated_at": datetime.fromtimestamp(stat.st_mtime),
                    "has_mcp_servers": bool(config.mcp_servers),
                })
            except ConfigurationError:
                # Skip invalid configs
                continue

        return configs

    def get_config_path(self, agent_name: str) -> Optional[str]:
        """
        Get the path to an agent's configuration file.

        Args:
            agent_name: Name of the agent

        Returns:
            Path to config file or None if not found
        """
        config_path = self.configs_dir / f"{agent_name}.yaml"
        return str(config_path) if config_path.exists() else None

    def delete_config(self, agent_name: str) -> bool:
        """
        Delete an agent configuration.

        Args:
            agent_name: Name of the agent

        Returns:
            True if deleted, False if not found
        """
        config_path = self.configs_dir / f"{agent_name}.yaml"
        if config_path.exists():
            config_path.unlink()
            return True
        return False

    def _validate_mcp_servers(self, mcp_servers: List[Any]) -> List[str]:
        """
        Validate MCP server configurations.

        Args:
            mcp_servers: List of MCP server configurations

        Returns:
            List of validation warnings/errors
        """
        errors = []
        server_names = set()

        for i, server in enumerate(mcp_servers):
            # Check for duplicate names
            if server.name in server_names:
                errors.append(f"Duplicate MCP server name: '{server.name}'")
            server_names.add(server.name)

            # Validate stdio transport
            if server.transport == "stdio":
                if not server.command:
                    errors.append(f"MCP server '{server.name}': 'command' is required for stdio transport")
                else:
                    # Check if command is a file path and validate it exists
                    import os
                    from pathlib import Path
                    if server.command.endswith('.py'):
                        # Check if it's a Python file path
                        if not os.path.isabs(server.command):
                            # Relative path - check from project root
                            project_root = Path(__file__).parent.parent.parent
                            full_path = project_root / server.command
                            if not full_path.exists():
                                errors.append(f"MCP server '{server.name}': command file not found: {server.command}")

            # Validate http/sse transports
            elif server.transport in ["http", "sse", "streamable_http"]:
                if not server.url:
                    errors.append(f"MCP server '{server.name}': 'url' is required for {server.transport} transport")
                elif not server.url.startswith(("http://", "https://")):
                    errors.append(f"MCP server '{server.name}': url must start with http:// or https://")

            # Unknown transport
            elif server.transport not in ["stdio", "http", "sse", "streamable_http"]:
                errors.append(f"MCP server '{server.name}': unsupported transport '{server.transport}'")

        return errors

    def validate_config_dict(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a configuration dictionary without saving.

        Args:
            config_data: Configuration data to validate

        Returns:
            Dictionary with validation results:
                - valid: bool
                - errors: List[Dict] (if invalid)
                - warnings: List[str]
        """
        try:
            # Attempt to parse with Pydantic
            config = AgentConfig(**config_data)

            # Collect warnings
            warnings = []

            # Check for recommended settings
            if not config.memory:
                warnings.append("No memory configuration provided. Agent won't persist state.")

            if not config.tools:
                warnings.append("No tools configured. Agent may have limited capabilities.")

            if config.llm.temperature > 1.0:
                warnings.append(f"High temperature ({config.llm.temperature}) may produce unpredictable outputs.")

            # MCP server validation
            if config.mcp_servers:
                mcp_errors = self._validate_mcp_servers(config.mcp_servers)
                if mcp_errors:
                    # Convert MCP errors to warnings for now (could be errors in strict mode)
                    warnings.extend(mcp_errors)

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

    def interpolate_variables(
        self,
        text: str,
        variables: Dict[str, Any]
    ) -> str:
        """
        Interpolate variables in text (e.g., {{variable_name}}).

        Args:
            text: Text with {{variable}} placeholders
            variables: Dictionary of variable values

        Returns:
            Text with variables replaced
        """
        result = text
        for key, value in variables.items():
            placeholder = f"{{{{{key}}}}}"
            result = result.replace(placeholder, str(value))
        return result

    def get_system_prompt(
        self,
        config: AgentConfig,
        variables: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Get the system prompt with variables interpolated.

        Args:
            config: Agent configuration
            variables: Optional variables to interpolate

        Returns:
            Processed system prompt
        """
        prompt = config.prompts.system

        if variables:
            prompt = self.interpolate_variables(prompt, variables)

        # Add default variables
        default_vars = {
            "agent_name": config.name,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M:%S"),
        }
        prompt = self.interpolate_variables(prompt, default_vars)

        return prompt

    def generate_tool_documentation(
        self,
        tools: List[BaseTool],
        tool_registry,
        include_examples: bool = True,
        emphasize_accuracy: bool = True
    ) -> str:
        """
        Generate dynamic tool documentation for system prompts.

        This method can be used to automatically generate tool documentation
        instead of manually writing it in the system prompt.

        Args:
            tools: List of tool instances to document
            tool_registry: ToolRegistry instance for categorization
            include_examples: Whether to include usage examples
            emphasize_accuracy: Whether to emphasize accuracy in tool selection

        Returns:
            Formatted tool documentation string

        Example usage in AgentFactory:
            tool_docs = config_manager.generate_tool_documentation(
                tools=all_tools,
                tool_registry=self.tool_registry
            )
            # Can be appended to system prompt or used standalone
        """
        return PromptHelper.generate_complete_tool_section(
            tools=tools,
            tool_registry=tool_registry,
            include_examples=include_examples,
            emphasize_accuracy=emphasize_accuracy
        )

    def export_config_to_dict(self, config: AgentConfig) -> Dict[str, Any]:
        """
        Export configuration to a dictionary (for API responses).

        Args:
            config: AgentConfig object

        Returns:
            Configuration as dictionary
        """
        return config.model_dump(exclude_none=True, by_alias=True)

    def load_template(self, template_name: str) -> AgentConfig:
        """
        Load a configuration template.

        Args:
            template_name: Name of the template

        Returns:
            AgentConfig from template

        Raises:
            ConfigurationError: If template not found
        """
        template_path = Path("./configs/templates") / f"{template_name}.yaml"
        if not template_path.exists():
            raise ConfigurationError(f"Template not found: {template_name}")

        return self.load_config(str(template_path))

    def merge_partial_config(
        self,
        base_config: AgentConfig,
        llm: Optional[Any] = None,
        tools: Optional[List[str]] = None,
        mcp_servers: Optional[List[Any]] = None,
        preserve_middleware: bool = True,
        preserve_memory: bool = True
    ) -> AgentConfig:
        """
        Merge partial configuration updates into base config.

        This enables selective reconfiguration of agents while preserving
        middleware and memory configuration.

        Args:
            base_config: Current agent configuration
            llm: New LLM configuration (optional)
            tools: New tools list (optional)
            mcp_servers: New MCP servers list (optional)
            preserve_middleware: Keep middleware config (default True)
            preserve_memory: Keep memory config (default True)

        Returns:
            Merged AgentConfig with updates applied

        Raises:
            ConfigurationError: If merged config is invalid
        """
        logger.info(f"Merging partial config for agent '{base_config.name}'")

        # Convert base config to dict for manipulation
        config_dict = self.export_config_to_dict(base_config)

        # Track changes for logging
        changes = []

        # Update LLM if provided
        if llm is not None:
            old_llm = f"{config_dict.get('llm', {}).get('provider')}/{config_dict.get('llm', {}).get('model')}"
            new_llm = f"{llm.get('provider') if isinstance(llm, dict) else llm.provider}/{llm.get('model') if isinstance(llm, dict) else llm.model}"
            config_dict['llm'] = llm.model_dump() if hasattr(llm, 'model_dump') else llm
            changes.append(f"LLM: {old_llm} → {new_llm}")
            logger.info(f"Updated LLM configuration: {old_llm} → {new_llm}")

        # Update tools if provided
        if tools is not None:
            old_count = len(config_dict.get('tools', []))
            config_dict['tools'] = tools
            new_count = len(tools)
            changes.append(f"Tools: {old_count} → {new_count} tools")
            logger.info(f"Updated tools: {old_count} → {new_count} tools")

        # Update MCP servers if provided
        if mcp_servers is not None:
            old_count = len(config_dict.get('mcp_servers', []))
            # Convert to dicts if they're Pydantic models
            if mcp_servers and hasattr(mcp_servers[0], 'model_dump'):
                config_dict['mcp_servers'] = [s.model_dump() for s in mcp_servers]
            else:
                config_dict['mcp_servers'] = mcp_servers
            new_count = len(mcp_servers)
            changes.append(f"MCP Servers: {old_count} → {new_count} servers")
            logger.info(f"Updated MCP servers: {old_count} → {new_count} servers")

        # Preserve middleware if requested (default)
        if preserve_middleware:
            if 'middleware' not in config_dict or not config_dict['middleware']:
                logger.warning("No middleware in base config to preserve")
            else:
                mw_count = len(config_dict['middleware'])
                changes.append(f"Middleware: preserved ({mw_count} items)")
                logger.info(f"Preserved middleware configuration ({mw_count} items)")

        # Preserve memory if requested (default)
        if preserve_memory:
            if 'memory' not in config_dict or not config_dict['memory']:
                logger.warning("No memory config in base config to preserve")
            else:
                changes.append("Memory: preserved")
                logger.info("Preserved memory configuration")

        # Validate merged configuration
        logger.info(f"Validating merged configuration with changes: {', '.join(changes)}")
        validation_result = self.validate_config_dict(config_dict)

        if not validation_result['valid']:
            error_messages = [err['message'] for err in validation_result['errors']]
            raise ConfigurationError(
                f"Merged configuration is invalid: {'; '.join(error_messages)}"
            )

        # Convert back to AgentConfig
        try:
            merged_config = self.load_config_from_dict(config_dict)
            logger.info(f"Successfully merged configuration for agent '{base_config.name}'")
            return merged_config
        except Exception as e:
            raise ConfigurationError(f"Failed to create merged config: {str(e)}")

    def list_templates(self) -> List[Dict[str, Any]]:
        """
        List all available configuration templates.

        Returns:
            List of template metadata
        """
        templates_dir = Path("./configs/templates")
        templates_dir.mkdir(parents=True, exist_ok=True)

        templates = []

        for template_file in templates_dir.glob("*.yaml"):
            try:
                config = self.load_config(str(template_file))

                templates.append({
                    "template_id": template_file.stem,
                    "name": config.name,
                    "description": config.description,
                    "tags": config.tags,
                    "category": "general",  # Could be extended
                })
            except ConfigurationError as e:
                logger.warning(f"Skipping invalid template '{template_file.name}': {str(e)}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error loading template '{template_file.name}': {str(e)}")
                continue

        return templates
