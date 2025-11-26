"""
Dependency injection functions for FastAPI routes.
Separated from main.py to avoid circular imports.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables BEFORE importing services that use them
# Find the project root (parent of agent_api directory)
project_root = Path(__file__).parent.parent
env_path = project_root / ".env"

# Load .env file if it exists
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    # Fallback to default load_dotenv() behavior
    load_dotenv()

from .services.agent_factory import AgentFactory
from .services.config_manager import ConfigManager
from .services.mcp_server_manager import MCPServerManager
from .services.middleware_factory import MiddlewareFactory
from .services.tool_registry import ToolRegistry


# Application state
class AppState:
    """Application state container."""

    def __init__(self):
        self.config_manager = ConfigManager()
        self.tool_registry = ToolRegistry()
        self.middleware_factory = MiddlewareFactory()
        self.mcp_server_manager = MCPServerManager()
        self.agent_factory = AgentFactory(
            tool_registry=self.tool_registry,
            middleware_factory=self.middleware_factory,
            config_manager=self.config_manager,
            mcp_server_manager=self.mcp_server_manager,
        )


# Global app state instance
app_state = AppState()


# Dependency injection functions for routes
def get_config_manager() -> ConfigManager:
    """Get config manager instance."""
    return app_state.config_manager


def get_tool_registry() -> ToolRegistry:
    """Get tool registry instance."""
    return app_state.tool_registry


def get_middleware_factory() -> MiddlewareFactory:
    """Get middleware factory instance."""
    return app_state.middleware_factory


def get_agent_factory() -> AgentFactory:
    """Get agent factory instance."""
    return app_state.agent_factory


def get_mcp_server_manager() -> MCPServerManager:
    """Get MCP server manager instance."""
    return app_state.mcp_server_manager
