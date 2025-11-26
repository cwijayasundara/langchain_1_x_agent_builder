"""
Agent factory for creating agents from configurations using LangChain's create_agent.
"""

import asyncio
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.sqlite import SqliteStore

from agent_api.models.config_schema import AgentConfig, MCPServerConfig, MCPServerReference
from agent_api.services.config_manager import ConfigManager
from agent_api.services.middleware_factory import MiddlewareFactory
from agent_api.services.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class AgentFactoryError(Exception):
    """Exception raised for agent factory errors."""
    pass


class AgentFactory:
    """Factory for creating LangChain agents from configurations."""

    def __init__(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        middleware_factory: Optional[MiddlewareFactory] = None,
        config_manager: Optional[ConfigManager] = None,
        mcp_server_manager: Optional["MCPServerManager"] = None
    ):
        """
        Initialize the agent factory.

        Args:
            tool_registry: Tool registry instance
            middleware_factory: Middleware factory instance
            config_manager: Configuration manager instance
            mcp_server_manager: MCP server manager instance for resolving references
        """
        self.tool_registry = tool_registry or ToolRegistry()
        self.middleware_factory = middleware_factory or MiddlewareFactory()
        self.config_manager = config_manager or ConfigManager()
        self.mcp_server_manager = mcp_server_manager  # May be None if not using references
        self._agents: Dict[str, Any] = {}  # Cache of created agents

        # Session override cache: Key = "{agent_name}:{thread_id}"
        # Value = {"override": RuntimeOverride, "variant_agent": agent, "system_prompt": str, "created_at": datetime}
        self._session_overrides: Dict[str, Dict[str, Any]] = {}

        # LLM instance cache for reuse: Key = "{provider}:{model}:{temperature}"
        self._llm_cache: Dict[str, BaseChatModel] = {}

    def _map_provider_name(self, provider: str) -> str:
        """
        Map provider names from config to LangChain's expected provider names.

        Args:
            provider: Provider name from config

        Returns:
            Mapped provider name for LangChain
        """
        # Map common provider names to LangChain's expected names
        provider_mapping = {
            "google": "google_genai",  # Map 'google' to 'google_genai'
        }
        return provider_mapping.get(provider.lower(), provider.lower())

    def create_llm(self, config: AgentConfig) -> BaseChatModel:
        """
        Create LLM instance from configuration.

        Args:
            config: Agent configuration

        Returns:
            Initialized chat model

        Raises:
            AgentFactoryError: If LLM creation fails
        """
        llm_config = config.llm

        try:
            # Special handling for OpenRouter - uses OpenAI-compatible API
            if llm_config.provider == "openrouter":
                from langchain_openai import ChatOpenAI

                api_key = llm_config.api_key or os.getenv("OPENROUTER_API_KEY")
                if not api_key:
                    raise AgentFactoryError("OpenRouter API key not provided. Set OPENROUTER_API_KEY environment variable or provide api_key in config.")

                return ChatOpenAI(
                    model=llm_config.model,  # e.g., "openai/gpt-4o", "anthropic/claude-3.5-sonnet"
                    openai_api_key=api_key,
                    openai_api_base="https://openrouter.ai/api/v1",
                    temperature=llm_config.temperature,
                    max_tokens=llm_config.max_tokens,
                    default_headers={
                        "HTTP-Referer": "https://langchain-agent-builder.local",
                        "X-Title": "LangChain Agent Builder"
                    }
                )

            from langchain.chat_models import init_chat_model

            # Build model parameters
            model_params = {
                "temperature": llm_config.temperature,
            }

            if llm_config.max_tokens:
                model_params["max_tokens"] = llm_config.max_tokens

            if llm_config.top_p is not None:
                model_params["top_p"] = llm_config.top_p

            # Add API key if provided (otherwise uses environment variables)
            if llm_config.api_key:
                model_params["api_key"] = llm_config.api_key

            # Map provider name to LangChain's expected format
            mapped_provider = self._map_provider_name(llm_config.provider)

            # Initialize chat model using LangChain's unified init_chat_model
            # This supports openai, anthropic, google_genai, and groq providers
            model = init_chat_model(
                model=llm_config.model,
                model_provider=mapped_provider,
                **model_params
            )

            return model

        except Exception as e:
            raise AgentFactoryError(f"Failed to create LLM: {str(e)}")

    def create_checkpointer(self, config: AgentConfig) -> Optional[SqliteSaver]:
        """
        Create checkpointer for short-term memory.

        Args:
            config: Agent configuration

        Returns:
            Checkpointer instance or None
        """
        if not config.memory or not config.memory.short_term:
            return None

        mem_config = config.memory.short_term

        if mem_config.type == "in_memory":
            # Note: InMemorySaver would be imported from langgraph.checkpoint.memory
            # For now, return None for in_memory (will use default)
            return None
        elif mem_config.type == "sqlite":
            if not mem_config.path:
                raise AgentFactoryError("SQLite checkpointer requires a path")

            # Ensure directory exists
            Path(mem_config.path).parent.mkdir(parents=True, exist_ok=True)

            # Create SQLite connection and SqliteSaver instance directly
            # SqliteSaver.from_conn_string() returns a context manager, so we use SqliteSaver() directly
            conn = sqlite3.connect(mem_config.path, check_same_thread=False)
            return SqliteSaver(conn)

        return None

    def create_store(self, config: AgentConfig):
        """
        Create store for long-term memory.

        Args:
            config: Agent configuration

        Returns:
            Store instance or None
        """
        if not config.memory or not config.memory.long_term:
            return None

        mem_config = config.memory.long_term

        if mem_config.type == "in_memory":
            return InMemoryStore()
        elif mem_config.type == "sqlite":
            if not mem_config.path:
                raise AgentFactoryError("SQLite store requires a path")

            # Ensure directory exists
            Path(mem_config.path).parent.mkdir(parents=True, exist_ok=True)

            return SqliteStore.from_conn_string(mem_config.path)

        return None

    def get_system_prompt(self, config: AgentConfig, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Get system prompt with variable interpolation.

        Args:
            config: Agent configuration
            context: Optional runtime context for interpolation

        Returns:
            Processed system prompt
        """
        variables = context or {}
        return self.config_manager.get_system_prompt(config, variables)

    def create_state_schema(self, config: AgentConfig) -> Optional[type]:
        """
        Create custom state schema if defined in config.

        Args:
            config: Agent configuration

        Returns:
            State schema class or None
        """
        if not config.memory or not config.memory.short_term:
            return None

        custom_state = config.memory.short_term.custom_state
        if not custom_state:
            return None

        # Dynamically create a dataclass for custom state
        from dataclasses import dataclass, field
        from typing import get_type_hints

        # Build field definitions
        annotations = {}
        defaults = {}

        for field_name, field_type_str in custom_state.items():
            # Parse type string (simplified - only handles basic types)
            type_map = {
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "dict": dict,
                "list": list,
                "Any": Any,
            }
            field_type = type_map.get(field_type_str, str)
            annotations[field_name] = field_type
            defaults[field_name] = field(default_factory=dict if field_type == dict else list if field_type == list else lambda: None)

        # Create custom state class
        # Note: This is a simplified version. In production, you'd want more robust type parsing
        return None  # For now, return None - will implement properly later

    def _resolve_mcp_servers(self, mcp_servers: List) -> List[MCPServerConfig]:
        """
        Resolve MCP server references to full MCPServerConfig objects.

        Args:
            mcp_servers: List of MCPServerConfig or MCPServerReference objects

        Returns:
            List of resolved MCPServerConfig objects

        Raises:
            AgentFactoryError: If reference resolution fails
        """
        if not mcp_servers:
            return []

        resolved = []
        for server in mcp_servers:
            if isinstance(server, MCPServerReference):
                # Resolve reference using mcp_server_manager
                if not self.mcp_server_manager:
                    raise AgentFactoryError(
                        f"Cannot resolve MCP server reference '{server.ref}': "
                        "MCPServerManager not configured"
                    )
                try:
                    resolved_config = self.mcp_server_manager.resolve_reference(server)
                    resolved.append(resolved_config)
                    logger.info(f"Resolved MCP server reference: '{server.ref}'")
                except Exception as e:
                    raise AgentFactoryError(
                        f"Failed to resolve MCP server reference '{server.ref}': {str(e)}"
                    )
            elif isinstance(server, MCPServerConfig):
                # Already a full config, keep as-is
                resolved.append(server)
            elif isinstance(server, dict):
                # Handle dict format (from YAML parsing)
                if 'ref' in server:
                    # It's a reference
                    if not self.mcp_server_manager:
                        raise AgentFactoryError(
                            f"Cannot resolve MCP server reference '{server['ref']}': "
                            "MCPServerManager not configured"
                        )
                    try:
                        ref = MCPServerReference(**server)
                        resolved_config = self.mcp_server_manager.resolve_reference(ref)
                        resolved.append(resolved_config)
                        logger.info(f"Resolved MCP server reference: '{server['ref']}'")
                    except Exception as e:
                        raise AgentFactoryError(
                            f"Failed to resolve MCP server reference '{server['ref']}': {str(e)}"
                        )
                else:
                    # It's an inline config
                    resolved.append(MCPServerConfig(**server))
            else:
                raise AgentFactoryError(f"Unknown MCP server format: {type(server)}")

        return resolved

    async def _create_mcp_tools_async(self, config: AgentConfig) -> tuple[Optional[Any], List[BaseTool]]:
        """
        Create tools from MCP servers (async version).

        Args:
            config: Agent configuration

        Returns:
            Tuple of (MCP client instance, list of tools from MCP servers)
            Client will be None if no MCP servers configured

        Raises:
            AgentFactoryError: If MCP tool creation fails
        """
        if not config.mcp_servers:
            return None, []

        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient

            # Resolve any MCP server references to full configs
            resolved_servers = self._resolve_mcp_servers(config.mcp_servers)

            if not resolved_servers:
                return None, []

            # Build connections dict from resolved configs
            connections = {}
            for server_config in resolved_servers:
                server_name = server_config.name

                if server_config.transport == "stdio":
                    # Get absolute path to the command if it's a relative path
                    command = server_config.command
                    if command and not os.path.isabs(command):
                        # Resolve relative to project root
                        project_root = Path(__file__).parent.parent.parent
                        command_path = project_root / command
                        if command_path.exists():
                            command = str(command_path)

                    connections[server_name] = {
                        "command": command,
                        "args": server_config.args or [],
                        "env": server_config.env or {},
                        "transport": "stdio",
                    }
                    logger.debug(f"Configured stdio MCP server '{server_name}': command={command}")

                elif server_config.transport == "streamable_http":
                    connections[server_name] = {
                        "url": server_config.url,
                        "transport": "streamable_http",
                    }
                    logger.debug(f"Configured HTTP MCP server '{server_name}': url={server_config.url}")

                elif server_config.transport == "http":
                    # Map "http" to "streamable_http" for compatibility
                    connections[server_name] = {
                        "url": server_config.url,
                        "transport": "streamable_http",
                    }
                    logger.debug(f"Configured HTTP MCP server '{server_name}': url={server_config.url}")

                elif server_config.transport == "sse":
                    connections[server_name] = {
                        "url": server_config.url,
                        "transport": "sse",
                    }
                    logger.debug(f"Configured SSE MCP server '{server_name}': url={server_config.url}")

            if not connections:
                logger.warning("No valid MCP server connections configured")
                return []

            # Create client and load tools
            logger.info(f"Connecting to {len(connections)} MCP server(s): {list(connections.keys())}")
            client = MultiServerMCPClient(connections=connections)

            # Get tools from MCP servers
            all_mcp_tools = await client.get_tools()

            # Filter tools based on selected_tools configuration
            # langchain-mcp-adapters prefixes tool names with server name: "{server_name}_{tool_name}"
            # Note: Use resolved_servers which has per-agent tool overrides already applied
            filtered_tools = []

            for tool in all_mcp_tools:
                tool_name = tool.name
                included = True

                # Check each server config to see if this tool should be included
                for server_config in resolved_servers:
                    server_name = server_config.name
                    server_prefix = f"{server_name}_"

                    # Check if this tool belongs to this server
                    if tool_name.startswith(server_prefix):
                        # Extract actual tool name (without server prefix)
                        actual_tool_name = tool_name[len(server_prefix):]

                        # If selected_tools is specified, filter based on it
                        # This includes per-agent overrides that were applied during reference resolution
                        if server_config.selected_tools is not None:
                            if actual_tool_name not in server_config.selected_tools:
                                included = False
                                logger.debug(f"Filtering out tool {tool_name} (not in selected_tools for {server_name})")
                                break
                        # If selected_tools is None, include all tools from this server
                        break

                if included:
                    filtered_tools.append(tool)

            logger.info(f"Loaded {len(filtered_tools)} tools from MCP servers (filtered from {len(all_mcp_tools)} total)")

            # Return both client and tools - client must be kept alive for tool execution
            return client, filtered_tools

        except ImportError as e:
            logger.error(f"langchain-mcp-adapters not available: {e}")
            raise AgentFactoryError(
                "MCP adapter library not installed. Install with: pip install langchain-mcp-adapters"
            )
        except Exception as e:
            logger.error(f"Failed to create MCP tools: {str(e)}", exc_info=True)
            raise AgentFactoryError(f"Failed to create MCP tools: {str(e)}")

    async def create_mcp_tools(self, config: AgentConfig) -> tuple[Optional[Any], List[BaseTool]]:
        """
        Create tools from MCP servers (async).

        Args:
            config: Agent configuration

        Returns:
            Tuple of (MCP client instance, list of tools from MCP servers)
            Client will be None if no MCP servers configured

        Raises:
            AgentFactoryError: If MCP tool creation fails
        """
        if not config.mcp_servers:
            return None, []

        try:
            # Await the async method directly
            return await self._create_mcp_tools_async(config)
        except Exception as e:
            logger.error(f"Error creating MCP tools: {str(e)}")
            raise AgentFactoryError(f"Failed to create MCP tools: {str(e)}")

    async def create_agent_from_config(
        self,
        config: AgentConfig,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Create an agent from configuration (async).

        Args:
            config: Agent configuration
            context: Optional runtime context

        Returns:
            Created agent instance

        Raises:
            AgentFactoryError: If agent creation fails
        """
        logger.info(f"Starting agent creation: name={config.name}, version={config.version}")

        try:
            # 1. Create LLM
            model = self.create_llm(config)
            logger.info(f"Created LLM: provider={config.llm.provider}, model={config.llm.model}")

            # 2. Get built-in tools
            builtin_tools = self.tool_registry.get_tools(config.tools)
            logger.info(f"Retrieved {len(builtin_tools)} built-in tools: {[t.name for t in builtin_tools]}")

            # 3. Get MCP tools and client
            mcp_client, mcp_tools = await self.create_mcp_tools(config)
            if mcp_tools:
                logger.info(f"Retrieved {len(mcp_tools)} MCP tools: {[t.name for t in mcp_tools]}")

                # Debug: Log tool attributes to verify async/sync setup
                for tool in mcp_tools[:3]:  # Log first 3 tools to avoid spam
                    has_func = hasattr(tool, 'func') and callable(getattr(tool, 'func', None))
                    has_coroutine = hasattr(tool, 'coroutine') and callable(getattr(tool, 'coroutine', None))
                    has_ainvoke = hasattr(tool, 'ainvoke') and callable(getattr(tool, 'ainvoke', None))
                    has_invoke = hasattr(tool, 'invoke') and callable(getattr(tool, 'invoke', None))
                    logger.debug(
                        f"🔍 Tool '{tool.name}' capabilities: "
                        f"func={has_func}, coroutine={has_coroutine}, "
                        f"invoke={has_invoke}, ainvoke={has_ainvoke}"
                    )

                # Test: Try invoking first calculator tool to validate it works
                test_tool = next((t for t in mcp_tools if 'add' in t.name.lower()), None)
                if test_tool:
                    try:
                        logger.debug(f"🧪 Testing tool '{test_tool.name}' with direct invocation")
                        test_result = await test_tool.ainvoke({"a": 2, "b": 3})
                        logger.info(f"✅ Tool validation PASSED: {test_tool.name} returned '{test_result}'")
                    except Exception as tool_error:
                        logger.error(
                            f"❌ Tool validation FAILED: {test_tool.name} raised {type(tool_error).__name__}: {str(tool_error)}",
                            exc_info=True,
                            extra={
                                "tool_name": test_tool.name,
                                "error_type": type(tool_error).__name__,
                                "error_details": str(tool_error)
                            }
                        )
                        # Don't fail agent creation, just warn
                        logger.warning(f"⚠️ MCP tools may not work correctly - validation test failed")

                # Fix: Add sync wrapper for async-only MCP tools
                # MCP tools from langchain-mcp-adapters are async-only (coroutine but no func)
                # LangGraph may call sync invoke() which raises NotImplementedError
                # We need to ensure tools support both sync and async invocation
                import asyncio
                from functools import wraps

                for tool in mcp_tools:
                    # Check if tool is async-only
                    has_func = hasattr(tool, 'func') and callable(getattr(tool, 'func', None))
                    has_coroutine = hasattr(tool, 'coroutine') and callable(getattr(tool, 'coroutine', None))

                    if has_coroutine and not has_func:
                        # Tool is async-only - add sync wrapper
                        logger.debug(f"🔧 Adding sync wrapper to async-only tool: {tool.name}")

                        # Store original coroutine
                        original_coroutine = tool.coroutine

                        # Create sync wrapper that runs async code
                        def make_sync_wrapper(async_func):
                            @wraps(async_func)
                            def sync_wrapper(*args, **kwargs):
                                try:
                                    # Try to get existing event loop
                                    loop = asyncio.get_event_loop()
                                    if loop.is_running():
                                        # Loop is already running (nested call)
                                        # Create new task and wait for it
                                        import concurrent.futures
                                        with concurrent.futures.ThreadPoolExecutor() as executor:
                                            future = executor.submit(
                                                asyncio.run, async_func(*args, **kwargs)
                                            )
                                            return future.result()
                                    else:
                                        # Loop exists but not running
                                        return loop.run_until_complete(async_func(*args, **kwargs))
                                except RuntimeError:
                                    # No event loop exists
                                    return asyncio.run(async_func(*args, **kwargs))
                            return sync_wrapper

                        # Set the sync wrapper as func
                        tool.func = make_sync_wrapper(original_coroutine)
                        logger.debug(f"✅ Sync wrapper added to {tool.name}")

            # Merge all tools
            tools = builtin_tools + mcp_tools
            logger.info(f"Total tools available: {len(tools)} ({len(builtin_tools)} built-in + {len(mcp_tools)} MCP)")

            # Auto-enable llm_tool_selector middleware for agents with many tools
            # Note: Threshold raised from 5 to 25 to avoid hallucination issues with gpt-4o-mini
            total_tools = len(tools)
            if total_tools >= 25:
                # Find llm_tool_selector middleware position
                selector_index = next((i for i, m in enumerate(config.middleware)
                                      if m.type == "llm_tool_selector"), None)

                if selector_index is None:
                    # Doesn't exist - add it at position 0
                    max_tools = min(8, max(5, total_tools // 3))

                    # Derive middleware model from agent's LLM config
                    middleware_model = f"{config.llm.provider}:{config.llm.model}"

                    # Create llm_tool_selector config
                    from agent_api.models.config_schema import MiddlewareConfig
                    selector_config = MiddlewareConfig(
                        type="llm_tool_selector",
                        params={
                            "model": middleware_model,
                            "max_tools": max_tools,
                            "always_include": []
                        },
                        enabled=True
                    )

                    # Insert at beginning of middleware list
                    config.middleware.insert(0, selector_config)
                    logger.info(
                        f"Auto-enabled llm_tool_selector middleware: "
                        f"{total_tools} tools detected → max_tools={max_tools}, model={middleware_model}"
                    )

                elif selector_index != 0:
                    # Exists but in wrong position - move it to position 0
                    selector = config.middleware.pop(selector_index)
                    config.middleware.insert(0, selector)
                    middleware_order = [m.type for m in config.middleware]
                    logger.warning(
                        f"Moved llm_tool_selector from position {selector_index} to position 0 "
                        f"(MUST be first middleware for proper tool filtering). "
                        f"Current middleware order: {middleware_order}"
                    )

            # 4. Get system prompt
            system_prompt = self.get_system_prompt(config, context)
            logger.debug(f"Generated system prompt (length: {len(system_prompt)} chars)")

            # 5. Create checkpointer
            checkpointer = self.create_checkpointer(config)
            if checkpointer:
                logger.info(f"Created checkpointer: type={config.memory.short_term.type}")
            else:
                logger.debug("No checkpointer configured")

            # 6. Create store
            store = self.create_store(config)
            if store:
                logger.info(f"Created store: type={config.memory.long_term.type if config.memory.long_term else 'none'}")
            else:
                logger.debug("No long-term store configured")

            # 7. Get middleware (pass LLM for middleware that may need it)
            middleware = self.middleware_factory.create_middleware_list(config, llm=model)
            logger.info(f"Created {len(middleware)} middleware instances: {[type(m).__name__ for m in middleware]}")

            # 8. Create state schema
            state_schema = self.create_state_schema(config)
            if state_schema:
                logger.debug(f"Created state schema")

            # Build create_agent parameters
            agent_params = {
                "model": model,
                "tools": tools,
                "system_prompt": system_prompt,
            }

            if checkpointer:
                agent_params["checkpointer"] = checkpointer

            if middleware:
                agent_params["middleware"] = middleware

            if state_schema:
                agent_params["state_schema"] = state_schema

            logger.debug(f"Calling create_agent with params: model, {len(tools)} tools, checkpointer={checkpointer is not None}, middleware={len(middleware)}")

            # Create agent using LangChain's create_agent
            agent = create_agent(**agent_params)

            logger.info(f"Successfully created agent '{config.name}' with create_agent()")

            # Store agent metadata (including MCP client to keep it alive)
            self._agents[config.name] = {
                "agent": agent,
                "config": config,
                "store": store,
                "mcp_client": mcp_client,  # Keep MCP client alive for tool execution
                "created_at": datetime.now(),
            }

            logger.info(f"Agent '{config.name}' stored in cache with MCP client: {mcp_client is not None}")

            return agent

        except Exception as e:
            logger.error(f"Failed to create agent '{config.name}': {str(e)}", exc_info=True)
            raise AgentFactoryError(f"Failed to create agent: {str(e)}")

    async def get_agent(self, agent_name: str):
        """
        Get a cached agent instance. If not cached, attempts to load and deploy on-demand - async.

        Args:
            agent_name: Name of the agent

        Returns:
            Agent instance or None if not found
        """
        # Check cache first
        agent_data = self._agents.get(agent_name)
        if agent_data:
            return agent_data["agent"]

        # Lazy loading: Try to load config and deploy if it exists
        try:
            logger.info(f"Agent '{agent_name}' not in cache, attempting lazy load...")

            # Try to find config file
            configs = self.config_manager.list_configs()
            matching_config = next((c for c in configs if c['name'] == agent_name), None)

            if matching_config:
                # Load and deploy
                config_dict = self.config_manager.load_config(matching_config['config_path'])
                await self.deploy_agent(config_dict)
                logger.info(f"Successfully lazy-loaded agent '{agent_name}'")

                # Return from cache
                agent_data = self._agents.get(agent_name)
                return agent_data["agent"] if agent_data else None
            else:
                logger.debug(f"No config found for agent '{agent_name}'")
                return None

        except Exception as e:
            logger.error(f"Failed to lazy-load agent '{agent_name}': {str(e)}")
            return None

    def get_agent_metadata(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a cached agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Agent metadata or None
        """
        return self._agents.get(agent_name)

    def list_agents(self) -> List[str]:
        """
        List all cached agent names.

        Returns:
            List of agent names
        """
        return list(self._agents.keys())

    def remove_agent(self, agent_name: str) -> bool:
        """
        Remove an agent from cache and cleanup resources.

        Args:
            agent_name: Name of the agent

        Returns:
            True if removed, False if not found
        """
        if agent_name in self._agents:
            agent_metadata = self._agents[agent_name]

            # Cleanup MCP client if it exists
            mcp_client = agent_metadata.get("mcp_client")
            if mcp_client:
                try:
                    # Try to close the MCP client gracefully
                    if hasattr(mcp_client, 'close'):
                        mcp_client.close()
                        logger.info(f"Closed MCP client for agent '{agent_name}'")
                    elif hasattr(mcp_client, '__aexit__'):
                        # If it's an async context manager, we can't close it synchronously
                        # Just log a warning
                        logger.warning(f"MCP client for agent '{agent_name}' is async, cannot close synchronously")
                except Exception as e:
                    logger.warning(f"Error closing MCP client for agent '{agent_name}': {e}")

            # Remove agent from cache
            del self._agents[agent_name]
            logger.info(f"Agent '{agent_name}' removed from cache")
            return True
        return False

    async def deploy_agent(self, config: AgentConfig) -> Dict[str, Any]:
        """
        Deploy an agent (create and cache) - async.

        Args:
            config: Agent configuration

        Returns:
            Deployment metadata
        """
        try:
            # Create agent
            agent = await self.create_agent_from_config(config)

            # Save configuration
            config_path = self.config_manager.save_config(config)

            return {
                "agent_name": config.name,
                "status": "deployed",
                "config_path": config_path,
                "deployed_at": datetime.now().isoformat(),
            }

        except Exception as e:
            raise AgentFactoryError(f"Failed to deploy agent: {str(e)}")

    async def redeploy_agent(self, agent_name: str) -> Dict[str, Any]:
        """
        Redeploy an agent (reload from config) - async.

        Args:
            agent_name: Name of the agent

        Returns:
            Redeployment metadata
        """
        # Remove from cache
        self.remove_agent(agent_name)

        # Load config
        config_path = self.config_manager.get_config_path(agent_name)
        if not config_path:
            raise AgentFactoryError(f"Configuration not found for agent: {agent_name}")

        config = self.config_manager.load_config(config_path)

        # Deploy
        return await self.deploy_agent(config)

    async def reconfigure_agent(
        self,
        agent_name: str,
        llm: Optional[Any] = None,
        tools: Optional[List[str]] = None,
        mcp_servers: Optional[List[Any]] = None,
        preserve_middleware: bool = True,
        preserve_memory: bool = True
    ) -> Dict[str, Any]:
        """
        Reconfigure a running agent with partial updates (LLM, tools, MCP servers).

        This method enables changing LLM, tools, and MCP servers while preserving
        middleware and memory configuration. Conversation threads are maintained
        through the reconfiguration process.

        Args:
            agent_name: Name of the agent to reconfigure
            llm: New LLM configuration (optional)
            tools: New tools list (optional)
            mcp_servers: New MCP servers list (optional)
            preserve_middleware: Keep middleware config (default True)
            preserve_memory: Keep memory config (default True)

        Returns:
            Dictionary with reconfiguration metadata:
                - agent_id: Agent identifier
                - reconfigured: Whether reconfiguration succeeded
                - changes: List of changes made
                - summary: Human-readable summary
                - redeployed_at: Timestamp
                - thread_continuity: Whether threads are preserved

        Raises:
            AgentFactoryError: If agent not found or reconfiguration fails
        """
        from datetime import datetime, timezone

        logger.info(f"🔄 Starting reconfiguration for agent '{agent_name}'")

        # Step 1: Validate agent exists and is deployed
        if agent_name not in self._agents:
            raise AgentFactoryError(
                f"Agent '{agent_name}' not deployed. Deploy it first before reconfiguring."
            )

        # Step 2: Load current full configuration from disk
        config_path = self.config_manager.get_config_path(agent_name)
        if not config_path:
            raise AgentFactoryError(f"Configuration file not found for agent: {agent_name}")

        current_config = self.config_manager.load_config(config_path)
        logger.info(f"Loaded current config for '{agent_name}' from {config_path}")

        # Step 3: Merge partial updates while preserving middleware/memory
        try:
            merged_config = self.config_manager.merge_partial_config(
                base_config=current_config,
                llm=llm,
                tools=tools,
                mcp_servers=mcp_servers,
                preserve_middleware=preserve_middleware,
                preserve_memory=preserve_memory
            )
            logger.info(f"Successfully merged partial configuration for '{agent_name}'")
        except Exception as e:
            raise AgentFactoryError(f"Failed to merge configuration: {str(e)}")

        # Step 4: Save merged configuration to disk
        try:
            saved_path = self.config_manager.save_config(merged_config, config_path)
            logger.info(f"Saved merged configuration to {saved_path}")
        except Exception as e:
            raise AgentFactoryError(f"Failed to save merged configuration: {str(e)}")

        # Step 5: Track changes for response
        changes = []
        summary = {}

        # Compare LLM
        if llm is not None:
            old_llm = f"{current_config.llm.provider}/{current_config.llm.model}"
            new_llm = f"{merged_config.llm.provider}/{merged_config.llm.model}"
            if old_llm != new_llm:
                changes.append({
                    "field": "llm.model",
                    "old_value": old_llm,
                    "new_value": new_llm,
                    "change_type": "changed"
                })
                summary["llm"] = f"changed: {old_llm} → {new_llm}"
            else:
                summary["llm"] = f"unchanged ({new_llm})"
        else:
            summary["llm"] = f"unchanged ({current_config.llm.provider}/{current_config.llm.model})"

        # Compare tools
        if tools is not None:
            old_count = len(current_config.tools)
            new_count = len(merged_config.tools)
            if old_count != new_count:
                changes.append({
                    "field": "tools",
                    "old_value": f"{old_count} tools",
                    "new_value": f"{new_count} tools",
                    "change_type": "changed"
                })
                summary["tools"] = f"changed: {old_count} → {new_count} tools"
            else:
                summary["tools"] = f"unchanged ({new_count} tools)"
        else:
            summary["tools"] = f"unchanged ({len(current_config.tools)} tools)"

        # Compare MCP servers
        if mcp_servers is not None:
            old_count = len(current_config.mcp_servers) if current_config.mcp_servers else 0
            new_count = len(merged_config.mcp_servers) if merged_config.mcp_servers else 0
            if old_count != new_count:
                changes.append({
                    "field": "mcp_servers",
                    "old_value": f"{old_count} servers",
                    "new_value": f"{new_count} servers",
                    "change_type": "changed"
                })
                summary["mcp_servers"] = f"changed: {old_count} → {new_count} servers"
            else:
                summary["mcp_servers"] = f"unchanged ({new_count} servers)"
        else:
            old_count = len(current_config.mcp_servers) if current_config.mcp_servers else 0
            summary["mcp_servers"] = f"unchanged ({old_count} servers)"

        # Middleware always preserved
        middleware_count = len(merged_config.middleware) if merged_config.middleware else 0
        changes.append({
            "field": "middleware",
            "old_value": f"{middleware_count} items",
            "new_value": f"{middleware_count} items",
            "change_type": "preserved"
        })
        summary["middleware"] = f"preserved ({middleware_count} items)"

        # Memory always preserved
        changes.append({
            "field": "memory",
            "old_value": "config preserved",
            "new_value": "config preserved",
            "change_type": "preserved"
        })
        summary["memory"] = "preserved"

        # Step 6: Gracefully remove old agent (cleanup MCP connections)
        logger.info(f"Removing old agent instance for '{agent_name}'")
        self.remove_agent(agent_name)

        # Step 7: Deploy new agent with merged configuration
        try:
            deployment_result = await self.deploy_agent(merged_config)
            logger.info(f"✅ Successfully reconfigured and redeployed agent '{agent_name}'")
        except Exception as e:
            raise AgentFactoryError(f"Failed to deploy reconfigured agent: {str(e)}")

        # Step 8: Return reconfiguration metadata
        return {
            "agent_id": agent_name,
            "reconfigured": True,
            "changes": changes,
            "summary": summary,
            "redeployed_at": datetime.now(timezone.utc),
            "thread_continuity": True,  # Always true since memory paths preserved
            "deployment_info": deployment_result
        }

    # ==================== Runtime Override Methods ====================

    async def get_agent_with_override(
        self,
        agent_name: str,
        thread_id: str,
        override: Optional["RuntimeOverride"] = None
    ) -> tuple[Any, str]:
        """
        Get agent instance, applying runtime overrides if specified.

        This method supports session-based persistence: if an override was previously
        applied with persist_for_session=True, it will be reused for subsequent
        requests on the same thread.

        Args:
            agent_name: Name of the base agent
            thread_id: Thread ID for session tracking
            override: Optional runtime override configuration

        Returns:
            Tuple of (agent_instance, effective_system_prompt)
            Returns (None, "") if agent not found
        """
        from agent_api.models.schemas import RuntimeOverride

        session_key = f"{agent_name}:{thread_id}"

        # Check for persisted session override first (if no new override provided)
        if override is None and session_key in self._session_overrides:
            cached = self._session_overrides[session_key]
            logger.info(f"Using cached session override for {session_key}")
            return cached["variant_agent"], cached["system_prompt"]

        # No override - return base agent
        if override is None:
            agent = await self.get_agent(agent_name)
            if not agent:
                return None, ""
            metadata = self.get_agent_metadata(agent_name)
            if not metadata:
                return None, ""
            config = metadata["config"]
            system_prompt = self.get_system_prompt(config)
            return agent, system_prompt

        # Apply override and create variant agent
        logger.info(f"Creating variant agent with override for {agent_name}")
        variant_agent, system_prompt = await self._create_variant_agent(agent_name, override)

        # Cache the variant if this is a session override (always persist for session per design)
        self._session_overrides[session_key] = {
            "override": override,
            "variant_agent": variant_agent,
            "system_prompt": system_prompt,
            "created_at": datetime.now()
        }
        logger.info(f"Cached session override for {session_key}")

        return variant_agent, system_prompt

    async def _create_variant_agent(
        self,
        agent_name: str,
        override: "RuntimeOverride"
    ) -> tuple[Any, str]:
        """
        Create a variant agent with applied overrides.

        This creates a NEW agent instance with the overridden configuration.
        The base agent in cache is NOT modified.

        Args:
            agent_name: Name of the base agent
            override: Runtime override configuration

        Returns:
            Tuple of (variant_agent_instance, effective_system_prompt)

        Raises:
            AgentFactoryError: If agent not found or creation fails
        """
        # Get base agent metadata
        metadata = self.get_agent_metadata(agent_name)
        if not metadata:
            # Try lazy loading
            agent = await self.get_agent(agent_name)
            metadata = self.get_agent_metadata(agent_name)
            if not metadata:
                raise AgentFactoryError(f"Agent not found: {agent_name}")

        base_config = metadata["config"]

        # 1. Resolve LLM (use cached or create new)
        model = await self._resolve_llm_override(base_config, override.llm)

        # 2. Resolve tools
        tools = await self._resolve_tools_override(base_config, override.tools)

        # 3. Determine if tools changed for prompt auto-update
        tools_changed = override.tools is not None
        tools_for_doc = tools if override.auto_update_prompt and tools_changed else None

        # 4. Resolve system prompt (with auto-update if tools changed)
        system_prompt = self._resolve_prompt_override(
            base_config,
            override.prompt,
            tools_for_doc
        )

        # 5. Reuse checkpointer from base agent (for conversation continuity)
        checkpointer = self.create_checkpointer(base_config)

        # 6. Create middleware (using overridden LLM)
        middleware = self.middleware_factory.create_middleware_list(base_config, llm=model)

        # 7. Build create_agent parameters
        agent_params = {
            "model": model,
            "tools": tools,
            "system_prompt": system_prompt,
        }

        if checkpointer:
            agent_params["checkpointer"] = checkpointer

        if middleware:
            agent_params["middleware"] = middleware

        # 8. Create variant agent
        variant_agent = create_agent(**agent_params)
        logger.info(
            f"Created variant agent for '{agent_name}' with "
            f"{len(tools)} tools, LLM={override.llm.provider if override.llm and override.llm.provider else 'base'}:"
            f"{override.llm.model if override.llm and override.llm.model else 'base'}"
        )

        return variant_agent, system_prompt

    async def _resolve_llm_override(
        self,
        base_config: AgentConfig,
        llm_override: Optional["LLMOverride"]
    ) -> BaseChatModel:
        """
        Resolve LLM with override, using cache when possible.

        Args:
            base_config: Base agent configuration
            llm_override: Optional LLM override

        Returns:
            LLM instance (may be cached)
        """
        from agent_api.models.config_schema import LLMConfig

        if llm_override is None:
            # Return LLM from base config
            return self.create_llm(base_config)

        # Build effective LLM config by merging override with base
        effective_provider = llm_override.provider or base_config.llm.provider
        effective_model = llm_override.model or base_config.llm.model
        effective_temperature = llm_override.temperature if llm_override.temperature is not None else base_config.llm.temperature
        effective_max_tokens = llm_override.max_tokens or base_config.llm.max_tokens
        effective_api_key = llm_override.api_key or base_config.llm.api_key

        # Check LLM cache
        cache_key = f"{effective_provider}:{effective_model}:{effective_temperature}"
        if cache_key in self._llm_cache:
            logger.debug(f"Using cached LLM: {cache_key}")
            return self._llm_cache[cache_key]

        # Create effective config
        effective_llm_config = LLMConfig(
            provider=effective_provider,
            model=effective_model,
            temperature=effective_temperature,
            max_tokens=effective_max_tokens,
            api_key=effective_api_key,
        )

        # Create temporary AgentConfig for LLM creation
        temp_config = AgentConfig(
            name=base_config.name,
            llm=effective_llm_config,
            prompts=base_config.prompts,
        )

        # Create new LLM and cache it
        llm = self.create_llm(temp_config)
        self._llm_cache[cache_key] = llm
        logger.info(f"Created and cached LLM: {cache_key}")
        return llm

    async def _resolve_tools_override(
        self,
        base_config: AgentConfig,
        tools_override: Optional["ToolsOverride"]
    ) -> List[BaseTool]:
        """
        Resolve tools with override.

        Args:
            base_config: Base agent configuration
            tools_override: Optional tools override

        Returns:
            List of resolved tools
        """
        if tools_override is None:
            # Return base tools
            builtin_tools = self.tool_registry.get_tools(base_config.tools)
            _, mcp_tools = await self.create_mcp_tools(base_config)
            return builtin_tools + mcp_tools

        tools = []

        # Built-in tools
        if tools_override.builtin_tools is not None:
            tools.extend(self.tool_registry.get_tools(tools_override.builtin_tools))
            logger.debug(f"Using override builtin tools: {tools_override.builtin_tools}")
        else:
            # Use base config tools
            tools.extend(self.tool_registry.get_tools(base_config.tools))

        # MCP tools
        if tools_override.mcp_servers is not None:
            # Create temporary config with overridden MCP servers
            mcp_configs = []
            for server_name in tools_override.mcp_servers:
                try:
                    server_def = self.mcp_server_manager.load_server(server_name)
                    selected_tools = None
                    if tools_override.mcp_selected_tools and server_name in tools_override.mcp_selected_tools:
                        selected_tools = tools_override.mcp_selected_tools[server_name]

                    mcp_configs.append(MCPServerConfig(
                        name=server_def.name,
                        description=server_def.description,
                        transport=server_def.transport,
                        url=server_def.url,
                        command=server_def.command,
                        args=server_def.args,
                        env=server_def.env,
                        stateful=server_def.stateful,
                        selected_tools=selected_tools,
                    ))
                except Exception as e:
                    logger.warning(f"Failed to load MCP server {server_name}: {e}")

            if mcp_configs:
                temp_config = AgentConfig(
                    name=base_config.name,
                    llm=base_config.llm,
                    prompts=base_config.prompts,
                    mcp_servers=mcp_configs,
                )
                _, mcp_tools = await self.create_mcp_tools(temp_config)
                tools.extend(mcp_tools)
                logger.debug(f"Using override MCP servers: {tools_override.mcp_servers}")
        elif base_config.mcp_servers:
            # Use base config MCP servers
            _, mcp_tools = await self.create_mcp_tools(base_config)
            tools.extend(mcp_tools)

        return tools

    def _resolve_prompt_override(
        self,
        base_config: AgentConfig,
        prompt_override: Optional["PromptOverride"],
        tools_for_doc: Optional[List[BaseTool]] = None
    ) -> str:
        """
        Resolve system prompt with override.

        If tools_for_doc is provided, regenerates tool documentation section.

        Args:
            base_config: Base agent configuration
            prompt_override: Optional prompt override (prepend/append)
            tools_for_doc: Optional tools list for auto-documentation

        Returns:
            Effective system prompt
        """
        base_prompt = self.get_system_prompt(base_config)

        # Auto-update tool documentation if tools changed
        if tools_for_doc:
            tool_docs = self._generate_tool_documentation(tools_for_doc)
            base_prompt = f"{base_prompt}\n\n## Available Tools\n{tool_docs}"
            logger.debug(f"Auto-updated system prompt with {len(tools_for_doc)} tool docs")

        if prompt_override is None:
            return base_prompt

        # Apply prepend/append
        result = base_prompt
        if prompt_override.prepend:
            result = f"{prompt_override.prepend}\n\n{result}"
            logger.debug("Applied prompt prepend")
        if prompt_override.append:
            result = f"{result}\n\n{prompt_override.append}"
            logger.debug("Applied prompt append")

        return result

    def _generate_tool_documentation(self, tools: List[BaseTool]) -> str:
        """
        Generate tool documentation for system prompt.

        Args:
            tools: List of tools to document

        Returns:
            Formatted tool documentation string
        """
        if not tools:
            return "No tools available."

        doc_lines = []
        for tool in tools:
            name = tool.name
            description = tool.description or "No description"
            doc_lines.append(f"- **{name}**: {description}")

        return "\n".join(doc_lines)

    def clear_session_override(self, agent_name: str, thread_id: str) -> bool:
        """
        Clear a persisted session override, reverting to base agent.

        Args:
            agent_name: Name of the agent
            thread_id: Thread ID for the session

        Returns:
            True if override was cleared, False if not found
        """
        session_key = f"{agent_name}:{thread_id}"
        if session_key in self._session_overrides:
            del self._session_overrides[session_key]
            logger.info(f"Cleared session override for {session_key}")
            return True
        return False

    def get_session_override(self, agent_name: str, thread_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current session override if any.

        Args:
            agent_name: Name of the agent
            thread_id: Thread ID for the session

        Returns:
            Session override data or None
        """
        session_key = f"{agent_name}:{thread_id}"
        return self._session_overrides.get(session_key)

    def list_session_overrides(self, agent_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all active session overrides, optionally filtered by agent.

        Args:
            agent_name: Optional agent name to filter by

        Returns:
            List of session override summaries
        """
        result = []
        for key, data in self._session_overrides.items():
            agent, thread = key.split(":", 1)
            if agent_name is None or agent == agent_name:
                result.append({
                    "agent_name": agent,
                    "thread_id": thread,
                    "created_at": data["created_at"],
                    "has_llm_override": data["override"].llm is not None,
                    "has_tools_override": data["override"].tools is not None,
                    "has_prompt_override": data["override"].prompt is not None,
                })
        return result
