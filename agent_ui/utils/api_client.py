"""
API Client for Agent UI - handles communication with the Agent Builder API.
Includes methods for agent execution, deployment, and management.
"""

import requests
from typing import Dict, Any, List, Optional
import streamlit as st


class APIClient:
    """HTTP client for interacting with the Agent Builder API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the API client.

        Args:
            base_url: Base URL of the Agent Builder API
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = 30

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request to the API.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint path
            data: Request body data
            params: Query parameters

        Returns:
            Response dictionary with data or error
        """
        url = f"{self.base_url}{endpoint}"

        try:
            response = requests.request(
                method=method,
                url=url,
                json=data,
                params=params,
                timeout=self.timeout
            )

            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "success": False
                }

        except requests.exceptions.ConnectionError:
            return {"error": "Failed to connect to API. Is the server running?", "success": False}
        except requests.exceptions.Timeout:
            return {"error": "Request timed out", "success": False}
        except Exception as e:
            return {"error": str(e), "success": False}

    def test_connection(self) -> Dict[str, Any]:
        """
        Test connection to the API.

        Returns:
            Dict with status information
        """
        return self._make_request("GET", "/health")

    # Agent Management Methods

    def list_agents(self) -> Dict[str, Any]:
        """
        Get list of all available agents.

        Returns:
            Dict with agents list and total count
        """
        return self._make_request("GET", "/agents/list")

    def get_agent_details(self, agent_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Dict with agent info and full configuration
        """
        return self._make_request("GET", f"/agents/{agent_id}")

    def reconfigure_agent(
        self,
        agent_id: str,
        llm: Optional[Dict[str, Any]] = None,
        tools: Optional[List[str]] = None,
        mcp_servers: Optional[List[Dict[str, Any]]] = None,
        preserve_middleware: bool = True,
        preserve_memory: bool = True
    ) -> Dict[str, Any]:
        """
        Reconfigure a running agent (LLM, tools, MCP servers) while preserving middleware.

        This method allows changing the LLM model, tools, and MCP servers for a deployed
        agent without affecting middleware or memory configuration. Conversation threads
        are preserved across the reconfiguration.

        Args:
            agent_id: Agent identifier
            llm: Optional new LLM configuration dict with keys:
                - provider: str (e.g., "openai", "anthropic")
                - model: str (e.g., "gpt-4o", "claude-3-5-sonnet-20241022")
                - temperature: float (optional)
                - max_tokens: int (optional)
            tools: Optional new list of built-in tool names
            mcp_servers: Optional new list of MCP server configurations
            preserve_middleware: Whether to preserve middleware config (default: True)
            preserve_memory: Whether to preserve memory config (default: True)

        Returns:
            Dict with reconfiguration result including:
                - agent_id: str
                - reconfigured: bool
                - changes: List[Dict] (detailed change list)
                - summary: Dict[str, str] (human-readable summary)
                - redeployed_at: datetime
                - thread_continuity: bool

        Example:
            ```python
            result = client.reconfigure_agent(
                agent_id="research_assistant",
                llm={"provider": "openai", "model": "gpt-4o", "temperature": 0.7},
                tools=["tavily_search", "calculator", "get_current_datetime"]
            )
            ```
        """
        # Build request payload
        data = {
            "preserve_middleware": preserve_middleware,
            "preserve_memory": preserve_memory
        }

        # Add optional fields only if provided
        if llm is not None:
            data["llm"] = llm

        if tools is not None:
            data["tools"] = tools

        if mcp_servers is not None:
            data["mcp_servers"] = mcp_servers

        return self._make_request("PUT", f"/agents/{agent_id}/reconfigure", data=data)

    # Execution Methods

    def invoke_agent(
        self,
        agent_id: str,
        messages: List[Dict[str, str]],
        thread_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        runtime_override: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Invoke agent with messages (non-streaming).

        Supports runtime overrides for LLM, tools, and prompt. Overrides persist
        for the entire thread/session until cleared.

        Args:
            agent_id: Agent identifier
            messages: List of message dicts with role and content
            thread_id: Optional thread ID for conversation continuity
            context: Optional runtime context values
            runtime_override: Optional runtime override configuration with structure:
                {
                    "llm": {"provider": str, "model": str, "temperature": float, "max_tokens": int},
                    "tools": {"builtin_tools": List[str], "mcp_servers": List[str]},
                    "prompt": {"prepend": str, "append": str},
                    "auto_update_prompt": bool  # default True
                }

        Returns:
            Dict with messages, thread_id, and metadata
        """
        data = {
            "messages": messages
        }

        if thread_id:
            data["thread_id"] = thread_id

        if context:
            data["context"] = context

        if runtime_override:
            data["runtime_override"] = runtime_override

        return self._make_request("POST", f"/execution/{agent_id}/invoke", data=data)

    def deploy_agent(self, agent_id: str) -> Dict[str, Any]:
        """
        Deploy an agent (load into memory if not already deployed).

        Args:
            agent_id: Agent identifier

        Returns:
            Dict with success status and message
        """
        return self._make_request("POST", f"/execution/{agent_id}/deploy")

    def undeploy_agent(self, agent_id: str) -> Dict[str, Any]:
        """
        Undeploy an agent (remove from memory).

        Args:
            agent_id: Agent identifier

        Returns:
            Dict with success status and message
        """
        return self._make_request("POST", f"/execution/{agent_id}/undeploy")

    # Tool Management Methods (for reference)

    def list_tools(self) -> Dict[str, Any]:
        """
        Get list of all available tools.

        Returns:
            Dict with tools list
        """
        return self._make_request("GET", "/tools/list")

    def get_tool_details(self, tool_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific tool.

        Args:
            tool_id: Tool identifier

        Returns:
            Dict with tool details
        """
        return self._make_request("GET", f"/tools/{tool_id}")

    # Runtime Override Methods

    def get_available_tools(self, agent_id: str) -> Dict[str, Any]:
        """
        Get all available tools for runtime override selection.

        Args:
            agent_id: Agent identifier

        Returns:
            Dict with:
                - builtin_tools: List of all available built-in tools
                - mcp_servers: List of all available MCP servers
                - current_tools: Tools currently configured for this agent
                - current_mcp_servers: MCP servers currently configured for this agent
        """
        return self._make_request("GET", f"/execution/{agent_id}/available-tools")

    def get_session_override(self, agent_id: str, thread_id: str) -> Dict[str, Any]:
        """
        Get current session override for a thread.

        Args:
            agent_id: Agent identifier
            thread_id: Thread/session identifier

        Returns:
            Dict with:
                - has_override: bool
                - override: Override configuration if any
                - created_at: When override was applied
        """
        return self._make_request("GET", f"/execution/{agent_id}/session-override/{thread_id}")

    def clear_session_override(self, agent_id: str, thread_id: str) -> Dict[str, Any]:
        """
        Clear session override for a thread, reverting to base agent.

        Args:
            agent_id: Agent identifier
            thread_id: Thread/session identifier

        Returns:
            Dict with:
                - cleared: bool
                - message: str
        """
        return self._make_request("DELETE", f"/execution/{agent_id}/session-override/{thread_id}")

    def list_session_overrides(self, agent_id: str) -> Dict[str, Any]:
        """
        List all active session overrides for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Dict with:
                - overrides: List of active session overrides
                - total: Total count
        """
        return self._make_request("GET", f"/execution/{agent_id}/session-overrides")


@st.cache_resource
def get_api_client(base_url: str = None) -> APIClient:
    """
    Get cached API client instance.

    Args:
        base_url: Optional base URL (defaults to session state or localhost)

    Returns:
        Cached APIClient instance
    """
    if base_url is None:
        base_url = st.session_state.get('api_base_url', 'http://localhost:8000')

    return APIClient(base_url)


def check_api_availability() -> bool:
    """
    Check if the API is available and reachable.

    Returns:
        True if API is reachable, False otherwise
    """
    try:
        client = get_api_client()
        result = client.test_connection()
        return result.get('success', False) if isinstance(result, dict) else False
    except Exception:
        return False
