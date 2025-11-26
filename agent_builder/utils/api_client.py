"""
API client for communicating with the agent_api backend.
"""

import requests
import streamlit as st
from typing import Dict, Any, Optional, List

from .constants import API_ENDPOINTS


class APIClient:
    """Client for interacting with the Agent Builder API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the API client.

        Args:
            base_url: Base URL of the API
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
        Make an HTTP request to the API.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            data: Request body data
            params: Query parameters

        Returns:
            Response data as dictionary
        """
        url = f"{self.base_url}{endpoint}"

        try:
            if method == "GET":
                response = requests.get(url, params=params, timeout=self.timeout)
            elif method == "POST":
                response = requests.post(url, json=data, timeout=self.timeout)
            elif method == "PUT":
                response = requests.put(url, json=data, timeout=self.timeout)
            elif method == "DELETE":
                response = requests.delete(url, timeout=self.timeout)
            else:
                return {"error": f"Unsupported HTTP method: {method}"}

            response.raise_for_status()
            return response.json()

        except requests.exceptions.ConnectionError:
            return {"error": "Could not connect to API. Is the server running?"}
        except requests.exceptions.Timeout:
            return {"error": "Request timed out"}
        except requests.exceptions.HTTPError as e:
            return {"error": f"HTTP error: {str(e)}"}
        except requests.exceptions.RequestException as e:
            return {"error": f"Request error: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}

    def test_connection(self) -> Dict[str, Any]:
        """
        Test API connection.

        Returns:
            Health check response
        """
        return self._make_request("GET", API_ENDPOINTS["health"])

    def get_templates(self) -> Dict[str, Any]:
        """
        Get available agent templates.

        Returns:
            List of templates
        """
        return self._make_request("GET", API_ENDPOINTS["templates_list"])

    def get_template(self, template_id: str) -> Dict[str, Any]:
        """
        Get a specific template by ID.

        Args:
            template_id: Template identifier

        Returns:
            Template configuration
        """
        endpoint = API_ENDPOINTS["template_get"].format(template_id=template_id)
        return self._make_request("GET", endpoint)

    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate agent configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Validation result
        """
        return self._make_request("POST", API_ENDPOINTS["validate"], data={"config": config})

    def create_agent(self, config: Dict[str, Any], deploy: bool = True) -> Dict[str, Any]:
        """
        Create a new agent.

        Args:
            config: Agent configuration
            deploy: Whether to deploy immediately

        Returns:
            Creation result
        """
        return self._make_request(
            "POST",
            API_ENDPOINTS["create"],
            data={"config": config, "deploy": deploy}
        )

    def list_agents(self) -> Dict[str, Any]:
        """
        List all agents.

        Returns:
            List of agents
        """
        return self._make_request("GET", API_ENDPOINTS["list"])

    def get_tools_list(self, include_pending: bool = False) -> Dict[str, Any]:
        """
        Get list of available tools.

        Args:
            include_pending: Whether to include pending tools

        Returns:
            List of tools
        """
        return self._make_request(
            "GET",
            API_ENDPOINTS["tools_list"],
            params={"include_pending": include_pending}
        )

    def generate_tool(self, description: str, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a custom tool from description.

        Args:
            description: Natural language description of the tool
            name: Optional tool name

        Returns:
            Generated tool information
        """
        data = {"description": description}
        if name:
            data["name"] = name

        return self._make_request("POST", API_ENDPOINTS["tool_generate"], data=data)

    def invoke_agent(
        self,
        agent_id: str,
        messages: List[Dict[str, str]],
        thread_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        runtime_override: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Invoke agent with messages (non-streaming execution).

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
        data = {"messages": messages}

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


@st.cache_resource
def get_api_client() -> APIClient:
    """
    Get cached API client instance.

    Returns:
        APIClient instance
    """
    base_url = st.session_state.get('api_base_url', 'http://localhost:8000')
    return APIClient(base_url)


def check_api_availability() -> bool:
    """
    Check if the API is available.

    Returns:
        True if API is available
    """
    client = get_api_client()
    result = client.test_connection()

    # Check if there's an error (either from exception or API response)
    if result.get("error"):
        st.session_state.api_available = False
        return False
    elif result.get("success"):
        st.session_state.api_available = True
        return True
    else:
        # Unexpected response format
        st.session_state.api_available = False
        return False
