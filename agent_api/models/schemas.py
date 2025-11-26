"""
API request and response schemas.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field

from .config_schema import (
    AgentConfig,
    LLMConfig,
    MCPServerConfig,
    MCPServerDefinition,
    MCPServerReference,
)


# ==================== Base Response Models ====================


class ErrorDetail(BaseModel):
    """Error detail information."""
    code: str = Field(description="Error code")
    message: str = Field(description="Error message")
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details"
    )


class APIResponse(BaseModel):
    """Standard API response wrapper."""
    success: bool = Field(description="Whether the request was successful")
    data: Optional[Any] = Field(default=None, description="Response data")
    error: Optional[ErrorDetail] = Field(default=None, description="Error details if failed")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response timestamp"
    )
    request_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique request identifier"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "data": {"agent_id": "research_assistant"},
                "error": None,
                "timestamp": "2025-01-14T10:00:00Z",
                "request_id": "550e8400-e29b-41d4-a716-446655440000"
            }
        }


# ==================== Agent Management ====================


class AgentCreateRequest(BaseModel):
    """Request to create a new agent."""
    config: AgentConfig = Field(description="Agent configuration")
    deploy: bool = Field(
        default=True,
        description="Whether to immediately deploy the agent"
    )


class AgentInfo(BaseModel):
    """Agent information summary."""
    agent_id: str = Field(description="Unique agent identifier")
    name: str = Field(description="Agent name")
    version: str = Field(description="Agent version")
    description: Optional[str] = Field(default=None, description="Agent description")
    tags: List[str] = Field(default_factory=list, description="Agent tags")
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")
    deployed: bool = Field(description="Whether agent is deployed")
    config_path: str = Field(description="Path to configuration file")
    has_mcp_servers: bool = Field(default=False, description="Whether agent uses MCP servers")


class AgentListResponse(BaseModel):
    """Response with list of agents."""
    agents: List[AgentInfo] = Field(description="List of agents")
    total: int = Field(description="Total number of agents")


class AgentDetailResponse(BaseModel):
    """Response with full agent details."""
    agent_info: AgentInfo = Field(description="Agent information")
    config: AgentConfig = Field(description="Full agent configuration")


class AgentUpdateRequest(BaseModel):
    """Request to update an existing agent."""
    config: AgentConfig = Field(description="Updated agent configuration")
    redeploy: bool = Field(
        default=True,
        description="Whether to redeploy the agent"
    )


class AgentReconfigureRequest(BaseModel):
    """Request to reconfigure a running agent (LLM, tools, MCP servers only)."""
    llm: Optional[LLMConfig] = Field(
        default=None,
        description="New LLM configuration (provider, model, temperature, etc.)"
    )
    tools: Optional[List[str]] = Field(
        default=None,
        description="New list of built-in tools"
    )
    mcp_servers: Optional[List[MCPServerConfig]] = Field(
        default=None,
        description="New list of MCP servers"
    )
    preserve_middleware: bool = Field(
        default=True,
        description="Preserve middleware configuration (always true)"
    )
    preserve_memory: bool = Field(
        default=True,
        description="Preserve memory paths and configuration (always true)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "llm": {
                    "provider": "openai",
                    "model": "gpt-4o",
                    "temperature": 0.7
                },
                "tools": ["tavily_search", "calculator"],
                "preserve_middleware": True,
                "preserve_memory": True
            }
        }


class ConfigChange(BaseModel):
    """Description of a configuration change."""
    field: str = Field(description="Configuration field that changed")
    old_value: Optional[str] = Field(default=None, description="Previous value summary")
    new_value: Optional[str] = Field(default=None, description="New value summary")
    change_type: str = Field(description="Type of change (changed/added/removed/preserved)")


class AgentReconfigureResponse(BaseModel):
    """Response from agent reconfiguration."""
    agent_id: str = Field(description="Agent identifier")
    reconfigured: bool = Field(description="Whether reconfiguration was successful")
    changes: List[ConfigChange] = Field(
        description="List of configuration changes made"
    )
    summary: Dict[str, str] = Field(
        description="Human-readable summary of changes"
    )
    redeployed_at: datetime = Field(
        description="Timestamp when agent was redeployed"
    )
    thread_continuity: bool = Field(
        default=True,
        description="Whether conversation threads are preserved"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "agent_id": "research_assistant",
                "reconfigured": True,
                "changes": [
                    {
                        "field": "llm.model",
                        "old_value": "gpt-4o-mini",
                        "new_value": "gpt-4o",
                        "change_type": "changed"
                    },
                    {
                        "field": "middleware",
                        "old_value": "5 items",
                        "new_value": "5 items",
                        "change_type": "preserved"
                    }
                ],
                "summary": {
                    "llm": "changed: gpt-4o-mini → gpt-4o",
                    "tools": "unchanged (1 tool)",
                    "mcp_servers": "unchanged (2 servers)",
                    "middleware": "preserved (5 items)",
                    "memory": "preserved"
                },
                "redeployed_at": "2025-01-15T10:30:00Z",
                "thread_continuity": True
            }
        }


class AgentDeleteResponse(BaseModel):
    """Response after deleting an agent."""
    agent_id: str = Field(description="Deleted agent ID")
    message: str = Field(description="Deletion confirmation message")


# ==================== Agent Execution ====================


class MessageInput(BaseModel):
    """User message input."""
    role: str = Field(default="user", description="Message role")
    content: str = Field(description="Message content")


class AgentInvokeRequest(BaseModel):
    """Request to invoke an agent."""
    messages: List[MessageInput] = Field(
        description="Conversation messages"
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Runtime context values"
    )
    thread_id: Optional[str] = Field(
        default=None,
        description="Thread ID for conversation continuity"
    )
    config_overrides: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Runtime configuration overrides"
    )


class ToolCall(BaseModel):
    """Tool call information."""
    id: str = Field(description="Tool call ID")
    name: str = Field(description="Tool name")
    args: Dict[str, Any] = Field(description="Tool arguments")
    result: Optional[Any] = Field(default=None, description="Tool result")


class AgentMessage(BaseModel):
    """Agent message response."""
    id: Optional[str] = Field(
        default=None,
        description="Unique message ID for deduplication"
    )
    role: str = Field(description="Message role (ai/user/tool)")
    content: str = Field(description="Message content")
    tool_calls: Optional[List[ToolCall]] = Field(
        default=None,
        description="Tool calls if any"
    )


class AgentInvokeResponse(BaseModel):
    """Response from agent invocation."""
    messages: List[AgentMessage] = Field(description="All messages in response")
    thread_id: str = Field(description="Thread ID for conversation")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (token usage, costs, etc.)"
    )


class StreamEvent(BaseModel):
    """Streaming event."""
    event_type: str = Field(description="Event type (updates/messages/custom)")
    data: Any = Field(description="Event data")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Event metadata"
    )


# ==================== Tool Management ====================


class ToolGenerateRequest(BaseModel):
    """Request to generate a custom tool."""
    description: str = Field(
        description="Natural language description of the tool"
    )
    name: Optional[str] = Field(
        default=None,
        description="Optional tool name (auto-generated if not provided)"
    )
    examples: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Example inputs and expected outputs"
    )


class ToolInfo(BaseModel):
    """Tool information."""
    tool_id: str = Field(description="Tool identifier")
    name: str = Field(description="Tool name")
    description: str = Field(description="Tool description")
    type: str = Field(description="Tool type (builtin/custom/mcp)")
    status: str = Field(description="Tool status (active/pending/rejected)")
    code: Optional[str] = Field(default=None, description="Tool code (for custom tools)")
    created_at: datetime = Field(description="Creation timestamp")


class ToolListResponse(BaseModel):
    """Response with list of tools."""
    tools: List[ToolInfo] = Field(description="List of tools")
    total: int = Field(description="Total number of tools")


class ToolApproveRequest(BaseModel):
    """Request to approve a pending tool."""
    tool_id: str = Field(description="Tool identifier")
    approved: bool = Field(description="Whether to approve or reject")
    modifications: Optional[str] = Field(
        default=None,
        description="Optional code modifications before approval"
    )


class ToolTestRequest(BaseModel):
    """Request to test a tool."""
    tool_id: str = Field(description="Tool identifier")
    test_input: Dict[str, Any] = Field(description="Test input arguments")


class ToolTestResponse(BaseModel):
    """Response from tool testing."""
    success: bool = Field(description="Whether the test succeeded")
    output: Optional[Any] = Field(default=None, description="Tool output")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    execution_time: float = Field(description="Execution time in seconds")


# ==================== MCP Server Management ====================


class MCPServerCreateRequest(BaseModel):
    """Request to create a new MCP server configuration."""
    config: MCPServerDefinition = Field(description="MCP server configuration")
    overwrite: bool = Field(
        default=False,
        description="Whether to overwrite existing server with same name"
    )


class MCPServerInfo(BaseModel):
    """MCP server information summary."""
    name: str = Field(description="Unique server name")
    description: Optional[str] = Field(default=None, description="Server description")
    transport: str = Field(description="Transport type (stdio, http, sse, streamable_http)")
    url: Optional[str] = Field(default=None, description="Server URL (for http/sse transports)")
    command: Optional[str] = Field(default=None, description="Command (for stdio transport)")
    stateful: bool = Field(default=False, description="Whether server uses stateful sessions")
    version: str = Field(default="1.0.0", description="Server config version")
    tags: List[str] = Field(default_factory=list, description="Tags for organization")
    selected_tools: Optional[List[str]] = Field(
        default=None,
        description="Default tool selection (None = all tools)"
    )
    tool_count: Optional[int] = Field(
        default=None,
        description="Number of discovered tools (if server is reachable)"
    )
    config_path: str = Field(description="Path to configuration file")
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")


class MCPServerListResponse(BaseModel):
    """Response with list of MCP servers."""
    servers: List[MCPServerInfo] = Field(description="List of MCP servers")
    total: int = Field(description="Total number of servers")


class MCPServerDetailResponse(BaseModel):
    """Response with full MCP server details."""
    server_info: MCPServerInfo = Field(description="Server information summary")
    config: MCPServerDefinition = Field(description="Full server configuration")
    discovered_tools: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="List of discovered tools (if server is reachable)"
    )


class MCPServerValidateRequest(BaseModel):
    """Request to validate an MCP server configuration."""
    config: Dict[str, Any] = Field(description="Server configuration to validate")


class MCPServerValidateResponse(BaseModel):
    """Response from MCP server validation."""
    valid: bool = Field(description="Whether configuration is valid")
    errors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Validation errors if any"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Validation warnings"
    )


class MCPServerDiscoverToolsRequest(BaseModel):
    """Request to discover tools from an MCP server."""
    timeout: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Timeout in seconds for tool discovery"
    )


class MCPToolInfo(BaseModel):
    """MCP tool information."""
    name: str = Field(description="Tool name (may include server prefix)")
    description: Optional[str] = Field(default=None, description="Tool description")
    input_schema: Optional[Dict[str, Any]] = Field(
        default=None,
        description="JSON schema for tool input"
    )


class MCPServerDiscoverToolsResponse(BaseModel):
    """Response from MCP server tool discovery."""
    server_name: str = Field(description="Server name")
    tools: List[MCPToolInfo] = Field(description="Discovered tools")
    total: int = Field(description="Total number of tools discovered")
    discovery_time: float = Field(description="Time taken to discover tools (seconds)")


class MCPServerDeleteResponse(BaseModel):
    """Response after deleting an MCP server."""
    server_name: str = Field(description="Deleted server name")
    message: str = Field(description="Deletion confirmation message")


# ==================== Configuration Management ====================


class ConfigValidateRequest(BaseModel):
    """Request to validate a configuration."""
    config: Dict[str, Any] = Field(description="Configuration to validate")


class ValidationError(BaseModel):
    """Validation error detail."""
    field: str = Field(description="Field with error")
    message: str = Field(description="Error message")
    value: Optional[Any] = Field(default=None, description="Invalid value")


class ConfigValidateResponse(BaseModel):
    """Response from configuration validation."""
    valid: bool = Field(description="Whether configuration is valid")
    errors: List[ValidationError] = Field(
        default_factory=list,
        description="Validation errors if any"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Validation warnings"
    )


# ==================== Template Management ====================


class TemplateInfo(BaseModel):
    """Template information."""
    template_id: str = Field(description="Template identifier")
    name: str = Field(description="Template name")
    description: str = Field(description="Template description")
    category: str = Field(description="Template category")
    tags: List[str] = Field(default_factory=list, description="Template tags")
    config: AgentConfig = Field(description="Template configuration")


class TemplateListResponse(BaseModel):
    """Response with list of templates."""
    templates: List[TemplateInfo] = Field(description="List of templates")
    total: int = Field(description="Total number of templates")


# ==================== Memory Management ====================


class MemoryEntry(BaseModel):
    """Memory store entry."""
    namespace: List[str] = Field(description="Namespace path")
    key: str = Field(description="Entry key")
    value: Dict[str, Any] = Field(description="Entry value")
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")


class MemoryListRequest(BaseModel):
    """Request to list memory entries."""
    namespace: Optional[List[str]] = Field(
        default=None,
        description="Filter by namespace"
    )
    limit: int = Field(default=100, gt=0, le=1000, description="Maximum entries to return")
    offset: int = Field(default=0, ge=0, description="Offset for pagination")


class MemoryListResponse(BaseModel):
    """Response with memory entries."""
    entries: List[MemoryEntry] = Field(description="Memory entries")
    total: int = Field(description="Total number of entries")


class MemoryPutRequest(BaseModel):
    """Request to store a memory entry."""
    namespace: List[str] = Field(description="Namespace path")
    key: str = Field(description="Entry key")
    value: Dict[str, Any] = Field(description="Entry value")


class MemoryDeleteRequest(BaseModel):
    """Request to delete memory entries."""
    namespace: List[str] = Field(description="Namespace path")
    key: Optional[str] = Field(default=None, description="Specific key (deletes all if not provided)")


# ==================== Health & Status ====================


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(description="Service status")
    version: str = Field(description="API version")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Current timestamp")
    components: Dict[str, str] = Field(
        description="Status of individual components"
    )
