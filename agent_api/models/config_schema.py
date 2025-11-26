"""
Configuration schema for agent configurations.
Defines Pydantic models for validating YAML agent configs.
"""

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator


class LLMConfig(BaseModel):
    """LLM provider and model configuration."""
    provider: Literal["openai", "anthropic", "google", "groq", "openrouter"] = Field(
        description="LLM provider to use"
    )
    model: str = Field(
        description="Model identifier (e.g., 'gpt-4o', 'claude-sonnet-4')"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )
    max_tokens: Optional[int] = Field(
        default=4096,
        gt=0,
        description="Maximum tokens to generate"
    )
    top_p: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="Optional API key override"
    )

    class Config:
        extra = "allow"  # Allow additional provider-specific parameters


class PromptsConfig(BaseModel):
    """System and user prompt configuration."""
    system: str = Field(
        description="System prompt with optional {{variables}}"
    )
    user_template: Optional[str] = Field(
        default=None,
        description="User message template with {{variables}}"
    )
    few_shot_examples: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Few-shot examples as list of {role: content} dicts"
    )

    @field_validator('system')
    @classmethod
    def validate_system_prompt(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("System prompt cannot be empty")
        return v


class ShortTermMemoryConfig(BaseModel):
    """Short-term memory (checkpointer) configuration."""
    type: Literal["in_memory", "sqlite"] = Field(
        default="sqlite",
        description="Checkpointer type"
    )
    path: Optional[str] = Field(
        default=None,
        description="Path for SQLite database (required if type=sqlite)"
    )
    custom_state: Optional[Dict[str, str]] = Field(
        default=None,
        description="Custom state schema fields with types"
    )
    message_management: Optional[Literal["trim", "summarize", "none"]] = Field(
        default="none",
        description="Message management strategy"
    )

    @field_validator('path')
    @classmethod
    def validate_path(cls, v: Optional[str], info) -> Optional[str]:
        if info.data.get('type') == 'sqlite' and not v:
            raise ValueError("Path is required when type is 'sqlite'")
        return v


class LongTermMemoryConfig(BaseModel):
    """Long-term memory (store) configuration."""
    type: Literal["in_memory", "sqlite"] = Field(
        default="sqlite",
        description="Store type"
    )
    path: Optional[str] = Field(
        default=None,
        description="Path for SQLite database (required if type=sqlite)"
    )
    namespaces: List[str] = Field(
        default_factory=list,
        description="Namespace templates (can use {{variables}})"
    )
    enable_vector_search: bool = Field(
        default=False,
        description="Enable embedding-based vector search"
    )


class MemoryConfig(BaseModel):
    """Memory configuration (short-term and long-term)."""
    short_term: Optional[ShortTermMemoryConfig] = None
    long_term: Optional[LongTermMemoryConfig] = None


class MiddlewareConfig(BaseModel):
    """Individual middleware configuration."""
    type: str = Field(
        description="Middleware type (e.g., 'summarization', 'pii_detection')"
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Middleware-specific parameters"
    )
    enabled: bool = Field(
        default=True,
        description="Whether this middleware is enabled"
    )


class StreamingConfig(BaseModel):
    """Streaming configuration."""
    enabled: bool = Field(
        default=True,
        description="Whether streaming is enabled"
    )
    modes: List[Literal["updates", "messages", "custom"]] = Field(
        default=["updates"],
        description="Streaming modes to enable"
    )


class RuntimeContextField(BaseModel):
    """Runtime context field definition."""
    name: str = Field(description="Field name")
    type: str = Field(description="Python type as string (e.g., 'str', 'int')")
    required: bool = Field(default=True, description="Whether field is required")
    default: Optional[Any] = Field(default=None, description="Default value")


class RuntimeConfig(BaseModel):
    """Runtime configuration."""
    context_schema: Optional[List[RuntimeContextField]] = Field(
        default=None,
        description="Runtime context schema definition"
    )


class RAGConfig(BaseModel):
    """RAG (Retrieval Augmented Generation) configuration."""
    enabled: bool = Field(default=False, description="Whether RAG is enabled")
    pattern: Literal["2step", "agentic", "hybrid"] = Field(
        default="2step",
        description="RAG pattern to use"
    )
    vector_store: Literal["chroma", "pinecone", "in_memory"] = Field(
        default="chroma",
        description="Vector store to use"
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model identifier"
    )
    chunk_size: int = Field(
        default=1000,
        gt=0,
        description="Text chunk size"
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        description="Overlap between chunks"
    )
    top_k: int = Field(
        default=4,
        gt=0,
        description="Number of documents to retrieve"
    )
    documents_path: Optional[str] = Field(
        default=None,
        description="Path to documents directory"
    )


class MCPServerConfig(BaseModel):
    """MCP server configuration (inline in agent config - legacy/backwards compatible)."""
    name: str = Field(description="Server name")
    description: Optional[str] = Field(default=None, description="Server description")
    transport: Literal["stdio", "http", "sse", "streamable_http"] = Field(
        default="stdio",
        description="Transport type"
    )
    command: Optional[str] = Field(
        default=None,
        description="Command to run (for stdio transport)"
    )
    url: Optional[str] = Field(
        default=None,
        description="URL (for http/sse transports)"
    )
    args: Optional[List[str]] = Field(
        default=None,
        description="Command arguments"
    )
    env: Optional[Dict[str, str]] = Field(
        default=None,
        description="Environment variables"
    )
    stateful: bool = Field(
        default=False,
        description="Whether to use stateful sessions"
    )
    selected_tools: Optional[List[str]] = Field(
        default=None,
        description="List of tool names to include from this server. If None, all tools are included."
    )


class MCPServerDefinition(BaseModel):
    """Standalone MCP server definition for configs/mcp_servers/*.yaml files.

    This is the schema for MCP server configuration files that can be
    referenced by multiple agents.
    """
    name: str = Field(description="Unique server name (used as reference key)")
    description: Optional[str] = Field(default=None, description="Server description")
    transport: Literal["stdio", "http", "sse", "streamable_http"] = Field(
        default="stdio",
        description="Transport type"
    )
    command: Optional[str] = Field(
        default=None,
        description="Command to run (for stdio transport)"
    )
    url: Optional[str] = Field(
        default=None,
        description="URL (for http/sse transports)"
    )
    args: Optional[List[str]] = Field(
        default=None,
        description="Command arguments"
    )
    env: Optional[Dict[str, str]] = Field(
        default=None,
        description="Environment variables"
    )
    stateful: bool = Field(
        default=False,
        description="Whether to use stateful sessions"
    )
    selected_tools: Optional[List[str]] = Field(
        default=None,
        description="Default list of tool names to include. If None, all tools are included."
    )
    version: str = Field(
        default="1.0.0",
        description="Server config version"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for organization and filtering"
    )

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Server name cannot be empty")
        # Normalize name to valid identifier
        return v.strip().replace(' ', '_').lower()


class MCPServerReference(BaseModel):
    """Reference to an MCP server defined in configs/mcp_servers/.

    Used in agent configs to link to standalone MCP server definitions
    with optional per-agent tool filtering override.
    """
    ref: str = Field(description="Name of the MCP server to reference (must exist in configs/mcp_servers/)")
    selected_tools: Optional[List[str]] = Field(
        default=None,
        description="Override server's default tool selection. If None, uses server's default. "
                    "If specified, filters to only these tools (must be subset of server's available tools)."
    )

    @field_validator('ref')
    @classmethod
    def validate_ref(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Server reference cannot be empty")
        return v.strip().lower()


class OutputFormatterConfig(BaseModel):
    """Output formatter configuration."""
    enabled: bool = Field(default=False, description="Whether output formatting is enabled")
    schema_description: Optional[str] = Field(
        default=None,
        description="Natural language description of desired output structure"
    )
    pydantic_model: Optional[str] = Field(
        default=None,
        description="Generated Pydantic model code"
    )


class AgentConfig(BaseModel):
    """Complete agent configuration schema."""

    # Basic info
    name: str = Field(description="Agent name")
    version: str = Field(default="1.0.0", description="Agent version")
    description: Optional[str] = Field(default=None, description="Agent description")
    tags: List[str] = Field(default_factory=list, description="Tags for organization")

    # LLM configuration
    llm: LLMConfig = Field(description="LLM configuration")

    # Tools (before prompts to match UI flow)
    tools: List[str] = Field(
        default_factory=list,
        description="List of tool identifiers"
    )

    # MCP Servers - supports both inline configs and references
    mcp_servers: Optional[List[Union[MCPServerConfig, MCPServerReference]]] = Field(
        default=None,
        description="MCP server configurations. Can be inline MCPServerConfig or MCPServerReference (ref: server_name)"
    )

    # Prompts (after tools to match UI flow)
    prompts: PromptsConfig = Field(description="Prompt configuration")

    # RAG
    rag: Optional[RAGConfig] = Field(
        default=None,
        description="RAG configuration"
    )

    # Memory
    memory: Optional[MemoryConfig] = Field(
        default=None,
        description="Memory configuration"
    )

    # Middleware
    middleware: List[MiddlewareConfig] = Field(
        default_factory=list,
        description="Middleware configurations"
    )

    # Streaming
    streaming: Optional[StreamingConfig] = Field(
        default=None,
        description="Streaming configuration"
    )

    # Runtime
    runtime: Optional[RuntimeConfig] = Field(
        default=None,
        description="Runtime configuration"
    )

    # Output formatting
    output_formatter: Optional[OutputFormatterConfig] = Field(
        default=None,
        description="Output formatter configuration"
    )

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Agent name cannot be empty")
        # Convert to valid identifier
        return v.strip().replace(' ', '_').lower()

    class Config:
        extra = "forbid"  # Strict validation - no extra fields allowed
        json_schema_extra = {
            "example": {
                "name": "research_assistant",
                "version": "1.0.0",
                "description": "AI research assistant with web search",
                "tags": ["research", "web-search"],
                "llm": {
                    "provider": "openai",
                    "model": "gpt-4o",
                    "temperature": 0.7,
                    "max_tokens": 4096
                },
                "prompts": {
                    "system": "You are a helpful research assistant named {{agent_name}}."
                },
                "tools": ["tavily_search"],
                "memory": {
                    "short_term": {
                        "type": "sqlite",
                        "path": "./data/checkpoints/research_assistant.db"
                    }
                },
                "middleware": [
                    {
                        "type": "model_call_limit",
                        "params": {"thread_limit": 50}
                    }
                ],
                "streaming": {
                    "enabled": True,
                    "modes": ["updates", "messages"]
                }
            }
        }
