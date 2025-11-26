# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **LangChain 1.x Agent Builder** - a configuration-driven system for creating, managing, and deploying LangChain 1.0 agents. The project enables users to build agents entirely through YAML configurations without writing code, using LangChain's `create_agent` abstraction.

**Architecture**: Three separate applications
1. **Agent Builder API** (FastAPI) - Core backend service
2. **Agent Builder UI** (Streamlit) - 8-page configuration wizard for creating agents
3. **Agent UI** (Streamlit) - Full-featured chat interface for interacting with deployed agents

## Core Commands

### Development Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment (copy API keys from .env.template)
cp .env.template .env
# Edit .env with your API keys
```

### Running the Agent Builder API
```bash
# Run with Python module
python -m agent_api.main

# Run with uvicorn (hot reload)
uvicorn agent_api.main:app --reload --host 0.0.0.0 --port 8000
```

### Running the Agent Builder UI
```bash
# From the agent_builder directory
cd agent_builder
streamlit run app.py

# Or from the root directory
streamlit run agent_builder/app.py

# Access at: http://localhost:8501
```

**Agent Builder UI Features**:
- 8-page wizard for agent configuration
- Template selection (start from preset or blank)
- Live YAML preview on every page
- Form validation with inline errors
- API integration for validation and deployment
- Session state persistence across pages

**Page Flow**:
1. Basic Info - Name, description, version, tags
2. LLM Config - Provider, model, temperature, tokens
3. Tools - Built-in tool selection + MCP server selection with per-agent tool filtering
4. Prompts - System prompt, user template, variables
5. Memory - Short-term (checkpointer) and long-term (store) configuration
6. Middleware - Processing middleware with presets
7. Advanced - Streaming modes and runtime options
8. Deploy - Review, validate, and deploy agent

### Running the Agent UI
```bash
# From the agent_ui directory
cd agent_ui
streamlit run app.py

# Or from the root directory
streamlit run agent_ui/app.py

# Access at: http://localhost:8502 (different port from Builder UI)
```

**Agent UI Features**:
- Agent selection with search and filtering
- Real-time chat interface with streaming support
- Thread/session management (multiple conversations per agent)
- Context editor for runtime context values
- Message history with tool call visualization
- Export conversations (JSON, Markdown, CSV)
- WebSocket streaming with fallback to non-streaming mode
- Full UI customization (themes, preferences)

**Page Structure**:
1. **Main App** - Agent selection and overview
2. **Chat Page** - Interactive chat interface with real-time messaging
3. **Sessions Page** - Thread management, history, bulk export
4. **Settings Page** - API config, UI preferences, data management

**Key Features**:
- **Dual Mode Support**: Both streaming and non-streaming execution
- **Multi-Thread**: Manage multiple conversation threads per agent
- **Tool Visualization**: See tool calls with arguments and results
- **Context Management**: Edit runtime context based on agent's schema
- **Data Export**: Export individual threads or all conversations
- **Responsive UI**: Two-column layout with chat and agent info sidebar

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/unit/test_config_manager.py

# Run specific test by name
pytest tests/ -k "test_config_validation"

# Run with coverage
pytest tests/ --cov=agent_api --cov-report=html

# Run tests in verbose mode
pytest tests/ -v
```

### Code Quality
```bash
# Format code
black agent_api/ agent_builder/ agent_ui/

# Sort imports
isort agent_api/ agent_builder/ agent_ui/

# Check formatting without modifying
black agent_api/ agent_builder/ agent_ui/ --check
```

### Running All Applications
To run the complete system, start all three applications in separate terminal windows:

```bash
# Terminal 1: API (must start first)
python -m agent_api.main
# or: uvicorn agent_api.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Agent Builder UI
streamlit run agent_builder/app.py

# Terminal 3: Agent UI
streamlit run agent_ui/app.py
```

**Ports**:
- API: `http://localhost:8000` (docs at `/docs`)
- Builder UI: `http://localhost:8501`
- Agent UI: `http://localhost:8502`

### Debugging Tips

**Agent not working after deployment?**
- Check API logs for errors during agent creation
- Verify all required environment variables are set in `.env`
- Ensure memory database paths exist and are writable
- Check that tools specified in config are registered in `ToolRegistry`

**Config validation errors?**
- Use the `/agents/validate` endpoint to get detailed validation errors
- Check `config_schema.py` for required fields and constraints
- Verify YAML syntax is correct (proper indentation, no tabs)

**Streaming not working?**
- Ensure agent config has `streaming.enabled: true`
- Check WebSocket connection in browser console
- Verify agent is deployed before streaming

**MCP server not working?**
- Check server definition exists in `configs/mcp_servers/{name}.yaml`
- Verify transport type and URL/command are correct
- For HTTP transports: ensure MCP server is running at the configured URL
- For stdio transports: verify command and args are installed and accessible
- Check API logs for reference resolution errors during agent creation
- Use `/mcp-servers/{name}/discover-tools` endpoint to test connectivity

**Testing API endpoints directly:**
```bash
# List all agents
curl http://localhost:8000/agents/list

# Get agent details
curl http://localhost:8000/agents/{agent_id}

# Invoke agent (non-streaming)
curl -X POST http://localhost:8000/execution/{agent_id}/invoke \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'

# List available tools
curl http://localhost:8000/tools/list

# List MCP servers
curl http://localhost:8000/mcp-servers/list

# Get MCP server details
curl http://localhost:8000/mcp-servers/{server_name}

# Discover tools from MCP server
curl -X POST http://localhost:8000/mcp-servers/{server_name}/discover-tools
```

## Architecture Deep Dive

### Configuration Flow
The entire system is built around YAML configuration files that drive agent creation:

1. **User creates/edits YAML** → 2. **ConfigManager validates** → 3. **AgentFactory creates agent** → 4. **Agent deployed and cached**

### Key Service Dependencies

**AgentFactory** orchestrates the entire agent creation pipeline:
- Depends on: `ConfigManager`, `ToolRegistry`, `MiddlewareFactory`
- Creates: LLM instances, tools, checkpointers (memory), stores, middleware
- Calls: LangChain's `create_agent(**params)` with assembled components
- Caches: Created agents in `_agents` dict with metadata (config, store, timestamp)

**Dependency Injection Pattern**:
All services are instantiated in `agent_api/dependencies.py:AppState` (not main.py!) and injected via FastAPI's `Depends()`. This singleton pattern ensures:
- Single instance of each service across the application
- Shared caches (agents, tools) across all requests
- Centralized lifecycle management
- Avoids circular import issues by separating DI from main app

The `app_state` global instance in `dependencies.py` is the single source of truth for all service instances.

### Critical Design Patterns

**1. Configuration-Driven Everything**
- Agents, tools, middleware all defined in YAML
- Pydantic models (`agent_api/models/config_schema.py`) validate all configs
- No code changes needed to create new agents

**2. Factory Pattern**
- `AgentFactory`: Creates agents from configs
- `MiddlewareFactory`: Creates middleware from configs with registry pattern
- `ToolRegistry`: Manages built-in and custom tools
- `MCPServerManager`: Manages MCP server definitions and reference resolution

**3. Custom Tool Workflow**
Custom tools go through a **review-before-activation** flow:
1. User requests tool via natural language → LLM generates Python code
2. Code saved to `custom_tools/pending_review/{tool_id}.py`
3. Tool metadata stored in `ToolRegistry._pending_tools`
4. User reviews/tests/approves → Code moved to `custom_tools/{tool_id}.py`
5. Tool registered in `ToolRegistry._custom_tools` and available for agents

**4. Memory Architecture**
Two separate persistence layers:
- **Short-term** (Checkpointer): Conversation state, SQLite via `langgraph.checkpoint.sqlite.SqliteSaver`
- **Long-term** (Store): Cross-conversation facts, SQLite via `langgraph.store.sqlite.SqliteStore`

Both are created per-agent and passed to `create_agent()`.

**5. MCP Server Management**
MCP (Model Context Protocol) servers are managed as standalone configurations with reference-based linking:

**Directory Structure**:
- MCP server definitions: `configs/mcp_servers/*.yaml`
- Agent configs reference servers by name (foreign key pattern)

**Key Models** (`agent_api/models/config_schema.py`):
- `MCPServerDefinition`: Standalone server config (name, transport, url/command, etc.)
- `MCPServerReference`: Reference with optional per-agent tool override

**Reference Resolution Flow**:
1. Agent config specifies `mcp_servers: [{ref: "server_name", selected_tools: [...]}]`
2. `AgentFactory._resolve_mcp_servers()` resolves references via `MCPServerManager`
3. Per-agent `selected_tools` override server defaults
4. Resolved `MCPServerConfig` objects passed to tool creation

**Example MCP Server Definition** (`configs/mcp_servers/calculator.yaml`):
```yaml
name: calculator
version: 1.0.0
description: Mathematical operations via MCP
transport: streamable_http
url: http://localhost:8005/mcp
stateful: false
tags: []
```

**Example Agent Reference**:
```yaml
# In agent config
mcp_servers:
  - ref: calculator
  - ref: rag
    selected_tools: [search, retrieve]  # Override: only use these tools
```

**Migration Script**: `scripts/migrate_mcp_servers.py` extracts inline MCP configs from existing agent files to standalone definitions.

### LLM Provider Support

Uses `langchain.chat_models.init_chat_model()` for provider-agnostic initialization:
- API keys from environment or config override
- Provider parameter mapping in `AgentFactory.create_llm()`
- Supported: OpenAI, Anthropic, Google, Groq

**Key LangChain 1.x Imports:**
```python
from langchain.agents import create_agent  # Core agent creation
from langchain.chat_models import init_chat_model  # LLM initialization
from langchain_core.tools import BaseTool, tool  # Tool definitions
from langchain_core.language_models import BaseChatModel  # LLM interface
from langgraph.checkpoint.sqlite import SqliteSaver  # Short-term memory
from langgraph.store.sqlite import SqliteStore  # Long-term memory
from langgraph.store.memory import InMemoryStore  # In-memory store
```

### API Response Pattern

All API endpoints return standardized `APIResponse`:
```python
{
  "success": bool,
  "data": Any | None,
  "error": {"code": str, "message": str, "details": dict} | None,
  "timestamp": datetime,
  "request_id": str
}
```

Error handling uses custom exceptions (`AgentFactoryError`, `ConfigurationError`, `ToolRegistryError`) caught in routers and converted to error responses.

### Streaming Architecture

Two execution modes:
1. **Non-streaming** (`/execution/{agent_id}/invoke`): Returns complete response
2. **Streaming** (`/execution/{agent_id}/stream` via WebSocket):
   - Agent configuration specifies modes: `updates`, `messages`, `custom`
   - Uses LangGraph's `.stream()` method
   - Chunks sent as JSON over WebSocket

## Important Implementation Details

### Variable Interpolation
System prompts support `{{variable}}` placeholders:
- Interpolated in `ConfigManager.get_system_prompt()`
- Default variables: `{{agent_name}}`, `{{date}}`, `{{time}}`
- Runtime context variables from request

### Agent Caching
Agents are created once and cached in `AgentFactory._agents`:
- Key: `agent_name` (from config)
- Value: `{"agent": agent_instance, "config": AgentConfig, "store": Store, "created_at": datetime}`
- Deploy = create + cache, Undeploy = remove from cache
- Redeploy = remove + reload config + create + cache

### Middleware Order Matters
Middleware executes in the order defined in YAML config. The `MiddlewareFactory.create_middleware_list()` maintains this order when passing to `create_agent()`.

### Middleware Performance Considerations

**CRITICAL**: Some middleware types add LLM call overhead. Choose middleware carefully based on your agent's use case.

#### Middleware Performance Impact

| Middleware | LLM Calls Added | When to Use | When to Avoid |
|------------|-----------------|-------------|---------------|
| **llm_tool_selector** | +1 per request | Agents with 5+ tools (auto-enabled) | Agents with <5 tools |
| **summarization** | +1 when triggered | Long conversations (>100K tokens) | Short sessions |
| **model_call_limit** | 0 (tracking only) | All agents (safety) | Never avoid |
| **tool_retry** | 0 (retries only) | Agents with unreliable tools | Stable tools |
| **todo_list** | +1 per request cycle | Complex multi-step workflows (5+ steps) | **Simple Q&A, math, search** |
| **pii_detection** | 0 (regex-based) | Customer support, data handling | Internal tools |
| **model_fallback** | 0 (only on failure) | Production agents | Development |
| **human_in_the_loop** | 0 (pauses execution) | High-risk operations | Autonomous agents |

#### TodoListMiddleware: Use Case Guidelines

**⚠️ PERFORMANCE WARNING**: TodoListMiddleware adds ~200 tokens of system prompt overhead on EVERY request and can cause an extra LLM call even for simple queries.

**Use TodoListMiddleware for:**
- ✅ Project management agents
- ✅ Multi-step workflow automation (5+ steps)
- ✅ Task planning and tracking systems
- ✅ Complex research projects with subtasks

**DO NOT use TodoListMiddleware for:**
- ❌ Simple Q&A agents
- ❌ Math/calculation agents
- ❌ Web search agents
- ❌ Single-purpose tools
- ❌ Any agent handling queries in <3 steps

**Example Performance Impact:**
```yaml
# Query: "What's 15% of 345?"
# WITHOUT todo_list: 3 LLM calls (selector + planning + result)
# WITH todo_list: 4 LLM calls (selector + planning + result + todo evaluation)
# Cost increase: +33% per query
```

**Recommendation**: Start without TodoListMiddleware. Add it only if users explicitly request task planning features or your agent handles genuinely complex multi-step workflows.

#### Choosing the Right Middleware

**For Simple Agents** (Q&A, search, math):
```yaml
middleware:
  - type: model_call_limit
    params: {thread_limit: 50}
  - type: tool_retry  # Only if using unreliable external tools
```

**For Multi-Tool Agents** (5+ tools, MCP servers):
```yaml
middleware:
  - type: llm_tool_selector  # Auto-enabled for 5+ tools
    params: {model: openai:gpt-4o-mini, max_tools: 7}
  - type: tool_retry
  - type: model_call_limit
```

**For Complex Workflow Agents** (task planning, project management):
```yaml
middleware:
  - type: todo_list  # Only for genuinely complex workflows
  - type: model_call_limit
    params: {thread_limit: 100, run_limit: 30}  # Higher limits for multi-step tasks
```

**For Production Agents** (customer-facing, high volume):
```yaml
middleware:
  - type: llm_tool_selector  # If many tools
  - type: pii_detection
    params: {strategy: redact}
  - type: model_fallback
    params: {fallback_models: ["gpt-4o-mini", "gpt-3.5-turbo"]}
  - type: model_call_limit
  - type: summarization  # For long conversations
```

### Configuration Validation Strategy
Two-phase validation:
1. **Pydantic schema validation** (structure, types, constraints)
2. **Business logic validation** (e.g., SQLite requires path, custom warnings)

Validation happens in `ConfigManager.validate_config_dict()` and returns structured errors.

### Tool Selection Optimization

**Problem**: Agents with many tools (especially MCP servers) suffer from:
- Token waste (all tool definitions sent every request)
- Tool confusion (LLM struggles with 10+ tool choices)
- Higher costs and slower responses

**Solution**: Three-layer optimization approach:

#### Layer 1: LLMToolSelectorMiddleware
Intelligent tool filtering before the main request:
```yaml
middleware:
  - type: llm_tool_selector
    params:
      model: openai:gpt-4o-mini  # Fast, cheap selector
      max_tools: 7                # ~1/3 of total, between 5-8
      always_include: []
    enabled: true
```

**Results**: 85-90% reduction in tool definition tokens (18 tools → 7 selected tools).

#### Layer 2: Tool Categorization
All tools are automatically categorized:
- **File**: `agent_api/services/tool_registry.py` - `ToolCategory` class + `infer_tool_category()`
- **Categories**: computation, search, retrieval, code_execution, utility, data_processing
- **MCP tools**: Auto-categorized using keyword patterns
- **UI**: Tools grouped by category in Builder UI (page 3)

#### Layer 3: Prompt Engineering Patterns
System prompts should include:
- **Tool categories** with clear "when to use" guidance
- **Decision tree** (e.g., "Math? → Computation tools")
- **Concrete examples** for each category
- **Critical rules** emphasized (e.g., "MUST use calculator for ALL math")

**Dynamic Documentation**:
```python
# agent_api/services/prompt_helper.py
tool_docs = PromptHelper.generate_complete_tool_section(
    tools=agent_tools,
    tool_registry=tool_registry,
    include_examples=True,
    emphasize_accuracy=True
)
```

**Middleware Recommendations**:
```python
# agent_api/services/middleware_factory.py
recommended = middleware_factory.recommend_middleware_for_agent(
    total_tool_count=18,
    use_mcp=True,
    priority="accuracy"  # or "cost", "balanced"
)
# Auto-enables llm_tool_selector for agents with 5+ tools
```

**Presets**:
- `multi_tool_optimized`: Pre-configured for agents with many tools
- Available in: `MiddlewareFactory.get_middleware_presets()`
- UI constants: `agent_builder/utils/constants.py::MIDDLEWARE_PRESETS`

**Templates**:
- `configs/templates/multi_tool_assistant.yaml`: Reference implementation
- `docs/TOOL_SELECTION_GUIDE.md`: Comprehensive user guide

**Key Files**:
- `agent_api/services/tool_registry.py`: Tool categorization system
- `agent_api/services/prompt_helper.py`: Dynamic tool documentation
- `agent_api/services/middleware_factory.py`: Smart presets + recommendations
- `agent_builder/pages/3_🔧_Tools.py`: Category-based tool selection UI

**Auto-Default Behavior** (NEW - `agent_factory.py:417-444`):
The `AgentFactory` now **automatically enables** `llm_tool_selector` middleware for agents with 5+ tools:

```python
# In create_agent_from_config()
total_tools = len(tools)
if total_tools >= 5:
    has_selector = any(m.type == "llm_tool_selector" for m in config.middleware)
    if not has_selector:
        max_tools = min(8, max(5, total_tools // 3))  # Smart sizing
        selector_config = MiddlewareConfig(...)
        config.middleware.insert(0, selector_config)  # Insert at beginning
        logger.info(f"Auto-enabled llm_tool_selector: {total_tools} tools → max_tools={max_tools}")
```

**Behavior**:
- ✅ **Automatic**: Agents with 5+ tools get llm_tool_selector without configuration
- ✅ **Smart**: Calculates optimal `max_tools` (1/3 of total, between 5-8)
- ✅ **Non-intrusive**: Only adds if user hasn't configured it
- ✅ **Override**: Users can still configure manually to customize or disable

**When to Use**:
- Agents with 5+ built-in tools: **AUTO-ENABLED** ✅
- Agents using MCP servers: **AUTO-ENABLED** ✅ (MCP servers add 5-20 tools each)
- Agents with <5 tools: Not auto-enabled (not needed)

## File Organization Logic

```
agent_api/
├── main.py              # FastAPI app, dependency injection, lifespan
├── models/
│   ├── config_schema.py # Pydantic models for YAML configs (single source of truth)
│   └── schemas.py       # Pydantic models for API requests/responses
├── services/            # Business logic (no FastAPI dependencies)
│   ├── config_manager.py    # YAML load/save/validate
│   ├── agent_factory.py     # create_agent wrapper + caching
│   ├── tool_registry.py     # Tool management + LLM generation
│   ├── middleware_factory.py # Middleware instantiation + registry
│   └── mcp_server_manager.py # MCP server config management + reference resolution
└── routers/             # FastAPI route handlers (thin layer)
    ├── agents.py        # CRUD for agent configs
    ├── execution.py     # Agent invocation + streaming
    ├── tools.py         # Tool generation + approval workflow
    └── mcp_servers.py   # MCP server CRUD + tool discovery

agent_builder/
├── app.py               # Main entry point, template selector, navigation grid
├── pages/               # Multi-page Streamlit app (8 pages)
│   ├── 1_📝_Basic_Info.py      # Name, description, version, tags
│   ├── 2_🤖_LLM_Config.py      # LLM provider, model, parameters
│   ├── 3_🔧_Tools.py           # Built-in tools + MCP server selection
│   ├── 4_💬_Prompts.py         # System prompt, user template
│   ├── 5_🧠_Memory.py          # Short-term and long-term memory config
│   ├── 6_⚙️_Middleware.py      # Middleware selection with presets
│   ├── 7_🚀_Advanced.py        # Streaming, runtime context
│   └── 8_✅_Deploy.py          # Review, validate, deploy agent
├── utils/               # Shared utilities
│   ├── constants.py     # LLM providers, middleware types, defaults
│   ├── state_manager.py # Session state management across pages
│   ├── api_client.py    # HTTP client for agent_api communication
│   ├── validators.py    # Form validation logic
│   └── yaml_generator.py # Generate YAML from session state
└── components/          # Reusable UI components
    ├── yaml_preview.py  # Live YAML preview with download
    ├── navigation.py    # Page header, progress tracking
    └── template_selector.py # Template selection UI

agent_ui/
├── app.py               # Main entry point, agent selection
├── pages/               # Multi-page Streamlit app (3 pages)
│   ├── 1_💬_Chat.py          # Chat interface with messaging
│   ├── 2_📊_Sessions.py      # Thread/session management, history
│   └── 3_⚙️_Settings.py      # UI preferences, API config, data management
├── utils/               # Shared utilities
│   ├── api_client.py    # HTTP client with execution methods
│   ├── websocket_client.py # WebSocket client for streaming
│   ├── state_manager.py # Chat session state, threads
│   ├── message_formatter.py # Message display formatting
│   └── export_utils.py  # Export conversations (JSON/MD/CSV)
└── components/          # Reusable UI components
    ├── agent_selector.py # Agent selection with filters
    ├── chat_interface.py # Chat container and input
    ├── message_renderer.py # Individual message display
    ├── streaming_handler.py # WebSocket streaming logic
    ├── thread_panel.py  # Thread list sidebar
    └── context_editor.py # Runtime context editor
```

**Separation of concerns**:
- **agent_api/services/** = pure Python, testable in isolation
- **agent_api/routers/** = FastAPI-specific, handles HTTP/WebSocket, calls services
- **agent_api/models/** = data contracts, shared across layers
- **agent_builder/pages/** = Streamlit page implementations, form handling
- **agent_builder/utils/** = business logic, state management, API integration
- **agent_builder/components/** = reusable UI components
- **agent_ui/pages/** = Chat pages (Chat, Sessions, Settings)
- **agent_ui/utils/** = Chat-specific utilities (messaging, export, WebSocket)
- **agent_ui/components/** = Chat UI components (messages, threads, streaming)

## Configuration File Locations

- **Templates**: `configs/templates/*.yaml` - Pre-built agent examples
- **User agents**: `configs/agents/*.yaml` - User-created configurations
- **MCP servers**: `configs/mcp_servers/*.yaml` - Standalone MCP server definitions
- **Custom tools**: `custom_tools/*.py` (approved), `custom_tools/pending_review/*.py` (pending)
- **Data storage**: `data/checkpoints/` (short-term), `data/store/` (long-term)

## Development Workflow for New Features

### Adding a New Middleware Type
1. Update `MiddlewareFactory._build_middleware_registry()` with import and registration
2. Add metadata to `MiddlewareFactory.list_available_middleware()`
3. Add validation in `MiddlewareFactory.validate_middleware_config()`
4. Optionally add to presets in `get_middleware_presets()`
5. Update `config_schema.py` if new config fields needed

### Adding a New LLM Provider
1. Update `LLMConfig.provider` enum in `config_schema.py`
2. Add API key mapping in `AgentFactory.create_llm()` (env var name)
3. Add to `.env.template` with placeholder
4. Test with `init_chat_model(model_provider="new_provider")`
5. Update agent_builder UI constants if adding to Builder UI

### Adding a Built-in Tool
1. Register in `ToolRegistry._register_builtin_tools()`
2. Add to `_builtin_tools` dict with tool instance
3. Tool becomes available via `tools: [tool_id]` in YAML
4. Update agent_builder UI if the tool should appear in Builder UI tool selection

### Adding a New API Endpoint
1. Create handler in appropriate router file (`routers/agents.py`, `routers/execution.py`, or `routers/tools.py`)
2. Use dependency injection: `service: ServiceClass = Depends(get_service)`
3. Return `APIResponse` with success/error structure
4. Add request/response models to `models/schemas.py`
5. Document endpoint behavior with docstring
6. Include router in `main.py` if it's a new router file

### Adding a New MCP Server
1. Create YAML file in `configs/mcp_servers/{server_name}.yaml`
2. Define server config following `MCPServerDefinition` schema:
   ```yaml
   name: my_server
   version: 1.0.0
   description: Description of the server
   transport: streamable_http  # or stdio, http, sse
   url: http://localhost:8005/mcp  # for HTTP transports
   # command: npx  # for stdio transport
   # args: ["-y", "@some/mcp-server"]  # for stdio transport
   stateful: false
   tags: []
   ```
3. Server becomes available in Agent Builder UI (Tools page) for selection
4. Reference in agent configs: `mcp_servers: [{ref: "my_server"}]`
5. Optionally filter tools per-agent: `{ref: "my_server", selected_tools: ["tool1", "tool2"]}`

### Adding a New Streamlit UI Page
**For Agent Builder UI:**
1. Create page file as `agent_builder/pages/N_emoji_PageName.py` (N = page number)
2. Initialize session state using `state_manager.py` helpers
3. Add YAML preview using `yaml_preview.py` component
4. Call API validation before allowing progression
5. Update navigation in `app.py` if needed

**For Agent UI:**
1. Create page file as `agent_ui/pages/N_emoji_PageName.py`
2. Use `state_manager.py` for session/thread management
3. Handle both streaming and non-streaming modes
4. Add error handling with user-friendly messages

## Testing Strategy

- **Unit tests**: Mock external dependencies (LLMs, file I/O), test services in isolation
- **Integration tests**: Use `pytest-recording` to record/replay LLM API calls
- **AgentEvals**: For trajectory testing (planned, not yet implemented)

When testing components that use LLMs, use `GenericFakeChatModel` for mocking or recorded responses to avoid API costs.

## Current Development Status

**Phase 1 COMPLETE**: Core API foundation with all CRUD operations, tool management, and execution endpoints.

**Phase 2 COMPLETE**: Agent Builder UI (Streamlit) - 8-page wizard for creating agents through an intuitive interface with template selection, live YAML preview, form validation, and API integration.

**Phase 3 COMPLETE**: Agent UI (Streamlit) - Full-featured chat interface with agent selection, real-time messaging (streaming and non-streaming), thread management, context editor, tool visualization, and conversation export.

**Phase 4 COMPLETE**: MCP Server Management - Standalone MCP server definitions in `configs/mcp_servers/`, reference-based linking in agent configs, per-agent tool filtering override, API endpoints for CRUD and tool discovery, Agent Builder UI integration for server selection, and migration script for existing configs.

**Not yet implemented**:
- RAG integration (config schema exists, implementation pending)
- Output formatters (Pydantic model generation)
- AgentEvals integration
- Docker deployment
- Custom tool generation via LLM (API endpoints exist, LLM integration pending)

When implementing these features, follow the established patterns:
- Services in `agent_api/services/` with factory pattern
- Pydantic models for all configuration
- YAML-driven whenever possible
- Proper error handling with custom exceptions
