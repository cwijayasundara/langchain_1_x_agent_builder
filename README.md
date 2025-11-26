# LangChain 1.x Agent Builder

A comprehensive agent configurator system for creating, managing, and deploying LangChain 1.0 agents through configuration files and intuitive UIs.

## Features

- **Configuration-Driven Agent Creation**: Define agents using simple YAML files
- **Multi-LLM Support**: OpenAI, Anthropic, Google Gemini, and Groq
- **Built-in Tools**: Web search (Tavily), code execution, and more
- **Tool Categorization**: Automatic categorization of tools (computation, search, retrieval, etc.)
- **Smart Tool Selection**: Auto-enabled tool filtering for agents with 5+ tools (85-90% token reduction)
- **Custom Tool Generation**: Create tools from natural language descriptions using LLMs
- **Memory Management**: Short-term (checkpointer) and long-term (store) memory
- **Middleware System**: Summarization, PII detection, rate limiting, retries, tool selection, and more
- **Streaming Support**: Real-time token streaming via WebSocket
- **REST API**: Complete FastAPI-based API for agent management
- **Interactive UIs**: Separate interfaces for building and interacting with agents

## Architecture

The system consists of three main components:

1. **Agent Builder API** (FastAPI): Core backend for agent creation and management
2. **Agent Builder UI** (Streamlit): Configuration interface for designing agents
3. **Agent UI** (Streamlit): Runtime interface for interacting with deployed agents

## Quick Start

### Prerequisites

- Python 3.10 or higher
- API keys for your preferred LLM provider(s)
- Tavily API key (for web search)

### Installation

1. **Clone the repository**:
```bash
cd langchain_1_x_agent_builder
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**:
```bash
cp .env.template .env
```

Edit `.env` and add your API keys:
```env
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GOOGLE_API_KEY=your_google_key_here
GROQ_API_KEY=your_groq_key_here
TAVILY_API_KEY=your_tavily_key_here
```

### Running the API

Start the FastAPI server:

```bash
python -m agent_api.main
```

Or with uvicorn directly:

```bash
uvicorn agent_api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- API: http://localhost:8000
- Interactive docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

### Running the Agent Builder UI

The Agent Builder UI is an 8-page wizard for configuring agents through an intuitive interface.

Start the Agent Builder UI:

```bash
streamlit run agent_builder/app.py
```

The UI will be available at http://localhost:8501

**Features**:
- Template selection (start from preset or blank)
- Live YAML preview on every page
- Form validation with inline errors
- Direct integration with the Agent Builder API
- Session state persistence across pages

**Page Flow**:
1. **Basic Info** - Agent name, description, version, tags
2. **LLM Config** - LLM provider, model, temperature, tokens
3. **Prompts** - System prompt, user template, variables
4. **Tools** - Built-in tool selection
5. **Memory** - Short-term and long-term memory configuration
6. **Middleware** - Processing middleware with presets
7. **Advanced** - Streaming modes and runtime options
8. **Deploy** - Review, validate, and deploy your agent

### Running the Agent UI

The Agent UI is a full-featured chat interface for interacting with your deployed agents.

Start the Agent UI:

```bash
streamlit run agent_ui/app.py
```

The UI will be available at http://localhost:8502

**Features**:
- **Agent Selection**: Browse and select from deployed agents with search and filtering
- **Real-time Chat**: Interactive chat interface with support for both streaming and non-streaming modes
- **Thread Management**: Create and manage multiple conversation threads per agent
- **Context Editor**: Configure runtime context values based on agent's schema
- **Tool Visualization**: See tool calls with arguments and results in expandable sections
- **Message Export**: Export conversations to JSON, Markdown, or CSV formats
- **Session History**: View and manage all your conversation sessions
- **Customization**: Configure UI preferences, themes, and behavior

**Pages**:
1. **Main App** - Agent selection with detailed info cards
2. **Chat** - Interactive chat interface with message history and input
3. **Sessions** - Thread management, history browsing, bulk export
4. **Settings** - API configuration, UI preferences, data management

**Dual Mode Support**:
- **Non-streaming**: Traditional request/response with loading indicator
- **Streaming**: Real-time token-by-token display via WebSocket (when agent supports it)

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
- API: http://localhost:8000 (docs at `/docs`)
- Builder UI: http://localhost:8501
- Agent UI: http://localhost:8502

### Creating Your First Agent

#### Using the API

1. **Create an agent configuration** (`my_agent.yaml`):

```yaml
name: my_first_agent
version: 1.0.0
description: My first LangChain agent

llm:
  provider: openai
  model: gpt-4o
  temperature: 0.7

prompts:
  system: |
    You are a helpful AI assistant named {{agent_name}}.
    Today's date is {{date}}.

tools:
  - tavily_search

memory:
  short_term:
    type: sqlite
    path: ./data/checkpoints/my_first_agent.db

streaming:
  enabled: true
  modes:
    - updates
```

2. **Create the agent via API**:

```bash
curl -X POST "http://localhost:8000/agents/create" \
  -H "Content-Type: application/json" \
  -d @my_agent.json
```

#### Using Configuration Templates

Use the provided templates as starting points:

```bash
# List available templates
curl http://localhost:8000/agents/templates/list

# Get a specific template
curl http://localhost:8000/agents/templates/research_assistant
```

Available templates:
- `research_assistant`: Web search and analysis capabilities
- `customer_support`: Friendly customer service agent

### Interacting with Agents

#### REST API (Non-Streaming)

```bash
curl -X POST "http://localhost:8000/execution/my_first_agent/invoke" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is LangChain?"}
    ]
  }'
```

#### WebSocket (Streaming)

```python
import asyncio
import websockets
import json

async def chat_with_agent():
    uri = "ws://localhost:8000/execution/my_first_agent/stream"

    async with websockets.connect(uri) as websocket:
        # Send message
        await websocket.send(json.dumps({
            "messages": [
                {"role": "user", "content": "Hello!"}
            ]
        }))

        # Receive streaming responses
        while True:
            response = await websocket.recv()
            data = json.loads(response)

            if data.get("type") == "complete":
                break

            print(data)

asyncio.run(chat_with_agent())
```

## Configuration Reference

### Agent Configuration Schema

```yaml
# Basic Information
name: string                    # Required: Agent identifier
version: string                 # Default: "1.0.0"
description: string             # Optional: Agent description
tags: list[string]              # Optional: Tags for organization

# LLM Configuration
llm:
  provider: openai|anthropic|google|groq  # Required
  model: string                           # Required: Model identifier
  temperature: float                      # Default: 0.7 (0.0-2.0)
  max_tokens: int                         # Default: 4096
  top_p: float                            # Optional: (0.0-1.0)
  api_key: string                         # Optional: Override env var

# Prompts
prompts:
  system: string                          # Required: System prompt
  user_template: string                   # Optional: User message template
  few_shot_examples: list[dict]           # Optional: Example conversations

# Tools
tools: list[string]                       # List of tool identifiers

# MCP Servers (Optional)
mcp_servers:
  - name: string
    transport: stdio|http|sse
    command: string                       # For stdio
    url: string                           # For http/sse
    stateful: bool

# Memory (Optional)
memory:
  short_term:
    type: in_memory|sqlite
    path: string                          # Required for sqlite
    custom_state: dict                    # Custom state fields
    message_management: trim|summarize|none

  long_term:
    type: in_memory|sqlite
    path: string                          # Required for sqlite
    namespaces: list[string]
    enable_vector_search: bool

# Middleware (Optional)
middleware:
  - type: string                          # Middleware type
    params: dict                          # Middleware parameters
    enabled: bool                         # Default: true

# Streaming (Optional)
streaming:
  enabled: bool                           # Default: true
  modes: list[updates|messages|custom]    # Default: ["updates"]

# Runtime (Optional)
runtime:
  context_schema:
    - name: string
      type: string
      required: bool
      default: any
```

### Available Middleware Types

| Type | Description | Required Params | LLM Overhead |
|------|-------------|-----------------|--------------|
| `llm_tool_selector` | Intelligent tool filtering for agents with many tools | `model`, `max_tools` | +1 call/request |
| `summarization` | Auto-compress conversation history | `model`, `max_tokens_before_summary` | +1 when triggered |
| `model_call_limit` | Restrict model invocations | `thread_limit` or `run_limit` | None |
| `tool_call_limit` | Control tool execution counts | `thread_limit` or `run_limit` | None |
| `pii_detection` | Detect/handle PII | `strategy` (block/redact/mask/hash) | None |
| `model_fallback` | Switch models on failure | `fallback_models` | None (on failure only) |
| `tool_retry` | Retry failed tools | `max_retries`, `initial_delay` | None |
| `human_in_the_loop` | Require human approval | Optional | None |
| `todo_list` | Task planning for complex multi-step workflows | Optional | +1 call/request |
| `anthropic_prompt_caching` | Cache prompts (Anthropic) | Optional | None |

### Tool Selection Optimization

For agents with many tools (especially when using MCP servers), the system provides automatic optimization:

**Auto-Enabled Behavior**: When an agent has 5+ tools, the `llm_tool_selector` middleware is automatically enabled. This uses a fast, cheap model (e.g., `gpt-4o-mini`) to filter tools before the main request, reducing token usage by 85-90%.

```yaml
# Auto-configured for agents with 5+ tools
middleware:
  - type: llm_tool_selector
    params:
      model: openai:gpt-4o-mini  # Fast selector model
      max_tools: 7               # ~1/3 of total tools
```

**When to Use**:
- Agents with 5+ built-in tools: **Auto-enabled**
- Agents using MCP servers: **Auto-enabled** (MCP servers typically add 5-20 tools each)
- Agents with <5 tools: Not needed

### Middleware Performance Considerations

Choose middleware carefully based on your agent's use case:

**For Simple Agents** (Q&A, search, math):
```yaml
middleware:
  - type: model_call_limit
    params: {thread_limit: 50}
```

**For Multi-Tool Agents** (5+ tools):
```yaml
middleware:
  - type: llm_tool_selector  # Auto-enabled
    params: {model: openai:gpt-4o-mini, max_tools: 7}
  - type: tool_retry
  - type: model_call_limit
```

**For Production Agents**:
```yaml
middleware:
  - type: llm_tool_selector
  - type: pii_detection
    params: {strategy: redact}
  - type: model_fallback
    params: {fallback_models: ["gpt-4o-mini", "gpt-3.5-turbo"]}
  - type: model_call_limit
```

> ⚠️ **Note**: The `todo_list` middleware adds overhead on every request. Only use it for agents handling genuinely complex multi-step workflows (5+ steps), not for simple Q&A or search agents.

## API Reference

### Agent Management

- `POST /agents/create` - Create a new agent
- `GET /agents/list` - List all agents
- `GET /agents/{agent_id}` - Get agent details
- `PUT /agents/{agent_id}` - Update agent configuration
- `DELETE /agents/{agent_id}` - Delete agent
- `POST /agents/validate` - Validate configuration
- `GET /agents/templates/list` - List templates
- `GET /agents/templates/{template_id}` - Get template

### Agent Execution

- `POST /execution/{agent_id}/invoke` - Invoke agent (non-streaming)
- `WS /execution/{agent_id}/stream` - Stream agent execution (WebSocket)
- `POST /execution/{agent_id}/deploy` - Deploy agent
- `POST /execution/{agent_id}/undeploy` - Undeploy agent

### Tool Management

- `GET /tools/list` - List all tools
- `GET /tools/{tool_id}` - Get tool details
- `POST /tools/generate` - Generate custom tool from description
- `POST /tools/{tool_id}/approve` - Approve/reject pending tool
- `POST /tools/{tool_id}/test` - Test tool with input
- `DELETE /tools/{tool_id}` - Delete custom tool

## Custom Tools

### Generating Tools

Generate custom tools using natural language:

```bash
curl -X POST "http://localhost:8000/tools/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "description": "A tool that converts temperatures between Celsius and Fahrenheit",
    "name": "temperature_converter"
  }'
```

### Approving Tools

1. **Review generated code**:
```bash
curl http://localhost:8000/tools/temperature_converter
```

2. **Test the tool**:
```bash
curl -X POST "http://localhost:8000/tools/temperature_converter/test" \
  -H "Content-Type: application/json" \
  -d '{
    "test_input": {"celsius": 25}
  }'
```

3. **Approve or reject**:
```bash
curl -X POST "http://localhost:8000/tools/temperature_converter/approve" \
  -H "Content-Type: application/json" \
  -d '{
    "tool_id": "temperature_converter",
    "approved": true
  }'
```

## Project Structure

```
langchain_1_x_agent_builder/
├── agent_api/               # FastAPI backend
│   ├── main.py              # Application entry point
│   ├── dependencies.py      # Dependency injection (AppState singleton)
│   ├── routers/             # API route handlers
│   ├── services/            # Business logic
│   │   ├── agent_factory.py     # Agent creation with auto-optimization
│   │   ├── config_manager.py    # YAML config load/save/validate
│   │   ├── tool_registry.py     # Tool management + categorization
│   │   ├── middleware_factory.py # Middleware creation + presets
│   │   └── prompt_helper.py     # Dynamic tool documentation
│   └── models/              # Pydantic schemas
├── agent_builder/           # Streamlit configuration UI ✅
│   ├── app.py               # Main entry point
│   ├── pages/               # 8-page wizard
│   ├── utils/               # State management, API client, validators
│   └── components/          # Reusable UI components
├── agent_ui/                # Streamlit chat interface ✅
│   ├── app.py               # Agent selection
│   ├── pages/               # Chat, Sessions, Settings
│   ├── utils/               # Chat utilities, WebSocket, export
│   └── components/          # Chat components, message rendering
├── configs/
│   ├── templates/           # Pre-built agent templates
│   └── agents/              # User-created agents
├── data/
│   ├── checkpoints/         # Short-term memory storage
│   └── store/               # Long-term memory storage
├── custom_tools/            # Custom tool code
│   └── pending_review/      # Tools awaiting approval
└── tests/                   # Test suites
```

## Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Run tests
pytest tests/

# With coverage
pytest tests/ --cov=agent_api --cov-report=html
```

### Code Quality

```bash
# Format code
black agent_api/ agent_builder/ agent_ui/

# Sort imports
isort agent_api/ agent_builder/ agent_ui/
```

## Troubleshooting

### Agent not working after deployment?
- Check API logs for errors during agent creation
- Verify all required environment variables are set in `.env`
- Ensure memory database paths exist and are writable
- Check that tools specified in config are registered in `ToolRegistry`

### Config validation errors?
- Use the `/agents/validate` endpoint to get detailed validation errors
- Check `config_schema.py` for required fields and constraints
- Verify YAML syntax is correct (proper indentation, no tabs)

### Streaming not working?
- Ensure agent config has `streaming.enabled: true`
- Check WebSocket connection in browser console
- Verify agent is deployed before streaming

### Testing API endpoints directly

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
```

## Roadmap

### Phase 1: Core Foundation ✅
- [x] Project structure
- [x] Configuration schema
- [x] Agent factory with create_agent
- [x] Multi-LLM support
- [x] Tool registry
- [x] Middleware factory
- [x] FastAPI endpoints
- [x] Example configurations

### Phase 2: Agent Builder UI ✅
- [x] 8-page configuration wizard
- [x] Template selection system
- [x] Live YAML preview
- [x] Form validation
- [x] API integration for deployment
- [x] Session state management

### Phase 3: Agent UI ✅
- [x] Agent selection with search and filtering
- [x] Real-time chat interface
- [x] Streaming and non-streaming modes
- [x] Thread/session management
- [x] Context editor for runtime values
- [x] Message export (JSON, Markdown, CSV)
- [x] Tool call visualization
- [x] UI preferences and settings

### Phase 4: Advanced Features (Current)
- [ ] Memory management UI enhancements
- [ ] RAG integration
- [ ] MCP server support
- [ ] Output formatters
- [ ] Testing framework with AgentEvals
- [ ] Custom tool builder UI
- [ ] Template marketplace
- [ ] Multi-agent orchestration

### Phase 5: Production Ready
- [ ] Docker deployment
- [ ] LangSmith integration
- [ ] Performance optimization
- [ ] Comprehensive documentation

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

[Add your license here]

## Support

For questions and support, please [open an issue](https://github.com/your-repo/issues).

## Acknowledgments

Built with:
- [LangChain](https://python.langchain.com/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Streamlit](https://streamlit.io/)
- [Pydantic](https://docs.pydantic.dev/)
