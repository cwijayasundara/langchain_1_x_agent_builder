"""
Constants and configuration values for the agent builder UI.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any


def _get_default_providers() -> Dict[str, Any]:
    """
    Fallback hardcoded LLM providers.
    Used if YAML config file is not found or invalid.
    """
    return {
        "openai": {
            "name": "OpenAI",
            "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
            "env_key": "OPENAI_API_KEY"
        },
        "anthropic": {
            "name": "Anthropic",
            "models": ["claude-sonnet-4-5-20250929", "claude-haiku-4-5-20251001", "claude-opus-4-1-20250805"],
            "env_key": "ANTHROPIC_API_KEY"
        },
        "google": {
            "name": "Google",
            "models": ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"],
            "env_key": "GOOGLE_API_KEY"
        },
        "groq": {
            "name": "Groq",
            "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "groq/compound"],
            "env_key": "GROQ_API_KEY"
        }
    }


def load_llm_providers() -> Dict[str, Any]:
    """
    Load LLM providers from YAML configuration file.

    Returns:
        Dictionary of LLM providers in the format:
        {
            "provider_id": {
                "name": "Display Name",
                "models": ["model1", "model2", ...],
                "env_key": "ENV_VAR_NAME"
            }
        }
    """
    # Path to YAML config: agent_builder/utils/../../../configs/llm_providers.yaml
    config_path = Path(__file__).parent.parent.parent / "configs" / "llm_providers.yaml"

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        if not data or 'providers' not in data:
            print(f"Warning: Invalid structure in {config_path}, using defaults")
            return _get_default_providers()

        # Transform YAML structure to match existing dictionary format
        providers = {}
        for provider_id, provider_data in data.get('providers', {}).items():
            providers[provider_id] = {
                "name": provider_data.get('name', provider_id),
                "env_key": provider_data.get('env_key', f"{provider_id.upper()}_API_KEY"),
                "models": [model['id'] for model in provider_data.get('models', [])]
            }

        return providers

    except FileNotFoundError:
        print(f"Warning: {config_path} not found, using hardcoded providers")
        return _get_default_providers()
    except yaml.YAMLError as e:
        print(f"Error parsing YAML in {config_path}: {e}, using defaults")
        return _get_default_providers()
    except Exception as e:
        print(f"Error loading LLM providers from {config_path}: {e}, using defaults")
        return _get_default_providers()


# Load LLM providers from YAML configuration
# Falls back to hardcoded defaults if config file is missing/invalid
LLM_PROVIDERS = load_llm_providers()

# Tool categories
TOOL_CATEGORIES = {
    "search": "Search & Information",
    "computation": "Computation & Math",
    "utility": "Utility",
    "data_processing": "Data Processing",
    "code_execution": "Code Execution",
    "retrieval": "Document Retrieval"
}

# Built-in tool identifiers
BUILTIN_TOOLS = [
    {
        "id": "tavily_search",
        "name": "Tavily Web Search",
        "description": "Search the web using Tavily API for current information",
        "category": "search"
    },
    {
        "id": "python_repl",
        "name": "Python REPL",
        "description": "Execute Python code in a sandboxed environment",
        "category": "code_execution"
    },
    {
        "id": "get_current_datetime",
        "name": "DateTime",
        "description": "Get current date and time in various formats",
        "category": "utility"
    },
    {
        "id": "calculator",
        "name": "Calculator",
        "description": "Perform mathematical calculations safely",
        "category": "computation"
    },
    {
        "id": "string_tool",
        "name": "String Manipulation",
        "description": "Perform text operations like uppercase, lowercase, reverse, length",
        "category": "data_processing"
    },
    {
        "id": "generate_uuid",
        "name": "UUID Generator",
        "description": "Generate unique identifiers (UUIDs)",
        "category": "utility"
    },
    {
        "id": "wikipedia_search",
        "name": "Wikipedia Search",
        "description": "Search Wikipedia for information on any topic",
        "category": "search"
    },
    {
        "id": "random_number",
        "name": "Random Number",
        "description": "Generate random numbers within a specified range",
        "category": "utility"
    }
]

# Middleware types with descriptions and parameter schemas
MIDDLEWARE_TYPES = [
    {
        "type": "summarization",
        "name": "Summarization",
        "description": "Automatically compress conversation history when approaching token limits",
        "category": "memory",
        "params": {
            "model": {
                "type": "text",
                "default": "openai:gpt-4o-mini",
                "label": "Summary Model",
                "help_text": "LLM model used to generate summaries of conversation history"
            },
            "max_tokens_before_summary": {
                "type": "number",
                "default": 100000,
                "min": 10000,
                "max": 500000,
                "label": "Token Threshold",
                "help_text": "Token count that triggers automatic summarization"
            },
            "messages_to_keep": {
                "type": "number",
                "default": 20,
                "min": 1,
                "max": 100,
                "label": "Messages to Keep",
                "help_text": "Number of recent messages to preserve after summarization"
            }
        }
    },
    {
        "type": "model_call_limit",
        "name": "Model Call Limit",
        "description": "Restrict the number of model invocations",
        "category": "reliability",
        "params": {
            "thread_limit": {
                "type": "number",
                "default": 50,
                "min": 1,
                "max": 1000,
                "label": "Thread Limit",
                "help_text": "Maximum model calls allowed across entire conversation thread"
            },
            "run_limit": {
                "type": "number",
                "default": None,
                "min": 1,
                "max": 100,
                "label": "Run Limit",
                "help_text": "Maximum model calls allowed per single invocation (optional)"
            },
            "exit_behavior": {
                "type": "select",
                "options": ["end", "error"],
                "default": "end",
                "label": "Exit Behavior",
                "help_text": "Behavior when limit reached: 'end' (graceful stop) or 'error' (raise exception)"
            }
        }
    },
    {
        "type": "tool_call_limit",
        "name": "Tool Call Limit",
        "description": "Control tool execution counts (global or per-tool)",
        "category": "reliability",
        "params": {
            "thread_limit": {
                "type": "number",
                "default": 20,
                "min": 1,
                "max": 500,
                "label": "Thread Limit",
                "help_text": "Maximum tool calls across entire conversation thread"
            },
            "run_limit": {
                "type": "number",
                "default": None,
                "min": 1,
                "max": 100,
                "label": "Run Limit",
                "help_text": "Maximum tool calls per single invocation (optional)"
            },
            "tool_name": {
                "type": "text",
                "default": None,
                "label": "Tool Name",
                "help_text": "Specific tool to limit (leave empty to apply to all tools)"
            },
            "exit_behavior": {
                "type": "select",
                "options": ["continue", "error", "end"],
                "default": "continue",
                "label": "Exit Behavior",
                "help_text": "Response when limit reached: 'continue' (ignore), 'error' (raise), or 'end' (stop)"
            }
        }
    },
    {
        "type": "pii_detection",
        "name": "PII Detection",
        "description": "Detect and handle personally identifiable information",
        "category": "safety",
        "params": {
            "strategy": {
                "type": "select",
                "options": ["block", "redact", "mask", "hash"],
                "default": "redact",
                "label": "Strategy",
                "help_text": "How to handle detected PII: 'block' (reject), 'redact' ([REDACTED]), 'mask' (***), 'hash' (SHA-256)"
            }
        }
    },
    {
        "type": "model_fallback",
        "name": "Model Fallback",
        "description": "Switch to alternative models on failure",
        "category": "reliability",
        "params": {
            "fallback_models": {
                "type": "list",
                "default": ["gpt-4o-mini", "gpt-3.5-turbo"],
                "label": "Fallback Models",
                "help_text": "List of alternative models to try in sequence when primary fails (e.g., gpt-4o-mini, gpt-3.5-turbo)"
            }
        }
    },
    {
        "type": "tool_retry",
        "name": "Tool Retry",
        "description": "Automatically retry failed tools with exponential backoff",
        "category": "reliability",
        "params": {
            "max_retries": {
                "type": "number",
                "default": 3,
                "min": 1,
                "max": 10,
                "label": "Max Retries",
                "help_text": "Number of retry attempts after initial failure"
            },
            "initial_delay": {
                "type": "number",
                "default": 1.0,
                "min": 0.1,
                "max": 10.0,
                "label": "Initial Delay",
                "help_text": "Starting delay in seconds before first retry (uses exponential backoff)"
            }
        }
    },
    {
        "type": "human_in_the_loop",
        "name": "Human in the Loop",
        "description": "Pause for human approval of tool calls",
        "category": "safety",
        "params": {}
    },
    {
        "type": "todo_list",
        "name": "To-Do List",
        "description": "Provides write_todos tool for task planning",
        "category": "planning",
        "params": {}
    },
    {
        "type": "llm_tool_selector",
        "name": "LLM Tool Selector",
        "description": "Use separate LLM to filter relevant tools",
        "category": "optimization",
        "params": {
            "model": {
                "type": "text",
                "default": "gpt-4o-mini",
                "label": "Selector Model",
                "help_text": "LLM model used to intelligently select relevant tools for each query"
            },
            "max_tools": {
                "type": "number",
                "default": None,
                "min": 1,
                "max": 100,
                "label": "Max Tools",
                "help_text": "Maximum number of tools to select (leave empty for unlimited)"
            },
            "always_include": {
                "type": "list",
                "default": [],
                "label": "Always Include Tools",
                "help_text": "Tool names to always include regardless of query (comma-separated, e.g., calculator, tavily_search)"
            },
            "system_prompt": {
                "type": "textarea",
                "default": None,
                "label": "Custom Selection Prompt",
                "help_text": "Optional custom instructions for tool selection. Leave empty to use LangChain's default prompt. Use this to prioritize specific tool categories or enforce domain-specific selection rules."
            }
        }
    },
    {
        "type": "context_editing",
        "name": "Context Editing",
        "description": "Manage context by clearing older tool outputs",
        "category": "memory",
        "params": {
            "token_count_method": {
                "type": "select",
                "options": ["approximate", "model"],
                "default": "approximate",
                "label": "Token Count Method",
                "help_text": "'approximate' (fast estimation) or 'model' (accurate but slower)"
            }
        }
    },
    {
        "type": "anthropic_prompt_caching",
        "name": "Anthropic Prompt Caching",
        "description": "Reduce costs by caching prompts (Anthropic-specific)",
        "category": "optimization",
        "params": {}
    }
]

# Middleware presets
MIDDLEWARE_PRESETS = {
    "production_safe": {
        "name": "Production Safe",
        "description": "Essential middleware for production deployments",
        "middleware": [
            {"type": "pii_detection", "params": {"strategy": "redact"}, "enabled": True},
            {"type": "model_call_limit", "params": {"thread_limit": 50}, "enabled": True},
            {"type": "tool_retry", "params": {"max_retries": 3}, "enabled": True},
            {"type": "summarization", "params": {"model": "openai:gpt-4o-mini", "max_tokens_before_summary": 100000}, "enabled": True}
        ]
    },
    "cost_optimized": {
        "name": "Cost Optimized",
        "description": "Minimize API costs",
        "middleware": [
            {"type": "summarization", "params": {"model": "openai:gpt-4o-mini", "max_tokens_before_summary": 50000}, "enabled": True},
            {"type": "model_call_limit", "params": {"thread_limit": 20}, "enabled": True},
            {"type": "anthropic_prompt_caching", "params": {}, "enabled": True}
        ]
    },
    "development": {
        "name": "Development",
        "description": "Minimal middleware for development",
        "middleware": [
            {"type": "model_call_limit", "params": {"thread_limit": 100}, "enabled": True}
        ]
    },
    "high_reliability": {
        "name": "High Reliability",
        "description": "Maximum reliability with fallbacks and retries",
        "middleware": [
            {"type": "model_fallback", "params": {"fallback_models": ["gpt-4o-mini", "gpt-3.5-turbo"]}, "enabled": True},
            {"type": "tool_retry", "params": {"max_retries": 5, "initial_delay": 1.0}, "enabled": True},
            {"type": "model_call_limit", "params": {"thread_limit": 100}, "enabled": True}
        ]
    },
    "multi_tool_optimized": {
        "name": "Multi-Tool Optimized",
        "description": "Optimized for agents with many tools (5+) - includes intelligent tool selection",
        "middleware": [
            {"type": "llm_tool_selector", "params": {"model": "openai:gpt-4o-mini", "max_tools": 7, "always_include": []}, "enabled": True},
            {"type": "tool_retry", "params": {"max_retries": 3, "initial_delay": 1.0}, "enabled": True},
            {"type": "summarization", "params": {"model": "openai:gpt-4o-mini", "max_tokens_before_summary": 100000}, "enabled": True}
        ]
    }
}

# MCP Server presets
MCP_SERVER_PRESETS = {
    "local_calculator": {
        "name": "Calculator Server (Port 8005)",
        "description": "Mathematical operations - add, subtract, multiply, divide, power, factorial, etc.",
        "server": {
            'name': 'calculator',
            'description': 'Mathematical operations via MCP',
            'transport': 'streamable_http',
            'url': 'http://localhost:8005/mcp',
            'stateful': False
        }
    },
    "local_rag": {
        "name": "RAG Server (Port 8006)",
        "description": "Document retrieval and knowledge base - search, retrieve context, summarize documents",
        "server": {
            'name': 'rag',
            'description': 'Document retrieval and knowledge base',
            'transport': 'streamable_http',
            'url': 'http://localhost:8006/mcp',
            'stateful': False
        }
    },
    "custom": {
        "name": "Custom MCP Server",
        "description": "Manually configure a custom MCP server with your own settings",
        "server": {
            'name': '',
            'description': '',
            'transport': 'streamable_http',
            'url': 'http://localhost:8000/mcp',
            'stateful': False
        }
    }
}

# Streaming modes
STREAMING_MODES = [
    {"value": "updates", "label": "Agent Progress", "description": "Emits state changes after each step"},
    {"value": "messages", "label": "LLM Tokens", "description": "Streams individual tokens as generated"},
    {"value": "custom", "label": "Custom Updates", "description": "Progress indicators from tools"}
]

# Variable placeholders for prompts
PROMPT_VARIABLES = [
    {"var": "{{agent_name}}", "description": "Name of the agent"},
    {"var": "{{date}}", "description": "Current date (YYYY-MM-DD)"},
    {"var": "{{time}}", "description": "Current time (HH:MM:SS)"},
    {"var": "{{user_id}}", "description": "User identifier (if provided in runtime context)"},
    {"var": "{{session_id}}", "description": "Session identifier (if provided in runtime context)"}
]

# Default values
DEFAULTS = {
    "version": "1.0.0",
    "temperature": 0.7,
    "max_tokens": 4096,
    "streaming_enabled": True,
    "streaming_modes": ["updates"],
    "memory_type": "sqlite",
    "message_management": "none"
}

# Page titles and icons
PAGES = [
    {"number": 1, "title": "Basic Info", "icon": "📝"},
    {"number": 2, "title": "LLM Config", "icon": "🤖"},
    {"number": 3, "title": "Tools", "icon": "🔧"},
    {"number": 4, "title": "Prompts", "icon": "💬"},
    {"number": 5, "title": "Memory", "icon": "🧠"},
    {"number": 6, "title": "Middleware", "icon": "⚙️"},
    {"number": 7, "title": "Advanced", "icon": "🚀"},
    {"number": 8, "title": "Deploy", "icon": "✅"}
]

# API endpoints (relative to base URL)
API_ENDPOINTS = {
    "health": "/health",
    "templates_list": "/agents/templates/list",
    "template_get": "/agents/templates/{template_id}",
    "validate": "/agents/validate",
    "create": "/agents/create",
    "list": "/agents/list",
    "tools_list": "/tools/list",
    "tool_generate": "/tools/generate"
}

# Memory types
MEMORY_TYPES = ["in_memory", "sqlite"]
MESSAGE_MANAGEMENT = ["none", "trim", "summarize"]

# Python types for runtime context fields
PYTHON_TYPES = ["str", "int", "float", "bool", "dict", "list", "Any"]
