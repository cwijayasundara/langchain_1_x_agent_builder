# Tool Selection Optimization Guide

## Overview

This guide explains how to optimize tool selection in your LangChain 1.x agents to ensure accurate tool usage, reduce token costs, and improve response quality.

## The Problem

When agents have access to many tools (especially with MCP servers), several issues can occur:

1. **Tool Confusion**: LLMs struggle to choose the correct tool from 10+ options
2. **Token Waste**: All tool definitions are sent in every request, consuming tokens unnecessarily
3. **Slower Responses**: More tools = larger context = slower processing
4. **Higher Costs**: More tokens = higher API costs

### Example Scenario

Your research assistant has:
- 1 built-in tool (Tavily Search)
- 10 calculator MCP tools
- 7 RAG MCP tools
- **Total: 18 tools**

Without optimization, all 18 tool definitions (~3000-5000 tokens) are sent with every request, even for simple queries like "What is 2+2?" that only need one calculator tool.

## The Solution: Three-Layer Optimization

### Layer 1: Intelligent Tool Selection (LLMToolSelectorMiddleware)

**What it does**: Uses a fast, cheap LLM to filter tools before the main request.

**How it works**:
1. User asks a question
2. Selector LLM receives the question + all available tools
3. Selector picks the N most relevant tools
4. Main LLM only sees those N tools
5. Response generated with reduced context

**Benefits**:
- 85-90% reduction in tool-related tokens
- Faster responses
- More accurate tool selection
- Lower costs

**Configuration**:

```yaml
middleware:
  - type: llm_tool_selector
    params:
      model: openai:gpt-4o-mini          # Fast, cheap selector model
      max_tools: 7                        # Limit to 7 most relevant tools
      always_include: []                  # Tools to always include
    enabled: true
```

**Recommendations**:
- Use for agents with **5+ tools**
- Set `max_tools` to ~1/3 of total tools (minimum 5, maximum 8)
- Use a fast, cheap model like `gpt-4o-mini` for selection
- For accuracy-critical agents, you can use the same model as the main agent

**Auto-Default Behavior** (NEW):
The framework now **automatically enables** `llm_tool_selector` middleware for agents with 5+ tools!

- ✅ **Automatic**: No configuration needed for agents with 5+ tools
- ✅ **Smart Sizing**: Automatically calculates optimal `max_tools` (1/3 of total, 5-8 range)
- ✅ **Non-intrusive**: Only adds if not already configured
- ✅ **Override**: Can disable by explicitly configuring your own middleware list

You'll see this in logs:
```
INFO - Auto-enabled llm_tool_selector middleware: 18 tools detected → max_tools=7
```

**Manual Configuration** (Optional):
You can still manually configure if you want custom settings:

```yaml
middleware:
  - type: llm_tool_selector
    params:
      model: openai:gpt-4o-mini
      max_tools: 10                         # Custom max
      always_include: ["calculator"]        # Always include specific tools
    enabled: true
```

#### Custom Selection Instructions (system_prompt)

**What it does**: Customize how the selector LLM chooses tools by providing your own selection instructions.

By default, LangChain's `llm_tool_selector` uses a built-in prompt that asks the LLM to select the most relevant tools for the user's query. You can override this with your own `system_prompt` to:

- **Prioritize specific tool categories** for your agent's domain
- **Add domain-specific selection logic** (e.g., "always prefer local tools over remote")
- **Enforce custom rules** (e.g., "never select more than 2 search tools")
- **Tailor selection to your use case** (research vs. data analysis vs. customer support)

**Configuration**:

```yaml
middleware:
  - type: llm_tool_selector
    params:
      model: openai:gpt-4o-mini
      max_tools: 7
      system_prompt: |
        You are selecting tools for a multi-purpose assistant.

        Guidelines:
        - Prioritize calculation tools for math queries
        - Prioritize search tools for current events/information
        - Prioritize document tools for knowledge retrieval

        Only select tools that are clearly relevant to the user's query.
        When in doubt, prefer fewer tools over more tools.
    enabled: true
```

**When to Use Custom system_prompt**:

✅ **Use custom prompt when**:
- Your agent has specific tool priority requirements
- You need domain-specific selection logic
- Default selection is too broad or too narrow
- You want to enforce custom constraints on tool selection

❌ **Use default prompt when**:
- Agent has general-purpose tools
- No special prioritization needed
- LangChain's default logic works well for your use case

**Examples**:

**Research-focused agent** (prioritize search and retrieval):
```yaml
system_prompt: |
  You are selecting tools for a research assistant.

  Rules:
  - For "latest", "recent", "current" queries → prioritize search tools
  - For "according to", "in the documentation" → prioritize retrieval tools
  - For calculations → only select if explicitly needed

  Select the minimum tools needed to answer the query accurately.
```

**Data analysis agent** (prioritize computation and code execution):
```yaml
system_prompt: |
  You are selecting tools for a data analysis assistant.

  Rules:
  - For numerical operations → prioritize calculator tools
  - For data manipulation → prioritize python_repl
  - For visualization → prioritize code execution tools
  - Avoid search tools unless external data is explicitly needed

  Prefer code execution over calculator for complex operations.
```

**Customer support agent** (balanced selection with safety):
```yaml
system_prompt: |
  You are selecting tools for a customer support assistant.

  Rules:
  - For product info → prioritize document retrieval
  - For order status → prioritize API tools
  - For troubleshooting → select diagnostic tools
  - Never select more than 5 tools total

  Prioritize customer-facing tools over internal tools.
```

**Note**: The `system_prompt` parameter is **optional**. If not provided, LangChain uses its default tool selection prompt, which works well for most general-purpose agents.

### Layer 2: Prompt Engineering with Tool Categorization

**What it does**: Provides clear, structured guidance on when to use each tool category.

**Tool Categories**:
- **Computation**: Mathematical operations, calculations
- **Search**: Web search, current information
- **Retrieval**: Document retrieval, knowledge base queries
- **Code Execution**: Python REPL, script execution
- **Utility**: DateTime, UUID generation, random numbers
- **Data Processing**: String manipulation, data transformation

**Best Practices**:

#### 1. Categorize Tools in System Prompt

```yaml
prompts:
  system: |
    ## Available Tools (By Category)

    ### 1. COMPUTATION TOOLS
    Use for ANY mathematical operations:
    - add, subtract, multiply, divide: Basic arithmetic
    - calculate_percentage: Percentage calculations

    **CRITICAL**: For ANY math, you MUST use calculator tools.

    ### 2. SEARCH TOOLS
    Use for current events and recent information:
    - tavily_search: Web search
```

#### 2. Include Decision Tree

```yaml
## Tool Selection Decision Tree

1. Does the query involve calculations? → Use COMPUTATION tools
2. Does it ask about current events? → Use SEARCH tools
3. Does it reference documents? → Use RETRIEVAL tools
4. Multiple capabilities needed? → Use tools in sequence
```

#### 3. Provide Concrete Examples

```yaml
Examples requiring computation tools:
- "What is 15% of 250?" → use calculate_percentage
- "Calculate 45 * 67" → use multiply
```

### Layer 3: Framework-Level Tool Management

The framework now includes:

#### Tool Registry with Categories

All tools are automatically categorized:

```python
from agent_api.services.tool_registry import ToolCategory

# Built-in categories
BUILTIN_TOOL_CATEGORIES = {
    "tavily_search": ToolCategory.SEARCH,
    "calculator": ToolCategory.COMPUTATION,
    "python_repl": ToolCategory.CODE_EXECUTION,
    # ...
}
```

#### Dynamic Tool Documentation

Generate tool documentation automatically:

```python
from agent_api.services.prompt_helper import PromptHelper

# Generate structured tool documentation
tool_docs = PromptHelper.generate_complete_tool_section(
    tools=agent_tools,
    tool_registry=tool_registry,
    include_examples=True,
    emphasize_accuracy=True
)
```

#### Smart Middleware Presets

Use pre-configured middleware for common scenarios:

```yaml
# Option 1: Use preset
middleware_preset: multi_tool_optimized

# Option 2: Manual configuration (same as preset)
middleware:
  - type: llm_tool_selector
    params: {model: "openai:gpt-4o-mini", max_tools: 7}
    enabled: true
  - type: tool_retry
    params: {max_retries: 3}
    enabled: true
  - type: summarization
    params: {model: "openai:gpt-4o-mini"}
    enabled: true
```

## Implementation Guide

### For Your Research Assistant Agent

#### Step 1: Enable Tool Selection Middleware

```yaml
# configs/agents/research_assistant.yaml
middleware:
  - type: llm_tool_selector
    params:
      model: openai:gpt-4o-mini
      max_tools: 7
      always_include: []
    enabled: true
```

#### Step 2: Enhance System Prompt

```yaml
prompts:
  system: |
    You are {{agent_name}}.

    ## Available Tools (By Category)

    ### 1. COMPUTATION TOOLS (Calculator MCP)
    For ANY math operations:
    - add, subtract, multiply, divide, power, square_root, etc.

    **CRITICAL**: ALWAYS use calculator tools for math. Never estimate.

    Examples:
    - "What's 15% of 250?" → calculate_percentage
    - "45 * 67 = ?" → multiply

    ### 2. SEARCH TOOLS
    For current information:
    - tavily_search: Web search

    ### 3. DOCUMENT RETRIEVAL TOOLS (RAG MCP)
    For knowledge base queries:
    - search_documents, retrieve_context, summarize_documents

    ## Decision Tree
    1. Math? → COMPUTATION tools
    2. Current info? → SEARCH tools
    3. Documents? → RETRIEVAL tools
```

### For New Agents with Many Tools

When creating agents with 5+ tools:

1. **Start with Multi-Tool Preset**: Use `multi_tool_optimized` middleware preset
2. **Categorize Tools**: Group tools by purpose in system prompt
3. **Add Decision Tree**: Help LLM choose correct tool category
4. **Include Examples**: Show concrete usage examples
5. **Emphasize Critical Rules**: Make important distinctions bold/explicit

## Performance Metrics

### Before Optimization

- **18 tools** sent to LLM on every request
- **~4500 tokens** for tool definitions
- **Slower responses** due to large context
- **Higher costs** from token usage
- **Tool confusion** - wrong tool selection

### After Optimization

- **~7 tools** sent to LLM (filtered by selector)
- **~1500 tokens** for tool definitions (**67% reduction**)
- **Faster responses** from smaller context
- **Lower costs** from reduced tokens
- **Better accuracy** from clearer guidance + filtering

## Advanced Topics

### Custom Tool Categorization

For MCP tools, categories are inferred from tool names and descriptions:

```python
# agent_api/services/tool_registry.py
MCP_CATEGORY_PATTERNS = {
    ToolCategory.COMPUTATION: ["add", "subtract", "multiply", "calculator", "math"],
    ToolCategory.SEARCH: ["search", "query", "find"],
    ToolCategory.RETRIEVAL: ["retrieve", "get", "document", "context"],
}
```

To override category inference, extend the patterns or categorize explicitly.

### Dynamic vs. Manual Prompts

**Manual Approach** (current):
- Write tool documentation in YAML system prompt
- Full control over wording and structure
- Must update manually when tools change

**Dynamic Approach** (framework feature):
- Auto-generate tool documentation from metadata
- Always up-to-date with tool changes
- Consistent formatting across agents

```python
# Example: Dynamic generation in AgentFactory
tool_docs = config_manager.generate_tool_documentation(
    tools=all_tools,
    tool_registry=self.tool_registry
)
# Append to or replace parts of system prompt
```

### Middleware Recommendations API

The framework can recommend middleware based on agent characteristics:

```python
# agent_api/services/middleware_factory.py
recommended = middleware_factory.recommend_middleware_for_agent(
    total_tool_count=18,
    use_mcp=True,
    priority="accuracy"  # or "cost" or "balanced"
)
# Returns list of MiddlewareConfig objects
```

## Best Practices Summary

### ✅ Do

1. **Enable `llm_tool_selector` for agents with 5+ tools**
2. **Categorize tools** in system prompts by purpose
3. **Provide concrete examples** for each tool category
4. **Use decision trees** to guide tool selection
5. **Emphasize critical rules** (e.g., "ALWAYS use calculator for math")
6. **Test with realistic queries** to verify tool selection
7. **Monitor tool usage** to identify patterns and issues

### ❌ Don't

1. **Don't send all tools** to the LLM if you have many
2. **Don't rely on tool descriptions alone** - add usage guidance
3. **Don't use vague examples** - be specific and concrete
4. **Don't skip middleware** for multi-tool agents
5. **Don't use expensive models** for tool selection (use gpt-4o-mini)
6. **Don't forget to update prompts** when tools change

## Troubleshooting

### Agent Still Choosing Wrong Tools

**Symptoms**: Agent uses web search instead of calculator, or vice versa.

**Solutions**:
1. Make tool selection rules MORE explicit in prompt (use "CRITICAL", "MUST", "ALWAYS")
2. Add more specific examples for each tool category
3. Reduce `max_tools` in llm_tool_selector (forces more selective filtering)
4. Check that tool descriptions are clear and distinct

### Too Many/Too Few Tools Selected

**Symptoms**: Selector middleware picking wrong number of tools.

**Solutions**:
- **Too many**: Decrease `max_tools` parameter
- **Too few**: Increase `max_tools` or use `always_include` for critical tools
- **Wrong tools**: Improve tool descriptions, add category keywords

### High Token Usage Still

**Symptoms**: Token usage not decreasing as expected.

**Solutions**:
1. Verify `llm_tool_selector` middleware is enabled and running
2. Check logs to see how many tools are being selected
3. Ensure selector model is configured correctly
4. Consider reducing `max_tools` further

## References

- [Anthropic: Code Execution with MCP](https://www.anthropic.com/engineering/code-execution-with-mcp)
- [LangChain Tool Calling Guide](https://docs.langchain.com/)
- Tool Registry: `agent_api/services/tool_registry.py`
- Prompt Helper: `agent_api/services/prompt_helper.py`
- Middleware Factory: `agent_api/services/middleware_factory.py`

## Examples

### Research Assistant (Simple - 1 Tool)

```yaml
tools: [tavily_search]
middleware: [tool_retry, summarization]
# No tool selector needed - only 1 tool
```

### Multi-Tool Agent (18 Tools - Calculator + RAG + Search)

```yaml
tools: [tavily_search]
mcp_servers:
  - {name: calculator, url: http://localhost:8005/mcp}
  - {name: rag, url: http://localhost:8006/mcp}

middleware:
  - type: llm_tool_selector
    params: {model: "openai:gpt-4o-mini", max_tools: 7}
    enabled: true
  - type: tool_retry
    params: {max_retries: 3}
    enabled: true

prompts:
  system: |
    ## Available Tools (By Category)
    ### 1. COMPUTATION TOOLS
    [10 calculator tools with examples]
    ### 2. SEARCH TOOLS
    [tavily_search with examples]
    ### 3. RETRIEVAL TOOLS
    [7 RAG tools with examples]
```

### Data Analysis Agent (Many Tools)

```yaml
tools: [python_repl, calculator]
middleware_preset: multi_tool_optimized  # Auto-configured
```

## Conclusion

Tool selection optimization is critical for agents with multiple tools. The three-layer approach (middleware filtering + prompt engineering + framework features) provides:

- **85-90% token reduction** for tool definitions
- **Better accuracy** in tool selection
- **Lower costs** from reduced API usage
- **Faster responses** from smaller context

Start with Layer 1 (middleware) for immediate impact, then enhance with Layer 2 (prompts) for maximum accuracy.
