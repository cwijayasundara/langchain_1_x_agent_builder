# Middleware Configuration Guide

## Overview

Middleware in the LangChain 1.x Agent Builder provides powerful capabilities for enhancing agent behavior, but each middleware type has different performance characteristics and use cases. This guide helps you choose the right middleware for your agent.

## Table of Contents

1. [Middleware Performance Overview](#middleware-performance-overview)
2. [Detailed Middleware Reference](#detailed-middleware-reference)
3. [Choosing the Right Middleware](#choosing-the-right-middleware)
4. [Common Middleware Patterns](#common-middleware-patterns)
5. [Performance Optimization](#performance-optimization)
6. [Troubleshooting](#troubleshooting)

---

## Middleware Performance Overview

### Performance Impact Matrix

| Middleware | LLM Calls | Token Overhead | Latency Impact | When Triggered |
|------------|-----------|----------------|----------------|----------------|
| **llm_tool_selector** | +1 per request | +500-1000 tokens | +0.5-1s | Every request |
| **summarization** | +1 when triggered | +200-2000 tokens | +1-3s | When >100K tokens |
| **model_call_limit** | 0 | Minimal (~10 tokens) | Negligible | Never (tracking) |
| **tool_retry** | 0 | None | Only on failure | Tool failures |
| **todo_list** | +1 per cycle | +200 tokens | +0.5-1s | Every request |
| **pii_detection** | 0 | None | Negligible | Pre-processing |
| **model_fallback** | 0 | None | Only on failure | Model errors |
| **human_in_the_loop** | 0 | None | User-dependent | Tool calls |
| **context_editing** | 0 | None (saves tokens) | Negligible | High token usage |
| **anthropic_prompt_caching** | 0 | None (saves tokens) | Negligible | Repeated prompts |

### Cost Impact Examples

**Simple Query: "What's 2+2?"**

| Configuration | LLM Calls | Approx. Cost (gpt-4o-mini) |
|---------------|-----------|---------------------------|
| No middleware | 2 | $0.0004 |
| + model_call_limit | 2 | $0.0004 |
| + tool_retry | 2 | $0.0004 |
| + llm_tool_selector | 3 | $0.0006 |
| + todo_list | 4 | $0.0008 |
| + todo_list + llm_tool_selector | 5 | $0.0010 |

**Complex Query: "Research climate change and create a report"**

| Configuration | LLM Calls | Approx. Cost (gpt-4o-mini) |
|---------------|-----------|---------------------------|
| Minimal middleware | 5-8 | $0.001-0.002 |
| + llm_tool_selector | 6-9 | $0.0012-0.0022 |
| + todo_list | 8-12 | $0.0016-0.003 |
| Fully optimized | 6-10 | $0.0012-0.0024 |

---

## Detailed Middleware Reference

### 1. LLM Tool Selector

**Type**: `llm_tool_selector`
**Category**: Optimization
**LLM Calls**: +1 per request

#### What It Does
Uses a separate LLM call to intelligently filter available tools before the main agent execution. Reduces token usage and tool confusion for agents with many tools.

#### Configuration
```yaml
middleware:
  - type: llm_tool_selector
    params:
      model: openai:gpt-4o-mini  # Cheap, fast selector model
      max_tools: 7                # Limit to N most relevant tools
      always_include: []          # Tools to always include (optional)
      system_prompt: |            # Custom selection instructions (optional)
        Prioritize calculation tools for math queries.
        Prioritize search tools for current information.
    enabled: true
```

#### When to Use
- ✅ Agents with 5+ tools (auto-enabled)
- ✅ Agents using MCP servers (often 10-20 tools per server)
- ✅ Multi-purpose agents with diverse toolsets

#### When to Avoid
- ❌ Agents with <5 tools
- ❌ Agents where all tools are always relevant
- ❌ Extremely latency-sensitive applications

#### Performance Impact
- **Extra LLM call**: 1 per request
- **Token savings**: 85-90% reduction in tool definition tokens for large toolsets
- **Net effect**: Usually cost-negative (saves more than it costs) for 10+ tools

#### Example
```python
# Before: 18 tools → ~4500 tokens per request
# After: 18 tools → 7 selected → ~1500 tokens per request
# Savings: 3000 tokens per request
# Cost of selector: ~500 tokens
# Net savings: 2500 tokens per request (56% reduction)
```

---

### 2. Summarization

**Type**: `summarization`
**Category**: Memory
**LLM Calls**: +1 when triggered

#### What It Does
Automatically compresses conversation history when token count exceeds threshold. Prevents context window overflow in long conversations.

#### Configuration
```yaml
middleware:
  - type: summarization
    params:
      model: openai:gpt-4o-mini       # Model for summarization
      max_tokens_before_summary: 100000  # Trigger threshold
      messages_to_keep: 20            # Recent messages to preserve
    enabled: true
```

#### When to Use
- ✅ Long-running conversations
- ✅ Multi-turn sessions (>50 messages)
- ✅ Agents that accumulate context over time

#### When to Avoid
- ❌ Short sessions (single-turn Q&A)
- ❌ Stateless agents
- ❌ Agents with short-term memory only

#### Performance Impact
- **Trigger frequency**: Depends on conversation length
- **Cost when triggered**: 1 LLM call for summarization (~1000-5000 tokens)
- **Long-term benefit**: Prevents hitting context limits, enables longer conversations

---

### 3. Model Call Limit

**Type**: `model_call_limit`
**Category**: Reliability
**LLM Calls**: 0 (tracking only)

#### What It Does
Limits the number of model invocations per conversation thread or per run. Prevents runaway loops and excessive costs.

#### Configuration
```yaml
middleware:
  - type: model_call_limit
    params:
      thread_limit: 50      # Max calls per conversation thread
      run_limit: 10         # Max calls per single invocation (optional)
      exit_behavior: end    # 'end' (graceful) or 'error' (raise exception)
    enabled: true
```

#### When to Use
- ✅ **ALL agents** (safety measure)
- ✅ Production deployments
- ✅ Experimental agents with unknown behavior

#### When to Avoid
- ❌ Never - this is a safety feature with minimal overhead

#### Performance Impact
- **LLM calls**: None
- **Overhead**: Negligible (~10 tokens for tracking)
- **Benefit**: Prevents catastrophic cost scenarios

#### Recommendations
- **thread_limit**: 50-100 for most agents
- **run_limit**: 10-30 for agentic loops, null for simple agents
- **exit_behavior**: `end` for production, `error` for debugging

---

### 4. Tool Retry

**Type**: `tool_retry`
**Category**: Reliability
**LLM Calls**: 0 (only retries tool calls)

#### What It Does
Automatically retries failed tool calls with exponential backoff. Improves reliability for agents using external APIs or MCP servers.

#### Configuration
```yaml
middleware:
  - type: tool_retry
    params:
      max_retries: 3        # Number of retry attempts
      initial_delay: 1.0    # Starting delay in seconds (exponential backoff)
    enabled: true
```

#### When to Use
- ✅ Agents using external APIs
- ✅ MCP servers with potential network issues
- ✅ Tools with transient failures

#### When to Avoid
- ❌ Purely local tools (python_repl, calculator)
- ❌ Tools where retry doesn't help (logic errors)

#### Performance Impact
- **LLM calls**: None (retries tools, not model)
- **Latency**: Only on tool failures (1s, 2s, 4s exponential backoff)
- **Reliability improvement**: Significantly reduces failure rate

---

### 5. TodoList

**Type**: `todo_list`
**Category**: Planning
**LLM Calls**: +1 per request cycle

#### What It Does
Provides a `write_todos` tool for task planning and management. Injects system prompt encouraging structured task breakdown.

#### Configuration
```yaml
middleware:
  - type: todo_list
    params: {}  # No parameters
    enabled: true
```

#### When to Use
- ✅ Project management agents
- ✅ Multi-step workflows (5+ steps genuinely required)
- ✅ Task planning and tracking systems
- ✅ Complex research projects with subtasks

#### When to Avoid
- ❌ **Simple Q&A agents** (e.g., "What's the capital of France?")
- ❌ **Math/calculation agents** (e.g., "What's 15% of 250?")
- ❌ **Web search agents** (e.g., "Latest news on AI?")
- ❌ **Single-purpose tools**
- ❌ **Any agent handling queries in <3 steps**

#### Performance Impact
- **Extra LLM call**: 1 per request cycle (even if todo not created)
- **Token overhead**: ~200 tokens per request (system prompt injection)
- **Cost increase**: ~33% for simple queries
- **Benefit**: Only valuable for genuinely complex workflows

#### ⚠️ CRITICAL WARNING
**TodoListMiddleware adds overhead on EVERY request**, even when not needed. This is the #1 cause of unexpected LLM call increases.

**Real-world example**:
```yaml
Query: "What's 15% of 345?"
WITHOUT todo_list: 3 LLM calls
WITH todo_list: 4 LLM calls (+33%)

The extra call evaluates whether to use write_todos (decides not to)
but still costs time and money.
```

**Recommendation**: Start without TodoListMiddleware. Add it only after confirming your agent genuinely needs task planning.

---

### 6. PII Detection

**Type**: `pii_detection`
**Category**: Safety
**LLM Calls**: 0 (regex-based)

#### What It Does
Detects and handles personally identifiable information (PII) in user messages and agent responses.

#### Configuration
```yaml
middleware:
  - type: pii_detection
    params:
      strategy: redact  # 'block', 'redact', 'mask', or 'hash'
    enabled: true
```

#### When to Use
- ✅ Customer support agents
- ✅ Healthcare/legal agents
- ✅ Agents handling sensitive data
- ✅ GDPR/CCPA compliance requirements

#### When to Avoid
- ❌ Internal tools with trusted users
- ❌ Agents that need to process PII (with proper authorization)

#### Performance Impact
- **LLM calls**: None
- **Latency**: Negligible (regex pattern matching)
- **Overhead**: None

---

### 7. Model Fallback

**Type**: `model_fallback`
**Category**: Reliability
**LLM Calls**: 0 (only on primary model failure)

#### What It Does
Automatically switches to backup models if primary model fails.

#### Configuration
```yaml
middleware:
  - type: model_fallback
    params:
      fallback_models:
        - gpt-4o-mini
        - gpt-3.5-turbo
    enabled: true
```

#### When to Use
- ✅ Production agents requiring high uptime
- ✅ Agents using newer/experimental models
- ✅ Multi-cloud deployments

#### When to Avoid
- ❌ Development/testing
- ❌ Agents requiring specific model capabilities

#### Performance Impact
- **Normal operation**: Zero overhead
- **On failure**: Retry with fallback model
- **Reliability**: Significantly improved uptime

---

### 8. Human in the Loop

**Type**: `human_in_the_loop`
**Category**: Safety
**LLM Calls**: 0 (pauses for approval)

#### What It Does
Pauses execution before tool calls, requiring human approval.

#### Configuration
```yaml
middleware:
  - type: human_in_the_loop
    params: {}
    enabled: true
```

#### When to Use
- ✅ High-risk operations (data deletion, payments)
- ✅ Agents in testing/validation phase
- ✅ Regulatory compliance requirements

#### When to Avoid
- ❌ Autonomous agents
- ❌ High-throughput applications
- ❌ Latency-sensitive use cases

#### Performance Impact
- **LLM calls**: None
- **Latency**: User-dependent (seconds to minutes)
- **Safety**: Maximum control over agent actions

---

### 9. Context Editing

**Type**: `context_editing`
**Category**: Memory
**LLM Calls**: 0 (clears old tool outputs)

#### What It Does
Manages context window by removing older tool call results when token limit approached.

#### Configuration
```yaml
middleware:
  - type: context_editing
    params:
      token_count_method: approximate  # 'approximate' or 'model'
    enabled: true
```

#### When to Use
- ✅ Agents with many tool calls
- ✅ Long-running sessions
- ✅ Agents approaching context limits

#### When to Avoid
- ❌ Agents where tool history is critical
- ❌ Short sessions

#### Performance Impact
- **LLM calls**: None
- **Token savings**: Significant (removes old tool outputs)
- **Trade-off**: May lose important context

---

### 10. Anthropic Prompt Caching

**Type**: `anthropic_prompt_caching`
**Category**: Optimization
**LLM Calls**: 0 (caches prompts)

#### What It Does
Enables Anthropic's prompt caching feature to reduce costs for repeated prompts.

#### Configuration
```yaml
middleware:
  - type: anthropic_prompt_caching
    params: {}
    enabled: true
```

#### When to Use
- ✅ Anthropic models only (Claude)
- ✅ Agents with large, consistent system prompts
- ✅ High-volume applications

#### When to Avoid
- ❌ Non-Anthropic models
- ❌ Frequently changing prompts
- ❌ Low-volume applications

#### Performance Impact
- **Cost savings**: Up to 90% on cached prompt tokens
- **Overhead**: None
- **Benefit**: Significant for large system prompts

---

## Choosing the Right Middleware

### Decision Tree

```
1. Does your agent have 5+ tools?
   YES → Enable llm_tool_selector (auto-enabled)
   NO  → Skip

2. Does your agent handle long conversations (>50 messages)?
   YES → Enable summarization (threshold: 100K tokens)
   NO  → Skip

3. Is your agent in production or experimental?
   ALL → Enable model_call_limit (safety)

4. Does your agent use external/unreliable tools?
   YES → Enable tool_retry
   NO  → Skip

5. Does your agent GENUINELY need task planning (5+ step workflows)?
   YES → Enable todo_list
   NO  → SKIP (this is critical - avoid unnecessary overhead)

6. Does your agent handle sensitive data?
   YES → Enable pii_detection
   NO  → Skip

7. Do you need high uptime?
   YES → Enable model_fallback
   NO  → Skip

8. Do you need approval for sensitive actions?
   YES → Enable human_in_the_loop
   NO  → Skip
```

---

## Common Middleware Patterns

### Pattern 1: Simple Q&A Agent

**Use Case**: FAQ bot, knowledge assistant

```yaml
middleware:
  - type: model_call_limit
    params: {thread_limit: 50}
```

**Expected LLM calls per query**: 1-2

---

### Pattern 2: Multi-Tool Agent (Research, Analysis)

**Use Case**: Research assistant with web search + calculator + documents

```yaml
middleware:
  - type: llm_tool_selector
    params: {model: openai:gpt-4o-mini, max_tools: 7}
  - type: tool_retry
    params: {max_retries: 3}
  - type: model_call_limit
    params: {thread_limit: 50, run_limit: 20}
```

**Expected LLM calls per query**: 3-5 (1 selector + 2-4 agent)

---

### Pattern 3: Complex Workflow Agent

**Use Case**: Project management, multi-step automation

```yaml
middleware:
  - type: todo_list
  - type: llm_tool_selector
    params: {model: openai:gpt-4o-mini, max_tools: 10}
  - type: model_call_limit
    params: {thread_limit: 100, run_limit: 30}
  - type: summarization
    params: {max_tokens_before_summary: 100000}
```

**Expected LLM calls per query**: 5-15 (complex workflows need more iterations)

---

### Pattern 4: Production Agent

**Use Case**: Customer-facing, high availability

```yaml
middleware:
  - type: pii_detection
    params: {strategy: redact}
  - type: llm_tool_selector
    params: {model: openai:gpt-4o-mini, max_tools: 7}
  - type: model_fallback
    params: {fallback_models: ["gpt-4o-mini", "gpt-3.5-turbo"]}
  - type: tool_retry
    params: {max_retries: 3}
  - type: model_call_limit
    params: {thread_limit: 50}
  - type: summarization
    params: {max_tokens_before_summary: 100000}
```

**Expected LLM calls per query**: 3-6

---

## Performance Optimization

### Best Practices

1. **Start Minimal**: Begin with only `model_call_limit`, add middleware as needed
2. **Measure Impact**: Track LLM call count and costs before/after middleware changes
3. **Avoid TodoList**: Unless genuinely needed for complex workflows
4. **Use Cheap Selectors**: gpt-4o-mini for llm_tool_selector, not gpt-4
5. **Set Appropriate Limits**: run_limit should allow for task complexity

### Optimization Checklist

- [ ] Removed todo_list from simple agents
- [ ] Using gpt-4o-mini for llm_tool_selector
- [ ] Set reasonable max_tools (5-8, not 15+)
- [ ] Configured summarization threshold appropriately
- [ ] Set run_limit to match agent complexity
- [ ] Enabled tool_retry only for external tools
- [ ] Using Anthropic caching if applicable

### Monitoring

Track these metrics:
- **Average LLM calls per query**
- **Average cost per query**
- **Tool call success rate**
- **Conversation length before summarization**

---

## Troubleshooting

### Issue: Too Many LLM Calls

**Symptoms**: Agent making 5+ LLM calls for simple queries

**Diagnosis**:
1. Check if `todo_list` is enabled → Remove if not needed
2. Check `run_limit` → Ensure it's not forcing extra loops
3. Check tool failures → May be retrying unnecessarily

**Solution**:
```yaml
# Remove todo_list for simple agents
middleware:
  # - type: todo_list  # REMOVE THIS
  - type: model_call_limit
    params: {thread_limit: 50}
```

---

### Issue: High Costs

**Symptoms**: API costs higher than expected

**Diagnosis**:
1. Count LLM calls in logs
2. Check if expensive models used for llm_tool_selector
3. Verify summarization isn't triggering too frequently

**Solution**:
```yaml
middleware:
  - type: llm_tool_selector
    params:
      model: openai:gpt-4o-mini  # Use cheap model, not gpt-4
      max_tools: 7
```

---

### Issue: Agent Stopping Mid-Conversation

**Symptoms**: "Model call limit reached" errors

**Diagnosis**:
- `run_limit` too low for complex tasks

**Solution**:
```yaml
middleware:
  - type: model_call_limit
    params:
      thread_limit: 50
      run_limit: 30  # Increase from default 10
```

---

### Issue: Tool Failures

**Symptoms**: NotImplementedError, connection errors

**Diagnosis**:
- Missing `tool_retry` middleware
- MCP server connectivity issues

**Solution**:
```yaml
middleware:
  - type: tool_retry
    params:
      max_retries: 3
      initial_delay: 1.0
```

---

## Summary

**Key Takeaways**:

1. **model_call_limit** - Enable on ALL agents (safety)
2. **llm_tool_selector** - Auto-enabled for 5+ tools (cost-effective)
3. **todo_list** - ONLY for complex workflows (high overhead)
4. **tool_retry** - Enable for external/unreliable tools
5. **summarization** - Enable for long conversations

**Performance Rule of Thumb**:
- Simple agent: 1-2 LLM calls per query
- Multi-tool agent: 3-5 LLM calls per query
- Complex workflow agent: 5-15 LLM calls per query

**Cost Optimization**:
- Use gpt-4o-mini for middleware (selectors, summarization)
- Avoid todo_list unless genuinely needed
- Set appropriate run_limit based on task complexity

For more information, see:
- `CLAUDE.md` - Project architecture and middleware design patterns
- `TOOL_SELECTION_GUIDE.md` - Deep dive into llm_tool_selector
- `configs/templates/` - Example configurations
