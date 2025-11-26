"""
LLM-based prompt enhancement utility.
Uses the selected LLM provider to enhance system prompts based on agent capabilities.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
import openai

# Load environment variables from .env file
project_root = Path(__file__).parent.parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

# Import LLM_PROVIDERS for env key mapping
try:
    from utils.constants import LLM_PROVIDERS
except ImportError:
    # Fallback if running standalone
    LLM_PROVIDERS = {
        "openai": {"env_key": "OPENAI_API_KEY"},
        "anthropic": {"env_key": "ANTHROPIC_API_KEY"},
        "google": {"env_key": "GOOGLE_API_KEY"},
        "groq": {"env_key": "GROQ_API_KEY"},
        "openrouter": {"env_key": "OPENROUTER_API_KEY"},
    }


def enhance_prompt(
    current_prompt: str,
    agent_name: str,
    agent_description: str,
    selected_tools: List[str],
    mcp_servers: List[Dict[str, Any]],
    llm_provider: str,
    llm_model: str,
    builtin_tools_metadata: List[Dict[str, str]]
) -> str:
    """
    Enhance a system prompt using the selected LLM based on agent capabilities.

    Args:
        current_prompt: Current system prompt text
        agent_name: Name of the agent
        agent_description: Description of the agent's purpose
        selected_tools: List of selected built-in tool IDs
        mcp_servers: List of MCP server configurations
        llm_provider: The LLM provider (openai, anthropic, google, groq, openrouter)
        llm_model: The LLM model being used
        builtin_tools_metadata: Metadata for all builtin tools

    Returns:
        Enhanced system prompt

    Raises:
        ValueError: If required API key is not set
        Exception: If API call fails
    """
    # Get the correct API key for the provider
    env_key = LLM_PROVIDERS.get(llm_provider, {}).get('env_key', f"{llm_provider.upper()}_API_KEY")
    api_key = os.getenv(env_key)
    if not api_key:
        raise ValueError(
            f"{env_key} not found in environment. "
            f"Please set it in your .env file to use prompt enhancement with {llm_provider}."
        )

    # Build tool descriptions
    tool_descriptions = []

    # Add built-in tools
    for tool in builtin_tools_metadata:
        if tool['id'] in selected_tools:
            tool_descriptions.append(f"- **{tool['name']}**: {tool['description']}")

    # Add MCP server tools
    for server in mcp_servers:
        server_name = server.get('name', 'Unknown')
        server_desc = server.get('description', 'No description')
        transport = server.get('transport', 'unknown')

        if transport == 'streamable_http':
            url = server.get('url', '')
            tool_descriptions.append(
                f"- **MCP Server: {server_name}** ({url}): {server_desc}"
            )
        elif transport == 'stdio':
            command = server.get('command', '')
            tool_descriptions.append(
                f"- **MCP Server: {server_name}** (stdio): {server_desc}"
            )

    tools_text = "\n".join(tool_descriptions) if tool_descriptions else "No tools selected"

    # Build meta-prompt for enhancement
    meta_prompt = f"""You are an expert at writing effective system prompts for LLM agents. Your task is to enhance the given system prompt to make it more effective at guiding the agent in using its available tools.

**Agent Information:**
- Name: {agent_name}
- Purpose: {agent_description}
- LLM Model: {llm_model}

**Available Tools:**
{tools_text}

**Current System Prompt:**
{current_prompt if current_prompt else "(No prompt provided yet)"}

**Enhancement Guidelines:**
1. If the current prompt exists, preserve its core intent, personality, and tone
2. Add clear, specific guidance on when and how to use each available tool
3. Include decision criteria for tool selection based on user queries
4. Emphasize multi-step reasoning when appropriate (use tools → analyze → respond)
5. Add fallback strategies for when tools aren't needed or fail
6. Keep the prompt concise but comprehensive (aim for 200-400 words)
7. Use second person ("you") to address the agent
8. Include relevant variables like {{{{agent_name}}}}, {{{{date}}}}, {{{{time}}}} where appropriate
9. For MCP servers, explain their purpose without assuming specific tool names (those will be discovered at runtime)

**Important:**
- Return ONLY the enhanced system prompt text
- Do NOT include explanations, metadata, or comments
- Do NOT use markdown code blocks
- The output should be ready to use directly as a system prompt

Enhanced prompt:"""

    try:
        enhanced_prompt = _call_llm_provider(
            provider=llm_provider,
            model=llm_model,
            api_key=api_key,
            system_message="You are an expert at writing clear, effective system prompts for LLM agents. You understand how to guide agents in tool usage, multi-step reasoning, and providing helpful responses.",
            user_message=meta_prompt
        )

        # Basic validation
        if not enhanced_prompt or len(enhanced_prompt) < 50:
            raise ValueError("Enhanced prompt is too short or empty")

        return enhanced_prompt

    except Exception as e:
        raise Exception(f"Failed to enhance prompt with {llm_provider}: {str(e)}")


def _call_llm_provider(
    provider: str,
    model: str,
    api_key: str,
    system_message: str,
    user_message: str
) -> str:
    """
    Call the appropriate LLM provider API.

    Args:
        provider: LLM provider name
        model: Model identifier
        api_key: API key for the provider
        system_message: System prompt for the LLM
        user_message: User message/prompt

    Returns:
        Generated text response
    """
    if provider == "openrouter":
        # OpenRouter uses OpenAI-compatible API
        # Free models require HTTP-Referer and X-Title headers for data policy
        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/langchain-agent-builder",
                "X-Title": "LangChain Agent Builder"
            }
        )
        response = client.chat.completions.create(
            model=model,
            temperature=0.7,
            max_tokens=2000,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
        )
        return response.choices[0].message.content.strip()

    elif provider == "openai":
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            temperature=0.7,
            max_tokens=2000,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
        )
        return response.choices[0].message.content.strip()

    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=2000,
            system=system_message,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )
        return response.content[0].text.strip()

    elif provider == "google":
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        gen_model = genai.GenerativeModel(
            model_name=model,
            system_instruction=system_message
        )
        response = gen_model.generate_content(user_message)
        return response.text.strip()

    elif provider == "groq":
        # Groq uses OpenAI-compatible API
        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        response = client.chat.completions.create(
            model=model,
            temperature=0.7,
            max_tokens=2000,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
        )
        return response.choices[0].message.content.strip()

    else:
        # Fallback: try OpenAI-compatible API
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            temperature=0.7,
            max_tokens=2000,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
        )
        return response.choices[0].message.content.strip()


def estimate_enhancement_cost(
    current_prompt: str,
    selected_tools: List[str],
    mcp_servers: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Estimate the API cost for prompt enhancement.

    Args:
        current_prompt: Current prompt text
        selected_tools: List of selected tool IDs
        mcp_servers: List of MCP server configs

    Returns:
        Dictionary with cost estimate information
    """
    # Rough estimation based on typical token counts
    # GPT-4o-mini pricing (as of 2024): ~$0.15 per 1M input tokens, ~$0.60 per 1M output tokens

    estimated_input_tokens = (
        len(current_prompt.split()) * 1.3 +  # Current prompt
        len(selected_tools) * 20 +  # Tool descriptions
        len(mcp_servers) * 30 +  # MCP server descriptions
        200  # Meta-prompt overhead
    )

    estimated_output_tokens = 600  # Typical enhanced prompt

    input_cost = (estimated_input_tokens / 1_000_000) * 0.15
    output_cost = (estimated_output_tokens / 1_000_000) * 0.60
    total_cost = input_cost + output_cost

    return {
        "estimated_input_tokens": int(estimated_input_tokens),
        "estimated_output_tokens": estimated_output_tokens,
        "estimated_cost_usd": round(total_cost, 6),
        "cost_display": f"~${total_cost:.6f}" if total_cost > 0.001 else "< $0.001"
    }
