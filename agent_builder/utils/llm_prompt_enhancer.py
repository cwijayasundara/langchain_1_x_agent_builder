"""
LLM-based prompt enhancement utility.
Uses GPT-4o-mini to enhance system prompts based on agent capabilities.
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


def enhance_prompt(
    current_prompt: str,
    agent_name: str,
    agent_description: str,
    selected_tools: List[str],
    mcp_servers: List[Dict[str, Any]],
    llm_model: str,
    builtin_tools_metadata: List[Dict[str, str]]
) -> str:
    """
    Enhance a system prompt using GPT-4o-mini based on agent capabilities.

    Args:
        current_prompt: Current system prompt text
        agent_name: Name of the agent
        agent_description: Description of the agent's purpose
        selected_tools: List of selected built-in tool IDs
        mcp_servers: List of MCP server configurations
        llm_model: The LLM model being used
        builtin_tools_metadata: Metadata for all builtin tools

    Returns:
        Enhanced system prompt

    Raises:
        ValueError: If OPENAI_API_KEY is not set
        Exception: If API call fails
    """
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment. "
            "Please set it in your .env file to use prompt enhancement."
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
        # Call OpenAI API
        client = openai.OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=2000,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at writing clear, effective system prompts for LLM agents. You understand how to guide agents in tool usage, multi-step reasoning, and providing helpful responses."
                },
                {
                    "role": "user",
                    "content": meta_prompt
                }
            ]
        )

        enhanced_prompt = response.choices[0].message.content.strip()

        # Basic validation
        if not enhanced_prompt or len(enhanced_prompt) < 50:
            raise ValueError("Enhanced prompt is too short or empty")

        return enhanced_prompt

    except openai.APIError as e:
        raise Exception(f"OpenAI API error: {str(e)}")
    except openai.RateLimitError:
        raise Exception("OpenAI API rate limit exceeded. Please try again in a moment.")
    except openai.AuthenticationError:
        raise Exception("OpenAI API authentication failed. Please check your OPENAI_API_KEY.")
    except Exception as e:
        raise Exception(f"Failed to enhance prompt: {str(e)}")


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
