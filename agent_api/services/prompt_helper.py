"""
Prompt helper for generating dynamic tool documentation and usage guidance.
"""

from typing import Any, Dict, List
from langchain_core.tools import BaseTool
from agent_api.services.tool_registry import ToolCategory


class PromptHelper:
    """Helper class for generating tool-related prompt components."""

    # Tool usage examples by category
    CATEGORY_EXAMPLES = {
        ToolCategory.COMPUTATION: [
            '"What is 15% of 250?" → use calculate_percentage',
            '"Calculate 45 * 67" → use multiply',
            '"What\'s the square root of 144?" → use square_root',
            '"If I have $1000 and spend $234.56, how much is left?" → use subtract',
        ],
        ToolCategory.SEARCH: [
            '"What\'s the latest news about AI?" → use web search',
            '"Who won the election in 2024?" → use web search',
            '"Current stock price of Apple" → use web search',
            '"Recent research on climate change" → use web search',
        ],
        ToolCategory.RETRIEVAL: [
            '"What does our policy say about X?" → use document retrieval',
            '"Find information about Y in our knowledge base" → use document retrieval',
            '"Summarize all documents related to Z" → use document summarization',
        ],
        ToolCategory.CODE_EXECUTION: [
            '"Run this Python code" → use python_repl',
            '"Execute a script to analyze data" → use python_repl',
        ],
        ToolCategory.UTILITY: [
            '"What time is it?" → use get_current_datetime',
            '"Generate a unique ID" → use generate_uuid',
            '"Pick a random number between 1 and 100" → use random_number',
        ],
        ToolCategory.DATA_PROCESSING: [
            '"Convert this text to uppercase" → use string_tool',
            '"Get the length of this string" → use string_tool',
        ],
    }

    # Category descriptions
    CATEGORY_DESCRIPTIONS = {
        ToolCategory.COMPUTATION: "Use these tools for ANY mathematical operations, calculations, or numerical analysis",
        ToolCategory.SEARCH: "Use these tools for current events, recent information, or facts you don't have",
        ToolCategory.RETRIEVAL: "Use these tools to access your knowledge base and retrieve stored documents",
        ToolCategory.CODE_EXECUTION: "Use these tools to execute code and perform complex programmatic tasks",
        ToolCategory.UTILITY: "Use these tools for general utility functions like time, UUIDs, random values",
        ToolCategory.DATA_PROCESSING: "Use these tools to process, transform, or manipulate text and data",
    }

    # Category display names
    CATEGORY_NAMES = {
        ToolCategory.COMPUTATION: "COMPUTATION TOOLS",
        ToolCategory.SEARCH: "SEARCH & INFORMATION TOOLS",
        ToolCategory.RETRIEVAL: "DOCUMENT RETRIEVAL TOOLS",
        ToolCategory.CODE_EXECUTION: "CODE EXECUTION TOOLS",
        ToolCategory.UTILITY: "UTILITY TOOLS",
        ToolCategory.DATA_PROCESSING: "DATA PROCESSING TOOLS",
    }

    @staticmethod
    def categorize_tools(tools: List[BaseTool], tool_registry) -> Dict[str, List[BaseTool]]:
        """
        Categorize tools into groups.

        Args:
            tools: List of tool instances
            tool_registry: ToolRegistry instance for category inference

        Returns:
            Dictionary mapping category to list of tools
        """
        categorized = {}

        for tool in tools:
            category = tool_registry.infer_tool_category(
                tool.name,
                tool.description or ""
            )

            if category not in categorized:
                categorized[category] = []

            categorized[category].append(tool)

        return categorized

    @staticmethod
    def generate_tool_documentation(
        tools: List[BaseTool],
        tool_registry,
        include_examples: bool = True
    ) -> str:
        """
        Generate formatted tool documentation for system prompts.

        Args:
            tools: List of tool instances
            tool_registry: ToolRegistry instance for category inference
            include_examples: Whether to include usage examples

        Returns:
            Formatted tool documentation string
        """
        if not tools:
            return "No tools available."

        categorized = PromptHelper.categorize_tools(tools, tool_registry)

        doc_lines = ["## Available Tools (By Category)", ""]

        # Sort categories for consistent output
        sorted_categories = sorted(categorized.keys())

        for idx, category in enumerate(sorted_categories, 1):
            category_tools = categorized[category]

            # Category header
            category_name = PromptHelper.CATEGORY_NAMES.get(category, category.upper())
            doc_lines.append(f"### {idx}. {category_name}")
            doc_lines.append("")

            # Category description
            description = PromptHelper.CATEGORY_DESCRIPTIONS.get(
                category,
                f"Tools in the {category} category"
            )
            doc_lines.append(description + ":")
            doc_lines.append("")

            # List tools in this category
            for tool in category_tools:
                tool_line = f"- {tool.name}"
                if tool.description:
                    # Take first line of description for brevity
                    first_line = tool.description.split('\n')[0].strip()
                    tool_line += f": {first_line}"
                doc_lines.append(tool_line)

            doc_lines.append("")

            # Add examples if requested
            if include_examples and category in PromptHelper.CATEGORY_EXAMPLES:
                doc_lines.append(f"Examples requiring {category} tools:")
                doc_lines.append("")
                for example in PromptHelper.CATEGORY_EXAMPLES[category]:
                    doc_lines.append(f"- {example}")
                doc_lines.append("")

        return "\n".join(doc_lines)

    @staticmethod
    def generate_tool_selection_guidance(
        categories_present: List[str],
        emphasize_accuracy: bool = True
    ) -> str:
        """
        Generate tool selection decision tree and guidelines.

        Args:
            categories_present: List of tool categories that are available
            emphasize_accuracy: Whether to emphasize using the right tool

        Returns:
            Formatted guidance string
        """
        guidance_lines = ["## Tool Selection Decision Tree", ""]

        # Build decision tree based on available categories
        decision_rules = []

        if ToolCategory.COMPUTATION in categories_present:
            decision_rules.append(
                "1. Does the query involve ANY calculations or numbers? → Use COMPUTATION tools"
            )

        if ToolCategory.SEARCH in categories_present:
            decision_rules.append(
                f"{len(decision_rules) + 1}. Does the query ask about current events or recent information? → Use SEARCH tools"
            )

        if ToolCategory.RETRIEVAL in categories_present:
            decision_rules.append(
                f"{len(decision_rules) + 1}. Does the query reference documents, policies, or your knowledge base? → Use RETRIEVAL tools"
            )

        if ToolCategory.CODE_EXECUTION in categories_present:
            decision_rules.append(
                f"{len(decision_rules) + 1}. Does the query need code execution or complex data analysis? → Use CODE EXECUTION tools"
            )

        if ToolCategory.DATA_PROCESSING in categories_present:
            decision_rules.append(
                f"{len(decision_rules) + 1}. Does the query need text/data transformation? → Use DATA PROCESSING tools"
            )

        decision_rules.append(
            f"{len(decision_rules) + 1}. Does the query need multiple capabilities? → Use multiple tool categories in sequence"
        )

        guidance_lines.extend(decision_rules)
        guidance_lines.append("")

        # Add critical guidance for computation if present
        if emphasize_accuracy and ToolCategory.COMPUTATION in categories_present:
            guidance_lines.extend([
                "**CRITICAL**: For ANY question involving numbers, calculations, percentages, or math, you MUST use the calculator tools. Do NOT attempt mental math or estimation.",
                ""
            ])

        return "\n".join(guidance_lines)

    @staticmethod
    def generate_research_guidelines() -> str:
        """
        Generate general research and tool usage guidelines.

        Returns:
            Formatted guidelines string
        """
        return """## Research Guidelines

1. **Always Use the Right Tool**: Match the query type to the appropriate tool category above
2. **Multi-Source Analysis**: For research queries, analyze multiple sources to ensure accuracy
3. **Show Your Work**: For calculations, show the operation and result clearly
4. **Cite Sources**: When using web search or documents, cite your sources
5. **Structured Responses**: Provide clear, well-organized answers
6. **Acknowledge Uncertainty**: If information is uncertain or unavailable, say so clearly

If a tool is unavailable, use your existing knowledge to provide the best possible answer."""

    @staticmethod
    def generate_complete_tool_section(
        tools: List[BaseTool],
        tool_registry,
        include_examples: bool = True,
        emphasize_accuracy: bool = True
    ) -> str:
        """
        Generate a complete tool section for system prompts.

        Args:
            tools: List of tool instances
            tool_registry: ToolRegistry instance
            include_examples: Whether to include usage examples
            emphasize_accuracy: Whether to emphasize accuracy in tool selection

        Returns:
            Complete formatted tool section
        """
        sections = []

        # Tool documentation
        tool_docs = PromptHelper.generate_tool_documentation(
            tools, tool_registry, include_examples
        )
        sections.append(tool_docs)

        # Selection guidance
        categorized = PromptHelper.categorize_tools(tools, tool_registry)
        categories = list(categorized.keys())

        selection_guidance = PromptHelper.generate_tool_selection_guidance(
            categories, emphasize_accuracy
        )
        sections.append(selection_guidance)

        # Research guidelines
        guidelines = PromptHelper.generate_research_guidelines()
        sections.append(guidelines)

        return "\n".join(sections)
