"""
Tool registry for managing built-in and custom tools.
"""

import importlib
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from langchain_core.tools import BaseTool, tool

logger = logging.getLogger(__name__)


# Tool category constants
class ToolCategory:
    """Tool category constants for classification."""
    SEARCH = "search"
    COMPUTATION = "computation"
    UTILITY = "utility"
    DATA_PROCESSING = "data_processing"
    RETRIEVAL = "retrieval"
    CODE_EXECUTION = "code_execution"


# Built-in tool category mappings
BUILTIN_TOOL_CATEGORIES = {
    "tavily_search": ToolCategory.SEARCH,
    "wikipedia_search": ToolCategory.SEARCH,
    "calculator": ToolCategory.COMPUTATION,
    "python_repl": ToolCategory.CODE_EXECUTION,
    "get_current_datetime": ToolCategory.UTILITY,
    "string_tool": ToolCategory.DATA_PROCESSING,
    "generate_uuid": ToolCategory.UTILITY,
    "random_number": ToolCategory.UTILITY,
}

# MCP tool category inference patterns (keyword-based)
MCP_CATEGORY_PATTERNS = {
    ToolCategory.COMPUTATION: ["add", "subtract", "multiply", "divide", "calculator", "math", "calculate", "percentage", "power", "sqrt", "square_root", "factorial", "modulo", "absolute"],
    ToolCategory.SEARCH: ["search", "query", "find", "lookup"],
    ToolCategory.RETRIEVAL: ["retrieve", "get", "fetch", "document", "context", "chunk", "list", "summarize"],
    ToolCategory.DATA_PROCESSING: ["process", "transform", "parse", "format", "convert"],
    ToolCategory.UTILITY: ["generate", "create", "random", "uuid", "datetime", "time"],
}


class ToolRegistryError(Exception):
    """Exception raised for tool registry errors."""
    pass


class ToolRegistry:
    """Registry for managing agent tools."""

    def __init__(self):
        """Initialize the tool registry."""
        self._builtin_tools: Dict[str, BaseTool] = {}
        self._custom_tools: Dict[str, BaseTool] = {}
        self._pending_tools: Dict[str, Dict[str, Any]] = {}
        self._register_builtin_tools()

    def _register_builtin_tools(self):
        """Register built-in tools."""
        # Register Tavily search
        self._register_tavily_search()

        # Register Python REPL
        self._register_python_repl()

        # Register simple utility tools
        self._register_datetime_tool()
        self._register_calculator_tool()
        self._register_string_tool()
        self._register_uuid_tool()
        self._register_wikipedia_tool()
        self._register_random_number_tool()

    def _register_tavily_search(self):
        """Register Tavily search tool."""
        try:
            from langchain_tavily import TavilySearch

            tavily_api_key = os.getenv("TAVILY_API_KEY")
            if tavily_api_key:
                tavily_tool = TavilySearch(
                    max_results=5,
                    api_key=tavily_api_key
                )
                self._builtin_tools["tavily_search"] = tavily_tool
                logger.info(f"Registered Tavily search tool with API key: {'***' + tavily_api_key[-4:]}")
            else:
                logger.warning("TAVILY_API_KEY not found in environment - tavily_search tool will not be available")
        except ImportError as e:
            logger.warning(f"Tavily search tool not available: {str(e)}")

    def _register_python_repl(self):
        """Register Python REPL tool for code execution."""
        try:
            from langchain_experimental.tools import PythonREPLTool

            python_repl_tool = PythonREPLTool()
            self._builtin_tools["python_repl"] = python_repl_tool
            logger.info("Registered Python REPL tool")
        except ImportError as e:
            logger.warning(f"Python REPL tool not available: {str(e)}")

    def _register_datetime_tool(self):
        """Register datetime utility tool."""
        from datetime import datetime as dt

        @tool
        def get_current_datetime(format_string: str = "%Y-%m-%d %H:%M:%S") -> str:
            """
            Get the current date and time in the specified format.

            Args:
                format_string: Datetime format string (default: YYYY-MM-DD HH:MM:SS).
                    Common formats:
                    - %Y-%m-%d: 2024-01-15
                    - %H:%M:%S: 14:30:00
                    - %Y-%m-%d %H:%M:%S: 2024-01-15 14:30:00
                    - %B %d, %Y: January 15, 2024

            Returns:
                Current datetime as a formatted string
            """
            try:
                return dt.now().strftime(format_string)
            except Exception as e:
                return f"Error formatting datetime: {str(e)}"

        self._builtin_tools["get_current_datetime"] = get_current_datetime
        logger.info("Registered datetime tool")

    def _register_calculator_tool(self):
        """Register calculator tool for mathematical calculations."""
        try:
            from langchain_community.tools.calculator import Calculator

            calculator_tool = Calculator()
            self._builtin_tools["calculator"] = calculator_tool
            logger.info("Registered calculator tool")
        except ImportError:
            # Fallback to simple calculator if community tool not available
            @tool
            def calculator(expression: str) -> str:
                """
                Safely evaluate mathematical expressions.

                Args:
                    expression: Mathematical expression to evaluate (e.g., "2 + 2", "5 * 10 / 2")

                Returns:
                    Result of the calculation as a string
                """
                try:
                    # Simple safe evaluation for basic math
                    # Remove any potentially dangerous characters
                    allowed_chars = set('0123456789+-*/().%** ')
                    if not all(c in allowed_chars for c in expression):
                        return "Error: Expression contains invalid characters"

                    result = eval(expression, {"__builtins__": {}}, {})
                    return str(result)
                except Exception as e:
                    return f"Error calculating: {str(e)}"

            self._builtin_tools["calculator"] = calculator
            logger.info("Registered simple calculator tool (fallback)")

    def _register_string_tool(self):
        """Register string manipulation tool."""
        @tool
        def string_tool(text: str, operation: str) -> str:
            """
            Perform various string operations on text.

            Args:
                text: The text to manipulate
                operation: Operation to perform. Options:
                    - uppercase: Convert to uppercase
                    - lowercase: Convert to lowercase
                    - reverse: Reverse the text
                    - length: Get the length of the text
                    - title: Convert to title case
                    - strip: Remove leading/trailing whitespace

            Returns:
                The manipulated text or operation result
            """
            operations = {
                "uppercase": text.upper(),
                "lowercase": text.lower(),
                "reverse": text[::-1],
                "length": str(len(text)),
                "title": text.title(),
                "strip": text.strip()
            }

            result = operations.get(operation.lower())
            if result is None:
                return f"Invalid operation '{operation}'. Valid options: {', '.join(operations.keys())}"
            return result

        self._builtin_tools["string_tool"] = string_tool
        logger.info("Registered string manipulation tool")

    def _register_uuid_tool(self):
        """Register UUID generation tool."""
        import uuid

        @tool
        def generate_uuid(version: int = 4) -> str:
            """
            Generate a universally unique identifier (UUID).

            Args:
                version: UUID version to generate (1 or 4, default: 4)
                    - Version 1: Time-based UUID
                    - Version 4: Random UUID

            Returns:
                A UUID string
            """
            try:
                if version == 1:
                    return str(uuid.uuid1())
                elif version == 4:
                    return str(uuid.uuid4())
                else:
                    return f"Invalid version {version}. Use 1 or 4."
            except Exception as e:
                return f"Error generating UUID: {str(e)}"

        self._builtin_tools["generate_uuid"] = generate_uuid
        logger.info("Registered UUID generator tool")

    def _register_wikipedia_tool(self):
        """Register Wikipedia search tool."""
        try:
            from langchain_community.tools import WikipediaQueryRun
            from langchain_community.utilities import WikipediaAPIWrapper

            wikipedia_tool = WikipediaQueryRun(
                api_wrapper=WikipediaAPIWrapper(
                    top_k_results=3,
                    doc_content_chars_max=4000
                )
            )
            self._builtin_tools["wikipedia_search"] = wikipedia_tool
            logger.info("Registered Wikipedia search tool")
        except ImportError as e:
            logger.warning(f"Wikipedia search tool not available: {str(e)}")

    def _register_random_number_tool(self):
        """Register random number generator tool."""
        import random

        @tool
        def random_number(min_val: int = 0, max_val: int = 100) -> int:
            """
            Generate a random integer between min and max (inclusive).

            Args:
                min_val: Minimum value (default: 0)
                max_val: Maximum value (default: 100)

            Returns:
                A random integer between min_val and max_val
            """
            try:
                return random.randint(min_val, max_val)
            except Exception as e:
                return f"Error generating random number: {str(e)}"

        self._builtin_tools["random_number"] = random_number
        logger.info("Registered random number generator tool")

    def register_tool(
        self,
        tool_id: str,
        tool_instance: BaseTool,
        is_custom: bool = True
    ):
        """
        Register a tool.

        Args:
            tool_id: Unique tool identifier
            tool_instance: Tool instance
            is_custom: Whether this is a custom tool
        """
        if is_custom:
            self._custom_tools[tool_id] = tool_instance
        else:
            self._builtin_tools[tool_id] = tool_instance

    def get_tool(self, tool_id: str) -> Optional[BaseTool]:
        """
        Get a tool by ID.

        Args:
            tool_id: Tool identifier

        Returns:
            Tool instance or None if not found
        """
        # Check builtin first, then custom
        return self._builtin_tools.get(tool_id) or self._custom_tools.get(tool_id)

    def get_tools(self, tool_ids: List[str]) -> List[BaseTool]:
        """
        Get multiple tools by IDs.

        Args:
            tool_ids: List of tool identifiers

        Returns:
            List of tool instances

        Raises:
            ToolRegistryError: If any tool is not found
        """
        logger.debug(f"Requested tools: {tool_ids}")

        tools = []
        missing = []

        for tool_id in tool_ids:
            tool = self.get_tool(tool_id)
            if tool:
                tools.append(tool)
                logger.debug(f"Found tool: {tool_id}")
            else:
                missing.append(tool_id)

        if missing:
            available_tools = list(self._builtin_tools.keys()) + list(self._custom_tools.keys())
            logger.error(f"Tools not found: {missing}. Available tools: {available_tools}")
            raise ToolRegistryError(f"Tools not found: {', '.join(missing)}")

        return tools

    def infer_tool_category(self, tool_name: str, tool_description: str = "") -> str:
        """
        Infer tool category based on tool name and description.

        Args:
            tool_name: Name of the tool
            tool_description: Description of the tool

        Returns:
            Category string (from ToolCategory constants)
        """
        # Check if it's a built-in tool with known category
        if tool_name in BUILTIN_TOOL_CATEGORIES:
            return BUILTIN_TOOL_CATEGORIES[tool_name]

        # For MCP and custom tools, infer from name and description
        search_text = f"{tool_name} {tool_description}".lower()

        # Check each category pattern
        category_scores = {}
        for category, patterns in MCP_CATEGORY_PATTERNS.items():
            score = sum(1 for pattern in patterns if pattern in search_text)
            if score > 0:
                category_scores[category] = score

        # Return category with highest score, or UTILITY as default
        if category_scores:
            return max(category_scores, key=category_scores.get)

        return ToolCategory.UTILITY

    def get_tools_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get all tools in a specific category.

        Args:
            category: Category to filter by (from ToolCategory constants)

        Returns:
            List of tools in that category
        """
        all_tools = self.list_tools()
        return [tool for tool in all_tools if tool.get("category") == category]

    def list_tools(self, include_pending: bool = False) -> List[Dict[str, Any]]:
        """
        List all available tools.

        Args:
            include_pending: Whether to include pending tools

        Returns:
            List of tool metadata
        """
        tools = []

        # Built-in tools
        for tool_id, tool_instance in self._builtin_tools.items():
            category = self.infer_tool_category(tool_id, tool_instance.description or "")
            tools.append({
                "tool_id": tool_id,
                "name": tool_instance.name,
                "description": tool_instance.description,
                "type": "builtin",
                "status": "active",
                "category": category,
            })

        # Custom tools
        for tool_id, tool_instance in self._custom_tools.items():
            category = self.infer_tool_category(tool_id, tool_instance.description or "")
            tools.append({
                "tool_id": tool_id,
                "name": tool_instance.name,
                "description": tool_instance.description,
                "type": "custom",
                "status": "active",
                "category": category,
            })

        # Pending tools
        if include_pending:
            for tool_id, tool_data in self._pending_tools.items():
                category = self.infer_tool_category(tool_data["name"], tool_data.get("description", ""))
                tools.append({
                    "tool_id": tool_id,
                    "name": tool_data["name"],
                    "description": tool_data["description"],
                    "type": "custom",
                    "status": "pending",
                    "created_at": tool_data["created_at"],
                    "category": category,
                })

        return tools

    def add_pending_tool(
        self,
        tool_id: str,
        name: str,
        description: str,
        code: str
    ):
        """
        Add a tool to pending review.

        Args:
            tool_id: Tool identifier
            name: Tool name
            description: Tool description
            code: Generated tool code
        """
        # Save code to pending_review directory
        pending_dir = Path("./custom_tools/pending_review")
        pending_dir.mkdir(parents=True, exist_ok=True)

        code_file = pending_dir / f"{tool_id}.py"
        code_file.write_text(code, encoding="utf-8")

        # Store metadata
        self._pending_tools[tool_id] = {
            "name": name,
            "description": description,
            "code": code,
            "code_path": str(code_file),
            "created_at": datetime.now(),
        }

    def get_pending_tool(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """
        Get pending tool information.

        Args:
            tool_id: Tool identifier

        Returns:
            Pending tool metadata or None
        """
        return self._pending_tools.get(tool_id)

    def approve_tool(self, tool_id: str, modified_code: Optional[str] = None) -> bool:
        """
        Approve a pending tool and make it active.

        Args:
            tool_id: Tool identifier
            modified_code: Optional modified code

        Returns:
            True if approved successfully

        Raises:
            ToolRegistryError: If tool not found or approval fails
        """
        if tool_id not in self._pending_tools:
            raise ToolRegistryError(f"Pending tool not found: {tool_id}")

        tool_data = self._pending_tools[tool_id]
        code = modified_code or tool_data["code"]

        try:
            # Execute code to define the tool
            namespace = {}
            exec(code, namespace)

            # Find the tool function (decorated with @tool)
            tool_func = None
            for item in namespace.values():
                if callable(item) and hasattr(item, 'name'):
                    # This is likely a tool
                    tool_func = item
                    break

            if not tool_func:
                raise ToolRegistryError("No valid tool function found in code")

            # Register as custom tool
            self._custom_tools[tool_id] = tool_func

            # Move from pending
            del self._pending_tools[tool_id]

            # Move code file to custom_tools
            pending_path = Path(tool_data["code_path"])
            active_path = Path("./custom_tools") / f"{tool_id}.py"
            active_path.parent.mkdir(parents=True, exist_ok=True)

            if pending_path.exists():
                pending_path.rename(active_path)

            return True

        except Exception as e:
            raise ToolRegistryError(f"Failed to approve tool: {str(e)}")

    def reject_tool(self, tool_id: str) -> bool:
        """
        Reject a pending tool.

        Args:
            tool_id: Tool identifier

        Returns:
            True if rejected successfully
        """
        if tool_id not in self._pending_tools:
            return False

        tool_data = self._pending_tools[tool_id]

        # Delete code file
        code_path = Path(tool_data["code_path"])
        if code_path.exists():
            code_path.unlink()

        # Remove from pending
        del self._pending_tools[tool_id]

        return True

    def test_tool(self, tool_id: str, test_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test a tool with given input.

        Args:
            tool_id: Tool identifier
            test_input: Test input arguments

        Returns:
            Test result with output, error, and execution time
        """
        import time

        tool = self.get_tool(tool_id)
        if not tool:
            # Check if it's a pending tool
            pending = self.get_pending_tool(tool_id)
            if not pending:
                raise ToolRegistryError(f"Tool not found: {tool_id}")

            # Execute pending tool code temporarily for testing
            try:
                namespace = {}
                exec(pending["code"], namespace)

                tool_func = None
                for item in namespace.values():
                    if callable(item) and hasattr(item, 'name'):
                        tool_func = item
                        break

                if not tool_func:
                    return {
                        "success": False,
                        "output": None,
                        "error": "No valid tool function found",
                        "execution_time": 0.0,
                    }

                tool = tool_func
            except Exception as e:
                return {
                    "success": False,
                    "output": None,
                    "error": f"Failed to load tool: {str(e)}",
                    "execution_time": 0.0,
                }

        try:
            start_time = time.time()
            output = tool.invoke(test_input)
            execution_time = time.time() - start_time

            return {
                "success": True,
                "output": output,
                "error": None,
                "execution_time": execution_time,
            }

        except Exception as e:
            execution_time = time.time() - start_time
            return {
                "success": False,
                "output": None,
                "error": str(e),
                "execution_time": execution_time,
            }

    def delete_tool(self, tool_id: str) -> bool:
        """
        Delete a custom tool.

        Args:
            tool_id: Tool identifier

        Returns:
            True if deleted, False if not found
        """
        if tool_id in self._custom_tools:
            del self._custom_tools[tool_id]

            # Delete code file
            code_path = Path(f"./custom_tools/{tool_id}.py")
            if code_path.exists():
                code_path.unlink()

            return True

        return False

    def create_python_repl_tool(self) -> BaseTool:
        """
        Create a Python REPL tool for code execution.

        Returns:
            Python REPL tool
        """
        try:
            from langchain_experimental.tools import PythonREPLTool

            return PythonREPLTool()
        except ImportError:
            raise ToolRegistryError("langchain-experimental required for Python REPL tool")

    def create_web_search_tool(self, provider: str = "tavily") -> BaseTool:
        """
        Create a web search tool.

        Args:
            provider: Search provider (tavily, google)

        Returns:
            Web search tool
        """
        if provider == "tavily":
            return self.get_tool("tavily_search")
        elif provider == "google":
            # Implement Google search
            raise NotImplementedError("Google search not yet implemented")
        else:
            raise ToolRegistryError(f"Unknown search provider: {provider}")
