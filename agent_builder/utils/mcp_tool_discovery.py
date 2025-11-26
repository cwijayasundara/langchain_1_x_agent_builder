"""
MCP Tool Discovery Utility
Discovers available tools from MCP servers for selection in the UI.
"""

import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
import os


async def discover_mcp_tools(
    server_name: str,
    transport: str,
    url: Optional[str] = None,
    command: Optional[str] = None,
    args: Optional[List[str]] = None,
    env: Optional[Dict[str, str]] = None
) -> List[Dict[str, Any]]:
    """
    Discover available tools from an MCP server.

    Args:
        server_name: Name of the MCP server
        transport: Transport type ("streamable_http", "stdio", "sse")
        url: Server URL (for HTTP transports)
        command: Command to run (for stdio transport)
        args: Command arguments (for stdio transport)
        env: Environment variables (for stdio transport)

    Returns:
        List of tool dictionaries with name, description, and input schema
    """
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        # Build connection config
        connection = {}
        
        if transport == "stdio":
            if not command:
                return []
            
            # Resolve relative paths
            if not os.path.isabs(command):
                project_root = Path(__file__).parent.parent.parent
                command_path = project_root / command
                if command_path.exists():
                    command = str(command_path)
            
            connection = {
                "command": command,
                "args": args or [],
                "env": env or {},
                "transport": "stdio",
            }
        elif transport in ["streamable_http", "http", "sse"]:
            if not url:
                return []
            
            transport_type = "streamable_http" if transport == "http" else transport
            connection = {
                "url": url,
                "transport": transport_type,
            }
        else:
            return []

        # Create client and discover tools
        connections = {server_name: connection}
        
        # Debug: print connection details (without sensitive info)
        print(f"Connecting to MCP server '{server_name}' with transport '{transport}'")
        if url:
            print(f"  URL: {url}")
        if command:
            print(f"  Command: {command}")
        
        client = MultiServerMCPClient(connections=connections)
        
        try:
            # Get tools from the server
            tools = await client.get_tools()
            
            print(f"Retrieved {len(tools)} tools from server '{server_name}'")
            
            # Convert tools to a simple format for UI display
            tool_list = []
            for tool in tools:
                # Extract tool name (remove server prefix if present)
                tool_name = tool.name
                original_name = tool_name
                
                # langchain-mcp-adapters prefixes tools with server name
                if tool_name.startswith(f"{server_name}_"):
                    tool_name = tool_name[len(f"{server_name}_"):]
                
                print(f"  Tool: {original_name} -> {tool_name}")
                
                tool_info = {
                    "name": tool_name,
                    "description": tool.description or "No description available",
                    "input_schema": {}
                }
                
                # Extract input schema if available
                if hasattr(tool, 'args_schema') and tool.args_schema:
                    try:
                        tool_info["input_schema"] = tool.args_schema.model_json_schema() if hasattr(tool.args_schema, 'model_json_schema') else {}
                    except:
                        pass
                
                tool_list.append(tool_info)
            
            print(f"Returning {len(tool_list)} tools")
            return tool_list
        except Exception as e:
            print(f"Error getting tools: {str(e)}")
            raise
        finally:
            # Clean up client connection
            try:
                if hasattr(client, 'close'):
                    await client.close()
            except Exception as cleanup_error:
                print(f"Warning: Error closing client: {cleanup_error}")

    except ImportError as e:
        raise Exception(f"langchain-mcp-adapters not installed: {str(e)}")
    except Exception as e:
        # Re-raise with more context for better error messages
        raise Exception(f"Failed to discover tools from {server_name}: {str(e)}")


def discover_mcp_tools_sync(
    server_name: str,
    transport: str,
    url: Optional[str] = None,
    command: Optional[str] = None,
    args: Optional[List[str]] = None,
    env: Optional[Dict[str, str]] = None
) -> List[Dict[str, Any]]:
    """
    Synchronous wrapper for discover_mcp_tools.
    Handles Streamlit's threading model properly by always using a new thread.
    
    Args:
        Same as discover_mcp_tools
        
    Returns:
        List of tool dictionaries
        
    Raises:
        Exception: If discovery fails
    """
    import threading
    import queue
    
    # Use a queue to pass results between threads
    result_queue = queue.Queue()
    exception_queue = queue.Queue()
    
    def run_in_new_loop():
        """Run async function in a new event loop in this thread."""
        try:
            # Create a new event loop for this thread
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                result = new_loop.run_until_complete(
                    discover_mcp_tools(server_name, transport, url, command, args, env)
                )
                result_queue.put(result)
            finally:
                new_loop.close()
        except Exception as e:
            exception_queue.put(e)
    
    # Always run in a new thread to avoid event loop conflicts with Streamlit
    thread = threading.Thread(target=run_in_new_loop, daemon=True)
    thread.start()
    thread.join(timeout=15)
    
    if thread.is_alive():
        raise Exception("Tool discovery timed out after 15 seconds")
    
    # Check for exceptions first
    if not exception_queue.empty():
        exception = exception_queue.get()
        raise Exception(f"Tool discovery failed: {str(exception)}")
    
    # Get result
    if not result_queue.empty():
        return result_queue.get()
    
    # If we get here, something went wrong
    raise Exception("Tool discovery completed but no result was returned")

