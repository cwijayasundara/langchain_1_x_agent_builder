"""
MCP server management endpoints.
"""

import logging
import time
from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status

from agent_api.dependencies import get_mcp_server_manager
from agent_api.models.schemas import (
    APIResponse,
    ErrorDetail,
    MCPServerCreateRequest,
    MCPServerDeleteResponse,
    MCPServerDetailResponse,
    MCPServerDiscoverToolsRequest,
    MCPServerDiscoverToolsResponse,
    MCPServerInfo,
    MCPServerListResponse,
    MCPServerValidateRequest,
    MCPServerValidateResponse,
    MCPToolInfo,
)
from agent_api.services.mcp_server_manager import MCPServerError, MCPServerManager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/list", response_model=APIResponse)
async def list_mcp_servers(
    mcp_server_manager: MCPServerManager = Depends(get_mcp_server_manager),
):
    """
    List all available MCP server configurations.

    Returns:
        API response with list of MCP servers
    """
    try:
        servers = mcp_server_manager.list_servers()

        # Convert to MCPServerInfo objects
        server_infos = [
            MCPServerInfo(
                name=s["name"],
                description=s.get("description"),
                transport=s["transport"],
                url=s.get("url"),
                command=s.get("command"),
                stateful=s.get("stateful", False),
                version=s.get("version", "1.0.0"),
                tags=s.get("tags", []),
                selected_tools=s.get("selected_tools"),
                tool_count=None,  # Would require connecting to server
                config_path=s["config_path"],
                created_at=s["created_at"],
                updated_at=s["updated_at"],
            )
            for s in servers
        ]

        return APIResponse(
            success=True,
            data=MCPServerListResponse(
                servers=server_infos,
                total=len(server_infos),
            ).model_dump(),
        )

    except Exception as e:
        logger.error(f"Error listing MCP servers: {str(e)}")
        return APIResponse(
            success=False,
            error=ErrorDetail(
                code="MCP_LIST_ERROR",
                message=f"Failed to list MCP servers: {str(e)}",
            ),
        )


@router.post("/create", response_model=APIResponse)
async def create_mcp_server(
    request: MCPServerCreateRequest,
    mcp_server_manager: MCPServerManager = Depends(get_mcp_server_manager),
):
    """
    Create a new MCP server configuration.

    Args:
        request: MCP server creation request with configuration

    Returns:
        API response with creation result
    """
    try:
        # Validate first
        validation = mcp_server_manager.validate_server(
            request.config.model_dump(exclude_none=True)
        )

        if not validation["valid"]:
            return APIResponse(
                success=False,
                error=ErrorDetail(
                    code="VALIDATION_ERROR",
                    message="MCP server configuration validation failed",
                    details={"errors": validation["errors"]},
                ),
            )

        # Save the configuration
        config_path = mcp_server_manager.save_server(
            request.config,
            overwrite=request.overwrite
        )

        return APIResponse(
            success=True,
            data={
                "server_name": request.config.name,
                "config_path": config_path,
                "message": f"MCP server '{request.config.name}' created successfully",
                "warnings": validation.get("warnings", []),
            },
        )

    except MCPServerError as e:
        return APIResponse(
            success=False,
            error=ErrorDetail(
                code="MCP_CREATE_ERROR",
                message=str(e),
            ),
        )
    except Exception as e:
        logger.error(f"Error creating MCP server: {str(e)}")
        return APIResponse(
            success=False,
            error=ErrorDetail(
                code="MCP_CREATE_ERROR",
                message=f"Failed to create MCP server: {str(e)}",
            ),
        )


@router.get("/{server_name}", response_model=APIResponse)
async def get_mcp_server(
    server_name: str,
    mcp_server_manager: MCPServerManager = Depends(get_mcp_server_manager),
):
    """
    Get a specific MCP server configuration.

    Args:
        server_name: Name of the MCP server

    Returns:
        API response with server details
    """
    try:
        server_info = mcp_server_manager.get_server_info(server_name)
        server_def = mcp_server_manager.load_server(server_name)

        info = MCPServerInfo(
            name=server_info["name"],
            description=server_info.get("description"),
            transport=server_info["transport"],
            url=server_info.get("url"),
            command=server_info.get("command"),
            stateful=server_info.get("stateful", False),
            version=server_info.get("version", "1.0.0"),
            tags=server_info.get("tags", []),
            selected_tools=server_info.get("selected_tools"),
            tool_count=None,
            config_path=server_info["config_path"],
            created_at=server_info["created_at"],
            updated_at=server_info["updated_at"],
        )

        return APIResponse(
            success=True,
            data=MCPServerDetailResponse(
                server_info=info,
                config=server_def,
                discovered_tools=None,
            ).model_dump(),
        )

    except MCPServerError as e:
        return APIResponse(
            success=False,
            error=ErrorDetail(
                code="MCP_NOT_FOUND",
                message=str(e),
            ),
        )
    except Exception as e:
        logger.error(f"Error getting MCP server '{server_name}': {str(e)}")
        return APIResponse(
            success=False,
            error=ErrorDetail(
                code="MCP_GET_ERROR",
                message=f"Failed to get MCP server: {str(e)}",
            ),
        )


@router.delete("/{server_name}", response_model=APIResponse)
async def delete_mcp_server(
    server_name: str,
    mcp_server_manager: MCPServerManager = Depends(get_mcp_server_manager),
):
    """
    Delete an MCP server configuration.

    Args:
        server_name: Name of the MCP server to delete

    Returns:
        API response with deletion result
    """
    try:
        deleted = mcp_server_manager.delete_server(server_name)

        if deleted:
            return APIResponse(
                success=True,
                data=MCPServerDeleteResponse(
                    server_name=server_name,
                    message=f"MCP server '{server_name}' deleted successfully",
                ).model_dump(),
            )
        else:
            return APIResponse(
                success=False,
                error=ErrorDetail(
                    code="MCP_NOT_FOUND",
                    message=f"MCP server '{server_name}' not found",
                ),
            )

    except Exception as e:
        logger.error(f"Error deleting MCP server '{server_name}': {str(e)}")
        return APIResponse(
            success=False,
            error=ErrorDetail(
                code="MCP_DELETE_ERROR",
                message=f"Failed to delete MCP server: {str(e)}",
            ),
        )


@router.post("/validate", response_model=APIResponse)
async def validate_mcp_server(
    request: MCPServerValidateRequest,
    mcp_server_manager: MCPServerManager = Depends(get_mcp_server_manager),
):
    """
    Validate an MCP server configuration without saving.

    Args:
        request: Validation request with server configuration

    Returns:
        API response with validation result
    """
    try:
        validation = mcp_server_manager.validate_server(request.config)

        return APIResponse(
            success=True,
            data=MCPServerValidateResponse(
                valid=validation["valid"],
                errors=validation.get("errors", []),
                warnings=validation.get("warnings", []),
            ).model_dump(),
        )

    except Exception as e:
        logger.error(f"Error validating MCP server config: {str(e)}")
        return APIResponse(
            success=False,
            error=ErrorDetail(
                code="MCP_VALIDATE_ERROR",
                message=f"Failed to validate MCP server config: {str(e)}",
            ),
        )


@router.post("/{server_name}/discover-tools", response_model=APIResponse)
async def discover_mcp_server_tools(
    server_name: str,
    request: MCPServerDiscoverToolsRequest = MCPServerDiscoverToolsRequest(),
    mcp_server_manager: MCPServerManager = Depends(get_mcp_server_manager),
):
    """
    Discover tools from a running MCP server.

    This endpoint connects to the MCP server and retrieves the list of
    available tools. The server must be running for this to succeed.

    Args:
        server_name: Name of the MCP server
        request: Optional request with timeout settings

    Returns:
        API response with discovered tools
    """
    try:
        # Load server config
        server_def = mcp_server_manager.load_server(server_name)

        start_time = time.time()

        # Try to discover tools using the MCP tool discovery utility
        try:
            from agent_builder.utils.mcp_tool_discovery import discover_mcp_tools_sync

            # Build server config for discovery
            server_config = {
                "name": server_def.name,
                "transport": server_def.transport,
            }
            if server_def.url:
                server_config["url"] = server_def.url
            if server_def.command:
                server_config["command"] = server_def.command
            if server_def.args:
                server_config["args"] = server_def.args

            discovered = discover_mcp_tools_sync([server_config], timeout=request.timeout)

            discovery_time = time.time() - start_time

            # Convert to MCPToolInfo
            tools = [
                MCPToolInfo(
                    name=tool["name"],
                    description=tool.get("description"),
                    input_schema=tool.get("input_schema"),
                )
                for tool in discovered
            ]

            return APIResponse(
                success=True,
                data=MCPServerDiscoverToolsResponse(
                    server_name=server_name,
                    tools=tools,
                    total=len(tools),
                    discovery_time=discovery_time,
                ).model_dump(),
            )

        except ImportError:
            # MCP discovery not available
            return APIResponse(
                success=False,
                error=ErrorDetail(
                    code="MCP_DISCOVERY_UNAVAILABLE",
                    message="MCP tool discovery is not available. Ensure langchain-mcp-adapters is installed.",
                ),
            )
        except Exception as e:
            return APIResponse(
                success=False,
                error=ErrorDetail(
                    code="MCP_DISCOVERY_ERROR",
                    message=f"Failed to discover tools: {str(e)}. Ensure the MCP server is running.",
                ),
            )

    except MCPServerError as e:
        return APIResponse(
            success=False,
            error=ErrorDetail(
                code="MCP_NOT_FOUND",
                message=str(e),
            ),
        )
    except Exception as e:
        logger.error(f"Error discovering tools from MCP server '{server_name}': {str(e)}")
        return APIResponse(
            success=False,
            error=ErrorDetail(
                code="MCP_DISCOVERY_ERROR",
                message=f"Failed to discover tools: {str(e)}",
            ),
        )
