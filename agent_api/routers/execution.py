"""
Agent execution endpoints.
"""

import logging
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from agent_api.dependencies import get_agent_factory
from agent_api.models.schemas import (
    AgentInvokeRequest,
    AgentInvokeResponse,
    AgentMessage,
    APIResponse,
    ErrorDetail,
    ToolCall,
)
from agent_api.services.agent_factory import AgentFactory

logger = logging.getLogger(__name__)

router = APIRouter()


def _extract_new_messages(result_messages: list, input_count: int) -> list:
    """
    Extract only new messages from LangGraph result.

    LangGraph with checkpointer returns the full conversation history.
    We need to return only the new messages from this invocation.

    Args:
        result_messages: All messages from agent result
        input_count: Number of messages sent in the request

    Returns:
        List of only new messages from this invocation
    """
    if len(result_messages) <= input_count:
        return []
    return result_messages[input_count:]


@router.post("/{agent_id}/invoke", response_model=APIResponse)
async def invoke_agent(
    agent_id: str,
    request: AgentInvokeRequest,
    agent_factory: AgentFactory = Depends(get_agent_factory),
):
    """
    Invoke an agent with a message (non-streaming).

    Supports runtime overrides for LLM, tools, and prompt via the
    runtime_override field. Overrides persist for the session (thread).

    Args:
        agent_id: Agent identifier
        request: Invocation request with messages, context, and optional runtime_override

    Returns:
        Agent response
    """
    has_override = request.runtime_override is not None
    logger.info(
        f"Agent invocation request: agent_id={agent_id}, thread_id={request.thread_id}, "
        f"message_count={len(request.messages)}, has_override={has_override}"
    )

    try:
        # Generate thread_id early since we need it for override lookup
        thread_id = request.thread_id or str(uuid4())

        # Get agent with override applied (uses session cache)
        agent, effective_system_prompt = await agent_factory.get_agent_with_override(
            agent_name=agent_id,
            thread_id=thread_id,
            override=request.runtime_override
        )

        if not agent:
            logger.warning(f"Agent not found or not deployed: {agent_id}")
            return APIResponse(
                success=False,
                error=ErrorDetail(
                    code="NOT_FOUND",
                    message=f"Agent not found or not deployed: {agent_id}",
                ),
            )

        logger.debug(f"Retrieved agent '{agent_id}' from factory (override={has_override})")

        # Get agent metadata for store access
        metadata = agent_factory.get_agent_metadata(agent_id)
        store = metadata.get("store") if metadata else None

        # Build input and track count for filtering response
        input_messages = [
            {"role": msg.role, "content": msg.content} for msg in request.messages
        ]
        input_message_count = len(input_messages)
        input_data = {"messages": input_messages}

        # Build config
        config = {
            "configurable": {}
        }

        # Add thread ID for conversation continuity (already generated above)
        config["configurable"]["thread_id"] = thread_id

        logger.debug(f"Using thread_id: {thread_id}")

        # Add context if provided
        if request.context:
            logger.debug(f"Adding context: {list(request.context.keys())}")
            # Context should be passed differently based on agent setup
            # For now, we'll add it to configurable
            for key, value in request.context.items():
                config["configurable"][key] = value

        # Add store if available
        if store:
            config["store"] = store

        logger.debug(f"Invoking agent with config: {list(config.keys())}")

        # Invoke agent with detailed error tracking
        try:
            logger.debug(f"Starting agent invocation for {agent_id}")
            result = await agent.ainvoke(input_data, config=config)
            logger.debug(f"Agent invocation completed, processing result")
        except Exception as invoke_error:
            logger.error(
                f"❌ Agent invocation FAILED during execution: {type(invoke_error).__name__}: {str(invoke_error)}",
                exc_info=True,
                extra={
                    "agent_id": agent_id,
                    "thread_id": thread_id,
                    "error_type": type(invoke_error).__name__,
                    "error_details": str(invoke_error)
                }
            )
            # Re-raise to be caught by outer exception handler
            raise

        total_result_messages = len(result.get('messages', []))
        logger.info(f"Agent invocation successful: agent_id={agent_id}, thread_id={thread_id}, total_messages={total_result_messages}")

        # Parse response with tool call tracking
        messages = []
        tool_calls_by_id = {}  # Track tool calls for correlation with results

        if "messages" in result:
            # Extract only NEW messages (skip input messages already known to client)
            new_messages = _extract_new_messages(result["messages"], input_message_count)
            logger.debug(f"Filtering messages: {total_result_messages} total → {len(new_messages)} new")

            for msg in new_messages:
                # Parse message based on type
                role = getattr(msg, "type", "ai")
                content = getattr(msg, "content", "")

                tool_calls_data = None

                # Handle AIMessage with tool_calls
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tool_calls_data = []
                    for tc in msg.tool_calls:
                        tool_call = ToolCall(
                            id=tc.get("id", ""),
                            name=tc.get("name", ""),
                            args=tc.get("args", {}),
                            result=None,
                        )
                        tool_calls_data.append(tool_call)
                        tool_calls_by_id[tc.get("id")] = tool_call

                        # Log tool call
                        logger.info(
                            f"🔧 Tool called: {tc.get('name')} | args={tc.get('args')}",
                            extra={
                                "agent_id": agent_id,
                                "thread_id": thread_id,
                                "tool_name": tc.get("name"),
                                "tool_args": tc.get("args"),
                                "tool_call_id": tc.get("id"),
                            }
                        )

                # Handle ToolMessage with results
                if role == "tool" and hasattr(msg, "tool_call_id"):
                    tool_call_id = msg.tool_call_id
                    if tool_call_id in tool_calls_by_id:
                        # Update the tool call with result
                        tool_calls_by_id[tool_call_id].result = content

                        # Log tool result
                        logger.info(
                            f"✅ Tool result: {tool_calls_by_id[tool_call_id].name} → {content}",
                            extra={
                                "agent_id": agent_id,
                                "thread_id": thread_id,
                                "tool_name": tool_calls_by_id[tool_call_id].name,
                                "tool_result": content,
                                "tool_call_id": tool_call_id,
                            }
                        )

                # Skip intermediate AI messages (have tool_calls but empty content)
                # These are mid-execution states that shouldn't be shown to users
                is_intermediate_ai = (
                    role == "ai" and
                    not content and
                    tool_calls_data is not None
                )

                if is_intermediate_ai:
                    logger.debug(f"Skipping intermediate AI message with {len(tool_calls_data)} tool calls")
                    continue

                # Generate unique message ID for deduplication
                message_id = str(uuid4())

                messages.append(
                    AgentMessage(
                        id=message_id,
                        role=role,
                        content=content,
                        tool_calls=tool_calls_data,
                    )
                )

        # Log summary of tools used
        if tool_calls_by_id:
            tools_used = [tc.name for tc in tool_calls_by_id.values()]
            logger.info(
                f"📊 Tool usage summary: {len(tools_used)} tool(s) invoked → {tools_used}",
                extra={
                    "agent_id": agent_id,
                    "thread_id": thread_id,
                    "tools_used": tools_used,
                    "tool_count": len(tools_used),
                }
            )

        # Extract metadata (token usage, costs, etc.)
        response_metadata = {}
        if "metadata" in result:
            response_metadata = result["metadata"]

        response = AgentInvokeResponse(
            messages=messages,
            thread_id=thread_id,
            metadata=response_metadata,
        )

        # Debug: Log response summary for UI troubleshooting
        logger.info(
            f"📤 Returning response to UI: {len(messages)} messages, "
            f"roles=[{', '.join(m.role for m in messages)}]"
        )
        for idx, msg in enumerate(messages):
            content_preview = msg.content[:100] if len(msg.content) > 100 else msg.content
            logger.info(
                f"  Message {idx+1}: role='{msg.role}', "
                f"content_length={len(msg.content)}, "
                f"has_tool_calls={msg.tool_calls is not None}, "
                f"preview='{content_preview}'"
            )

        return APIResponse(success=True, data=response.model_dump())

    except Exception as e:
        logger.error(
            f"Agent execution error: agent_id={agent_id}, error={str(e)}",
            exc_info=True,
            extra={"agent_id": agent_id, "thread_id": request.thread_id}
        )
        return APIResponse(
            success=False,
            error=ErrorDetail(
                code="EXECUTION_ERROR",
                message=str(e),
                details={"agent_id": agent_id, "error_type": type(e).__name__},
            ),
        )


@router.websocket("/{agent_id}/stream")
async def stream_agent(
    websocket: WebSocket,
    agent_id: str,
    agent_factory: AgentFactory = Depends(get_agent_factory),
):
    """
    Stream agent execution via WebSocket.

    Args:
        websocket: WebSocket connection
        agent_id: Agent identifier
        agent_factory: Agent factory dependency
    """
    await websocket.accept()

    try:
        # Get agent
        agent = await agent_factory.get_agent(agent_id)
        if not agent:
            await websocket.send_json({
                "error": {
                    "code": "NOT_FOUND",
                    "message": f"Agent not found or not deployed: {agent_id}",
                }
            })
            await websocket.close()
            return

        # Get agent metadata
        metadata = agent_factory.get_agent_metadata(agent_id)
        store = metadata.get("store") if metadata else None
        config_obj = metadata.get("config") if metadata else None

        # Receive messages from client
        while True:
            data = await websocket.receive_json()

            # Parse request
            messages = data.get("messages", [])
            context = data.get("context", {})
            thread_id = data.get("thread_id") or str(uuid4())

            # Build input
            input_data = {"messages": messages}

            # Build config
            config = {"configurable": {"thread_id": thread_id}}

            if context:
                for key, value in context.items():
                    config["configurable"][key] = value

            if store:
                config["store"] = store

            # Determine streaming mode
            stream_mode = "updates"
            if config_obj and config_obj.streaming:
                if len(config_obj.streaming.modes) > 1:
                    stream_mode = config_obj.streaming.modes
                else:
                    stream_mode = config_obj.streaming.modes[0]

            # Stream response
            try:
                for chunk in agent.stream(input_data, config=config, stream_mode=stream_mode):
                    # Send chunk to client
                    await websocket.send_json({
                        "type": "chunk",
                        "data": str(chunk),  # Simplified - would need better serialization
                    })

                # Send completion message
                await websocket.send_json({
                    "type": "complete",
                    "thread_id": thread_id,
                })

            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "error": {
                        "code": "EXECUTION_ERROR",
                        "message": str(e),
                    },
                })

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: agent_id={agent_id}")
    except Exception as e:
        logger.error(f"WebSocket error: agent_id={agent_id}, error={str(e)}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "error": {
                    "code": "WEBSOCKET_ERROR",
                    "message": str(e),
                },
            })
        except:
            pass
        finally:
            await websocket.close()


@router.post("/{agent_id}/deploy", response_model=APIResponse)
async def deploy_agent(
    agent_id: str,
    agent_factory: AgentFactory = Depends(get_agent_factory),
):
    """
    Deploy an agent (load from configuration).

    Args:
        agent_id: Agent identifier

    Returns:
        Deployment result
    """
    logger.info(f"Deploying agent: {agent_id}")

    try:
        deployment = await agent_factory.redeploy_agent(agent_id)

        logger.info(f"Agent deployed successfully: {agent_id}, deployment_info={deployment}")

        return APIResponse(
            success=True,
            data={
                "agent_id": agent_id,
                **deployment,
            },
        )

    except Exception as e:
        logger.error(f"Agent deployment failed: agent_id={agent_id}, error={str(e)}", exc_info=True)
        return APIResponse(
            success=False,
            error=ErrorDetail(
                code="DEPLOYMENT_ERROR",
                message=str(e),
            ),
        )


@router.post("/{agent_id}/undeploy", response_model=APIResponse)
async def undeploy_agent(
    agent_id: str,
    agent_factory: AgentFactory = Depends(get_agent_factory),
):
    """
    Undeploy an agent (remove from memory).

    Args:
        agent_id: Agent identifier

    Returns:
        Undeployment result
    """
    try:
        removed = agent_factory.remove_agent(agent_id)

        if not removed:
            return APIResponse(
                success=False,
                error=ErrorDetail(
                    code="NOT_FOUND",
                    message=f"Agent not deployed: {agent_id}",
                ),
            )

        return APIResponse(
            success=True,
            data={
                "agent_id": agent_id,
                "message": f"Agent '{agent_id}' undeployed successfully",
            },
        )

    except Exception as e:
        return APIResponse(
            success=False,
            error=ErrorDetail(
                code="UNDEPLOY_ERROR",
                message=str(e),
            ),
        )


# ==================== Runtime Override Endpoints ====================


@router.get("/{agent_id}/available-tools", response_model=APIResponse)
async def get_available_tools(
    agent_id: str,
    agent_factory: AgentFactory = Depends(get_agent_factory),
):
    """
    Get all available tools for runtime override selection.

    Returns:
        - builtin_tools: List of all registered built-in tools with metadata
        - mcp_servers: List of all available MCP servers
        - current_tools: Currently configured tools for this agent
        - current_mcp_servers: Currently configured MCP servers for this agent
    """
    from agent_api.dependencies import get_tool_registry, get_mcp_server_manager

    try:
        tool_registry = get_tool_registry()
        mcp_server_manager = get_mcp_server_manager()

        # Get all built-in tools with metadata
        builtin_tools = tool_registry.list_tools()

        # Get all MCP servers
        mcp_servers = []
        if mcp_server_manager:
            mcp_servers = mcp_server_manager.list_servers()

        # Get agent's current config
        metadata = agent_factory.get_agent_metadata(agent_id)
        current_config = metadata["config"] if metadata else None

        current_tools = []
        current_mcp_servers = []

        if current_config:
            current_tools = current_config.tools or []
            if current_config.mcp_servers:
                for s in current_config.mcp_servers:
                    if hasattr(s, 'name'):
                        current_mcp_servers.append(s.name)
                    elif hasattr(s, 'ref'):
                        current_mcp_servers.append(s.ref)
                    elif isinstance(s, dict):
                        current_mcp_servers.append(s.get('ref', s.get('name', '')))

        return APIResponse(
            success=True,
            data={
                "builtin_tools": builtin_tools,
                "mcp_servers": mcp_servers,
                "current_tools": current_tools,
                "current_mcp_servers": current_mcp_servers,
            }
        )

    except Exception as e:
        logger.error(f"Error getting available tools: {str(e)}", exc_info=True)
        return APIResponse(
            success=False,
            error=ErrorDetail(
                code="TOOLS_ERROR",
                message=str(e),
            ),
        )


@router.get("/{agent_id}/session-override/{thread_id}", response_model=APIResponse)
async def get_session_override(
    agent_id: str,
    thread_id: str,
    agent_factory: AgentFactory = Depends(get_agent_factory),
):
    """
    Get current session override for a thread.

    Args:
        agent_id: Agent identifier
        thread_id: Thread/session identifier

    Returns:
        Current override configuration if any
    """
    try:
        override_data = agent_factory.get_session_override(agent_id, thread_id)

        if override_data:
            # Serialize the override for response
            override = override_data.get("override")
            return APIResponse(
                success=True,
                data={
                    "has_override": True,
                    "override": override.model_dump() if override else None,
                    "created_at": override_data.get("created_at").isoformat() if override_data.get("created_at") else None,
                }
            )
        else:
            return APIResponse(
                success=True,
                data={
                    "has_override": False,
                    "override": None,
                    "created_at": None,
                }
            )

    except Exception as e:
        logger.error(f"Error getting session override: {str(e)}", exc_info=True)
        return APIResponse(
            success=False,
            error=ErrorDetail(
                code="SESSION_OVERRIDE_ERROR",
                message=str(e),
            ),
        )


@router.delete("/{agent_id}/session-override/{thread_id}", response_model=APIResponse)
async def clear_session_override(
    agent_id: str,
    thread_id: str,
    agent_factory: AgentFactory = Depends(get_agent_factory),
):
    """
    Clear session override for a thread, reverting to base agent.

    Args:
        agent_id: Agent identifier
        thread_id: Thread/session identifier

    Returns:
        Whether override was cleared
    """
    try:
        cleared = agent_factory.clear_session_override(agent_id, thread_id)

        return APIResponse(
            success=True,
            data={
                "cleared": cleared,
                "message": "Session override cleared, reverted to base agent" if cleared else "No session override found",
            }
        )

    except Exception as e:
        logger.error(f"Error clearing session override: {str(e)}", exc_info=True)
        return APIResponse(
            success=False,
            error=ErrorDetail(
                code="CLEAR_OVERRIDE_ERROR",
                message=str(e),
            ),
        )


@router.get("/{agent_id}/session-overrides", response_model=APIResponse)
async def list_session_overrides(
    agent_id: str,
    agent_factory: AgentFactory = Depends(get_agent_factory),
):
    """
    List all active session overrides for an agent.

    Args:
        agent_id: Agent identifier

    Returns:
        List of active session overrides
    """
    try:
        overrides = agent_factory.list_session_overrides(agent_name=agent_id)

        return APIResponse(
            success=True,
            data={
                "overrides": overrides,
                "total": len(overrides),
            }
        )

    except Exception as e:
        logger.error(f"Error listing session overrides: {str(e)}", exc_info=True)
        return APIResponse(
            success=False,
            error=ErrorDetail(
                code="LIST_OVERRIDES_ERROR",
                message=str(e),
            ),
        )
