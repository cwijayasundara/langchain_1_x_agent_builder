"""
Tool management endpoints.
"""

from fastapi import APIRouter, Depends

from agent_api.dependencies import get_tool_registry
from agent_api.models.schemas import (
    APIResponse,
    ErrorDetail,
    ToolApproveRequest,
    ToolGenerateRequest,
    ToolInfo,
    ToolListResponse,
    ToolTestRequest,
    ToolTestResponse,
)
from agent_api.services.tool_registry import ToolRegistry, ToolRegistryError

router = APIRouter()


@router.get("/list", response_model=APIResponse)
async def list_tools(
    include_pending: bool = False,
    tool_registry: ToolRegistry = Depends(get_tool_registry),
):
    """
    List all available tools.

    Args:
        include_pending: Whether to include pending tools
        tool_registry: Tool registry dependency

    Returns:
        List of tool information
    """
    try:
        tools_data = tool_registry.list_tools(include_pending=include_pending)

        tools = [
            ToolInfo(
                tool_id=t["tool_id"],
                name=t["name"],
                description=t["description"],
                type=t["type"],
                status=t["status"],
                code=None,  # Don't include code in list view
                created_at=t.get("created_at"),
            )
            for t in tools_data
        ]

        response = ToolListResponse(tools=tools, total=len(tools))

        return APIResponse(success=True, data=response.model_dump())

    except Exception as e:
        return APIResponse(
            success=False,
            error=ErrorDetail(
                code="LIST_ERROR",
                message=str(e),
            ),
        )


@router.get("/{tool_id}", response_model=APIResponse)
async def get_tool(
    tool_id: str,
    tool_registry: ToolRegistry = Depends(get_tool_registry),
):
    """
    Get details for a specific tool.

    Args:
        tool_id: Tool identifier

    Returns:
        Tool details including code for custom tools
    """
    try:
        # Try to get active tool
        tool = tool_registry.get_tool(tool_id)

        if tool:
            return APIResponse(
                success=True,
                data=ToolInfo(
                    tool_id=tool_id,
                    name=tool.name,
                    description=tool.description,
                    type="custom",  # Assume custom if retrievable
                    status="active",
                    code=None,  # Could add code retrieval if needed
                    created_at=None,
                ).model_dump(),
            )

        # Check if it's a pending tool
        pending = tool_registry.get_pending_tool(tool_id)
        if pending:
            return APIResponse(
                success=True,
                data=ToolInfo(
                    tool_id=tool_id,
                    name=pending["name"],
                    description=pending["description"],
                    type="custom",
                    status="pending",
                    code=pending["code"],
                    created_at=pending["created_at"],
                ).model_dump(),
            )

        return APIResponse(
            success=False,
            error=ErrorDetail(
                code="NOT_FOUND",
                message=f"Tool not found: {tool_id}",
            ),
        )

    except Exception as e:
        return APIResponse(
            success=False,
            error=ErrorDetail(
                code="GET_ERROR",
                message=str(e),
            ),
        )


@router.post("/generate", response_model=APIResponse)
async def generate_tool(
    request: ToolGenerateRequest,
    tool_registry: ToolRegistry = Depends(get_tool_registry),
):
    """
    Generate a custom tool from a natural language description.

    Args:
        request: Tool generation request
        tool_registry: Tool registry dependency

    Returns:
        Generated tool information (pending approval)
    """
    try:
        from langchain_chat_models import init_chat_model
        import os

        # Use an LLM to generate tool code
        llm = init_chat_model(
            model="gpt-4o",
            model_provider="openai",
            temperature=0.2,
        )

        # Create prompt for tool generation
        prompt = f"""Generate a Python tool function for LangChain using the @tool decorator.

Requirements:
- Description: {request.description}
- The function should have a clear docstring that explains what it does
- Use type hints for all parameters
- Include proper error handling
- The function should be self-contained and importable

Example format:
```python
from langchain_core.tools import tool
from typing import Any, Dict

@tool
def my_tool(param1: str, param2: int) -> str:
    \"\"\"
    Tool description here.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value
    \"\"\"
    try:
        # Implementation here
        result = f"Processed {{param1}} with {{param2}}"
        return result
    except Exception as e:
        return f"Error: {{str(e)}}"
```

Generate only the Python code, no explanations. Start with imports and end with the complete function."""

        if request.examples:
            prompt += f"\n\nExamples:\n{request.examples}"

        # Generate code
        response = llm.invoke(prompt)
        generated_code = response.content

        # Clean up code (remove markdown code blocks if present)
        if "```python" in generated_code:
            generated_code = generated_code.split("```python")[1].split("```")[0].strip()
        elif "```" in generated_code:
            generated_code = generated_code.split("```")[1].split("```")[0].strip()

        # Generate tool ID from name or description
        tool_id = request.name or request.description.lower().replace(" ", "_")[:50]
        tool_id = "".join(c if c.isalnum() or c == "_" else "_" for c in tool_id)

        # Add to pending tools
        tool_registry.add_pending_tool(
            tool_id=tool_id,
            name=request.name or tool_id.replace("_", " ").title(),
            description=request.description,
            code=generated_code,
        )

        return APIResponse(
            success=True,
            data={
                "tool_id": tool_id,
                "status": "pending",
                "code": generated_code,
                "message": "Tool generated successfully. Please review and approve.",
            },
        )

    except Exception as e:
        return APIResponse(
            success=False,
            error=ErrorDetail(
                code="GENERATION_ERROR",
                message=str(e),
            ),
        )


@router.post("/{tool_id}/approve", response_model=APIResponse)
async def approve_tool(
    tool_id: str,
    request: ToolApproveRequest,
    tool_registry: ToolRegistry = Depends(get_tool_registry),
):
    """
    Approve or reject a pending tool.

    Args:
        tool_id: Tool identifier
        request: Approval request
        tool_registry: Tool registry dependency

    Returns:
        Approval result
    """
    try:
        if request.approved:
            # Approve with optional modifications
            tool_registry.approve_tool(tool_id, request.modifications)

            return APIResponse(
                success=True,
                data={
                    "tool_id": tool_id,
                    "status": "approved",
                    "message": f"Tool '{tool_id}' approved and activated",
                },
            )
        else:
            # Reject
            tool_registry.reject_tool(tool_id)

            return APIResponse(
                success=True,
                data={
                    "tool_id": tool_id,
                    "status": "rejected",
                    "message": f"Tool '{tool_id}' rejected and removed",
                },
            )

    except ToolRegistryError as e:
        return APIResponse(
            success=False,
            error=ErrorDetail(
                code="APPROVAL_ERROR",
                message=str(e),
            ),
        )


@router.post("/{tool_id}/test", response_model=APIResponse)
async def test_tool(
    tool_id: str,
    request: ToolTestRequest,
    tool_registry: ToolRegistry = Depends(get_tool_registry),
):
    """
    Test a tool with given input.

    Args:
        tool_id: Tool identifier
        request: Test request with input
        tool_registry: Tool registry dependency

    Returns:
        Test result
    """
    try:
        result = tool_registry.test_tool(tool_id, request.test_input)

        response = ToolTestResponse(
            success=result["success"],
            output=result["output"],
            error=result["error"],
            execution_time=result["execution_time"],
        )

        return APIResponse(success=True, data=response.model_dump())

    except ToolRegistryError as e:
        return APIResponse(
            success=False,
            error=ErrorDetail(
                code="TEST_ERROR",
                message=str(e),
            ),
        )


@router.delete("/{tool_id}", response_model=APIResponse)
async def delete_tool(
    tool_id: str,
    tool_registry: ToolRegistry = Depends(get_tool_registry),
):
    """
    Delete a custom tool.

    Args:
        tool_id: Tool identifier

    Returns:
        Deletion result
    """
    try:
        # Try to delete from active tools
        deleted = tool_registry.delete_tool(tool_id)

        if not deleted:
            # Try to reject from pending
            deleted = tool_registry.reject_tool(tool_id)

        if not deleted:
            return APIResponse(
                success=False,
                error=ErrorDetail(
                    code="NOT_FOUND",
                    message=f"Tool not found: {tool_id}",
                ),
            )

        return APIResponse(
            success=True,
            data={
                "tool_id": tool_id,
                "message": f"Tool '{tool_id}' deleted successfully",
            },
        )

    except Exception as e:
        return APIResponse(
            success=False,
            error=ErrorDetail(
                code="DELETE_ERROR",
                message=str(e),
            ),
        )
