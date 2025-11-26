"""
Agent management endpoints.
"""

import logging
from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status

logger = logging.getLogger(__name__)

from agent_api.dependencies import get_agent_factory, get_config_manager
from agent_api.models.schemas import (
    AgentCreateRequest,
    AgentDeleteResponse,
    AgentDetailResponse,
    AgentInfo,
    AgentListResponse,
    AgentReconfigureRequest,
    AgentUpdateRequest,
    APIResponse,
    ConfigValidateRequest,
    ConfigValidateResponse,
    ErrorDetail,
    TemplateListResponse,
    TemplateInfo,
)
from agent_api.services.agent_factory import AgentFactory, AgentFactoryError
from agent_api.services.config_manager import ConfigManager, ConfigurationError

router = APIRouter()


@router.post("/create", response_model=APIResponse)
async def create_agent(
    request: AgentCreateRequest,
    agent_factory: AgentFactory = Depends(get_agent_factory),
    config_manager: ConfigManager = Depends(get_config_manager),
):
    """
    Create a new agent from configuration.

    Args:
        request: Agent creation request with configuration
        agent_factory: Agent factory dependency
        config_manager: Config manager dependency

    Returns:
        API response with agent creation result
    """
    try:
        # Validate configuration
        validation_result = config_manager.validate_config_dict(
            request.config.model_dump(exclude_none=True)
        )

        if not validation_result["valid"]:
            return APIResponse(
                success=False,
                error=ErrorDetail(
                    code="VALIDATION_ERROR",
                    message="Configuration validation failed",
                    details={"errors": validation_result["errors"]},
                ),
            )

        # Save configuration
        config_path = config_manager.save_config(request.config)

        # Deploy if requested
        if request.deploy:
            deployment = await agent_factory.deploy_agent(request.config)
            return APIResponse(
                success=True,
                data={
                    "agent_id": request.config.name,
                    "deployed": True,
                    "config_path": config_path,
                    **deployment,
                },
            )
        else:
            return APIResponse(
                success=True,
                data={
                    "agent_id": request.config.name,
                    "deployed": False,
                    "config_path": config_path,
                },
            )

    except ConfigurationError as e:
        return APIResponse(
            success=False,
            error=ErrorDetail(
                code="CONFIGURATION_ERROR",
                message=str(e),
            ),
        )
    except AgentFactoryError as e:
        return APIResponse(
            success=False,
            error=ErrorDetail(
                code="AGENT_CREATION_ERROR",
                message=str(e),
            ),
        )


@router.get("/list", response_model=APIResponse)
async def list_agents(
    config_manager: ConfigManager = Depends(get_config_manager),
    agent_factory: AgentFactory = Depends(get_agent_factory),
):
    """
    List all agent configurations.

    Returns:
        List of agent information
    """
    try:
        configs = config_manager.list_configs()
        deployed_agents = set(agent_factory.list_agents())

        agents = []
        for config_data in configs:
            agents.append(
                AgentInfo(
                    agent_id=config_data["agent_id"],
                    name=config_data["name"],
                    version=config_data["version"],
                    description=config_data.get("description"),
                    tags=config_data.get("tags", []),
                    created_at=config_data["created_at"],
                    updated_at=config_data["updated_at"],
                    deployed=config_data["name"] in deployed_agents,
                    config_path=config_data["config_path"],
                    has_mcp_servers=config_data.get("has_mcp_servers", False),
                )
            )

        response = AgentListResponse(agents=agents, total=len(agents))

        return APIResponse(success=True, data=response.model_dump())

    except Exception as e:
        return APIResponse(
            success=False,
            error=ErrorDetail(
                code="LIST_ERROR",
                message=str(e),
            ),
        )


@router.get("/{agent_id}", response_model=APIResponse)
async def get_agent(
    agent_id: str,
    config_manager: ConfigManager = Depends(get_config_manager),
    agent_factory: AgentFactory = Depends(get_agent_factory),
):
    """
    Get details for a specific agent.

    Args:
        agent_id: Agent identifier

    Returns:
        Agent details including full configuration
    """
    try:
        # Get configuration path
        config_path = config_manager.get_config_path(agent_id)
        if not config_path:
            return APIResponse(
                success=False,
                error=ErrorDetail(
                    code="NOT_FOUND",
                    message=f"Agent not found: {agent_id}",
                ),
            )

        # Load configuration
        config = config_manager.load_config(config_path)

        # Get deployment status
        deployed = agent_id in agent_factory.list_agents()

        # Get file stats
        from pathlib import Path

        stat = Path(config_path).stat()

        agent_info = AgentInfo(
            agent_id=agent_id,
            name=config.name,
            version=config.version,
            description=config.description,
            tags=config.tags,
            created_at=datetime.fromtimestamp(stat.st_ctime),
            updated_at=datetime.fromtimestamp(stat.st_mtime),
            deployed=deployed,
            config_path=config_path,
        )

        response = AgentDetailResponse(
            agent_info=agent_info,
            config=config,
        )

        return APIResponse(success=True, data=response.model_dump())

    except ConfigurationError as e:
        return APIResponse(
            success=False,
            error=ErrorDetail(
                code="CONFIGURATION_ERROR",
                message=str(e),
            ),
        )


@router.put("/{agent_id}", response_model=APIResponse)
async def update_agent(
    agent_id: str,
    request: AgentUpdateRequest,
    agent_factory: AgentFactory = Depends(get_agent_factory),
    config_manager: ConfigManager = Depends(get_config_manager),
):
    """
    Update an existing agent configuration.

    Args:
        agent_id: Agent identifier
        request: Update request with new configuration

    Returns:
        Update result
    """
    try:
        # Check if agent exists
        config_path = config_manager.get_config_path(agent_id)
        if not config_path:
            return APIResponse(
                success=False,
                error=ErrorDetail(
                    code="NOT_FOUND",
                    message=f"Agent not found: {agent_id}",
                ),
            )

        # Validate new configuration
        validation_result = config_manager.validate_config_dict(
            request.config.model_dump(exclude_none=True)
        )

        if not validation_result["valid"]:
            return APIResponse(
                success=False,
                error=ErrorDetail(
                    code="VALIDATION_ERROR",
                    message="Configuration validation failed",
                    details={"errors": validation_result["errors"]},
                ),
            )

        # Save updated configuration
        config_manager.save_config(request.config, config_path)

        # Redeploy if requested and agent was deployed
        if request.redeploy and agent_id in agent_factory.list_agents():
            deployment = await agent_factory.redeploy_agent(agent_id)
            return APIResponse(
                success=True,
                data={
                    "agent_id": agent_id,
                    "updated": True,
                    "redeployed": True,
                    **deployment,
                },
            )
        else:
            return APIResponse(
                success=True,
                data={
                    "agent_id": agent_id,
                    "updated": True,
                    "redeployed": False,
                },
            )

    except ConfigurationError as e:
        return APIResponse(
            success=False,
            error=ErrorDetail(
                code="CONFIGURATION_ERROR",
                message=str(e),
            ),
        )
    except AgentFactoryError as e:
        return APIResponse(
            success=False,
            error=ErrorDetail(
                code="AGENT_UPDATE_ERROR",
                message=str(e),
            ),
        )


@router.put("/{agent_id}/reconfigure", response_model=APIResponse)
async def reconfigure_agent(
    agent_id: str,
    request: AgentReconfigureRequest,
    agent_factory: AgentFactory = Depends(get_agent_factory),
):
    """
    Reconfigure a running agent (LLM, tools, MCP servers) while preserving middleware.

    This endpoint allows selective updates to an agent's configuration without
    affecting middleware or memory settings. Conversation threads are preserved
    across the reconfiguration.

    Args:
        agent_id: Agent identifier
        request: Reconfiguration request with partial updates

    Returns:
        Reconfiguration result with change summary

    Examples:
        Change LLM model:
        ```json
        {
            "llm": {
                "provider": "openai",
                "model": "gpt-4o",
                "temperature": 0.7
            }
        }
        ```

        Add tools:
        ```json
        {
            "tools": ["tavily_search", "calculator", "get_current_datetime"]
        }
        ```

        Update MCP servers:
        ```json
        {
            "mcp_servers": [
                {
                    "name": "calculator",
                    "transport": "streamable_http",
                    "url": "http://localhost:8005/mcp"
                }
            ]
        }
        ```
    """
    logger.info(f"Reconfigure request for agent '{agent_id}'")

    try:
        # Validate at least one field is provided
        if (request.llm is None and
            request.tools is None and
            request.mcp_servers is None):
            return APIResponse(
                success=False,
                error=ErrorDetail(
                    code="VALIDATION_ERROR",
                    message="At least one of 'llm', 'tools', or 'mcp_servers' must be provided",
                ),
            )

        # Call agent factory reconfigure method
        result = await agent_factory.reconfigure_agent(
            agent_name=agent_id,
            llm=request.llm,
            tools=request.tools,
            mcp_servers=request.mcp_servers,
            preserve_middleware=request.preserve_middleware,
            preserve_memory=request.preserve_memory,
        )

        logger.info(f"Successfully reconfigured agent '{agent_id}'")

        return APIResponse(
            success=True,
            data=result,
        )

    except AgentFactoryError as e:
        logger.error(f"Agent reconfiguration failed: {str(e)}")
        return APIResponse(
            success=False,
            error=ErrorDetail(
                code="RECONFIGURATION_ERROR",
                message=str(e),
            ),
        )
    except ConfigurationError as e:
        logger.error(f"Configuration error during reconfiguration: {str(e)}")
        return APIResponse(
            success=False,
            error=ErrorDetail(
                code="CONFIGURATION_ERROR",
                message=str(e),
            ),
        )
    except Exception as e:
        logger.error(f"Unexpected error during reconfiguration: {str(e)}", exc_info=True)
        return APIResponse(
            success=False,
            error=ErrorDetail(
                code="INTERNAL_ERROR",
                message=f"An unexpected error occurred: {str(e)}",
            ),
        )


@router.delete("/{agent_id}", response_model=APIResponse)
async def delete_agent(
    agent_id: str,
    agent_factory: AgentFactory = Depends(get_agent_factory),
    config_manager: ConfigManager = Depends(get_config_manager),
):
    """
    Delete an agent and its configuration.

    Args:
        agent_id: Agent identifier

    Returns:
        Deletion result
    """
    try:
        # Remove from deployed agents
        agent_factory.remove_agent(agent_id)

        # Delete configuration
        deleted = config_manager.delete_config(agent_id)

        if not deleted:
            return APIResponse(
                success=False,
                error=ErrorDetail(
                    code="NOT_FOUND",
                    message=f"Agent not found: {agent_id}",
                ),
            )

        response = AgentDeleteResponse(
            agent_id=agent_id,
            message=f"Agent '{agent_id}' deleted successfully",
        )

        return APIResponse(success=True, data=response.model_dump())

    except Exception as e:
        return APIResponse(
            success=False,
            error=ErrorDetail(
                code="DELETE_ERROR",
                message=str(e),
            ),
        )


@router.post("/validate", response_model=APIResponse)
async def validate_config(
    request: ConfigValidateRequest,
    config_manager: ConfigManager = Depends(get_config_manager),
):
    """
    Validate an agent configuration without saving.

    Args:
        request: Configuration to validate

    Returns:
        Validation result
    """
    try:
        validation_result = config_manager.validate_config_dict(request.config)

        response = ConfigValidateResponse(
            valid=validation_result["valid"],
            errors=[
                {"field": err["field"], "message": err["message"]}
                for err in validation_result.get("errors", [])
            ],
            warnings=validation_result.get("warnings", []),
        )

        return APIResponse(success=True, data=response.model_dump())

    except Exception as e:
        return APIResponse(
            success=False,
            error=ErrorDetail(
                code="VALIDATION_ERROR",
                message=str(e),
            ),
        )


@router.get("/templates/list", response_model=APIResponse)
async def list_templates(
    config_manager: ConfigManager = Depends(get_config_manager),
):
    """
    List available agent configuration templates.

    Returns:
        List of template information
    """
    try:
        templates_data = config_manager.list_templates()

        templates = [
            TemplateInfo(
                template_id=t["template_id"],
                name=t["name"],
                description=t.get("description", ""),
                category=t.get("category", "general"),
                tags=t.get("tags", []),
                config=config_manager.load_template(t["template_id"]),
            )
            for t in templates_data
        ]

        response = TemplateListResponse(templates=templates, total=len(templates))

        return APIResponse(success=True, data=response.model_dump())

    except Exception as e:
        return APIResponse(
            success=False,
            error=ErrorDetail(
                code="TEMPLATE_ERROR",
                message=str(e),
            ),
        )


@router.get("/templates/{template_id}", response_model=APIResponse)
async def get_template(
    template_id: str,
    config_manager: ConfigManager = Depends(get_config_manager),
):
    """
    Get a specific template configuration.

    Args:
        template_id: Template identifier

    Returns:
        Template configuration
    """
    try:
        config = config_manager.load_template(template_id)

        return APIResponse(
            success=True,
            data={"template_id": template_id, "config": config.model_dump()},
        )

    except ConfigurationError as e:
        return APIResponse(
            success=False,
            error=ErrorDetail(
                code="NOT_FOUND",
                message=str(e),
            ),
        )
