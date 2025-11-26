"""
FastAPI application for the Agent Builder API.
"""

import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .dependencies import app_state
from .models.schemas import APIResponse, ErrorDetail, HealthResponse
from .routers import agents, execution, mcp_servers, tools

# Load environment variables (also loaded in dependencies.py, but ensure it's loaded here too)
# Find the project root (parent of agent_api directory)
project_root = Path(__file__).parent.parent
env_path = project_root / ".env"

if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    # Fallback to default load_dotenv() behavior
    load_dotenv()

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('agent_api.log')
    ]
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown logic.
    """
    # Startup
    logger.info("=" * 60)
    logger.info("Starting Agent Builder API")
    logger.info(f"Configs directory: {app_state.config_manager.configs_dir}")
    logger.info(f"Registered tools: {len(app_state.tool_registry.list_tools())}")
    logger.info(f"Available middleware: {len(app_state.middleware_factory.list_available_middleware())}")

    # Load and deploy any existing agent configurations
    auto_deploy = os.getenv("API_AUTO_DEPLOY", "false").lower() in ("true", "1", "yes")

    if auto_deploy:
        try:
            configs = app_state.config_manager.list_configs()
            logger.info(f"Found {len(configs)} existing agent configurations")

            deployed_count = 0
            failed_count = 0

            for config_info in configs:
                agent_name = config_info['name']
                try:
                    # Load the full config
                    config_dict = app_state.config_manager.load_config(config_info['config_path'])

                    # Deploy the agent (creates and caches in memory)
                    await app_state.agent_factory.deploy_agent(config_dict)

                    deployed_count += 1
                    logger.info(f"  ✓ Deployed: {agent_name} (v{config_info['version']})")

                except Exception as agent_error:
                    failed_count += 1
                    logger.warning(
                        f"  ✗ Failed to deploy {agent_name}: {str(agent_error)}",
                        exc_info=False
                    )

            logger.info(f"Auto-deployment complete: {deployed_count} deployed, {failed_count} failed")

        except Exception as e:
            logger.warning(f"Could not load existing configs: {str(e)}", exc_info=True)
    else:
        logger.info("Auto-deployment disabled (agents will deploy on first use)")

    logger.info("=" * 60)

    yield

    # Shutdown
    logger.info("Shutting down Agent Builder API")


# Create FastAPI app
app = FastAPI(
    title="Agent Builder API",
    description="API for creating and managing LangChain agents from configurations",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(
        f"Unhandled exception in {request.method} {request.url.path}: {str(exc)}",
        exc_info=True,
        extra={"method": request.method, "path": request.url.path}
    )
    return JSONResponse(
        status_code=500,
        content=APIResponse(
            success=False,
            error=ErrorDetail(
                code="INTERNAL_ERROR",
                message=str(exc),
                details={"type": type(exc).__name__},
            ),
        ).model_dump(),
    )


# Include routers
app.include_router(agents.router, prefix="/agents", tags=["Agents"])
app.include_router(execution.router, prefix="/execution", tags=["Execution"])
app.include_router(tools.router, prefix="/tools", tags=["Tools"])
app.include_router(mcp_servers.router, prefix="/mcp-servers", tags=["MCP Servers"])


# Root endpoints
@app.get("/", response_model=APIResponse)
async def root():
    """Root endpoint."""
    return APIResponse(
        success=True,
        data={
            "message": "Agent Builder API",
            "version": "1.0.0",
            "docs": "/docs",
        },
    )


@app.get("/health", response_model=APIResponse)
async def health():
    """Health check endpoint."""
    components = {
        "api": "healthy",
        "config_manager": "healthy",
        "tool_registry": "healthy",
        "agent_factory": "healthy",
        "mcp_server_manager": "healthy",
    }

    # Check if any agents are deployed
    try:
        deployed_agents = len(app_state.agent_factory.list_agents())
        components["deployed_agents"] = str(deployed_agents)
    except Exception:
        components["deployed_agents"] = "error"

    # Check MCP server configs
    try:
        mcp_servers = len(app_state.mcp_server_manager.list_servers())
        components["mcp_servers"] = str(mcp_servers)
    except Exception:
        components["mcp_servers"] = "error"

    health_response = HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.utcnow(),
        components=components,
    )

    return APIResponse(success=True, data=health_response.model_dump())


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("API_PORT", 8000))
    host = os.getenv("API_HOST", "0.0.0.0")
    # Reload mode disabled by default (set API_RELOAD=true to enable)
    reload = os.getenv("API_RELOAD", "false").lower() in ("true", "1", "yes")

    def signal_handler(sig, frame):
        """Handle shutdown signals gracefully."""
        logger.info("\nReceived shutdown signal, shutting down gracefully...")
        sys.exit(0)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        uvicorn.run(
            "agent_api.main:app",
            host=host,
            port=port,
            reload=reload,
            factory=False,  # Explicitly specify this is NOT a factory function
            reload_dirs=["./agent_api"] if reload else None,  # Only watch API source code, not data/config directories
            log_level="info",
        )
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        sys.exit(0)
