"""
Middleware factory for creating middleware from configurations.
"""

import logging
from typing import Any, Dict, List, Optional

from agent_api.models.config_schema import AgentConfig, MiddlewareConfig

logger = logging.getLogger(__name__)


class MiddlewareFactoryError(Exception):
    """Exception raised for middleware factory errors."""
    pass


class MiddlewareFactory:
    """Factory for creating LangChain middleware from configurations."""

    def __init__(self):
        """Initialize the middleware factory."""
        self._middleware_registry = self._build_middleware_registry()

    def _build_middleware_registry(self) -> Dict[str, Any]:
        """
        Build registry of available middleware types.

        Returns:
            Dictionary mapping middleware types to their classes/factories
        """
        registry = {}

        # Try to import middleware from langchain
        # Note: These imports may fail if middleware classes aren't available yet
        try:
            from langchain.agents.middleware import (
                SummarizationMiddleware,
                ModelCallLimitMiddleware,
                ToolCallLimitMiddleware,
                PIIMiddleware,
                ModelFallbackMiddleware,
                ToolRetryMiddleware,
                HumanInTheLoopMiddleware,
                TodoListMiddleware,
                LLMToolSelectorMiddleware,
                ContextEditingMiddleware,
            )

            registry.update({
                "summarization": SummarizationMiddleware,
                "model_call_limit": ModelCallLimitMiddleware,
                "tool_call_limit": ToolCallLimitMiddleware,
                "pii_detection": PIIMiddleware,
                "model_fallback": ModelFallbackMiddleware,
                "tool_retry": ToolRetryMiddleware,
                "human_in_the_loop": HumanInTheLoopMiddleware,
                "todo_list": TodoListMiddleware,
                "llm_tool_selector": LLMToolSelectorMiddleware,
                "context_editing": ContextEditingMiddleware,
            })
        except ImportError as e:
            logger.warning(
                f"Middleware classes not available - some middleware types will not work. "
                f"ImportError: {str(e)}"
            )

        # Provider-specific middleware
        try:
            from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
            registry["anthropic_prompt_caching"] = AnthropicPromptCachingMiddleware
        except ImportError as e:
            logger.warning(f"Anthropic middleware not available: {str(e)}")

        logger.info(f"Middleware registry initialized with {len(registry)} types: {', '.join(registry.keys())}")
        return registry

    def create_middleware(
        self,
        middleware_config: MiddlewareConfig,
        llm: Optional[Any] = None
    ) -> Optional[Any]:
        """
        Create a middleware instance from configuration.

        Args:
            middleware_config: Middleware configuration
            llm: Optional LLM instance to use for middleware that requires it

        Returns:
            Middleware instance or None if disabled

        Raises:
            MiddlewareFactoryError: If middleware creation fails
        """
        if not middleware_config.enabled:
            logger.debug(f"Middleware '{middleware_config.type}' is disabled, skipping")
            return None

        middleware_type = middleware_config.type
        # Copy params to avoid mutating the original config (which gets serialized to YAML)
        params = dict(middleware_config.params) if middleware_config.params else {}

        logger.debug(f"Creating middleware: type={middleware_type}, params={params}")

        # Check if middleware type is registered
        if middleware_type not in self._middleware_registry:
            logger.error(f"Unknown middleware type: {middleware_type}. Available: {', '.join(self._middleware_registry.keys())}")
            raise MiddlewareFactoryError(
                f"Unknown middleware type: {middleware_type}. "
                f"Available types: {', '.join(self._middleware_registry.keys())}"
            )

        try:
            middleware_class = self._middleware_registry[middleware_type]

            # Special handling for llm_tool_selector middleware
            if middleware_type == "llm_tool_selector":
                # Validate and log system_prompt if provided
                if "system_prompt" in params:
                    system_prompt = params["system_prompt"]
                    if not isinstance(system_prompt, str):
                        raise MiddlewareFactoryError(
                            f"llm_tool_selector 'system_prompt' must be a string, got {type(system_prompt).__name__}"
                        )
                    if not system_prompt.strip():
                        logger.warning("llm_tool_selector has empty system_prompt, removing parameter to use default")
                        del params["system_prompt"]
                    else:
                        logger.info(
                            f"llm_tool_selector using custom system_prompt "
                            f"(length: {len(system_prompt)} chars, "
                            f"preview: '{system_prompt[:80]}...')"
                        )

            # Special handling for summarization middleware which requires an LLM instance
            if middleware_type == "summarization":
                if "model" not in params:
                    if llm is None:
                        raise MiddlewareFactoryError(
                            "summarization middleware requires 'model' parameter or LLM instance"
                        )
                    params["model"] = llm
                    logger.debug("Using provided LLM for summarization middleware")
                elif isinstance(params["model"], str):
                    # If model is a string, create an LLM instance from it
                    model_str = params["model"]
                    try:
                        from langchain.chat_models import init_chat_model
                        # Parse format like "openai:gpt-4o-mini" or just "gpt-4o-mini"
                        if ":" in model_str:
                            provider, model_name = model_str.split(":", 1)
                            params["model"] = init_chat_model(
                                model=model_name,
                                model_provider=provider
                            )
                        else:
                            # Default to OpenAI if no provider specified
                            params["model"] = init_chat_model(
                                model=model_str,
                                model_provider="openai"
                            )
                        logger.debug(f"Created LLM from string '{model_str}' for summarization middleware")
                    except Exception as e:
                        raise MiddlewareFactoryError(
                            f"Failed to create LLM from '{model_str}' for summarization middleware: {str(e)}"
                        )

            # Create instance with parameters
            middleware_instance = middleware_class(**params)

            logger.info(f"Successfully created middleware: {middleware_type}")
            return middleware_instance

        except Exception as e:
            logger.error(f"Failed to create middleware '{middleware_type}': {str(e)}", exc_info=True)
            raise MiddlewareFactoryError(
                f"Failed to create middleware '{middleware_type}': {str(e)}"
            )

    def create_middleware_list(
        self,
        config: AgentConfig,
        llm: Optional[Any] = None
    ) -> List[Any]:
        """
        Create list of middleware instances from agent configuration.

        Args:
            config: Agent configuration
            llm: Optional LLM instance to use for middleware that requires it

        Returns:
            List of middleware instances
        """
        middleware_list = []

        if not config.middleware:
            return middleware_list

        logger.info(f"Creating middleware list for agent: {config.name}")

        for middleware_config in config.middleware:
            try:
                middleware = self.create_middleware(middleware_config, llm=llm)
                if middleware:
                    middleware_list.append(middleware)
            except MiddlewareFactoryError as e:
                # Log error but continue with other middleware
                logger.error(f"Middleware creation failed: {str(e)}", exc_info=True)
                continue

        logger.info(f"Created {len(middleware_list)} middleware instances for agent '{config.name}'")

        # Validate llm_tool_selector position (should always be first for proper tool filtering)
        selector_index = next((i for i, m in enumerate(config.middleware)
                              if m.type == "llm_tool_selector"), None)

        if selector_index is not None and selector_index != 0:
            middleware_types = [m.type for m in config.middleware]
            logger.warning(
                f"CONFIGURATION WARNING: llm_tool_selector is at position {selector_index} "
                f"but should be first (position 0) for proper tool filtering. "
                f"Current order: {middleware_types}. "
                f"This may cause tool selection issues. "
                f"Note: AgentFactory auto-default will fix this on deployment."
            )

        return middleware_list

    def validate_middleware_config(
        self,
        middleware_config: MiddlewareConfig
    ) -> Dict[str, Any]:
        """
        Validate a middleware configuration.

        Args:
            middleware_config: Middleware configuration to validate

        Returns:
            Validation result dictionary
        """
        errors = []
        warnings = []

        # Check if type is known
        if middleware_config.type not in self._middleware_registry:
            errors.append(
                f"Unknown middleware type: {middleware_config.type}"
            )

        # Validate parameters based on type
        # This would be extended with specific validation for each middleware type
        if middleware_config.type == "summarization":
            if "model" not in middleware_config.params:
                errors.append(
                    "summarization middleware requires 'model' parameter"
                )
            if "max_tokens_before_summary" not in middleware_config.params:
                warnings.append(
                    "summarization middleware: 'max_tokens_before_summary' parameter not set, using default"
                )

        if middleware_config.type == "model_call_limit":
            if "thread_limit" not in middleware_config.params and "run_limit" not in middleware_config.params:
                errors.append(
                    "model_call_limit middleware requires at least one of 'thread_limit' or 'run_limit' parameters"
                )

        if middleware_config.type == "pii_detection":
            valid_strategies = ["block", "redact", "mask", "hash"]
            strategy = middleware_config.params.get("strategy")
            if strategy and strategy not in valid_strategies:
                errors.append(
                    f"pii_detection middleware: invalid strategy '{strategy}'. "
                    f"Must be one of: {', '.join(valid_strategies)}"
                )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }

    def list_available_middleware(self) -> List[Dict[str, Any]]:
        """
        List all available middleware types with descriptions.

        Returns:
            List of middleware metadata
        """
        middleware_info = {
            "summarization": {
                "name": "Summarization",
                "description": "Automatically compress conversation history when approaching token limits",
                "category": "memory",
                "required_params": ["model"],
                "optional_params": ["max_tokens_before_summary", "messages_to_keep"],
            },
            "model_call_limit": {
                "name": "Model Call Limit",
                "description": "Restrict the number of model invocations",
                "category": "reliability",
                "required_params": [],
                "optional_params": ["thread_limit", "run_limit", "exit_behavior"],
            },
            "tool_call_limit": {
                "name": "Tool Call Limit",
                "description": "Control tool execution counts (global or per-tool)",
                "category": "reliability",
                "required_params": [],
                "optional_params": ["thread_limit", "run_limit", "tool_name", "exit_behavior"],
            },
            "pii_detection": {
                "name": "PII Detection",
                "description": "Detect and handle personally identifiable information",
                "category": "safety",
                "required_params": ["strategy"],
                "optional_params": ["pii_types"],
            },
            "model_fallback": {
                "name": "Model Fallback",
                "description": "Switch to alternative models on failure",
                "category": "reliability",
                "required_params": ["fallback_models"],
                "optional_params": [],
            },
            "tool_retry": {
                "name": "Tool Retry",
                "description": "Automatically retry failed tools with exponential backoff",
                "category": "reliability",
                "required_params": [],
                "optional_params": ["max_retries", "initial_delay"],
            },
            "human_in_the_loop": {
                "name": "Human in the Loop",
                "description": "Pause for human approval of tool calls",
                "category": "safety",
                "required_params": [],
                "optional_params": ["require_approval_for"],
            },
            "todo_list": {
                "name": "To-Do List",
                "description": "Provides write_todos tool for task planning",
                "category": "planning",
                "required_params": [],
                "optional_params": [],
            },
            "llm_tool_selector": {
                "name": "LLM Tool Selector",
                "description": "Use separate LLM to filter relevant tools",
                "category": "optimization",
                "required_params": [],
                "optional_params": ["model", "max_tools", "always_include"],
            },
            "context_editing": {
                "name": "Context Editing",
                "description": "Manage context by clearing older tool outputs",
                "category": "memory",
                "required_params": [],
                "optional_params": ["edits", "token_count_method"],
            },
            "anthropic_prompt_caching": {
                "name": "Anthropic Prompt Caching",
                "description": "Reduce costs by caching prompts (Anthropic-specific)",
                "category": "optimization",
                "required_params": [],
                "optional_params": ["cache_ttl"],
            },
        }

        result = []
        for middleware_type, info in middleware_info.items():
            if middleware_type in self._middleware_registry:
                result.append({
                    "type": middleware_type,
                    **info,
                    "available": True,
                })
            else:
                result.append({
                    "type": middleware_type,
                    **info,
                    "available": False,
                })

        return result

    def get_middleware_presets(self) -> Dict[str, List[MiddlewareConfig]]:
        """
        Get predefined middleware presets for common scenarios.

        Returns:
            Dictionary of preset name to middleware configurations
        """
        presets = {
            "production_safe": [
                MiddlewareConfig(
                    type="pii_detection",
                    params={"strategy": "redact"},
                    enabled=True
                ),
                MiddlewareConfig(
                    type="model_call_limit",
                    params={"thread_limit": 50},
                    enabled=True
                ),
                MiddlewareConfig(
                    type="tool_retry",
                    params={"max_retries": 3},
                    enabled=True
                ),
                MiddlewareConfig(
                    type="summarization",
                    params={"model": "openai:gpt-4o-mini", "max_tokens_before_summary": 100000},
                    enabled=True
                ),
            ],
            "cost_optimized": [
                MiddlewareConfig(
                    type="summarization",
                    params={"model": "openai:gpt-4o-mini", "max_tokens_before_summary": 50000},
                    enabled=True
                ),
                MiddlewareConfig(
                    type="model_call_limit",
                    params={"thread_limit": 20},
                    enabled=True
                ),
                MiddlewareConfig(
                    type="anthropic_prompt_caching",
                    params={},
                    enabled=True
                ),
            ],
            "development": [
                MiddlewareConfig(
                    type="model_call_limit",
                    params={"thread_limit": 100},
                    enabled=True
                ),
            ],
            "high_reliability": [
                MiddlewareConfig(
                    type="model_fallback",
                    params={"fallback_models": ["gpt-4o-mini", "gpt-3.5-turbo"]},
                    enabled=True
                ),
                MiddlewareConfig(
                    type="tool_retry",
                    params={"max_retries": 5, "initial_delay": 1.0},
                    enabled=True
                ),
                MiddlewareConfig(
                    type="model_call_limit",
                    params={"thread_limit": 100},
                    enabled=True
                ),
            ],
            "minimal": [],
            "multi_tool_optimized": [
                MiddlewareConfig(
                    type="llm_tool_selector",
                    params={
                        "model": "openai:gpt-4o-mini",
                        "max_tools": 7,
                        "always_include": []
                    },
                    enabled=True
                ),
                MiddlewareConfig(
                    type="tool_retry",
                    params={"max_retries": 3, "initial_delay": 1.0},
                    enabled=True
                ),
                MiddlewareConfig(
                    type="summarization",
                    params={"model": "openai:gpt-4o-mini", "max_tokens_before_summary": 100000},
                    enabled=True
                ),
            ],
        }

        return presets

    def recommend_middleware_for_agent(
        self,
        total_tool_count: int,
        use_mcp: bool = False,
        priority: str = "accuracy"
    ) -> List[MiddlewareConfig]:
        """
        Recommend middleware configuration based on agent characteristics.

        Args:
            total_tool_count: Total number of tools (built-in + MCP)
            use_mcp: Whether agent uses MCP servers
            priority: Optimization priority - "accuracy", "cost", or "balanced"

        Returns:
            List of recommended middleware configurations
        """
        recommended = []

        # For agents with 5+ tools, strongly recommend tool selector
        if total_tool_count >= 5:
            max_tools = min(8, max(5, total_tool_count // 3))  # ~1/3 of tools, between 5-8

            recommended.append(MiddlewareConfig(
                type="llm_tool_selector",
                params={
                    "model": "openai:gpt-4o-mini",
                    "max_tools": max_tools,
                    "always_include": []
                },
                enabled=True
            ))
            logger.info(
                f"Recommending llm_tool_selector middleware: "
                f"{total_tool_count} tools -> max_tools={max_tools}"
            )

        # Always recommend tool retry for reliability
        recommended.append(MiddlewareConfig(
            type="tool_retry",
            params={"max_retries": 3, "initial_delay": 1.0},
            enabled=True
        ))

        # Recommend summarization for long conversations
        if priority in ["accuracy", "balanced"]:
            recommended.append(MiddlewareConfig(
                type="summarization",
                params={
                    "model": "openai:gpt-4o-mini",
                    "max_tokens_before_summary": 100000 if priority == "accuracy" else 50000
                },
                enabled=True
            ))

        # Add model call limit
        thread_limit = {
            "accuracy": 100,
            "balanced": 50,
            "cost": 20
        }.get(priority, 50)

        recommended.append(MiddlewareConfig(
            type="model_call_limit",
            params={"thread_limit": thread_limit},
            enabled=True
        ))

        return recommended
