"""
Validation functions for agent configuration forms.
"""

import re
from typing import List, Any, Dict


class ValidationError:
    """Represents a validation error."""

    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message

    def __repr__(self):
        return f"{self.field}: {self.message}"


def validate_required(value: Any, field_name: str) -> List[ValidationError]:
    """Validate required field."""
    errors = []
    if not value or (isinstance(value, str) and not value.strip()):
        errors.append(ValidationError(field_name, f"{field_name} is required"))
    return errors


def validate_agent_name(name: str) -> List[ValidationError]:
    """Validate agent name format."""
    errors = []

    if not name:
        errors.append(ValidationError("Name", "Name is required"))
        return errors

    if len(name) < 3:
        errors.append(ValidationError("Name", "Name must be at least 3 characters"))
    if len(name) > 50:
        errors.append(ValidationError("Name", "Name must be at most 50 characters"))

    # Only alphanumeric and underscores
    if not re.match(r'^[a-zA-Z0-9_]+$', name):
        errors.append(ValidationError("Name", "Name can only contain letters, numbers, and underscores"))

    return errors


def validate_version(version: str) -> List[ValidationError]:
    """Validate version format (X.Y.Z)."""
    errors = []

    if not version:
        errors.append(ValidationError("Version", "Version is required"))
        return errors

    if not re.match(r'^\d+\.\d+\.\d+$', version):
        errors.append(ValidationError("Version", "Version must be in format X.Y.Z (e.g., 1.0.0)"))

    return errors


def validate_page_1(data: Dict[str, Any]) -> List[ValidationError]:
    """Validate Page 1: Basic Info."""
    errors = []

    errors.extend(validate_agent_name(data.get('name', '')))
    errors.extend(validate_required(data.get('description'), 'Description'))
    errors.extend(validate_version(data.get('version', '')))

    return errors


def validate_page_2(data: Dict[str, Any]) -> List[ValidationError]:
    """Validate Page 2: LLM Config."""
    errors = []

    errors.extend(validate_required(data.get('provider'), 'Provider'))
    errors.extend(validate_required(data.get('model'), 'Model'))

    # Validate temperature range
    temp = data.get('temperature')
    if temp is not None:
        if temp < 0.0 or temp > 2.0:
            errors.append(ValidationError("Temperature", "Temperature must be between 0.0 and 2.0"))

    # Validate max_tokens
    max_tokens = data.get('max_tokens')
    if max_tokens is not None:
        if max_tokens < 1:
            errors.append(ValidationError("Max Tokens", "Max tokens must be greater than 0"))

    return errors


def validate_page_4(data: Dict[str, Any]) -> List[ValidationError]:
    """Validate Page 4: Prompts."""
    errors = []

    errors.extend(validate_required(data.get('system_prompt'), 'System Prompt'))

    system_prompt = data.get('system_prompt', '')
    if system_prompt and len(system_prompt) < 10:
        errors.append(ValidationError("System Prompt", "System prompt should be at least 10 characters"))

    return errors


def validate_page_5(data: Dict[str, Any]) -> List[ValidationError]:
    """Validate Page 5: Memory."""
    errors = []

    # Validate short-term memory
    short_term = data.get('short_term', {})
    if short_term.get('enabled'):
        if short_term.get('type') == 'sqlite':
            if not short_term.get('path'):
                errors.append(ValidationError("Short-term Path", "SQLite path is required when type is sqlite"))

    # Validate long-term memory
    long_term = data.get('long_term', {})
    if long_term.get('enabled'):
        if long_term.get('type') == 'sqlite':
            if not long_term.get('path'):
                errors.append(ValidationError("Long-term Path", "SQLite path is required when type is sqlite"))

    return errors


def validate_all_pages(all_data: Dict[str, Any]) -> Dict[int, List[ValidationError]]:
    """
    Validate all pages.

    Args:
        all_data: Dictionary with all page data

    Returns:
        Dictionary mapping page numbers to error lists
    """
    validation_results = {}

    validation_results[1] = validate_page_1(all_data.get('page_1_data', {}))
    validation_results[2] = validate_page_2(all_data.get('page_2_data', {}))
    validation_results[4] = validate_page_4(all_data.get('page_4_data', {}))
    validation_results[5] = validate_page_5(all_data.get('page_5_data', {}))

    # Filter out pages with no errors
    return {page: errors for page, errors in validation_results.items() if errors}
