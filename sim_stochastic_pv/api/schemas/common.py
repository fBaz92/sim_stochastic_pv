"""
Common utilities and base schemas for API validation.

This module provides shared functionality used across all schema modules:
- Type coercion utilities for handling different input types
- Field merging utilities for flexible schema evolution
- Base validators and transformers

These utilities allow schemas to:
1. Accept both dict and SQLAlchemy model inputs (for ORM mode)
2. Auto-populate missing fields from JSON blob fields
3. Maintain backward compatibility while evolving schemas
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping


def _coerce_to_dict(data: Any) -> Dict[str, Any]:
    """
    Coerce various input types to dictionary for Pydantic validation.

    Handles multiple input types to support flexible schema validation:
    - Plain dicts: Pass through unchanged
    - Mappings: Convert to dict
    - SQLAlchemy models: Return as-is for Pydantic's from_attributes mode
    - Other types: Attempt dict conversion

    Args:
        data: Input data to coerce. Can be dict, Mapping, SQLAlchemy model,
            or any dict-like object.

    Returns:
        Dictionary representation of the input, or the original object
        if it's a SQLAlchemy model (for ORM mode compatibility).

    Notes:
        - SQLAlchemy models are detected by __tablename__ or _sa_instance_state
        - Pydantic's from_attributes mode (formerly orm_mode) handles models natively
        - This function is used in @root_validator(pre=True) decorators

    Example:
        >>> data = {"name": "test", "value": 123}
        >>> result = _coerce_to_dict(data)
        >>> assert result == data

        >>> from sqlalchemy.orm import DeclarativeBase
        >>> class Model(DeclarativeBase):
        ...     __tablename__ = "test"
        >>> model = Model()
        >>> result = _coerce_to_dict(model)
        >>> assert result is model  # Returns original for ORM mode
    """
    if data is None:
        return {}
    if isinstance(data, dict):
        return data
    if isinstance(data, Mapping):
        return dict(data)
    # Handle SQLAlchemy models (ORM mode) - return as-is for Pydantic to handle
    if hasattr(data, "__tablename__") or hasattr(data, "_sa_instance_state"):
        return data  # type: ignore
    return dict(data)


def _merge_specs_defaults(values: Dict[str, Any], fields: List[str]) -> Dict[str, Any]:
    """
    Merge missing fields from a 'specs' JSON blob into top-level fields.

    Enables flexible schema evolution: new fields can be added to the 'specs'
    JSON column without database migrations, and will be automatically extracted
    when present in API responses.

    This pattern allows:
    - Adding new optional fields without schema changes
    - Storing vendor-specific data in specs
    - Gradual migration from specs to dedicated columns

    Args:
        values: Dictionary of field values (can be dict or SQLAlchemy model).
        fields: List of field names to attempt extraction from specs.

    Returns:
        Enhanced values dictionary with specs fields merged into top-level.
        For SQLAlchemy models, returns the model unchanged (Pydantic handles it).

    Notes:
        - Only populates fields that are None/missing in values
        - Preserves explicit None values (won't override with specs data)
        - For SQLAlchemy models, relies on Pydantic's from_attributes mode

    Example:
        >>> values = {
        ...     "name": "Test",
        ...     "price": None,
        ...     "specs": {"price": 100.0, "warranty_years": 5}
        ... }
        >>> result = _merge_specs_defaults(values, ["price", "warranty_years"])
        >>> assert result["price"] == 100.0
        >>> assert result["warranty_years"] == 5

        >>> # Explicit values are preserved
        >>> values2 = {"name": "Test", "price": 50.0, "specs": {"price": 100.0}}
        >>> result2 = _merge_specs_defaults(values2, ["price"])
        >>> assert result2["price"] == 50.0  # Explicit value kept
    """
    # Handle SQLAlchemy models - get attribute instead of dict access
    if hasattr(values, "__tablename__") or hasattr(values, "_sa_instance_state"):
        # SQLAlchemy model - access specs attribute
        specs = getattr(values, "specs", None)
    else:
        # Regular dict
        specs = values.get("specs")

    if isinstance(specs, Mapping):
        for field in fields:
            # Handle both dict and model access
            if hasattr(values, "__tablename__") or hasattr(values, "_sa_instance_state"):
                current_val = getattr(values, field, None)
            else:
                current_val = values.get(field)

            if current_val is None and field in specs:
                if isinstance(values, dict):
                    values[field] = specs[field]
                # For models, let Pydantic handle it naturally - don't set attributes
    return values
