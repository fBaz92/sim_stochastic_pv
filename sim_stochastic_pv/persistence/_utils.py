"""
Shared utility functions for persistence layer.
"""

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Mapping


def asdict_safe(obj: Any) -> Dict[str, Any]:
    """
    Convert Python objects to plain dictionaries for JSON storage.

    Handles multiple object types commonly used in the application:
    - Dataclasses (Python standard library)
    - Pydantic models (v1 and v2)
    - Mappings (dict, OrderedDict, etc.)
    - None (returns empty dict)

    Args:
        obj: Object to convert. Can be:
            - Dataclass instance: Converted via asdict()
            - Pydantic model: Converted via model_dump() or dict()
            - Mapping: Converted via dict()
            - None: Returns {}

    Returns:
        Dictionary representation of the object, suitable for JSON serialization.
            Empty dict if obj is None.

    Raises:
        TypeError: If obj type is not supported (not dataclass, Pydantic, mapping, or None).

    Example:
        ```python
        from dataclasses import dataclass
        from pydantic import BaseModel

        @dataclass
        class Hardware:
            name: str
            power: float

        class Config(BaseModel):
            pv_kwp: float

        # Dataclass
        hw = Hardware(name="Inverter", power=5.0)
        d1 = asdict_safe(hw)  # {"name": "Inverter", "power": 5.0}

        # Pydantic
        cfg = Config(pv_kwp=6.5)
        d2 = asdict_safe(cfg)  # {"pv_kwp": 6.5}

        # Dict
        d3 = asdict_safe({"key": "value"})  # {"key": "value"}

        # None
        d4 = asdict_safe(None)  # {}
        ```

    Notes:
        - Pydantic v2 uses model_dump(), v1 uses dict() - both are supported
        - Nested dataclasses/Pydantic models are recursively converted
        - Used internally for database serialization
    """
    if obj is None:
        return {}
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Mapping):
        return dict(obj)
    # Handle Pydantic models (v2 style)
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    # Handle Pydantic models (v1 style)
    if hasattr(obj, "dict"):
        return obj.dict()
    raise TypeError(f"Unsupported object type for serialization: {type(obj)!r}")
