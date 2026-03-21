from __future__ import annotations

from typing import Any


def item_get(item: Any, key: str) -> Any:
    if item is None:
        return None
    if isinstance(item, dict):
        return item.get(key)
    try:
        return item.get(key)
    except Exception:
        pass
    try:
        return item[key]
    except Exception:
        return None


def get_field_value(item: Any, field_spec: Any) -> Any:
    if field_spec is None:
        return None
    if isinstance(field_spec, (list, tuple)):
        for spec in field_spec:
            val = get_field_value(item, spec)
            if val is not None:
                return val
        return None
    if isinstance(field_spec, str) and "." in field_spec:
        current = item
        for part in field_spec.split("."):
            if current is None:
                return None
            current = item_get(current, part)
        return current
    if isinstance(field_spec, str):
        return item_get(item, field_spec)
    return None
