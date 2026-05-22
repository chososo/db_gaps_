"""Strategy registry.

Use ``@register("name")`` on a Strategy subclass / factory to register it.
Strategies (or factory callables) become available via ``get("name")``.
"""

from __future__ import annotations

from typing import Callable

from .base import Strategy

_REGISTRY: dict[str, Callable[..., Strategy]] = {}


def register(name: str):
    def deco(factory: Callable[..., Strategy]):
        _REGISTRY[name] = factory
        return factory
    return deco


def get(name: str, **kwargs) -> Strategy:
    if name not in _REGISTRY:
        raise KeyError(f"Strategy '{name}' not found. Available: {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def available() -> list[str]:
    return sorted(_REGISTRY)
