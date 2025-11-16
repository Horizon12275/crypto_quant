import importlib
from typing import Callable, Dict


class FactorRegistry:
    """Registry mapping factor names to callables."""

    def __init__(self) -> None:
        self._registry: Dict[str, Callable] = {}

    def register(self, name: str, fn: Callable) -> None:
        if not callable(fn):
            raise ValueError("Factor must be callable")
        self._registry[name] = fn

    def get(self, name: str) -> Callable:
        if name not in self._registry:
            raise KeyError(f"Factor '{name}' not found in registry")
        return self._registry[name]

    def names(self):
        return list(self._registry.keys())


registry = FactorRegistry()


def register_builtin_factors() -> None:
    """Import and register built-in example factors."""
    module = importlib.import_module("factors.example_factors")
    if hasattr(module, "register"):
        module.register(registry)
    module = importlib.import_module("factors.volume_factors")
    if hasattr(module, "register"):
        module.register(registry)