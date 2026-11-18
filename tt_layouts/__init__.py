"""Compatibility imports for the original package name."""

from importlib import import_module
import sys

for _name in ("config", "geometry", "model", "inventory", "generate", "export"):
    _module = import_module(f"railroad_layouts.{_name}")
    globals()[_name] = sys.modules[f"{__name__}.{_name}"] = _module
