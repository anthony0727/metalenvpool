"""Packaged Metal shader sources."""

from __future__ import annotations

from importlib import resources


def shader_source(name: str) -> str:
    """Read a packaged Metal shader source file."""

    return resources.files(__package__).joinpath(name).read_text(encoding="utf-8")


__all__ = ["shader_source"]
