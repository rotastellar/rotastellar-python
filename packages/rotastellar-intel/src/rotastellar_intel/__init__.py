"""
RotaStellar Intel - Orbital Intelligence & Space Situational Awareness

Tools for tracking, analyzing, and monitoring orbital activity.

Coming Q1 2026.
"""

__version__ = "0.0.1"
__author__ = "Rota, Inc."

__all__ = ["__version__", "__author__"]


def __getattr__(name: str):
    raise NotImplementedError(
        f"rotastellar_intel.{name} is not yet available. "
        f"Launching Q1 2026. Visit https://rotastellar.com"
    )
