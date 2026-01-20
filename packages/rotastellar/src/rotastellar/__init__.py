"""
RotaStellar - Space Computing Infrastructure

Python SDK for orbital compute planning, simulation, and space intelligence.

Documentation: https://rotastellar.com/docs
GitHub: https://github.com/rotastellar/rotastellar-python

Coming Q1 2026.
"""

__version__ = "0.0.1"
__author__ = "Rota, Inc."
__email__ = "hello@rotastellar.com"
__url__ = "https://rotastellar.com"

__all__ = ["__version__", "__author__", "__email__", "__url__"]


def __getattr__(name: str):
    raise NotImplementedError(
        f"rotastellar.{name} is not yet available. "
        f"The RotaStellar SDK is launching Q1 2026. "
        f"Visit https://rotastellar.com for updates."
    )
