"""
RotaStellar Compute - Orbital Compute Planning & Simulation

Tools for planning and simulating space-based data centers.

Documentation: https://rotastellar.com/docs/compute
GitHub: https://github.com/rotastellar/rotastellar-python

Example:
    >>> from rotastellar_compute import FeasibilityCalculator, WorkloadProfile, WorkloadType
    >>>
    >>> calculator = FeasibilityCalculator(api_key="rs_live_xxx")
    >>> profile = WorkloadProfile(
    ...     workload_type=WorkloadType.INFERENCE,
    ...     compute_tflops=10,
    ...     memory_gb=32
    ... )
    >>> result = calculator.analyze(profile)
    >>> print(f"Feasible: {result.feasible}, Rating: {result.rating.value}")
"""

__version__ = "0.1.0"
__author__ = "Rota, Inc."
__email__ = "hello@rotastellar.com"
__url__ = "https://rotastellar.com"

# Feasibility
from .feasibility import (
    FeasibilityCalculator,
    FeasibilityResult,
    FeasibilityRating,
    WorkloadProfile,
    WorkloadType,
)

# Thermal
from .thermal import (
    ThermalSimulator,
    ThermalResult,
    ThermalConfig,
    ThermalEnvironment,
    OrbitType,
)

# Latency
from .latency import (
    LatencySimulator,
    LatencyResult,
    LatencyComponent,
    LinkType,
)

# Power
from .power import (
    PowerAnalyzer,
    PowerBudget,
    PowerProfile,
    SolarConfig,
    BatteryConfig,
    SolarCellType,
    BatteryChemistry,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    "__url__",
    # Feasibility
    "FeasibilityCalculator",
    "FeasibilityResult",
    "FeasibilityRating",
    "WorkloadProfile",
    "WorkloadType",
    # Thermal
    "ThermalSimulator",
    "ThermalResult",
    "ThermalConfig",
    "ThermalEnvironment",
    "OrbitType",
    # Latency
    "LatencySimulator",
    "LatencyResult",
    "LatencyComponent",
    "LinkType",
    # Power
    "PowerAnalyzer",
    "PowerBudget",
    "PowerProfile",
    "SolarConfig",
    "BatteryConfig",
    "SolarCellType",
    "BatteryChemistry",
]
