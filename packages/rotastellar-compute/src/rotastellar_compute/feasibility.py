"""
RotaStellar Compute - Feasibility Analysis

Analyze workload suitability for orbital compute environments.

subhadipmitra@: Not all workloads belong in space. This module helps users
understand whether their workload is a good fit before they commit.

Good candidates for orbital compute:
- Batch processing (tolerant of intermittent connectivity)
- Large-scale ML training (compute-bound, can checkpoint)
- Rendering farms (embarrassingly parallel)
- Scientific simulation (high compute:data ratio)

Poor candidates:
- Real-time trading (latency-critical)
- Interactive applications (user-facing latency)
- Database OLTP (requires persistent connections)
- Workloads with large data gravity (data transfer > compute)

The scoring algorithm weights factors based on our operational experience
with early customers. We'll tune these weights as we gather more data.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List

from rotastellar import RotaStellarClient

# TODO(subhadipmitra): Add cost estimation to feasibility report
# TODO: Factor in constellation coverage for latency-sensitive workloads


class WorkloadType(Enum):
    """Types of compute workloads."""

    INFERENCE = "inference"  # ML inference
    TRAINING = "training"  # ML model training
    BATCH = "batch"  # Batch processing
    STREAMING = "streaming"  # Real-time streaming
    RENDER = "render"  # Graphics rendering
    SIMULATION = "simulation"  # Scientific simulation
    ANALYTICS = "analytics"  # Data analytics


class FeasibilityRating(Enum):
    """Feasibility assessment rating."""

    EXCELLENT = "excellent"  # Highly suitable
    GOOD = "good"  # Suitable with minor adjustments
    MODERATE = "moderate"  # Possible but challenging
    POOR = "poor"  # Not recommended
    UNSUITABLE = "unsuitable"  # Not feasible


@dataclass
class WorkloadProfile:
    """Profile of a compute workload for feasibility analysis.

    Attributes:
        workload_type: Type of workload
        compute_tflops: Required compute in TFLOPS
        memory_gb: Required memory in GB
        storage_gb: Required storage in GB
        data_transfer_gb: Data to transfer per day
        latency_requirement_ms: Maximum acceptable latency (None = not latency-sensitive)
        batch_duration_hours: For batch workloads, typical job duration
        availability_requirement: Required uptime percentage (0-100)
    """

    workload_type: WorkloadType
    compute_tflops: float
    memory_gb: float = 16.0
    storage_gb: float = 100.0
    data_transfer_gb: float = 10.0
    latency_requirement_ms: Optional[float] = None
    batch_duration_hours: Optional[float] = None
    availability_requirement: float = 99.0


@dataclass
class FeasibilityResult:
    """Result of feasibility analysis.

    Attributes:
        feasible: Whether the workload is feasible for orbital compute
        rating: Overall feasibility rating
        score: Numeric score (0-100)
        compute_feasible: Whether compute requirements can be met
        thermal_feasible: Whether thermal constraints can be satisfied
        power_feasible: Whether power requirements can be met
        latency_feasible: Whether latency requirements can be met
        data_transfer_feasible: Whether data transfer requirements can be met
        recommendations: List of recommendations
        constraints: Key constraints identified
        estimated_cost_factor: Cost factor relative to terrestrial (1.0 = same)
    """

    feasible: bool
    rating: FeasibilityRating
    score: float
    compute_feasible: bool = True
    thermal_feasible: bool = True
    power_feasible: bool = True
    latency_feasible: bool = True
    data_transfer_feasible: bool = True
    recommendations: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    estimated_cost_factor: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feasible": self.feasible,
            "rating": self.rating.value,
            "score": self.score,
            "compute_feasible": self.compute_feasible,
            "thermal_feasible": self.thermal_feasible,
            "power_feasible": self.power_feasible,
            "latency_feasible": self.latency_feasible,
            "data_transfer_feasible": self.data_transfer_feasible,
            "recommendations": self.recommendations,
            "constraints": self.constraints,
            "estimated_cost_factor": self.estimated_cost_factor,
        }


class FeasibilityCalculator:
    """Analyze workload feasibility for orbital compute.

    Example:
        >>> from rotastellar_compute import FeasibilityCalculator, WorkloadProfile, WorkloadType
        >>> calculator = FeasibilityCalculator(api_key="rs_live_xxx")
        >>>
        >>> profile = WorkloadProfile(
        ...     workload_type=WorkloadType.INFERENCE,
        ...     compute_tflops=10,
        ...     memory_gb=32,
        ...     latency_requirement_ms=100
        ... )
        >>> result = calculator.analyze(profile)
        >>> print(f"Feasible: {result.feasible}, Rating: {result.rating.value}")
    """

    # Orbital constraints
    MAX_COMPUTE_TFLOPS = 100  # Current max per node
    MAX_MEMORY_GB = 256
    MAX_POWER_WATTS = 2000
    MIN_LATENCY_MS = 20  # Physical minimum (LEO round trip)
    MAX_DATA_TRANSFER_GB_DAY = 1000  # Ground station limitations

    # Workload type characteristics
    WORKLOAD_CHARACTERISTICS = {
        WorkloadType.INFERENCE: {
            "thermal_factor": 0.7,
            "power_factor": 0.6,
            "latency_sensitive": True,
            "batch_friendly": True,
        },
        WorkloadType.TRAINING: {
            "thermal_factor": 1.0,
            "power_factor": 1.0,
            "latency_sensitive": False,
            "batch_friendly": True,
        },
        WorkloadType.BATCH: {
            "thermal_factor": 0.8,
            "power_factor": 0.7,
            "latency_sensitive": False,
            "batch_friendly": True,
        },
        WorkloadType.STREAMING: {
            "thermal_factor": 0.5,
            "power_factor": 0.5,
            "latency_sensitive": True,
            "batch_friendly": False,
        },
        WorkloadType.RENDER: {
            "thermal_factor": 1.0,
            "power_factor": 0.9,
            "latency_sensitive": False,
            "batch_friendly": True,
        },
        WorkloadType.SIMULATION: {
            "thermal_factor": 0.9,
            "power_factor": 0.8,
            "latency_sensitive": False,
            "batch_friendly": True,
        },
        WorkloadType.ANALYTICS: {
            "thermal_factor": 0.6,
            "power_factor": 0.5,
            "latency_sensitive": False,
            "batch_friendly": True,
        },
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        client: Optional[RotaStellarClient] = None,
        orbit_altitude_km: float = 550,
        **kwargs,
    ):
        """Initialize the feasibility calculator.

        Args:
            api_key: RotaStellar API key
            client: Existing RotaStellarClient
            orbit_altitude_km: Target orbit altitude (affects latency)
            **kwargs: Additional client arguments
        """
        if client is not None:
            self._client = client
        else:
            self._client = RotaStellarClient(api_key=api_key, **kwargs)

        self.orbit_altitude_km = orbit_altitude_km

    @property
    def client(self) -> RotaStellarClient:
        """Get the underlying API client."""
        return self._client

    def analyze(
        self,
        profile: WorkloadProfile,
        orbit_altitude_km: Optional[float] = None,
    ) -> FeasibilityResult:
        """Analyze workload feasibility.

        Args:
            profile: Workload profile to analyze
            orbit_altitude_km: Override orbit altitude

        Returns:
            Feasibility analysis result
        """
        altitude = orbit_altitude_km or self.orbit_altitude_km
        characteristics = self.WORKLOAD_CHARACTERISTICS[profile.workload_type]

        # Check individual constraints
        compute_ok, compute_score = self._check_compute(profile)
        thermal_ok, thermal_score = self._check_thermal(profile, characteristics)
        power_ok, power_score = self._check_power(profile, characteristics)
        latency_ok, latency_score = self._check_latency(profile, altitude, characteristics)
        data_ok, data_score = self._check_data_transfer(profile)

        # Calculate overall score
        scores = [compute_score, thermal_score, power_score, latency_score, data_score]
        overall_score = sum(scores) / len(scores)

        # Determine feasibility
        feasible = all([compute_ok, thermal_ok, power_ok, latency_ok, data_ok])

        # Determine rating
        if overall_score >= 85:
            rating = FeasibilityRating.EXCELLENT
        elif overall_score >= 70:
            rating = FeasibilityRating.GOOD
        elif overall_score >= 50:
            rating = FeasibilityRating.MODERATE
        elif overall_score >= 30:
            rating = FeasibilityRating.POOR
        else:
            rating = FeasibilityRating.UNSUITABLE

        # Generate recommendations
        recommendations = self._generate_recommendations(
            profile, compute_ok, thermal_ok, power_ok, latency_ok, data_ok
        )

        # Estimate cost factor
        cost_factor = self._estimate_cost_factor(profile, characteristics)

        return FeasibilityResult(
            feasible=feasible,
            rating=rating,
            score=overall_score,
            compute_feasible=compute_ok,
            thermal_feasible=thermal_ok,
            power_feasible=power_ok,
            latency_feasible=latency_ok,
            data_transfer_feasible=data_ok,
            recommendations=recommendations,
            constraints={
                "compute_score": compute_score,
                "thermal_score": thermal_score,
                "power_score": power_score,
                "latency_score": latency_score,
                "data_transfer_score": data_score,
                "orbit_altitude_km": altitude,
            },
            estimated_cost_factor=cost_factor,
        )

    def _check_compute(self, profile: WorkloadProfile) -> tuple:
        """Check compute requirements."""
        if profile.compute_tflops > self.MAX_COMPUTE_TFLOPS:
            return False, 20.0
        if profile.memory_gb > self.MAX_MEMORY_GB:
            return False, 30.0

        # Score based on utilization
        compute_util = profile.compute_tflops / self.MAX_COMPUTE_TFLOPS
        memory_util = profile.memory_gb / self.MAX_MEMORY_GB

        if compute_util <= 0.5 and memory_util <= 0.5:
            return True, 100.0
        elif compute_util <= 0.8 and memory_util <= 0.8:
            return True, 80.0
        else:
            return True, 60.0

    def _check_thermal(
        self, profile: WorkloadProfile, characteristics: Dict
    ) -> tuple:
        """Check thermal constraints."""
        thermal_load = profile.compute_tflops * characteristics["thermal_factor"]

        # Simplified thermal model
        max_thermal_load = 70  # Arbitrary units

        if thermal_load > max_thermal_load:
            return False, 20.0

        score = 100 * (1 - thermal_load / max_thermal_load)
        return True, max(score, 40.0)

    def _check_power(
        self, profile: WorkloadProfile, characteristics: Dict
    ) -> tuple:
        """Check power constraints."""
        # Estimate power from compute
        estimated_power = profile.compute_tflops * 20 * characteristics["power_factor"]

        if estimated_power > self.MAX_POWER_WATTS:
            return False, 20.0

        score = 100 * (1 - estimated_power / self.MAX_POWER_WATTS)
        return True, max(score, 40.0)

    def _check_latency(
        self,
        profile: WorkloadProfile,
        altitude_km: float,
        characteristics: Dict,
    ) -> tuple:
        """Check latency constraints."""
        if profile.latency_requirement_ms is None:
            return True, 100.0

        # Calculate minimum achievable latency
        # Round trip to LEO + processing
        speed_of_light_km_s = 299792.458
        min_latency = (2 * altitude_km / speed_of_light_km_s) * 1000 + 5  # +5ms processing

        if not characteristics["latency_sensitive"]:
            return True, 90.0

        if profile.latency_requirement_ms < min_latency:
            return False, 10.0

        margin = profile.latency_requirement_ms - min_latency
        if margin > 50:
            return True, 100.0
        elif margin > 20:
            return True, 80.0
        else:
            return True, 60.0

    def _check_data_transfer(self, profile: WorkloadProfile) -> tuple:
        """Check data transfer requirements."""
        if profile.data_transfer_gb > self.MAX_DATA_TRANSFER_GB_DAY:
            return False, 20.0

        util = profile.data_transfer_gb / self.MAX_DATA_TRANSFER_GB_DAY
        if util <= 0.3:
            return True, 100.0
        elif util <= 0.6:
            return True, 80.0
        else:
            return True, 60.0

    def _generate_recommendations(
        self,
        profile: WorkloadProfile,
        compute_ok: bool,
        thermal_ok: bool,
        power_ok: bool,
        latency_ok: bool,
        data_ok: bool,
    ) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        if not compute_ok:
            recommendations.append(
                "Consider partitioning workload across multiple orbital nodes"
            )

        if not thermal_ok:
            recommendations.append(
                "Implement duty cycling to manage thermal constraints"
            )

        if not power_ok:
            recommendations.append(
                "Schedule compute-intensive tasks during solar exposure windows"
            )

        if not latency_ok:
            recommendations.append(
                "Consider edge caching or predictive pre-computation"
            )

        if not data_ok:
            recommendations.append(
                "Implement data compression or delta-sync strategies"
            )

        # Positive recommendations
        if profile.workload_type in [WorkloadType.BATCH, WorkloadType.TRAINING]:
            recommendations.append(
                "Batch workloads are well-suited for orbital compute"
            )

        return recommendations

    def _estimate_cost_factor(
        self, profile: WorkloadProfile, characteristics: Dict
    ) -> float:
        """Estimate cost factor relative to terrestrial compute."""
        base_factor = 2.5  # Base orbital premium

        # Adjust for workload type
        if characteristics["batch_friendly"]:
            base_factor *= 0.8

        # Adjust for compute intensity
        if profile.compute_tflops > 50:
            base_factor *= 1.2

        # Adjust for data transfer
        if profile.data_transfer_gb > 500:
            base_factor *= 1.3

        return round(base_factor, 2)

    def compare_scenarios(
        self,
        profile: WorkloadProfile,
        altitudes: List[float],
    ) -> List[Dict[str, Any]]:
        """Compare feasibility across different orbit scenarios.

        Args:
            profile: Workload profile
            altitudes: List of orbit altitudes to compare

        Returns:
            List of results for each altitude
        """
        results = []
        for altitude in altitudes:
            result = self.analyze(profile, orbit_altitude_km=altitude)
            results.append({
                "altitude_km": altitude,
                "feasible": result.feasible,
                "rating": result.rating.value,
                "score": result.score,
            })
        return results
