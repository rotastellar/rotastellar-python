"""
RotaStellar Distributed - Model Partitioning

Optimal model partitioning across Earth and orbital compute nodes.

subhadipmitra@: This is essentially pipeline parallelism but with *much* higher
inter-node latency (20-40ms vs ~0.1ms in a datacenter). The key insight is that
we want to minimize the amount of data that crosses the Earth-space boundary.

For transformer models, this usually means:
- Keep embeddings on ground (large vocab = large activations)
- Keep attention layers on orbital (compute-heavy, small activations)
- Keep output projection on ground (large vocab again)

The optimizer finds the best split point by simulating latency for all possible
cut points. For a 12-layer transformer, that's only 13 options so brute force works.

Related work:
- "PipeDream" (Microsoft, 2019) - pipeline parallelism
- "GPipe" (Google, 2019) - micro-batch pipelining
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Tuple

# TODO(subhadipmitra): Add memory constraints to the optimization
# TODO: Support tensor parallelism within orbital nodes
# NOTE: The FLOP estimates are approximate - real values need profiling


class LayerType(Enum):
    """Type of neural network layer."""
    LINEAR = "linear"
    CONV2D = "conv2d"
    ATTENTION = "attention"
    EMBEDDING = "embedding"
    NORMALIZATION = "normalization"
    ACTIVATION = "activation"
    POOLING = "pooling"
    OTHER = "other"


class PlacementLocation(Enum):
    """Where to place a layer for computation."""
    GROUND = "ground"
    ORBITAL = "orbital"
    SPLIT = "split"  # Split across ground and orbital


class OptimizationObjective(Enum):
    """Objective for partition optimization."""
    MINIMIZE_LATENCY = "minimize_latency"
    MINIMIZE_BANDWIDTH = "minimize_bandwidth"
    BALANCE = "balance"
    MAXIMIZE_THROUGHPUT = "maximize_throughput"


@dataclass
class LayerProfile:
    """Profile of a single layer's compute characteristics.

    Attributes:
        name: Layer name
        layer_type: Type of layer
        params: Number of parameters
        flops: FLOPs required for forward pass
        input_size: Input tensor size in bytes
        output_size: Output tensor size in bytes
        activation_memory: Memory for activations in bytes
    """
    name: str
    layer_type: LayerType
    params: int
    flops: int
    input_size: int
    output_size: int
    activation_memory: int = 0

    @property
    def compute_intensity(self) -> float:
        """FLOPs per byte of memory access."""
        memory_access = self.input_size + self.output_size
        if memory_access == 0:
            return 0.0
        return self.flops / memory_access


@dataclass
class ModelProfile:
    """Profile of a model's layers and compute requirements.

    Example:
        >>> profile = ModelProfile.from_layers([
        ...     LayerProfile("embed", LayerType.EMBEDDING, 50000, 1000000, 1024, 4096),
        ...     LayerProfile("attn", LayerType.ATTENTION, 100000, 5000000, 4096, 4096),
        ...     LayerProfile("fc", LayerType.LINEAR, 200000, 2000000, 4096, 1024),
        ... ])
        >>> print(f"Total params: {profile.total_params}")
    """
    layers: List[LayerProfile] = field(default_factory=list)
    name: str = "model"

    @classmethod
    def from_layers(cls, layers: List[LayerProfile], name: str = "model") -> "ModelProfile":
        """Create profile from layer list."""
        return cls(layers=layers, name=name)

    @classmethod
    def create_transformer(
        cls,
        num_layers: int = 12,
        hidden_size: int = 768,
        vocab_size: int = 50000,
        seq_length: int = 512,
        name: str = "transformer"
    ) -> "ModelProfile":
        """Create a typical transformer model profile."""
        layers = []

        # Embedding layer
        embed_params = vocab_size * hidden_size
        embed_flops = seq_length * hidden_size
        layers.append(LayerProfile(
            name="embedding",
            layer_type=LayerType.EMBEDDING,
            params=embed_params,
            flops=embed_flops,
            input_size=seq_length * 4,
            output_size=seq_length * hidden_size * 4
        ))

        # Transformer layers
        for i in range(num_layers):
            # Self-attention
            attn_params = 4 * hidden_size * hidden_size
            attn_flops = 2 * seq_length * seq_length * hidden_size + 4 * seq_length * hidden_size * hidden_size
            layers.append(LayerProfile(
                name=f"layer_{i}_attention",
                layer_type=LayerType.ATTENTION,
                params=attn_params,
                flops=attn_flops,
                input_size=seq_length * hidden_size * 4,
                output_size=seq_length * hidden_size * 4
            ))

            # FFN
            ffn_params = 2 * hidden_size * (4 * hidden_size)
            ffn_flops = 2 * seq_length * hidden_size * (4 * hidden_size)
            layers.append(LayerProfile(
                name=f"layer_{i}_ffn",
                layer_type=LayerType.LINEAR,
                params=ffn_params,
                flops=ffn_flops,
                input_size=seq_length * hidden_size * 4,
                output_size=seq_length * hidden_size * 4
            ))

        # Output layer
        output_params = hidden_size * vocab_size
        output_flops = seq_length * hidden_size * vocab_size
        layers.append(LayerProfile(
            name="output",
            layer_type=LayerType.LINEAR,
            params=output_params,
            flops=output_flops,
            input_size=seq_length * hidden_size * 4,
            output_size=seq_length * vocab_size * 4
        ))

        return cls(layers=layers, name=name)

    @property
    def total_params(self) -> int:
        """Total number of parameters."""
        return sum(l.params for l in self.layers)

    @property
    def total_flops(self) -> int:
        """Total FLOPs for forward pass."""
        return sum(l.flops for l in self.layers)

    @property
    def num_layers(self) -> int:
        """Number of layers."""
        return len(self.layers)

    def summary(self) -> Dict:
        """Get model summary."""
        return {
            "name": self.name,
            "num_layers": self.num_layers,
            "total_params": self.total_params,
            "total_params_millions": round(self.total_params / 1e6, 2),
            "total_flops": self.total_flops,
            "total_gflops": round(self.total_flops / 1e9, 2),
        }


@dataclass
class LayerPlacement:
    """Placement decision for a single layer.

    Attributes:
        layer_name: Name of the layer
        location: Where to place the layer
        node_id: Specific node ID (if assigned)
        estimated_latency_ms: Estimated latency for this layer
        data_transfer_bytes: Data to transfer to next layer
    """
    layer_name: str
    location: PlacementLocation
    node_id: Optional[str] = None
    estimated_latency_ms: float = 0.0
    data_transfer_bytes: int = 0


@dataclass
class PartitionPlan:
    """Complete partitioning plan for a model.

    Attributes:
        model_name: Name of the model
        placements: List of layer placements
        total_latency_ms: Total estimated latency
        ground_orbital_transfers: Number of ground-orbital data transfers
        total_transfer_bytes: Total bytes transferred between ground and orbital
        objective: Optimization objective used
    """
    model_name: str
    placements: List[LayerPlacement]
    total_latency_ms: float
    ground_orbital_transfers: int
    total_transfer_bytes: int
    objective: OptimizationObjective

    @property
    def ground_layers(self) -> List[LayerPlacement]:
        """Layers placed on ground."""
        return [p for p in self.placements if p.location == PlacementLocation.GROUND]

    @property
    def orbital_layers(self) -> List[LayerPlacement]:
        """Layers placed on orbital nodes."""
        return [p for p in self.placements if p.location == PlacementLocation.ORBITAL]

    def summary(self) -> Dict:
        """Get partition plan summary."""
        return {
            "model_name": self.model_name,
            "total_layers": len(self.placements),
            "ground_layers": len(self.ground_layers),
            "orbital_layers": len(self.orbital_layers),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "ground_orbital_transfers": self.ground_orbital_transfers,
            "total_transfer_mb": round(self.total_transfer_bytes / 1e6, 2),
            "objective": self.objective.value,
        }


class LatencyEstimator:
    """Estimate latency for ground-orbital communication.

    Accounts for propagation delay, transmission time, and processing overhead.
    """

    SPEED_OF_LIGHT_KM_S = 299792.458
    DEFAULT_PROCESSING_OVERHEAD_MS = 5.0

    def __init__(
        self,
        orbit_altitude_km: float = 550.0,
        uplink_bandwidth_mbps: float = 100.0,
        downlink_bandwidth_mbps: float = 200.0,
        processing_overhead_ms: float = DEFAULT_PROCESSING_OVERHEAD_MS
    ):
        self.orbit_altitude_km = orbit_altitude_km
        self.uplink_bandwidth_mbps = uplink_bandwidth_mbps
        self.downlink_bandwidth_mbps = downlink_bandwidth_mbps
        self.processing_overhead_ms = processing_overhead_ms

    @property
    def propagation_delay_ms(self) -> float:
        """One-way propagation delay in ms."""
        return (self.orbit_altitude_km / self.SPEED_OF_LIGHT_KM_S) * 1000

    def estimate_transfer_time_ms(self, bytes_size: int, is_uplink: bool = True) -> float:
        """Estimate time to transfer data."""
        bandwidth = self.uplink_bandwidth_mbps if is_uplink else self.downlink_bandwidth_mbps
        bits = bytes_size * 8
        transfer_time_s = bits / (bandwidth * 1e6)
        return transfer_time_s * 1000

    def estimate_round_trip_ms(self, uplink_bytes: int, downlink_bytes: int) -> float:
        """Estimate round-trip latency for a ground-orbital transfer."""
        uplink_time = self.estimate_transfer_time_ms(uplink_bytes, is_uplink=True)
        downlink_time = self.estimate_transfer_time_ms(downlink_bytes, is_uplink=False)
        propagation = 2 * self.propagation_delay_ms
        return uplink_time + downlink_time + propagation + 2 * self.processing_overhead_ms


class PartitionOptimizer:
    """Optimize model partitioning across Earth and orbital nodes.

    Finds the optimal split point(s) to minimize latency, bandwidth usage,
    or achieve a balanced objective.

    Example:
        >>> model = ModelProfile.create_transformer(num_layers=12)
        >>> optimizer = PartitionOptimizer(
        ...     ground_compute_tflops=100.0,
        ...     orbital_compute_tflops=10.0,
        ... )
        >>> plan = optimizer.optimize(model, OptimizationObjective.MINIMIZE_LATENCY)
        >>> print(plan.summary())
    """

    def __init__(
        self,
        ground_compute_tflops: float = 100.0,
        orbital_compute_tflops: float = 10.0,
        orbit_altitude_km: float = 550.0,
        uplink_bandwidth_mbps: float = 100.0,
        downlink_bandwidth_mbps: float = 200.0
    ):
        self.ground_compute_tflops = ground_compute_tflops
        self.orbital_compute_tflops = orbital_compute_tflops
        self.latency_estimator = LatencyEstimator(
            orbit_altitude_km=orbit_altitude_km,
            uplink_bandwidth_mbps=uplink_bandwidth_mbps,
            downlink_bandwidth_mbps=downlink_bandwidth_mbps
        )

    def optimize(
        self,
        model: ModelProfile,
        objective: OptimizationObjective = OptimizationObjective.BALANCE
    ) -> PartitionPlan:
        """Find optimal partition for the model."""
        if objective == OptimizationObjective.MINIMIZE_LATENCY:
            return self._optimize_latency(model, objective)
        elif objective == OptimizationObjective.MINIMIZE_BANDWIDTH:
            return self._optimize_bandwidth(model, objective)
        else:
            return self._optimize_balanced(model, objective)

    def _optimize_latency(self, model: ModelProfile, objective: OptimizationObjective) -> PartitionPlan:
        """Minimize end-to-end latency."""
        # Try all possible split points
        best_plan = None
        best_latency = float('inf')

        for split_idx in range(len(model.layers) + 1):
            plan = self._create_plan(model, split_idx, objective)
            if plan.total_latency_ms < best_latency:
                best_latency = plan.total_latency_ms
                best_plan = plan

        return best_plan or self._create_plan(model, 0, objective)

    def _optimize_bandwidth(self, model: ModelProfile, objective: OptimizationObjective) -> PartitionPlan:
        """Minimize data transfer between ground and orbital."""
        # Find split point with minimum activation size
        min_transfer_idx = 0
        min_transfer_size = float('inf')

        for i in range(len(model.layers)):
            transfer_size = model.layers[i].output_size
            if transfer_size < min_transfer_size:
                min_transfer_size = transfer_size
                min_transfer_idx = i + 1

        return self._create_plan(model, min_transfer_idx, objective)

    def _optimize_balanced(self, model: ModelProfile, objective: OptimizationObjective) -> PartitionPlan:
        """Balance latency and bandwidth."""
        # Use compute ratio to determine split
        total_flops = model.total_flops
        target_ground_flops = total_flops * self.ground_compute_tflops / (
            self.ground_compute_tflops + self.orbital_compute_tflops
        )

        cumulative_flops = 0
        split_idx = 0
        for i, layer in enumerate(model.layers):
            cumulative_flops += layer.flops
            if cumulative_flops >= target_ground_flops:
                split_idx = i + 1
                break

        return self._create_plan(model, split_idx, objective)

    def _create_plan(
        self,
        model: ModelProfile,
        split_idx: int,
        objective: OptimizationObjective
    ) -> PartitionPlan:
        """Create a partition plan with given split point."""
        placements = []
        total_latency_ms = 0.0
        total_transfer = 0
        num_transfers = 0

        for i, layer in enumerate(model.layers):
            if i < split_idx:
                location = PlacementLocation.GROUND
                compute_tflops = self.ground_compute_tflops
            else:
                location = PlacementLocation.ORBITAL
                compute_tflops = self.orbital_compute_tflops

            # Compute time for this layer
            layer_latency_ms = (layer.flops / (compute_tflops * 1e12)) * 1000

            # Add transfer latency at split point
            transfer_bytes = 0
            if i == split_idx and split_idx > 0 and split_idx < len(model.layers):
                transfer_bytes = layer.input_size
                transfer_latency = self.latency_estimator.estimate_transfer_time_ms(
                    transfer_bytes, is_uplink=True
                )
                # Add propagation delay
                transfer_latency += self.latency_estimator.propagation_delay_ms
                layer_latency_ms += transfer_latency
                total_transfer += transfer_bytes
                num_transfers += 1

            placements.append(LayerPlacement(
                layer_name=layer.name,
                location=location,
                estimated_latency_ms=layer_latency_ms,
                data_transfer_bytes=transfer_bytes
            ))

            total_latency_ms += layer_latency_ms

        return PartitionPlan(
            model_name=model.name,
            placements=placements,
            total_latency_ms=total_latency_ms,
            ground_orbital_transfers=num_transfers,
            total_transfer_bytes=total_transfer,
            objective=objective
        )

    def compare_strategies(self, model: ModelProfile) -> Dict[str, PartitionPlan]:
        """Compare all optimization strategies."""
        return {
            obj.value: self.optimize(model, obj)
            for obj in OptimizationObjective
        }
