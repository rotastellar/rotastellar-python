"""
RotaStellar Distributed - Federated Learning

Federated learning components for Earth-space distributed training.

subhadipmitra@: This module is inspired by the communication constraints we face
in Earth-to-orbit scenarios. Standard federated learning assumes ~100Mbps links,
but LEO uplinks are often 10-50Mbps with 20-40ms latency. The compression here
is aggressive by design.

References:
- "Communication-Efficient Learning" (McMahan et al., 2017)
- "Deep Gradient Compression" (Lin et al., 2018)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Callable, Any
import math

# TODO(subhadipmitra): Add support for momentum correction in error feedback
# TODO: Benchmark against PyTorch's built-in gradient compression


class CompressionMethod(Enum):
    """Gradient compression method."""
    NONE = "none"
    TOP_K = "topk"
    TOP_K_QUANTIZED = "topk_quantized"
    RANDOM_K = "random_k"
    QUANTIZATION = "quantization"


class AggregationStrategy(Enum):
    """Strategy for aggregating gradients from multiple nodes."""
    FEDAVG = "fedavg"
    ASYNC_FEDAVG = "async_fedavg"
    WEIGHTED_AVG = "weighted_avg"


@dataclass
class CompressionConfig:
    """Configuration for gradient compression.

    Compression is critical for Earth-space distributed training due to
    limited uplink bandwidth (typically 50-100 Mbps).

    Attributes:
        method: Compression method to use
        k_ratio: For Top-K methods, fraction of gradients to keep (e.g., 0.01 = top 1%)
        quantization_bits: Number of bits for quantization (8, 4, or 2)
        error_feedback: Whether to accumulate compression error for future rounds
        seed: Random seed for reproducible compression
    """
    method: CompressionMethod = CompressionMethod.TOP_K_QUANTIZED
    k_ratio: float = 0.01
    quantization_bits: int = 8
    error_feedback: bool = True
    seed: Optional[int] = None

    def __post_init__(self):
        if not 0 < self.k_ratio <= 1.0:
            raise ValueError("k_ratio must be between 0 and 1")
        if self.quantization_bits not in (2, 4, 8, 16, 32):
            raise ValueError("quantization_bits must be 2, 4, 8, 16, or 32")

    @property
    def theoretical_compression_ratio(self) -> float:
        """Theoretical compression ratio (smaller = more compression).

        subhadipmitra@: Note that actual compression may differ due to:
        - Index encoding overhead (we use 32-bit indices, could use varint)
        - Metadata headers not accounted for here
        - Wire format overhead (protobuf, etc.)
        """
        if self.method == CompressionMethod.NONE:
            return 1.0
        elif self.method == CompressionMethod.TOP_K:
            # subhadipmitra@: 32 bits for value + 32 bits for index = 2x per element
            return self.k_ratio * (1 + 32 / 32)  # value + index
        elif self.method == CompressionMethod.TOP_K_QUANTIZED:
            return self.k_ratio * (self.quantization_bits / 32 + 32 / 32)
        elif self.method == CompressionMethod.QUANTIZATION:
            return self.quantization_bits / 32
        else:  # RANDOM_K
            return self.k_ratio

    @classmethod
    def high_compression(cls) -> "CompressionConfig":
        """Configuration for maximum compression (1000x+)."""
        return cls(
            method=CompressionMethod.TOP_K_QUANTIZED,
            k_ratio=0.001,
            quantization_bits=4,
            error_feedback=True
        )

    @classmethod
    def balanced(cls) -> "CompressionConfig":
        """Balanced compression and accuracy."""
        return cls(
            method=CompressionMethod.TOP_K_QUANTIZED,
            k_ratio=0.01,
            quantization_bits=8,
            error_feedback=True
        )

    @classmethod
    def low_compression(cls) -> "CompressionConfig":
        """Minimal compression for high-bandwidth links."""
        return cls(
            method=CompressionMethod.QUANTIZATION,
            k_ratio=1.0,
            quantization_bits=16,
            error_feedback=False
        )


@dataclass
class CompressedGradient:
    """Compressed gradient representation.

    Stores sparse gradients with indices and optionally quantized values.
    """
    indices: List[int]
    values: List[float]
    shape: tuple
    original_size: int
    compressed_size: int
    compression_ratio: float
    quantization_bits: Optional[int] = None

    @property
    def sparsity(self) -> float:
        """Fraction of zero values after compression."""
        return 1.0 - len(self.indices) / self.original_size


class GradientCompressor:
    """Compress gradients for bandwidth-efficient synchronization.

    Example:
        >>> config = CompressionConfig.balanced()
        >>> compressor = GradientCompressor(config)
        >>> gradients = [0.1, 0.001, 0.5, -0.2, 0.002, 0.8]
        >>> compressed = compressor.compress(gradients)
        >>> print(f"Compression ratio: {compressed.compression_ratio:.2f}")
    """

    def __init__(self, config: CompressionConfig):
        self.config = config
        self._error_accumulator: Optional[List[float]] = None

    def compress(self, gradients: List[float]) -> CompressedGradient:
        """Compress gradients using configured method."""
        original_size = len(gradients)

        # Apply error feedback if enabled
        if self.config.error_feedback and self._error_accumulator is not None:
            gradients = [g + e for g, e in zip(gradients, self._error_accumulator)]

        if self.config.method == CompressionMethod.NONE:
            return CompressedGradient(
                indices=list(range(original_size)),
                values=gradients,
                shape=(original_size,),
                original_size=original_size,
                compressed_size=original_size * 4,
                compression_ratio=1.0
            )

        # Select top-k or random-k indices
        k = max(1, int(original_size * self.config.k_ratio))

        if self.config.method in (CompressionMethod.TOP_K, CompressionMethod.TOP_K_QUANTIZED):
            # Top-K selection
            indexed = [(i, abs(v), v) for i, v in enumerate(gradients)]
            indexed.sort(key=lambda x: x[1], reverse=True)
            selected = indexed[:k]
            indices = [x[0] for x in selected]
            values = [x[2] for x in selected]
        else:
            # Random-K selection
            import random
            if self.config.seed is not None:
                random.seed(self.config.seed)
            indices = random.sample(range(original_size), k)
            values = [gradients[i] for i in indices]

        # Apply quantization if needed
        if self.config.method == CompressionMethod.TOP_K_QUANTIZED:
            values = self._quantize(values)

        # Calculate compression error for error feedback
        if self.config.error_feedback:
            reconstructed = [0.0] * original_size
            for i, v in zip(indices, values):
                reconstructed[i] = v
            self._error_accumulator = [g - r for g, r in zip(gradients, reconstructed)]

        # Calculate compressed size (indices + values)
        bits_per_value = self.config.quantization_bits if self.config.method == CompressionMethod.TOP_K_QUANTIZED else 32
        compressed_bits = k * (32 + bits_per_value)  # 32 bits for index
        compressed_size = compressed_bits // 8

        return CompressedGradient(
            indices=indices,
            values=values,
            shape=(original_size,),
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compressed_size / (original_size * 4),
            quantization_bits=self.config.quantization_bits if self.config.method == CompressionMethod.TOP_K_QUANTIZED else None
        )

    def decompress(self, compressed: CompressedGradient) -> List[float]:
        """Decompress gradients back to dense representation."""
        result = [0.0] * compressed.original_size
        for i, v in zip(compressed.indices, compressed.values):
            result[i] = v
        return result

    def _quantize(self, values: List[float]) -> List[float]:
        """Quantize values to configured bit width.

        subhadipmitra@: Using linear quantization here. Could explore:
        - Non-uniform quantization (more levels near zero where most gradients are)
        - Learned quantization boundaries
        But linear is simple and works well enough for our use case.
        """
        if not values:
            return values

        # Find range for quantization
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val

        if range_val == 0:
            return values

        # TODO(subhadipmitra): Consider stochastic rounding for better convergence
        levels = 2 ** self.config.quantization_bits
        scale = range_val / (levels - 1)

        quantized = []
        for v in values:
            q = round((v - min_val) / scale)
            quantized.append(min_val + q * scale)

        return quantized


class FederatedClient:
    """Client for federated learning on Earth or orbital nodes.

    Handles local training, gradient compression, and synchronization
    with the central aggregator.

    Example:
        >>> from rotastellar_distributed import FederatedClient, CompressionConfig
        >>> client = FederatedClient(
        ...     node_id="orbital-1",
        ...     compression=CompressionConfig.balanced()
        ... )
        >>> # Simulate local training
        >>> gradients = client.compute_gradients(model_params, local_data)
        >>> compressed = client.compress(gradients)
        >>> # Send compressed gradients to aggregator
    """

    def __init__(
        self,
        node_id: str,
        compression: Optional[CompressionConfig] = None,
        node_type: str = "orbital"
    ):
        self.node_id = node_id
        self.node_type = node_type
        self.compression = compression or CompressionConfig.balanced()
        self._compressor = GradientCompressor(self.compression)
        self._local_steps = 0
        self._sync_round = 0

    def compute_gradients(
        self,
        model_params: List[float],
        local_data: List[Any],
        loss_fn: Optional[Callable] = None
    ) -> List[float]:
        """Compute gradients from local data (simulation).

        In production, this would integrate with PyTorch/TensorFlow.
        Here we simulate gradient computation.
        """
        # Simulated gradient computation
        import random
        gradients = [random.gauss(0, 0.1) for _ in model_params]
        self._local_steps += 1
        return gradients

    def compress(self, gradients: List[float]) -> CompressedGradient:
        """Compress gradients for transmission."""
        return self._compressor.compress(gradients)

    def apply_update(self, model_params: List[float], update: List[float]) -> List[float]:
        """Apply aggregated update to local model."""
        return [p - u * 0.01 for p, u in zip(model_params, update)]  # lr=0.01

    def get_stats(self) -> Dict:
        """Get client statistics."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "local_steps": self._local_steps,
            "sync_rounds": self._sync_round,
            "compression_method": self.compression.method.value,
            "compression_ratio": self.compression.theoretical_compression_ratio,
        }


class GradientAggregator:
    """Central aggregator for gradient synchronization.

    Collects compressed gradients from distributed nodes and computes
    the global model update using FedAvg or other strategies.

    Example:
        >>> aggregator = GradientAggregator(strategy=AggregationStrategy.FEDAVG)
        >>> aggregator.receive_gradients("node-1", compressed_grads_1, samples=1000)
        >>> aggregator.receive_gradients("node-2", compressed_grads_2, samples=500)
        >>> global_update = aggregator.aggregate()
    """

    def __init__(
        self,
        strategy: AggregationStrategy = AggregationStrategy.FEDAVG,
        min_participants: int = 1,
        model_size: Optional[int] = None
    ):
        self.strategy = strategy
        self.min_participants = min_participants
        self.model_size = model_size
        self._pending_gradients: Dict[str, tuple[CompressedGradient, int]] = {}
        self._round = 0

    def receive_gradients(
        self,
        node_id: str,
        gradients: CompressedGradient,
        samples: int = 1
    ) -> None:
        """Receive gradients from a node."""
        self._pending_gradients[node_id] = (gradients, samples)

    @property
    def num_participants(self) -> int:
        """Number of nodes that have submitted gradients."""
        return len(self._pending_gradients)

    def ready_to_aggregate(self) -> bool:
        """Check if enough participants for aggregation."""
        return self.num_participants >= self.min_participants

    def aggregate(self) -> List[float]:
        """Aggregate gradients using configured strategy."""
        if not self._pending_gradients:
            raise ValueError("No gradients to aggregate")

        # Determine model size from first gradient
        first = next(iter(self._pending_gradients.values()))[0]
        model_size = self.model_size or first.original_size

        if self.strategy == AggregationStrategy.FEDAVG:
            return self._fedavg(model_size)
        elif self.strategy == AggregationStrategy.WEIGHTED_AVG:
            return self._weighted_avg(model_size)
        else:  # ASYNC_FEDAVG
            return self._async_fedavg(model_size)

    def _fedavg(self, model_size: int) -> List[float]:
        """Standard FedAvg aggregation."""
        total_samples = sum(s for _, s in self._pending_gradients.values())
        aggregated = [0.0] * model_size

        for node_id, (grad, samples) in self._pending_gradients.items():
            weight = samples / total_samples
            # Decompress and weight
            for i, v in zip(grad.indices, grad.values):
                aggregated[i] += v * weight

        self._pending_gradients.clear()
        self._round += 1
        return aggregated

    def _weighted_avg(self, model_size: int) -> List[float]:
        """Weighted average based on sample counts."""
        return self._fedavg(model_size)  # Same as FedAvg

    def _async_fedavg(self, model_size: int) -> List[float]:
        """Asynchronous FedAvg (process gradients as they arrive)."""
        # For async, just average all available
        aggregated = [0.0] * model_size
        n = len(self._pending_gradients)

        for node_id, (grad, _) in self._pending_gradients.items():
            for i, v in zip(grad.indices, grad.values):
                aggregated[i] += v / n

        self._pending_gradients.clear()
        self._round += 1
        return aggregated

    def get_stats(self) -> Dict:
        """Get aggregator statistics."""
        return {
            "strategy": self.strategy.value,
            "round": self._round,
            "pending_participants": self.num_participants,
            "min_participants": self.min_participants,
        }
