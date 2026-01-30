"""
RotaStellar Distributed - Core Types

Core types for Earth-space distributed compute coordination.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List
import time


class NodeType(Enum):
    """Type of compute node in the Earth-space infrastructure."""
    GROUND = "ground"
    ORBITAL = "orbital"


@dataclass
class NodeConfig:
    """Configuration for a compute node.

    Attributes:
        node_id: Unique identifier for the node
        node_type: Type of node (ground or orbital)
        compute_tflops: Compute capacity in TFLOPS
        memory_gb: Memory capacity in GB
        bandwidth_mbps: Network bandwidth in Mbps
        orbit_altitude_km: Orbital altitude (for orbital nodes)
        location: Ground location tuple (lat, lon) for ground nodes
    """
    node_id: str
    node_type: NodeType
    compute_tflops: float = 10.0
    memory_gb: float = 32.0
    bandwidth_mbps: float = 100.0
    orbit_altitude_km: Optional[float] = None
    location: Optional[tuple[float, float]] = None

    def __post_init__(self):
        if self.node_type == NodeType.ORBITAL and self.orbit_altitude_km is None:
            self.orbit_altitude_km = 550.0

    @classmethod
    def orbital(cls, node_id: str, altitude_km: float = 550.0,
                compute_tflops: float = 10.0) -> "NodeConfig":
        """Create an orbital node configuration."""
        return cls(
            node_id=node_id,
            node_type=NodeType.ORBITAL,
            compute_tflops=compute_tflops,
            orbit_altitude_km=altitude_km
        )

    @classmethod
    def ground(cls, node_id: str, lat: float, lon: float,
               compute_tflops: float = 100.0) -> "NodeConfig":
        """Create a ground node configuration."""
        return cls(
            node_id=node_id,
            node_type=NodeType.GROUND,
            compute_tflops=compute_tflops,
            location=(lat, lon)
        )


@dataclass
class Topology:
    """Topology of Earth-space compute infrastructure.

    Manages the collection of ground and orbital nodes and their connections.
    """
    nodes: Dict[str, NodeConfig] = field(default_factory=dict)
    connections: List[tuple[str, str, float]] = field(default_factory=list)  # (node1, node2, bandwidth)

    def add_node(self, node: NodeConfig) -> None:
        """Add a node to the topology."""
        self.nodes[node.node_id] = node

    def add_connection(self, node1_id: str, node2_id: str, bandwidth_mbps: float) -> None:
        """Add a connection between two nodes."""
        if node1_id not in self.nodes or node2_id not in self.nodes:
            raise ValueError("Both nodes must exist in topology")
        self.connections.append((node1_id, node2_id, bandwidth_mbps))

    def get_ground_nodes(self) -> List[NodeConfig]:
        """Get all ground nodes."""
        return [n for n in self.nodes.values() if n.node_type == NodeType.GROUND]

    def get_orbital_nodes(self) -> List[NodeConfig]:
        """Get all orbital nodes."""
        return [n for n in self.nodes.values() if n.node_type == NodeType.ORBITAL]

    @property
    def total_compute_tflops(self) -> float:
        """Total compute capacity across all nodes."""
        return sum(n.compute_tflops for n in self.nodes.values())

    @property
    def ground_compute_tflops(self) -> float:
        """Total ground compute capacity."""
        return sum(n.compute_tflops for n in self.get_ground_nodes())

    @property
    def orbital_compute_tflops(self) -> float:
        """Total orbital compute capacity."""
        return sum(n.compute_tflops for n in self.get_orbital_nodes())


@dataclass
class TrainingMetrics:
    """Metrics for distributed training across Earth-space infrastructure.

    Tracks training progress, communication costs, and synchronization efficiency.
    """
    total_steps: int = 0
    total_samples: int = 0
    total_epochs: float = 0.0

    # Communication metrics
    bytes_uploaded: int = 0
    bytes_downloaded: int = 0
    sync_count: int = 0

    # Timing metrics
    compute_time_s: float = 0.0
    communication_time_s: float = 0.0
    idle_time_s: float = 0.0

    # Loss tracking
    loss_history: List[float] = field(default_factory=list)

    # Compression metrics
    compression_ratio: float = 1.0
    sparsity_achieved: float = 0.0

    _start_time: Optional[float] = field(default=None, repr=False)

    def start_step(self) -> None:
        """Mark the start of a training step."""
        self._start_time = time.time()

    def end_step(self, loss: Optional[float] = None, samples: int = 0) -> None:
        """Mark the end of a training step."""
        if self._start_time is not None:
            self.compute_time_s += time.time() - self._start_time
        self.total_steps += 1
        self.total_samples += samples
        if loss is not None:
            self.loss_history.append(loss)
        self._start_time = None

    def record_sync(self, bytes_up: int, bytes_down: int, duration_s: float) -> None:
        """Record a synchronization event."""
        self.bytes_uploaded += bytes_up
        self.bytes_downloaded += bytes_down
        self.communication_time_s += duration_s
        self.sync_count += 1

    @property
    def total_bytes_transferred(self) -> int:
        """Total bytes transferred."""
        return self.bytes_uploaded + self.bytes_downloaded

    @property
    def compute_efficiency(self) -> float:
        """Ratio of compute time to total time."""
        total = self.compute_time_s + self.communication_time_s + self.idle_time_s
        if total == 0:
            return 0.0
        return self.compute_time_s / total

    @property
    def communication_overhead(self) -> float:
        """Ratio of communication time to compute time."""
        if self.compute_time_s == 0:
            return float('inf')
        return self.communication_time_s / self.compute_time_s

    @property
    def average_loss(self) -> Optional[float]:
        """Average loss over all steps."""
        if not self.loss_history:
            return None
        return sum(self.loss_history) / len(self.loss_history)

    @property
    def latest_loss(self) -> Optional[float]:
        """Most recent loss value."""
        return self.loss_history[-1] if self.loss_history else None

    def summary(self) -> Dict:
        """Get a summary of training metrics."""
        return {
            "total_steps": self.total_steps,
            "total_samples": self.total_samples,
            "compute_time_s": round(self.compute_time_s, 2),
            "communication_time_s": round(self.communication_time_s, 2),
            "compute_efficiency": round(self.compute_efficiency, 4),
            "total_bytes_transferred": self.total_bytes_transferred,
            "sync_count": self.sync_count,
            "compression_ratio": round(self.compression_ratio, 4),
            "latest_loss": self.latest_loss,
        }
