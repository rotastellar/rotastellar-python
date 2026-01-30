"""
RotaStellar Distributed - Earth-Space AI Coordination

Coordinate AI workloads across Earth and orbital infrastructure with
federated learning, model partitioning, and bandwidth-optimized sync.

Documentation: https://rotastellar.com/docs/distributed
GitHub: https://github.com/rotastellar/rotastellar-python

Example:
    >>> from rotastellar_distributed import (
    ...     FederatedClient, CompressionConfig, GradientAggregator,
    ...     ModelProfile, PartitionOptimizer,
    ...     SyncScheduler, GroundStation, Priority,
    ...     SpaceMesh, OrbitalNode, create_constellation,
    ... )
    >>>
    >>> # Federated learning with gradient compression
    >>> client = FederatedClient("orbital-1", CompressionConfig.balanced())
    >>> gradients = client.compute_gradients(model_params, local_data)
    >>> compressed = client.compress(gradients)
    >>>
    >>> # Model partitioning
    >>> model = ModelProfile.create_transformer(num_layers=12)
    >>> optimizer = PartitionOptimizer()
    >>> plan = optimizer.optimize(model)
    >>>
    >>> # Sync scheduling
    >>> scheduler = SyncScheduler()
    >>> scheduler.schedule_sync("node-1", 1024*1024, Priority.HIGH)
    >>>
    >>> # Space mesh routing
    >>> mesh = create_constellation("test", num_planes=4, sats_per_plane=10)
    >>> route = mesh.find_route("test_P0_S0", "test_P2_S5")
"""

__version__ = "0.1.0"

# Core types
from .core import (
    NodeType,
    NodeConfig,
    Topology,
    TrainingMetrics,
)

# Federated learning
from .federated import (
    CompressionMethod,
    CompressionConfig,
    CompressedGradient,
    GradientCompressor,
    FederatedClient,
    AggregationStrategy,
    GradientAggregator,
)

# Model partitioning
from .partitioning import (
    LayerType,
    LayerProfile,
    ModelProfile,
    PlacementLocation,
    LayerPlacement,
    PartitionPlan,
    OptimizationObjective,
    LatencyEstimator,
    PartitionOptimizer,
)

# Sync scheduling
from .sync import (
    Priority,
    GroundStation,
    ContactWindow,
    SyncTask,
    PriorityQueue,
    SyncScheduler,
)

# Space mesh
from .mesh import (
    LinkType,
    OrbitalNode,
    ISLLink,
    Route,
    SpaceMesh,
    create_constellation,
)

__all__ = [
    # Version
    "__version__",
    # Core
    "NodeType",
    "NodeConfig",
    "Topology",
    "TrainingMetrics",
    # Federated
    "CompressionMethod",
    "CompressionConfig",
    "CompressedGradient",
    "GradientCompressor",
    "FederatedClient",
    "AggregationStrategy",
    "GradientAggregator",
    # Partitioning
    "LayerType",
    "LayerProfile",
    "ModelProfile",
    "PlacementLocation",
    "LayerPlacement",
    "PartitionPlan",
    "OptimizationObjective",
    "LatencyEstimator",
    "PartitionOptimizer",
    # Sync
    "Priority",
    "GroundStation",
    "ContactWindow",
    "SyncTask",
    "PriorityQueue",
    "SyncScheduler",
    # Mesh
    "LinkType",
    "OrbitalNode",
    "ISLLink",
    "Route",
    "SpaceMesh",
    "create_constellation",
]
