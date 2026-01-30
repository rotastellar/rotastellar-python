# rotastellar-distributed

**Distributed Computing for Space Infrastructure**

Federated learning, model partitioning, gradient synchronization, and mesh networking for orbital compute clusters.

## Installation

```bash
pip install rotastellar-distributed
```

With framework support:
```bash
pip install rotastellar-distributed[pytorch]
pip install rotastellar-distributed[tensorflow]
```

## Quick Start

### Federated Learning

```python
from rotastellar_distributed import (
    FederatedClient,
    GradientAggregator,
    AggregationStrategy,
    CompressionConfig,
    CompressionType
)

# Configure gradient compression for limited bandwidth
compression = CompressionConfig(
    compression_type=CompressionType.TOP_K,
    sparsity=0.99,  # Keep top 1% of gradients
    error_feedback=True
)

# Create federated client
client = FederatedClient(
    node_id="sat-001",
    compression=compression
)

# Compress gradients before transmission
gradients = model.get_gradients()
compressed = client.compress(gradients)
print(f"Compression ratio: {compressed.compression_ratio:.1f}x")

# Server-side aggregation
aggregator = GradientAggregator(strategy=AggregationStrategy.FEDAVG)
aggregated = aggregator.aggregate([grad1, grad2, grad3], weights=[0.4, 0.3, 0.3])
```

### Model Partitioning

```python
from rotastellar_distributed import (
    ModelProfile,
    PartitionOptimizer,
    NodeConfig,
    NodeType
)

# Profile your model
profile = ModelProfile(
    layers=[
        {"name": "embedding", "params_mb": 100, "flops": 1e9},
        {"name": "transformer_1", "params_mb": 200, "flops": 5e9},
        {"name": "transformer_2", "params_mb": 200, "flops": 5e9},
        {"name": "output", "params_mb": 50, "flops": 1e8},
    ]
)

# Define available nodes
nodes = [
    NodeConfig(node_id="sat-001", node_type=NodeType.SATELLITE, memory_gb=8, compute_tflops=2.0),
    NodeConfig(node_id="sat-002", node_type=NodeType.SATELLITE, memory_gb=8, compute_tflops=2.0),
    NodeConfig(node_id="ground-001", node_type=NodeType.GROUND, memory_gb=32, compute_tflops=10.0),
]

# Optimize partitioning
optimizer = PartitionOptimizer()
plan = optimizer.optimize(profile, nodes)
print(f"Partition plan: {plan.assignments}")
print(f"Estimated latency: {plan.estimated_latency_ms:.1f} ms")
```

### Synchronization Scheduling

```python
from rotastellar_distributed import SyncScheduler, GroundStation
from rotastellar import Position

# Define ground stations
stations = [
    GroundStation(
        name="KSC",
        position=Position(28.5729, -80.6490, 0.0),
        uplink_mbps=100.0,
        downlink_mbps=200.0
    ),
    GroundStation(
        name="Svalbard",
        position=Position(78.2297, 15.3975, 0.0),
        uplink_mbps=150.0,
        downlink_mbps=300.0
    ),
]

# Create scheduler
scheduler = SyncScheduler(ground_stations=stations)

# Get optimal sync windows
windows = scheduler.get_sync_windows(
    satellite_id="sat-001",
    duration_hours=24
)
for window in windows:
    print(f"Station: {window.station.name}")
    print(f"Start: {window.start_time}, Duration: {window.duration_seconds}s")
    print(f"Data capacity: {window.data_capacity_mb:.1f} MB")
```

### Space Mesh Networking

```python
from rotastellar_distributed import SpaceMesh, MeshNode
from rotastellar import Position

# Create mesh network
mesh = SpaceMesh()

# Add nodes
mesh.add_node(MeshNode(node_id="sat-001", position=Position(45.0, -122.0, 550.0)))
mesh.add_node(MeshNode(node_id="sat-002", position=Position(46.0, -120.0, 550.0)))
mesh.add_node(MeshNode(node_id="sat-003", position=Position(44.0, -118.0, 550.0)))

# Add inter-satellite links
mesh.add_link("sat-001", "sat-002", bandwidth_mbps=1000.0, latency_ms=2.0)
mesh.add_link("sat-002", "sat-003", bandwidth_mbps=1000.0, latency_ms=2.5)

# Find optimal route
route = mesh.find_route("sat-001", "sat-003")
print(f"Route: {' -> '.join(route.hops)}")
print(f"Total latency: {route.total_latency_ms:.1f} ms")
```

## Features

- **Federated Learning** — Privacy-preserving distributed training across orbital nodes
- **Gradient Compression** — TopK, random sparsification, quantization for bandwidth-limited links
- **Model Partitioning** — Intelligent layer placement across heterogeneous nodes
- **Sync Scheduling** — Optimal ground station contact windows for data synchronization
- **Mesh Networking** — Dynamic routing for inter-satellite communication

## Links

- **Website:** https://rotastellar.com/products/distributed
- **Documentation:** https://docs.rotastellar.com/sdks/python/distributed
- **Main SDK:** https://pypi.org/project/rotastellar/

## Author

Created by [Subhadip Mitra](mailto:subhadipmitra@rotastellar.com) at [RotaStellar](https://rotastellar.com).

## License

MIT License — Copyright (c) 2026 RotaStellar
