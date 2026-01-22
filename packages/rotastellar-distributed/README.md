# rotastellar-distributed

Distributed compute coordination for Earth-space AI workloads.

**Status:** Coming Q1 2026

## Overview

`rotastellar-distributed` enables AI training and inference across hybrid Earth-space infrastructure. Coordinate federated learning, partition models optimally, and synchronize through bandwidth-constrained orbital links.

## Installation

```bash
pip install rotastellar-distributed
```

With framework support:
```bash
pip install rotastellar-distributed[pytorch]
pip install rotastellar-distributed[tensorflow]
```

## Features

### Federated Learning
Train models across Earth and orbital nodes with gradient compression.

```python
from rotastellar_distributed import FederatedClient, CompressionConfig

compression = CompressionConfig(
    method="topk_quantized",
    k_ratio=0.01,           # Keep top 1% of gradients
    quantization_bits=8,    # 8-bit quantization
    error_feedback=True
)

client = FederatedClient(
    api_key="...",
    node_id="orbital-3",
    compression=compression
)

gradients = client.train_step(model, batch)
client.sync(gradients, priority="high")
```

### Model Partitioning
Find optimal layer placement across Earth and orbital infrastructure.

```python
from rotastellar_distributed import PartitionOptimizer, ModelProfile

profile = ModelProfile.from_pytorch(model)

optimizer = PartitionOptimizer(api_key="...")
partition = optimizer.optimize(
    model=profile,
    topology=topology,
    objective="minimize_latency"
)

print(partition.ground_layers)
print(partition.orbital_layers)
```

### Sync Scheduler
Schedule data synchronization across ground station passes.

```python
from rotastellar_distributed import SyncScheduler, GroundStation

scheduler = SyncScheduler(
    api_key="...",
    ground_stations=[
        GroundStation("svalbard", lat=78.2, lon=15.6),
        GroundStation("singapore", lat=1.3, lon=103.8),
    ]
)

windows = scheduler.get_windows(hours=24)
scheduler.schedule_sync(node="orbital-1", data_size=50e6, priority="critical")
```

### Space Mesh
ISL routing for orbital node communication.

```python
from rotastellar_distributed import SpaceMesh

mesh = SpaceMesh(api_key="...")
route = mesh.find_route(source="orbital-1", destination="ground-svalbard")
```

## Documentation

Full documentation: https://docs.rotastellar.com/sdks/python/distributed

## Links

- Website: https://rotastellar.com/products/distributed-compute
- Interactive Demo: https://rotastellar.com/products/distributed-compute/demo
- Research: https://rotastellar.com/research

## License

MIT License - see LICENSE for details.
