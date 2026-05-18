# GraphRTX

**GraphRTX** is a high-performance **RT-accelerated graph analytics framework** built on **CUDA** and **NVIDIA OptiX**. It accelerates traditional graph algorithms by mapping their core traversal and intersection patterns onto **ray tracing hardware (RT cores)**.

- **C++ core engine** with direct CUDA/OptiX integration
- **Python bindings** that load `.mtx` files or existing `networkx` graphs
- **Out-of-core memory manager** for analyzing graphs larger than GPU memory

---

## Overview

GraphRTX loads standard graph datasets (`.mtx` Matrix Market files), builds GPU-optimized adjacency structures (UASP, triangles, and AABBs), and runs a suite of graph algorithms via either:

- a pure **CUDA** pipeline, or
- a **hybrid OptiX + CUDA** pipeline that offloads neighbor traversal and intersection work to RT cores.

---

## Supported Algorithms

| Algorithm | CUDA | Hybrid (OptiX + CUDA) | Notes |
|-----------|:----:|:---------------------:|-------|
| **BFS** — Breadth-First Search | ✓ | ✓ | Parallel frontier expansion |
| **PR** — PageRank | ✓ | ✓ | Configurable damping factor |
| **SSSP** — Single-Source Shortest Path | ✓ | ✓ | Weighted relaxation |
| **BC** — Betweenness Centrality | ✓ | ✓ | Approximate version supported |
| **TC** — Triangle Counting | ✓ | ✓ | Geometric intersection variant |
| **CDLP** — Community Detection (Label Propagation) | ✓ | ✓ | RT-accelerated neighbor scans |
| **WCC** — Weakly Connected Components | ✓ | ✓ | RT-accelerated neighbor scans |
| **LCC** — Local Clustering Coefficient | ✓ | ✓ | Built on TC |

Hybrid variants split work between CUDA and OptiX to accelerate neighbor traversal and intersection.

---

## Python Example (NetworkX)

```python
import networkx as nx
from pygraph_rtx import Graph

G = nx.fast_gnp_random_graph(1000, 0.01)

grtx = Graph()
grtx.from_networkx(G)
grtx.prepare()

print(grtx.run_bfs(0))
```

---

## Build

### Prerequisites

- **CUDA** ≥ 13.0
- **OptiX SDK** ≥ 9.0
- **CMake** ≥ 3.13
- **GPU with RT cores** (e.g. RTX 4070)

Tested on RTX 4070, RTX 5090, and RTX A6000 Pro.

### Compile

Set `OPTIX_ROOT` to your OptiX SDK install path, then:

```bash
git clone https://github.com/anongraph/graphrtx.git
cd graphrtx
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

See [bench/README.md](bench/README.md) for benchmark CLI options.

---

## Evaluation Datasets

**General:** roadUSA, livegraph, kron21, weibo, twitter10, orkut

**LDBC:** datagen, graph500, sk-2005, friendster
