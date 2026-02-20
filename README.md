# ⚡ K8s Pod Sizing Simulator

> **[▶ Live Demo — request-queueing-simulator.streamlit.app](https://request-queueing-simulator.streamlit.app/)**

Should you run 24 single-core pods or 6 four-core pods? This simulator answers that question using **queuing theory** and **discrete event simulation**, showing exactly how multi-core pods fix tail latency caused by head-of-line blocking.

## The Queuing Theory Problem

Every Kubernetes pod is a **queue**. Requests arrive, wait if the CPU is busy, get processed, and leave. When your workload is a mix of fast requests (20ms) and slow ones (500ms), something nasty happens:

**Single-core pod = M/G/1 queue.** One server, one queue. A 500ms request blocks everything behind it. The 20ms requests that land on that pod wait 500ms+ even though other pods are idle. This is **head-of-line (HOL) blocking** — and it's why your P99 is 10× higher than it should be.

**Multi-core pod = M/G/c queue.** Same queue, but `c` servers (cores) pull from it in parallel. A 500ms request occupies 1 of 4 cores — the other 3 keep serving. The Linux kernel scheduler acts as an intelligent, CPU-aware load balancer *inside* the pod.

The key insight from queuing theory: **a single M/G/c queue always outperforms c separate M/G/1 queues** at the same total utilization. This is the *pooling effect* — and it's why multi-core pods help even though total compute stays constant.

### Why load balancers can't fix this

| Layer | Algorithm | What it sees | What it misses |
|-------|-----------|-------------|----------------|
| Envoy (L4) | Round-robin | Nothing — just cycles through pods | Everything |
| Istio (L7) | Least-connection | Active connection count | Whether those connections are 20ms or 500ms |
| Linux scheduler | CFS / work-stealing | Actual CPU run queue depth | — (it sees everything) |

The network-level load balancers don't know a request's CPU cost until it's done. The Linux scheduler *does* — but it can only help if there are multiple cores in the pod to distribute across.

### The math

For our bimodal workload (95% at 20ms, 5% at 500ms):

- **Average service time:** 44ms
- **Coefficient of variation squared (CV²):** 5.65 — extremely high variance
- **Pollaczek-Khinchine formula** predicts mean wait of 341ms for M/G/1 at 70% utilization
- A deterministic workload (same mean, CV²=0) would wait only 51ms — **the variance alone costs 290ms**

M/G/c has no closed-form solution for the general case — which is why we simulate.

## What We Simulate

We use **discrete event simulation** to model the full request lifecycle across different pod configurations, keeping total compute constant:

```
Scenario 1: 24 pods × 1 core    →  24 independent M/G/1 queues
Scenario 2:  6 pods × 4 cores   →   6 independent M/G/4 queues
Scenario 3:  3 pods × 8 cores   →   3 independent M/G/8 queues
Scenario 4:  1 pod  × 24 cores  →   1 M/G/24 queue (theoretical optimum)
```

Each pod is a Ciw network node. Requests arrive as a Poisson process (split equally across pods, approximating round-robin). Service times follow the bimodal distribution. Within each node, Ciw manages the multi-server shared queue — which is exactly how the Linux scheduler distributes work across cores.

We run 100K+ requests, discard warmup, and measure the full response time distribution — P50, P90, P99, P99.9, max, and standard deviation.

## Key Results

24 cores, 70% utilization, 5% heavy requests at 500ms:

| Config | P99 | P99.9 | Mean Wait | Improvement |
|--------|-----|-------|-----------|-------------|
| 1-core pods (×24) | 1,889ms | 2,681ms | 322ms | baseline |
| 4-core pods (×6) | 627ms | 896ms | 47ms | **P99 −67%** |
| 8-core pods (×3) | 511ms | 660ms | 14ms | **P99 −73%** |
| 24-core pod (×1) | 500ms | 508ms | 0.6ms | P99 −74% |

The biggest jump is 1-core → 4-core. Beyond 8 cores, returns diminish. The theoretical optimum (single shared queue) is only marginally better than 8-core.

## Why Ciw

We chose **[Ciw](https://ciw.readthedocs.io/)** (a Python discrete event simulation library for open queuing networks) because it maps directly to our problem:

| What we need | How Ciw provides it |
|-------------|-------------------|
| M/G/c queues (multi-core pods) | Native — set `number_of_servers=c` per node |
| Bimodal service times | Subclass `ciw.dists.Distribution`, return 20ms or 500ms |
| Independent pods | Multiple nodes, no inter-node routing — requests exit after service |
| Per-request records | `DataRecord` with arrival, wait, service start, service time, server ID |
| Tail latency analysis | Compute P99/P99.9 from the full record set |
| Gantt chart reconstruction | Server ID + timestamps → which core processed what, when |

**Alternatives considered:**

| Tool | Strengths | Why we chose Ciw instead |
|------|-----------|--------------------------|
| **[queueing-tool](https://github.com/djordon/queueing-tool)** | Python, NetworkX-based graph topology, good for routing-heavy models | Limited to exponential/gamma service distributions — no easy way to plug in our bimodal (20ms/500ms) distribution. Record-keeping is aggregate (queue-level stats), not per-request — we need individual request wait/service times for P99.9 and Gantt charts. Ciw's `Distribution` subclassing and `DataRecord` are purpose-built for this. |
| **SimPy** | Mature, flexible, large community | General-purpose DES — no queuing primitives. We'd hand-build M/G/c queue logic, FIFO discipline, server assignment, and per-request tracking. Ciw wraps all of this in `create_network()`. |
| **MATLAB / Simulink** | Powerful, good visualization | Not Python, can't deploy to Streamlit Cloud, license cost |
| **Custom event loop** | Full control | More code, more bugs, same result. Ciw is peer-reviewed ([Journal of Simulation, 2019](https://doi.org/10.1080/17477778.2018.1473909)) and well-tested |
| **Analytical only** | Fast, no simulation needed | M/G/c has no closed-form for tail percentiles. The Pollaczek-Khinchine formula works for M/G/1 mean wait, but not for P99. Approximations exist (Allen-Cunneen) but are inaccurate for high CV² like ours (5.65). Simulation is the practical path. |

## Quick Start

```bash
# Setup
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Interactive dashboard
streamlit run interactive_app.py
```

Open [http://localhost:8501](http://localhost:8501) — tune parameters in the sidebar, explore the tabs.

```bash
# Batch mode — generate static charts
python simulator.py
# → outputs PNG charts + GIF animation to output/
```

## What's Inside

| File | Purpose |
|------|---------|
| `interactive_app.py` | Streamlit dashboard — the main experience |
| `pages/` | Doc pages shown in sidebar (problem, theory, guide) |
| `simulator.py` | Batch simulator — generates static charts to `output/` |
| `docs/*.qmd` | Quarto source for the documentation pages |
| `queuing_theory_analysis.md` | Full mathematical analysis with derivations |
| `requirements.txt` | Python dependencies |

## Deploy

Push to GitHub → [Streamlit Community Cloud](https://share.streamlit.io) → set main file to `interactive_app.py` → Deploy. One host, everything included.

## Tech Stack

- **[Ciw](https://ciw.readthedocs.io/)** — discrete event simulation (queuing networks)
- **[Plotly](https://plotly.com/python/)** — interactive charts with hover, zoom, animation
- **[Streamlit](https://streamlit.io/)** — dashboard framework with multi-page support
