                #!/usr/bin/env python3
"""
Kubernetes Pod Load Balancing Simulator
=======================================

Simulates head-of-line (HOL) blocking in single-core vs multi-core pod
configurations using discrete event simulation with the Ciw library.

Models the queuing theory dynamics of:
  - 24 × 1-core pods  (M/G/1 queues)  — Current state
  - 6 × 4-core pods   (M/G/4 queues)  — Option A
  - 3 × 8-core pods   (M/G/8 queues)  — Option B
  - 1 × 24-core pod   (M/G/24 queue)  — Theoretical optimum (shared queue)

Request workload is bimodal:
  - 95% of requests: ~20ms CPU (light)
  - 5% of requests:  ~500ms CPU (heavy)

Usage:
    pip install -r requirements.txt
    python simulator.py
"""

import os
import random
import time
from collections import OrderedDict

import ciw
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving files
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter

# =============================================================================
# Configuration
# =============================================================================

TOTAL_CORES = 24                  # Total cores across all pods
LIGHT_SERVICE_TIME = 0.020        # 20ms in seconds
HEAVY_SERVICE_TIME = 0.500        # 500ms in seconds
HEAVY_REQUEST_PROB = 0.05         # 5% of requests are heavy
TARGET_UTILIZATION = 0.70         # 70% CPU utilization target
SIM_TIME = 300                    # Simulation duration (seconds)
WARMUP_FRACTION = 0.10            # Discard first 10% as warmup
RANDOM_SEED = 42

# Derived parameters
AVG_SERVICE_TIME = (
    (1 - HEAVY_REQUEST_PROB) * LIGHT_SERVICE_TIME
    + HEAVY_REQUEST_PROB * HEAVY_SERVICE_TIME
)
ARRIVAL_RATE_PER_CORE = TARGET_UTILIZATION / AVG_SERVICE_TIME
TOTAL_ARRIVAL_RATE = TOTAL_CORES * ARRIVAL_RATE_PER_CORE

# Output directory
OUTPUT_DIR = "output"

# Scenario definitions (ordered for display)
SCENARIOS = OrderedDict([
    ("1-core pods (×24)", {"cores_per_pod": 1,  "color": "#e74c3c", "marker": "o"}),
    ("4-core pods (×6)",  {"cores_per_pod": 4,  "color": "#f39c12", "marker": "s"}),
    ("8-core pods (×3)",  {"cores_per_pod": 8,  "color": "#27ae60", "marker": "D"}),
    ("24-core pod (×1)",  {"cores_per_pod": 24, "color": "#2980b9", "marker": "^"}),
])

# Plot style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})


# =============================================================================
# Custom Bimodal Service Time Distribution
# =============================================================================

class BimodalServiceDist(ciw.dists.Distribution):
    """
    Bimodal service time distribution for modeling heterogeneous request costs.

    With probability `heavy_prob`, returns `heavy_time`.
    Otherwise, returns `light_time`.
    """

    def __init__(self, light_time, heavy_time, heavy_prob):
        super().__init__()
        self.light_time = light_time
        self.heavy_time = heavy_time
        self.heavy_prob = heavy_prob

    def sample(self, t=None, ind=None):
        if random.random() < self.heavy_prob:
            return self.heavy_time
        return self.light_time


# =============================================================================
# Simulation Engine
# =============================================================================

def run_simulation(cores_per_pod, seed=RANDOM_SEED):
    """
    Run a discrete event simulation for a given pod configuration.

    Each pod is modeled as a separate Ciw node with `cores_per_pod` servers.
    Arrivals are Poisson with rate split equally across pods.
    After service, requests exit (no inter-node routing).

    Args:
        cores_per_pod: Number of CPU cores per pod (servers per node).
        seed: Random seed for reproducibility.

    Returns:
        List of Ciw DataRecord objects for completed requests.
    """
    num_pods = TOTAL_CORES // cores_per_pod
    arrival_rate_per_pod = TOTAL_ARRIVAL_RATE / num_pods

    # Seed both ciw and Python random for reproducibility
    ciw.seed(seed)
    random.seed(seed)

    # Build the queuing network
    network = ciw.create_network(
        arrival_distributions=[
            ciw.dists.Exponential(rate=arrival_rate_per_pod)
            for _ in range(num_pods)
        ],
        service_distributions=[
            BimodalServiceDist(
                LIGHT_SERVICE_TIME, HEAVY_SERVICE_TIME, HEAVY_REQUEST_PROB
            )
            for _ in range(num_pods)
        ],
        number_of_servers=[cores_per_pod] * num_pods,
        routing=[[0.0] * num_pods for _ in range(num_pods)],
    )

    # Run simulation
    simulation = ciw.Simulation(network)
    simulation.simulate_until_max_time(SIM_TIME)

    # Collect records, filtering out warmup period
    all_records = simulation.get_all_records()
    warmup_cutoff = SIM_TIME * WARMUP_FRACTION
    records = [r for r in all_records if r.arrival_date > warmup_cutoff]

    return records


def extract_metrics(records):
    """
    Extract response time metrics from simulation records.

    Returns:
        Tuple of (response_times, waiting_times, service_times, stats_dict)
        All times in the arrays are in seconds; stats_dict values are in ms.
    """
    if not records:
        empty = np.array([])
        return empty, empty, empty, {}

    response_times = np.array([r.waiting_time + r.service_time for r in records])
    waiting_times = np.array([r.waiting_time for r in records])
    service_times = np.array([r.service_time for r in records])

    to_ms = 1000.0  # Convert seconds to milliseconds
    stats = {
        "count":        len(response_times),
        "mean":         np.mean(response_times) * to_ms,
        "median":       np.median(response_times) * to_ms,
        "p90":          np.percentile(response_times, 90) * to_ms,
        "p95":          np.percentile(response_times, 95) * to_ms,
        "p99":          np.percentile(response_times, 99) * to_ms,
        "p999":         np.percentile(response_times, 99.9) * to_ms,
        "max":          np.max(response_times) * to_ms,
        "std":          np.std(response_times) * to_ms,
        "mean_wait":    np.mean(waiting_times) * to_ms,
        "mean_service": np.mean(service_times) * to_ms,
    }

    return response_times, waiting_times, service_times, stats


def run_all_scenarios():
    """Run all configured scenarios and return results dictionary."""
    results = OrderedDict()

    for name, config in SCENARIOS.items():
        cores = config["cores_per_pod"]
        num_pods = TOTAL_CORES // cores

        print(f"\n{'=' * 60}")
        print(f"  Scenario: {name}")
        print(f"  {num_pods} pods × {cores} core(s) = {TOTAL_CORES} total cores")
        print(f"  Arrival rate per pod: {TOTAL_ARRIVAL_RATE / num_pods:.1f} req/s")
        print(f"{'=' * 60}")

        t0 = time.time()
        records = run_simulation(cores)
        elapsed = time.time() - t0

        response_times, waiting_times, service_times, stats = extract_metrics(records)

        results[name] = {
            "records":        records,
            "response_times": response_times,
            "waiting_times":  waiting_times,
            "service_times":  service_times,
            "stats":          stats,
            "config":         config,
        }

        print(f"  Completed in {elapsed:.1f}s | {stats['count']:,} requests processed")
        print(f"  Mean: {stats['mean']:.1f}ms | Median: {stats['median']:.1f}ms")
        print(f"  P90: {stats['p90']:.1f}ms | P95: {stats['p95']:.1f}ms")
        print(f"  P99: {stats['p99']:.1f}ms | P99.9: {stats['p999']:.1f}ms")
        print(f"  Max: {stats['max']:.1f}ms | Std Dev: {stats['std']:.1f}ms")

    return results


# =============================================================================
# Visualization Functions
# =============================================================================

def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_cdf_comparison(results):
    """
    Plot Cumulative Distribution Function of response times for all scenarios.
    This is the most important plot — it shows tail latency behavior clearly.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Full CDF
    for name, data in results.items():
        sorted_times = np.sort(data["response_times"]) * 1000
        cdf = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
        ax1.plot(sorted_times, cdf, label=name,
                 color=data["config"]["color"], linewidth=2.5)

    for p, label in [(0.50, "P50"), (0.90, "P90"), (0.99, "P99"), (0.999, "P99.9")]:
        ax1.axhline(y=p, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
        ax1.text(ax1.get_xlim()[0] + 5, p + 0.003, label,
                 fontsize=9, color="gray", fontweight="bold")

    ax1.set_xlabel("Response Time (ms)")
    ax1.set_ylabel("Cumulative Probability")
    ax1.set_title("Full CDF of Response Times")
    ax1.legend(fontsize=10)
    ax1.set_xlim(left=0)

    # Tail CDF (zoom into P95+)
    for name, data in results.items():
        sorted_times = np.sort(data["response_times"]) * 1000
        cdf = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
        mask = cdf >= 0.90
        ax2.plot(sorted_times[mask], cdf[mask], label=name,
                 color=data["config"]["color"], linewidth=2.5)

    for p, label in [(0.95, "P95"), (0.99, "P99"), (0.999, "P99.9")]:
        ax2.axhline(y=p, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
        ax2.text(ax2.get_xlim()[0] + 5, p + 0.0005, label,
                 fontsize=9, color="gray", fontweight="bold")

    ax2.set_xlabel("Response Time (ms)")
    ax2.set_ylabel("Cumulative Probability")
    ax2.set_title("Tail Latency CDF (P90+) — Zoom")
    ax2.legend(fontsize=10)
    ax2.set_ylim(0.90, 1.001)

    fig.suptitle(
        "Head-of-Line Blocking: Response Time CDF Comparison\n"
        f"({TOTAL_CORES} cores, {HEAVY_REQUEST_PROB*100:.0f}% heavy requests @ "
        f"{HEAVY_SERVICE_TIME*1000:.0f}ms, {TARGET_UTILIZATION*100:.0f}% utilization)",
        fontsize=14, fontweight="bold", y=1.02
    )

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "cdf_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


def plot_percentile_bars(results):
    """Plot tail latency percentiles as a grouped bar chart."""
    percentiles = ["p90", "p95", "p99", "p999"]
    labels = ["P90", "P95", "P99", "P99.9"]

    x = np.arange(len(percentiles))
    n = len(results)
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(14, 8))

    for i, (name, data) in enumerate(results.items()):
        values = [data["stats"][p] for p in percentiles]
        offset = (i - n / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=name,
                      color=data["config"]["color"], alpha=0.85,
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 8,
                    f"{val:.0f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xlabel("Latency Percentile")
    ax.set_ylabel("Response Time (ms)")
    ax.set_title(
        "Tail Latency Comparison: Impact of Multi-Core Pods\n"
        "(Lower is better)",
        fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "percentile_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


def plot_response_time_histogram(results):
    """Plot histograms of response times for each scenario side-by-side."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (name, data) in zip(axes, results.items()):
        times_ms = data["response_times"] * 1000
        # Clip to 99.5th percentile for better visualization
        clip_val = np.percentile(times_ms, 99.5)
        clipped = times_ms[times_ms <= clip_val]

        ax.hist(clipped, bins=80, color=data["config"]["color"], alpha=0.75,
                edgecolor="white", linewidth=0.3)

        p99_val = data["stats"]["p99"]
        mean_val = data["stats"]["mean"]
        ax.axvline(p99_val, color="black", linestyle="--", linewidth=1.5,
                   label=f"P99 = {p99_val:.0f}ms")
        ax.axvline(mean_val, color="navy", linestyle="-", linewidth=1.5,
                   label=f"Mean = {mean_val:.0f}ms")

        ax.set_xlabel("Response Time (ms)")
        ax.set_title(name, fontweight="bold")
        ax.legend(fontsize=9)

    axes[0].set_ylabel("Request Count")
    fig.suptitle("Response Time Distribution by Pod Configuration",
                 fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "histogram_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


def plot_box_comparison(results):
    """Box plot comparison of response time distributions."""
    fig, ax = plt.subplots(figsize=(12, 7))

    data_list = []
    labels = []
    colors = []
    for name, data in results.items():
        data_list.append(data["response_times"] * 1000)
        labels.append(name)
        colors.append(data["config"]["color"])

    bp = ax.boxplot(
        data_list, tick_labels=labels, patch_artist=True,
        showfliers=True,
        flierprops=dict(marker="o", markersize=1.5, alpha=0.2, color="gray"),
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Response Time (ms)")
    ax.set_title("Response Time Variability: Box Plot Comparison\n"
                 "(Whiskers = 1.5×IQR, dots = outliers)",
                 fontweight="bold")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "boxplot_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


def _find_interesting_window(records, cores_per_pod, window_size=2.0,
                             search_start=40.0, min_heavy=2):
    """Find a time window that shows HOL blocking (has heavy requests)."""
    for t in np.arange(search_start, search_start + 60, 0.5):
        window_recs = [
            r for r in records
            if r.service_start_date < t + window_size
            and (r.service_start_date + r.service_time) > t
        ]
        heavy_count = sum(
            1 for r in window_recs
            if r.service_time > (LIGHT_SERVICE_TIME + HEAVY_SERVICE_TIME) / 2
        )
        if heavy_count >= min_heavy:
            return (t, t + window_size)
    return (search_start, search_start + window_size)


def plot_gantt_chart(results):
    """
    Gantt chart showing request processing on individual cores.
    Clearly visualizes HOL blocking: red blocks (heavy requests)
    occupying a core while other requests wait.
    """
    n_scenarios = len(results)
    fig, axes = plt.subplots(n_scenarios, 1,
                             figsize=(20, 3.5 * n_scenarios),
                             sharex=False)
    if n_scenarios == 1:
        axes = [axes]

    for ax, (name, data) in zip(axes, results.items()):
        records = data["records"]
        cores_per_pod = data["config"]["cores_per_pod"]
        num_pods = TOTAL_CORES // cores_per_pod

        # Show up to 8 total cores in the Gantt chart for comparability
        max_cores_shown = min(8, TOTAL_CORES)
        max_pods_shown = min(max_cores_shown // cores_per_pod, num_pods)
        actual_cores_shown = max_pods_shown * cores_per_pod

        # Find a window that contains heavy requests for visual interest
        visible_recs = [r for r in records if r.node <= max_pods_shown]
        t_start, t_end = _find_interesting_window(
            visible_recs, cores_per_pod, window_size=2.0
        )

        for rec in visible_recs:
            svc_end = rec.service_start_date + rec.service_time
            # Filter to time window
            if svc_end < t_start or rec.service_start_date > t_end:
                continue

            core_idx = (rec.node - 1) * cores_per_pod + (rec.server_id - 1)
            if core_idx >= actual_cores_shown:
                continue

            bar_start = max(rec.service_start_date, t_start)
            bar_end = min(svc_end, t_end)

            is_heavy = rec.service_time > (LIGHT_SERVICE_TIME + HEAVY_SERVICE_TIME) / 2
            color = "#e74c3c" if is_heavy else "#3498db"
            alpha = 0.95 if is_heavy else 0.65

            ax.barh(core_idx, bar_end - bar_start, left=bar_start,
                    height=0.8, color=color, alpha=alpha,
                    edgecolor="white", linewidth=0.3)

        # Pod separator lines
        for i in range(1, max_pods_shown):
            ax.axhline(y=i * cores_per_pod - 0.5, color="black",
                       linewidth=2, linestyle="-")

        # Y-axis labels
        ytick_labels = []
        for pod in range(max_pods_shown):
            for core in range(cores_per_pod):
                ytick_labels.append(f"P{pod + 1}:C{core + 1}")
        ax.set_yticks(range(actual_cores_shown))
        ax.set_yticklabels(ytick_labels, fontsize=8, fontfamily="monospace")
        ax.set_ylabel(name, fontsize=11, fontweight="bold")
        ax.set_xlim(t_start, t_end)
        ax.set_xlabel("Time (seconds)")
        ax.invert_yaxis()

    # Legend
    light_patch = mpatches.Patch(color="#3498db", alpha=0.65,
                                 label="Light request (~20ms)")
    heavy_patch = mpatches.Patch(color="#e74c3c", alpha=0.95,
                                 label="Heavy request (~500ms)")
    axes[0].legend(handles=[light_patch, heavy_patch],
                   loc="upper right", fontsize=9)

    fig.suptitle(
        "Request Processing Timeline — Gantt Chart\n"
        "Red blocks show heavy requests causing HOL blocking in single-core pods",
        fontsize=14, fontweight="bold", y=1.02
    )

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "gantt_chart.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


def plot_queue_depth_timeseries(results, time_window=(40, 55)):
    """
    Plot queue depth (requests in system) over time for Pod #1 in each scenario.
    Shows how HOL blocking creates spikes in single-core pods.
    """
    t_start, t_end = time_window
    dt = 0.02  # 20ms resolution
    time_points = np.arange(t_start, t_end, dt)

    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(18, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (name, data) in zip(axes, results.items()):
        records = data["records"]
        cores_per_pod = data["config"]["cores_per_pod"]

        # Filter to Pod #1
        pod1_recs = [
            r for r in records
            if r.node == 1
            and (r.service_start_date + r.service_time) >= t_start
            and r.arrival_date <= t_end
        ]

        queue_depths = []    # Waiting in queue
        in_service = []      # Being served
        for t in time_points:
            waiting = sum(
                1 for r in pod1_recs
                if r.arrival_date <= t < r.service_start_date
            )
            serving = sum(
                1 for r in pod1_recs
                if r.service_start_date <= t < (r.service_start_date + r.service_time)
            )
            queue_depths.append(waiting)
            in_service.append(serving)

        queue_arr = np.array(queue_depths)
        service_arr = np.array(in_service)
        total_arr = queue_arr + service_arr

        ax.fill_between(time_points, 0, service_arr, alpha=0.4,
                        color=data["config"]["color"], label="In service")
        ax.fill_between(time_points, service_arr, total_arr, alpha=0.25,
                        color="gray", label="Waiting in queue")
        ax.plot(time_points, total_arr, color=data["config"]["color"],
                linewidth=1, alpha=0.8)

        ax.axhline(y=cores_per_pod, color="black", linestyle="--",
                   linewidth=1.5, alpha=0.6,
                   label=f"Capacity ({cores_per_pod} core{'s' if cores_per_pod > 1 else ''})")

        ax.set_ylabel(f"{name}\nRequests", fontsize=10)
        ax.legend(fontsize=8, loc="upper right", ncol=3)
        ax.set_ylim(bottom=0)

    axes[-1].set_xlabel("Time (seconds)", fontsize=12)
    fig.suptitle("Queue Depth Over Time — Pod #1\n"
                 "(Spikes above capacity line = requests waiting = HOL blocking)",
                 fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "queue_timeseries.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


def plot_improvement_summary(results):
    """
    Summary dashboard showing % improvement in P99 and P99.9 vs single-core baseline.
    """
    if "1-core pods (×24)" not in results:
        return

    baseline = results["1-core pods (×24)"]["stats"]
    names = []
    p99_improvements = []
    p999_improvements = []
    colors = []

    for name, data in results.items():
        names.append(name)
        p99_improvements.append(
            (1 - data["stats"]["p99"] / baseline["p99"]) * 100
        )
        p999_improvements.append(
            (1 - data["stats"]["p999"] / baseline["p999"]) * 100
        )
        colors.append(data["config"]["color"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # P99 improvement
    bars1 = ax1.barh(names, p99_improvements, color=colors, alpha=0.85,
                     edgecolor="white", height=0.6)
    for bar, val in zip(bars1, p99_improvements):
        xpos = max(val + 1, 5)
        ax1.text(xpos, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}%", va="center", fontweight="bold", fontsize=11)
    ax1.set_xlabel("P99 Latency Reduction (%)")
    ax1.set_title("P99 Improvement vs 1-Core Baseline", fontweight="bold")
    ax1.set_xlim(left=min(0, min(p99_improvements) - 5))

    # P99.9 improvement
    bars2 = ax2.barh(names, p999_improvements, color=colors, alpha=0.85,
                     edgecolor="white", height=0.6)
    for bar, val in zip(bars2, p999_improvements):
        xpos = max(val + 1, 5)
        ax2.text(xpos, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}%", va="center", fontweight="bold", fontsize=11)
    ax2.set_xlabel("P99.9 Latency Reduction (%)")
    ax2.set_title("P99.9 Improvement vs 1-Core Baseline", fontweight="bold")
    ax2.set_xlim(left=min(0, min(p999_improvements) - 5))

    fig.suptitle("Tail Latency Improvement from Multi-Core Pods",
                 fontsize=15, fontweight="bold", y=1.02)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "improvement_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


# =============================================================================
# Animation
# =============================================================================

def _compute_pod_states_at_time(records, t, num_pods, cores_per_pod):
    """
    Compute per-pod queue state at a specific time.

    Returns:
        light_serving: list[int] — light requests being served per pod
        heavy_serving: list[int] — heavy requests being served per pod
        waiting:       list[int] — requests waiting in queue per pod
    """
    light_serving = [0] * num_pods
    heavy_serving = [0] * num_pods
    waiting = [0] * num_pods

    for rec in records:
        if rec.node > num_pods:
            continue
        pod_idx = rec.node - 1
        svc_end = rec.service_start_date + rec.service_time
        is_heavy = rec.service_time > (LIGHT_SERVICE_TIME + HEAVY_SERVICE_TIME) / 2

        if rec.arrival_date <= t:
            if rec.service_start_date <= t < svc_end:
                # Being served
                if is_heavy:
                    heavy_serving[pod_idx] += 1
                else:
                    light_serving[pod_idx] += 1
            elif t < rec.service_start_date:
                # Waiting in queue
                waiting[pod_idx] += 1

    return light_serving, heavy_serving, waiting


def create_queue_animation(results, anim_start=50.0, duration=10.0, fps=10):
    """
    Create an animated GIF showing queue states across all scenarios over time.

    Each frame shows a bar chart per scenario with:
    - Blue bars: light requests being served
    - Red bars: heavy requests being served
    - Gray bars: requests waiting in queue
    - Dashed line: pod capacity
    """
    total_frames = int(duration * fps)
    frame_times = np.linspace(anim_start, anim_start + duration, total_frames)

    # Pre-filter records for animation window and compute per-scenario metadata
    scenario_data = OrderedDict()
    for name, data in results.items():
        cores_per_pod = data["config"]["cores_per_pod"]
        num_pods = TOTAL_CORES // cores_per_pod
        pods_to_show = min(6, num_pods)

        # Pre-filter records to animation window and visible pods
        relevant = [
            r for r in data["records"]
            if r.node <= pods_to_show
            and (r.service_start_date + r.service_time) >= anim_start
            and r.arrival_date <= anim_start + duration
        ]

        scenario_data[name] = {
            "records": relevant,
            "cores_per_pod": cores_per_pod,
            "pods_to_show": pods_to_show,
            "color": data["config"]["color"],
        }

    # Pre-compute all frame states
    print("  Pre-computing animation frames...")
    all_states = {}
    for name, sd in scenario_data.items():
        states = []
        for t in frame_times:
            ls, hs, w = _compute_pod_states_at_time(
                sd["records"], t, sd["pods_to_show"], sd["cores_per_pod"]
            )
            states.append((ls, hs, w))
        all_states[name] = states

    # Create figure
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 7))
    if n == 1:
        axes = [axes]

    def update(frame_idx):
        t = frame_times[frame_idx]
        for ax, (name, sd) in zip(axes, scenario_data.items()):
            ax.clear()
            ls, hs, w = all_states[name][frame_idx]
            pods = sd["pods_to_show"]
            cpp = sd["cores_per_pod"]
            x = np.arange(pods)

            # Stacked bars: light serving, heavy serving, waiting
            ax.bar(x, ls, color="#3498db", alpha=0.7, label="Light (serving)",
                   edgecolor="white", linewidth=0.5)
            ax.bar(x, hs, bottom=ls, color="#e74c3c", alpha=0.85,
                   label="Heavy (serving)", edgecolor="white", linewidth=0.5)
            bottoms = [ls[i] + hs[i] for i in range(pods)]
            ax.bar(x, w, bottom=bottoms, color="#95a5a6", alpha=0.5,
                   label="Waiting", edgecolor="white", linewidth=0.5)

            ax.axhline(y=cpp, color="black", linestyle="--", linewidth=2,
                       alpha=0.7, label=f"Capacity ({cpp})")

            ax.set_title(f"{name}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Pod")
            ax.set_ylabel("Requests")
            max_y = max(cpp * 4, 12)
            ax.set_ylim(0, max_y)
            ax.set_xticks(x)
            ax.set_xticklabels([f"P{i+1}" for i in range(pods)], fontsize=8)
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.3, axis="y")

        fig.suptitle(
            f"Pod Queue State at t = {t:.1f}s\n"
            f"(Bars above dashed line = HOL blocking / queueing)",
            fontsize=13, fontweight="bold"
        )
        plt.tight_layout(rect=[0, 0, 1, 0.92])

    print("  Rendering animation frames...")
    anim = FuncAnimation(fig, update, frames=total_frames, interval=1000 / fps)

    path = os.path.join(OUTPUT_DIR, "queue_animation.gif")
    anim.save(path, writer=PillowWriter(fps=fps), dpi=80)
    print(f"  Saved: {path}")
    plt.close(fig)


# =============================================================================
# Summary Table
# =============================================================================

def print_summary_table(results):
    """Print a comprehensive formatted summary table to console."""
    print("\n" + "=" * 90)
    print("  SIMULATION RESULTS — COMPREHENSIVE SUMMARY")
    print("=" * 90)
    print(f"\n  Configuration:")
    print(f"    Total cores:              {TOTAL_CORES}")
    print(f"    Light request CPU time:   {LIGHT_SERVICE_TIME * 1000:.0f}ms")
    print(f"    Heavy request CPU time:   {HEAVY_SERVICE_TIME * 1000:.0f}ms")
    print(f"    Heavy request probability:{HEAVY_REQUEST_PROB * 100:.1f}%")
    print(f"    Average service time:     {AVG_SERVICE_TIME * 1000:.1f}ms")
    print(f"    Target utilization:       {TARGET_UTILIZATION * 100:.0f}%")
    print(f"    Arrival rate (total):     {TOTAL_ARRIVAL_RATE:.1f} req/s")
    print(f"    Simulation time:          {SIM_TIME}s (warmup: {SIM_TIME * WARMUP_FRACTION:.0f}s)")

    # Table header
    col_width = 20
    header = f"\n  {'Metric':<25}"
    for name in results:
        header += f"{name:>{col_width}}"
    print(header)
    print("  " + "-" * (25 + col_width * len(results)))

    # Metric rows
    metrics = [
        ("Pods",                lambda d: f"{TOTAL_CORES // d['config']['cores_per_pod']}"),
        ("Cores/Pod",           lambda d: f"{d['config']['cores_per_pod']}"),
        ("Requests Processed",  lambda d: f"{d['stats']['count']:,}"),
        ("",                    None),  # separator
        ("Mean (ms)",           lambda d: f"{d['stats']['mean']:.1f}"),
        ("Median (ms)",         lambda d: f"{d['stats']['median']:.1f}"),
        ("P90 (ms)",            lambda d: f"{d['stats']['p90']:.1f}"),
        ("P95 (ms)",            lambda d: f"{d['stats']['p95']:.1f}"),
        ("P99 (ms)",            lambda d: f"{d['stats']['p99']:.1f}"),
        ("P99.9 (ms)",          lambda d: f"{d['stats']['p999']:.1f}"),
        ("Max (ms)",            lambda d: f"{d['stats']['max']:.1f}"),
        ("Std Dev (ms)",        lambda d: f"{d['stats']['std']:.1f}"),
        ("",                    None),  # separator
        ("Mean Wait (ms)",      lambda d: f"{d['stats']['mean_wait']:.1f}"),
        ("Mean Service (ms)",   lambda d: f"{d['stats']['mean_service']:.1f}"),
    ]

    for metric_name, metric_fn in metrics:
        if metric_fn is None:
            print()
            continue
        row = f"  {metric_name:<25}"
        for name, data in results.items():
            row += f"{metric_fn(data):>{col_width}}"
        print(row)

    # Improvement vs baseline
    if "1-core pods (×24)" in results:
        baseline = results["1-core pods (×24)"]["stats"]
        print()
        for pct, label in [("p99", "P99 Improvement"), ("p999", "P99.9 Improvement")]:
            row = f"  {label:<25}"
            for name, data in results.items():
                improvement = (1 - data["stats"][pct] / baseline[pct]) * 100
                row += f"{improvement:>{col_width - 1}.1f}%"
            print(row)

    print("\n" + "=" * 90)


# =============================================================================
# Theoretical Analysis
# =============================================================================

def print_theoretical_analysis():
    """Print theoretical M/G/1 analysis for comparison with simulation."""
    print("\n" + "=" * 70)
    print("  THEORETICAL ANALYSIS (M/G/1 — Single-Core Pods)")
    print("=" * 70)

    es = AVG_SERVICE_TIME
    es2 = (1 - HEAVY_REQUEST_PROB) * LIGHT_SERVICE_TIME**2 + \
          HEAVY_REQUEST_PROB * HEAVY_SERVICE_TIME**2
    var_s = es2 - es**2
    cv2 = var_s / es**2
    rho = TARGET_UTILIZATION
    lam = rho / es

    wq = (lam * es2) / (2 * (1 - rho))
    w = wq + es
    lq = lam * wq
    l = lam * w

    print(f"\n  Service Time Distribution:")
    print(f"    E[S]   = {es * 1000:.2f} ms")
    print(f"    E[S²]  = {es2 * 1e6:.2f} ms²")
    print(f"    Var[S] = {var_s * 1e6:.2f} ms²")
    print(f"    σ[S]   = {var_s**0.5 * 1000:.2f} ms")
    print(f"    CV²    = {cv2:.2f}  (>>1 indicates high variance)")
    print(f"\n  Pollaczek-Khinchine Results (M/G/1, ρ={rho:.2f}):")
    print(f"    Mean wait in queue (Wq) = {wq * 1000:.1f} ms")
    print(f"    Mean response time (W)  = {w * 1000:.1f} ms")
    print(f"    Mean queue length  (Lq) = {lq:.2f}")
    print(f"    Mean in system     (L)  = {l:.2f}")

    # Compare with deterministic service (CV²=0)
    wq_det = (rho * es) / (2 * (1 - rho))
    w_det = wq_det + es
    print(f"\n  Comparison (if service were deterministic, same mean):")
    print(f"    Mean wait (deterministic) = {wq_det * 1000:.1f} ms")
    print(f"    Mean response (determ.)   = {w_det * 1000:.1f} ms")
    print(f"    Variance penalty factor   = {w / w_det:.1f}× "
          f"(bimodal is {w / w_det:.1f}× worse)")

    print("\n" + "=" * 70)


# =============================================================================
# Main
# =============================================================================

def main():
    print()
    print("╔" + "═" * 58 + "╗")
    print("║  Kubernetes Pod Load Balancing Simulator                  ║")
    print("║  Discrete Event Simulation using Ciw                     ║")
    print("║  Modeling Head-of-Line Blocking in Single vs Multi-Core  ║")
    print("╚" + "═" * 58 + "╝")
    print()

    ensure_output_dir()

    # --- Theoretical analysis ---
    print_theoretical_analysis()

    # --- Run simulations ---
    print("\n\n▶ Running simulations...")
    results = run_all_scenarios()

    # --- Print summary ---
    print_summary_table(results)

    # --- Generate visualizations ---
    print("\n▶ Generating visualizations...")
    plot_cdf_comparison(results)
    plot_percentile_bars(results)
    plot_response_time_histogram(results)
    plot_box_comparison(results)
    plot_gantt_chart(results)
    plot_queue_depth_timeseries(results)
    plot_improvement_summary(results)

    # --- Generate animation ---
    print("\n▶ Generating queue animation (this may take a minute)...")
    create_queue_animation(results, anim_start=50.0, duration=10.0, fps=10)

    # --- Done ---
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║  ✅ All done!                                             ║")
    print("║                                                          ║")
    print("║  Output files saved to ./output/:                        ║")
    print("║    • cdf_comparison.png       — Response time CDF        ║")
    print("║    • percentile_comparison.png— P90/P95/P99/P99.9 bars  ║")
    print("║    • histogram_comparison.png — Distribution histograms  ║")
    print("║    • boxplot_comparison.png   — Box plot comparison      ║")
    print("║    • gantt_chart.png          — Processing timeline      ║")
    print("║    • queue_timeseries.png     — Queue depth over time    ║")
    print("║    • improvement_summary.png  — % improvement dashboard  ║")
    print("║    • queue_animation.gif      — Animated queue states    ║")
    print("╚" + "═" * 58 + "╝")


if __name__ == "__main__":
    main()

