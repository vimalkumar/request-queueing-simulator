#!/usr/bin/env python3
"""
Interactive Kubernetes Pod Load Balancing Simulator
===================================================

Run with:  streamlit run interactive_app.py
"""

import random
from collections import OrderedDict

import ciw
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# =============================================================================
# Page Config & Styling
# =============================================================================

st.set_page_config(
    page_title="K8s Pod Sizing Simulator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] { padding: 10px 20px; }
    div[data-testid="stMetricValue"] { font-size: 1.3rem; }
    .insight-box {
        background: rgba(41,128,185,0.15); border-left: 4px solid #2980b9;
        padding: 0.8rem 1rem; margin: 0.8rem 0; border-radius: 0 6px 6px 0;
        color: inherit;
    }
    .warning-box {
        background: rgba(231,76,60,0.13); border-left: 4px solid #e74c3c;
        padding: 0.8rem 1rem; margin: 0.8rem 0; border-radius: 0 6px 6px 0;
        color: inherit;
    }
    .success-box {
        background: rgba(39,174,96,0.13); border-left: 4px solid #27ae60;
        padding: 0.8rem 1rem; margin: 0.8rem 0; border-radius: 0 6px 6px 0;
        color: inherit;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Simulation Engine
# =============================================================================

class BimodalServiceDist(ciw.dists.Distribution):
    def __init__(self, light_time, heavy_time, heavy_prob):
        super().__init__()
        self.light_time = light_time
        self.heavy_time = heavy_time
        self.heavy_prob = heavy_prob

    def sample(self, t=None, ind=None):
        return self.heavy_time if random.random() < self.heavy_prob else self.light_time


@st.cache_data(show_spinner="Simulating traffic...")
def run_simulation(cores_per_pod, total_cores, light_time_ms, heavy_time_ms,
                   heavy_prob, utilization, sim_time, seed=42):
    light_time = light_time_ms / 1000.0
    heavy_time = heavy_time_ms / 1000.0
    avg_service = (1 - heavy_prob) * light_time + heavy_prob * heavy_time
    total_arrival_rate = total_cores * (utilization / avg_service)
    num_pods = total_cores // cores_per_pod
    rate_per_pod = total_arrival_rate / num_pods

    ciw.seed(seed)
    random.seed(seed)

    network = ciw.create_network(
        arrival_distributions=[ciw.dists.Exponential(rate=rate_per_pod)] * num_pods,
        service_distributions=[BimodalServiceDist(light_time, heavy_time, heavy_prob)] * num_pods,
        number_of_servers=[cores_per_pod] * num_pods,
        routing=[[0.0] * num_pods for _ in range(num_pods)],
    )
    sim = ciw.Simulation(network)
    sim.simulate_until_max_time(sim_time)

    warmup = sim_time * 0.10
    return [
        {
            "arrival_date": r.arrival_date, "waiting_time": r.waiting_time,
            "service_start_date": r.service_start_date, "service_time": r.service_time,
            "exit_date": r.exit_date, "node": r.node, "server_id": r.server_id,
        }
        for r in sim.get_all_records() if r.arrival_date > warmup
    ]


def compute_metrics(rec_dicts):
    if not rec_dicts:
        return {}, np.array([])
    rt = np.array([r["waiting_time"] + r["service_time"] for r in rec_dicts])
    ms = 1000.0
    return {
        "count": len(rt), "mean": np.mean(rt) * ms, "median": np.median(rt) * ms,
        "p50": np.percentile(rt, 50) * ms, "p90": np.percentile(rt, 90) * ms,
        "p95": np.percentile(rt, 95) * ms, "p99": np.percentile(rt, 99) * ms,
        "p999": np.percentile(rt, 99.9) * ms, "max": np.max(rt) * ms,
        "std": np.std(rt) * ms,
        "mean_wait": np.mean([r["waiting_time"] for r in rec_dicts]) * ms,
        "mean_service": np.mean([r["service_time"] for r in rec_dicts]) * ms,
    }, rt


def compute_pod_states(rec_dicts, t, num_pods, cores_per_pod, max_pods=8):
    pods_shown = min(max_pods, num_pods)
    ls, hs, w = [0]*pods_shown, [0]*pods_shown, [0]*pods_shown
    matrix = np.zeros((pods_shown, cores_per_pod))
    for r in rec_dicts:
        if r["node"] > pods_shown:
            continue
        pi = r["node"] - 1
        end = r["service_start_date"] + r["service_time"]
        heavy = r["service_time"] > 0.1
        if r["arrival_date"] <= t:
            if r["service_start_date"] <= t < end:
                (hs if heavy else ls)[pi] += 1
                ci = r["server_id"] - 1
                if ci < cores_per_pod:
                    matrix[pi][ci] = 2 if heavy else 1
            elif t < r["service_start_date"]:
                w[pi] += 1
    return ls, hs, w, matrix


def hex_to_rgba(hex_color, alpha=0.3):
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{alpha})"


COLORS = ["#e74c3c", "#f39c12", "#27ae60", "#2980b9", "#9b59b6", "#1abc9c", "#e67e22"]



# =============================================================================
# Header — this IS the home page
# =============================================================================

st.markdown("# ⚡ K8s Pod Sizing Simulator")
st.caption("Should you use 1-core or multi-core Kubernetes pods? "
           "Simulate your workload and see the impact on tail latency.")

# Help / docs toggle right at the top
with st.expander("ℹ️  **What is this?** — quick explainer & links to full docs", expanded=False):
    st.markdown("""
    **The problem:** Single-core pods process requests one at a time. When a slow
    request (e.g. 500ms) arrives, every request behind it *waits* — even if other
    pods are idle. This is called **head-of-line (HOL) blocking**, and it's a major
    cause of P99 latency spikes in Kubernetes.

    **The fix:** Multi-core pods let the Linux scheduler spread requests across cores
    *within* each pod. A slow request blocks only 1 of N cores — the rest keep serving.

    **How this simulator works:** We model each pod as a queuing system (M/G/c) using
    [Ciw](https://ciw.readthedocs.io/) discrete event simulation. We keep the total
    number of cores constant and compare: many single-core pods vs. fewer multi-core pods.

    **Tune the parameters** in the sidebar to match your real workload, then explore
    the tabs below for latency charts, live pod visualizations, and Gantt timelines.

    ---
    📖 **Want the full details?** Use the **📖 Documentation** page in the sidebar
    for the complete theory, worked examples, and simulator guide.
    """)


# =============================================================================
# Sidebar — parameter controls
# =============================================================================

st.sidebar.markdown("## ⚙️ Tune the Model")
st.sidebar.caption("Match these to your infrastructure. Results update automatically.")

total_cores = st.sidebar.slider("Total CPU Cores", 8, 64, 24, step=8,
                                help="Total compute capacity — stays constant across scenarios")
light_time_ms = st.sidebar.slider("Light Request CPU (ms)", 5, 100, 20,
                                  help="CPU time for a typical request")
heavy_time_ms = st.sidebar.slider("Heavy Request CPU (ms)", 100, 2000, 500, step=50,
                                  help="CPU time for an expensive request")
heavy_prob_pct = st.sidebar.slider("Heavy Request %", 1, 20, 5,
                                   help="What fraction of requests are expensive")
utilization_pct = st.sidebar.slider("Target Utilization %", 30, 90, 70,
                                    help="How busy each core is on average")
sim_time = st.sidebar.slider("Simulation Duration (s)", 30, 300, 120,
                             help="Longer = more stable tail latency numbers")

heavy_prob = heavy_prob_pct / 100.0
utilization = utilization_pct / 100.0
avg_svc = (1 - heavy_prob) * light_time_ms + heavy_prob * heavy_time_ms
cv2 = ((1-heavy_prob)*light_time_ms**2 + heavy_prob*heavy_time_ms**2 - avg_svc**2) / avg_svc**2
arrival_rate = total_cores * (utilization / (avg_svc / 1000.0))

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Avg service:** {avg_svc:.1f}ms &nbsp;•&nbsp; **CV²:** {cv2:.1f} "
                    f"&nbsp;•&nbsp; **Throughput:** {arrival_rate:.0f} req/s")
if cv2 > 3:
    st.sidebar.warning("⚠️ High variance — expect significant HOL blocking with single-core pods.")

st.sidebar.markdown("---")
st.sidebar.markdown("**📖 Docs**")
st.sidebar.page_link("pages/1_📖_The_Problem.py", label="The Problem")
st.sidebar.page_link("pages/2_📐_Queuing_Theory.py", label="Queuing Theory")
st.sidebar.page_link("pages/3_🛠️_Simulator_Guide.py", label="Simulator Guide")


# =============================================================================
# Run simulations
# =============================================================================

SCENARIO_CONFIGS = OrderedDict()
for c in [c for c in [1, 2, 4, 8, 16, 24, 32] if total_cores % c == 0]:
    n = total_cores // c
    SCENARIO_CONFIGS[f"{c}-core pods (×{n})"] = {"cores_per_pod": c, "num_pods": n}

all_results = OrderedDict()
for i, (name, cfg) in enumerate(SCENARIO_CONFIGS.items()):
    recs = run_simulation(cfg["cores_per_pod"], total_cores, light_time_ms,
                          heavy_time_ms, heavy_prob, utilization, sim_time)
    stats, rt = compute_metrics(recs)
    all_results[name] = {"records": recs, "stats": stats, "response_times": rt,
                         "config": cfg, "color": COLORS[i % len(COLORS)]}


# =============================================================================
# Results at a glance
# =============================================================================

st.markdown("---")

# Compact problem/fix summary — one line each, not big boxes
first_key = list(all_results.keys())[0]
last_key = list(all_results.keys())[-1]
baseline_p99 = all_results[first_key]["stats"]["p99"]
best_p99 = all_results[last_key]["stats"]["p99"]
improvement = (1 - best_p99 / baseline_p99) * 100

st.markdown(f"""
<div class="warning-box">
<strong>Current state ({first_key}):</strong> P99 = <strong>{baseline_p99:.0f}ms</strong>
&nbsp;— heavy requests ({heavy_time_ms}ms) block the queue, inflating tail latency well above service time.
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="success-box">
<strong>With multi-core pods ({last_key}):</strong> P99 = <strong>{best_p99:.0f}ms</strong>
&nbsp;— a <strong>{improvement:.0f}% reduction</strong>. The Linux scheduler keeps other cores serving while one handles a heavy request.
</div>
""", unsafe_allow_html=True)

# Metrics row
cols = st.columns(len(all_results))
for col, (name, data) in zip(cols, all_results.items()):
    with col:
        p99 = data["stats"]["p99"]
        delta_str = f"{(1 - p99/baseline_p99)*100:+.0f}% vs baseline" if p99 < baseline_p99 else "baseline"
        st.metric(label=name, value=f"{p99:.0f}ms P99",
                  delta=delta_str, delta_color="normal" if p99 < baseline_p99 else "off")

# =============================================================================
# Tabs
# =============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Latency Analysis", "🖥️ Live Pod View", "📈 Queue Depth",
    "📋 Gantt Timeline", "🗂️ Raw Data",
])

# ── TAB 1: Latency Analysis ─────────────────────────────────────────────────

with tab1:
    st.markdown("""
    ### How does response time change with pod sizing?

    The **CDF** (cumulative distribution function) tells the full story.
    For any point on the curve, you can read: *"X% of requests completed within Y ms."*
    The further left the curve, the better.

    **What to look for:** Single-core pods (red) have a long, slow-rising tail — that's the HOL blocking.
    Multi-core pods (green/blue) snap sharply to 1.0, meaning almost all requests finish quickly.
    """)

    col_left, col_right = st.columns(2)

    with col_left:
        fig_cdf = go.Figure()
        for name, data in all_results.items():
            st_t = np.sort(data["response_times"]) * 1000
            cdf = np.arange(1, len(st_t) + 1) / len(st_t)
            step = max(1, len(st_t) // 2000)
            fig_cdf.add_trace(go.Scatter(
                x=st_t[::step], y=cdf[::step], mode="lines", name=name,
                line=dict(color=data["color"], width=2.5),
                hovertemplate="Response: %{x:.0f}ms<br>Percentile: %{y:.4f}<extra></extra>"))
        for p, lbl in [(0.90, "P90"), (0.99, "P99"), (0.999, "P99.9")]:
            fig_cdf.add_hline(y=p, line_dash="dot", line_color="gray", opacity=0.4,
                              annotation_text=lbl, annotation_position="top left")
        fig_cdf.update_layout(
            title="Full CDF", xaxis_title="Response Time (ms)",
            yaxis_title="Cumulative Probability", height=480, hovermode="x unified",
            legend=dict(yanchor="bottom", y=0.01, xanchor="left", x=0.01))
        st.plotly_chart(fig_cdf, use_container_width=True)

    with col_right:
        fig_tail = go.Figure()
        for name, data in all_results.items():
            st_t = np.sort(data["response_times"]) * 1000
            cdf = np.arange(1, len(st_t) + 1) / len(st_t)
            mask = cdf >= 0.90
            step = max(1, mask.sum() // 1000)
            fig_tail.add_trace(go.Scatter(
                x=st_t[mask][::step], y=cdf[mask][::step], mode="lines", name=name,
                line=dict(color=data["color"], width=2.5),
                hovertemplate="Response: %{x:.0f}ms<br>Percentile: %{y:.5f}<extra></extra>"))
        for p, lbl in [(0.95, "P95"), (0.99, "P99"), (0.999, "P99.9")]:
            fig_tail.add_hline(y=p, line_dash="dot", line_color="gray", opacity=0.4,
                               annotation_text=lbl, annotation_position="top left")
        fig_tail.update_layout(
            title="Tail Zoom (P90+)", xaxis_title="Response Time (ms)",
            yaxis_title="Cumulative Probability", yaxis_range=[0.90, 1.002],
            height=480, hovermode="x unified",
            legend=dict(yanchor="bottom", y=0.01, xanchor="left", x=0.01))
        st.plotly_chart(fig_tail, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    💡 <strong>Reading the tail zoom:</strong> Find the P99 line and see where each curve crosses it.
    That horizontal distance is the P99 latency difference between configurations.
    The gap between red (1-core) and green (8-core) is your potential improvement.
    </div>
    """, unsafe_allow_html=True)

    # Percentile bars
    st.markdown("### Percentile Comparison")
    st.markdown("Each bar group shows one percentile. "
                "The height difference between red and blue bars is the **latency you'd save** "
                "by switching to multi-core pods.")
    pct_labels = ["P50", "P90", "P95", "P99", "P99.9"]
    pct_keys = ["p50", "p90", "p95", "p99", "p999"]
    fig_bars = go.Figure()
    for name, data in all_results.items():
        vals = [data["stats"][k] for k in pct_keys]
        fig_bars.add_trace(go.Bar(
            x=pct_labels, y=vals, name=name, marker_color=data["color"], opacity=0.85,
            text=[f"{v:.0f}" for v in vals], textposition="outside",
            hovertemplate="%{x}: %{y:.1f}ms<extra></extra>"))
    fig_bars.update_layout(barmode="group", yaxis_title="Response Time (ms)", height=420,
                           legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig_bars, use_container_width=True)

    # Histograms
    st.markdown("### Response Time Distribution")
    st.markdown("Notice how the single-core histogram has a long right tail — "
                "those are the requests that got stuck behind heavy ones. "
                "Multi-core distributions are much tighter.")
    fig_hist = make_subplots(rows=1, cols=len(all_results),
                             subplot_titles=list(all_results.keys()), shared_yaxes=True)
    for i, (name, data) in enumerate(all_results.items()):
        t_ms = data["response_times"] * 1000
        clip = np.percentile(t_ms, 99.5)
        fig_hist.add_trace(go.Histogram(
            x=t_ms[t_ms <= clip], nbinsx=80, name=name,
            marker_color=data["color"], opacity=0.75,
            hovertemplate="Range: %{x}ms<br>Count: %{y}<extra></extra>"), row=1, col=i+1)
        fig_hist.add_vline(x=data["stats"]["p99"], line_dash="dash", line_color="black",
                           annotation_text=f"P99={data['stats']['p99']:.0f}", row=1, col=i+1)
    fig_hist.update_layout(height=380, showlegend=False)
    fig_hist.update_xaxes(title_text="Response Time (ms)")
    st.plotly_chart(fig_hist, use_container_width=True)


# ── TAB 2: Live Pod View ────────────────────────────────────────────────────

with tab2:
    st.markdown("""
    ### See Inside the Pods — in Real Time

    Drag the time slider to freeze the simulation at any moment and see what each pod
    is doing. Each **cell** represents one CPU core inside a pod.

    | Color | Meaning |
    |-------|---------|
    | 🔵 Blue | Core is processing a **light** request (~{light}ms) |
    | 🔴 Red | Core is processing a **heavy** request (~{heavy}ms) |
    | ⬜ Gray | Core is **idle** |
    | **Q:N** | **N requests waiting** in the pod's queue |

    **The key insight:** In 1-core pods, a single red cell means the entire pod is blocked.
    In multi-core pods, other cores keep working even when one is busy with a heavy request.
    """.format(light=light_time_ms, heavy=heavy_time_ms))

    view_start = sim_time * 0.15
    view_end = sim_time * 0.50
    t_val = st.slider("⏱️ Simulation Time", min_value=float(view_start),
                       max_value=float(view_end),
                       value=float(view_start + (view_end - view_start) * 0.3),
                       step=0.05, format="%.2fs")

    viz_cols = st.columns(len(all_results))
    for col, (name, data) in zip(viz_cols, all_results.items()):
        with col:
            cfg = data["config"]
            cpp, npods = cfg["cores_per_pod"], cfg["num_pods"]
            ps = min(8, npods)
            relevant = [r for r in data["records"]
                        if (r["service_start_date"] + r["service_time"]) >= t_val - 1
                        and r["arrival_date"] <= t_val + 1 and r["node"] <= ps]
            ls, hs, w, mx = compute_pod_states(relevant, t_val, npods, cpp, ps)

            colorscale = [[0, "#f0f0f0"], [0.49, "#f0f0f0"], [0.50, "#3498db"],
                          [0.74, "#3498db"], [0.75, "#e74c3c"], [1.0, "#e74c3c"]]
            fig_hm = go.Figure(data=go.Heatmap(
                z=mx, x=[f"C{j+1}" for j in range(cpp)],
                y=[f"Pod {i+1}" for i in range(ps)],
                colorscale=colorscale, zmin=0, zmax=2, showscale=False, xgap=3, ygap=3,
                hovertemplate="%{y}, %{x}: %{z:.0f}<extra></extra>"))
            for i in range(ps):
                fig_hm.add_annotation(x=cpp-0.5, y=i, xshift=40,
                                      text=f"Q:{w[i]}", showarrow=False,
                                      font=dict(size=11, color="red" if w[i] > 0 else "gray"))
            fig_hm.update_layout(title=dict(text=f"**{name}**", font=dict(size=12)),
                                 height=max(180, ps * 38 + 70),
                                 margin=dict(l=60, r=50, t=40, b=20),
                                 yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_hm, use_container_width=True)

            total_q = sum(w)
            total_h = sum(hs)
            if total_q > 0:
                st.markdown(f"⏳ **{total_q} requests waiting** ({total_h} heavy in service)")
            else:
                st.markdown("✅ No requests waiting")

    # Animated bar chart
    st.markdown("---")
    st.markdown("""
    ### Animated Queue Dynamics

    Press **▶ Play** to watch how queues build up and drain over a 10-second window.
    Bars above the dashed **capacity** line represent requests that are *waiting* —
    they can't be served because all cores are busy. Notice how single-core pods
    frequently spike above capacity while multi-core pods rarely do.
    """)

    anim_dur = min(10.0, view_end - view_start)
    n_frames = int(anim_dur / 0.2)
    anim_times = np.linspace(t_val, t_val + anim_dur, n_frames)

    rows = []
    for t in anim_times:
        for sn, sd in all_results.items():
            cfg = sd["config"]
            cpp, npods = cfg["cores_per_pod"], cfg["num_pods"]
            ps = min(6, npods)
            rel = [r for r in sd["records"]
                   if (r["service_start_date"]+r["service_time"]) >= t-1
                   and r["arrival_date"] <= t+1 and r["node"] <= ps]
            ls, hs, w, _ = compute_pod_states(rel, t, npods, cpp, ps)
            for pi in range(ps):
                rows.append({"time": round(t,1), "scenario": sn, "pod": f"P{pi+1}",
                             "Light (serving)": ls[pi], "Heavy (serving)": hs[pi], "Waiting": w[pi]})

    if rows:
        df_a = pd.DataFrame(rows)
        df_m = df_a.melt(id_vars=["time","scenario","pod"],
                         value_vars=["Light (serving)","Heavy (serving)","Waiting"],
                         var_name="type", value_name="count")
        scens = list(all_results.keys())
        fig_an = make_subplots(rows=1, cols=len(scens), subplot_titles=scens, shared_yaxes=True)
        tc = {"Light (serving)":"#3498db","Heavy (serving)":"#e74c3c","Waiting":"#95a5a6"}
        t0 = anim_times[0]
        for ci, sn in enumerate(scens):
            sub = df_m[(df_m["time"]==round(t0,1))&(df_m["scenario"]==sn)]
            for rt in tc:
                s = sub[sub["type"]==rt]
                fig_an.add_trace(go.Bar(x=s["pod"],y=s["count"],name=rt,marker_color=tc[rt],
                                        showlegend=(ci==0)), row=1,col=ci+1)
        for ci, sn in enumerate(scens):
            cpp = all_results[sn]["config"]["cores_per_pod"]
            fig_an.add_hline(y=cpp, line_dash="dash", line_color="black", opacity=0.6,
                             annotation_text=f"cap={cpp}", row=1, col=ci+1)
        frames = []
        for t in anim_times:
            fd = []
            for sn in scens:
                sub = df_m[(df_m["time"]==round(t,1))&(df_m["scenario"]==sn)]
                for rt in tc:
                    s = sub[sub["type"]==rt]
                    fd.append(go.Bar(x=s["pod"].tolist(), y=s["count"].tolist(), marker_color=tc[rt]))
            frames.append(go.Frame(data=fd, name=f"{t:.1f}"))
        fig_an.frames = frames
        mx_y = max(df_a[["Light (serving)","Heavy (serving)","Waiting"]].sum(axis=1).max(), 10)
        fig_an.update_layout(
            barmode="stack", height=430, yaxis_range=[0, mx_y*1.2],
            updatemenus=[{"type":"buttons","showactive":False,"y":-0.15,"x":0.5,"xanchor":"center",
                          "buttons":[
                              {"label":"▶ Play","method":"animate",
                               "args":[None,{"frame":{"duration":150,"redraw":True},"fromcurrent":True}]},
                              {"label":"⏸ Pause","method":"animate",
                               "args":[[None],{"frame":{"duration":0},"mode":"immediate"}]}]}],
            sliders=[{"active":0,"y":-0.05,"x":0.05,"len":0.9,
                       "currentvalue":{"prefix":"Time: ","suffix":"s","visible":True},
                       "steps":[{"args":[[f"{t:.1f}"],{"frame":{"duration":0,"redraw":True},"mode":"immediate"}],
                                 "label":f"{t:.1f}","method":"animate"} for t in anim_times]}])
        st.plotly_chart(fig_an, use_container_width=True)


# ── TAB 3: Queue Depth ──────────────────────────────────────────────────────

with tab3:
    st.markdown(f"""
    ### Queue Build-up Over Time

    Each chart tracks **Pod #1** in each scenario. The colored area shows requests *being served*,
    and the gray area on top shows requests *waiting in queue*.

    **The dashed line is the pod's capacity** (number of cores). Anything above it is a queue —
    those requests are blocked, experiencing HOL delay.

    For single-core pods, the queue spikes every time a {heavy_time_ms}ms request arrives (~every
    {1/(heavy_prob * arrival_rate / total_cores):.1f}s on average). Multi-core pods absorb the shock
    because the other cores keep draining the queue.
    """)

    qd_start = sim_time * 0.15
    qd_end = min(sim_time * 0.3, qd_start + 30)
    time_pts = np.arange(qd_start, qd_end, 0.05)

    fig_qd = make_subplots(rows=len(all_results), cols=1,
                           subplot_titles=list(all_results.keys()),
                           shared_xaxes=True, vertical_spacing=0.08)
    for ri, (name, data) in enumerate(all_results.items()):
        cpp = data["config"]["cores_per_pod"]
        pod1 = [r for r in data["records"] if r["node"]==1
                and (r["service_start_date"]+r["service_time"])>=qd_start
                and r["arrival_date"]<=qd_end]
        svc_arr, wait_arr = [], []
        for t in time_pts:
            svc_arr.append(sum(1 for r in pod1
                               if r["service_start_date"]<=t<(r["service_start_date"]+r["service_time"])))
            wait_arr.append(sum(1 for r in pod1
                                if r["arrival_date"]<=t<r["service_start_date"]))
        total = np.array(svc_arr) + np.array(wait_arr)
        fig_qd.add_trace(go.Scatter(
            x=time_pts, y=svc_arr, mode="lines", name="Serving", fill="tozeroy",
            line=dict(color=data["color"], width=1), fillcolor=hex_to_rgba(data["color"], 0.3),
            showlegend=(ri==0), hovertemplate="t=%{x:.2f}s — Serving: %{y}<extra></extra>"),
            row=ri+1, col=1)
        fig_qd.add_trace(go.Scatter(
            x=time_pts, y=total.tolist(), mode="lines", name="+ Waiting", fill="tonexty",
            line=dict(color="gray", width=1), fillcolor="rgba(150,150,150,0.2)",
            showlegend=(ri==0), hovertemplate="t=%{x:.2f}s — Total: %{y}<extra></extra>"),
            row=ri+1, col=1)
        fig_qd.add_hline(y=cpp, line_dash="dash", line_color="black", opacity=0.5,
                         annotation_text=f"capacity={cpp}", row=ri+1, col=1)
    fig_qd.update_layout(height=260*len(all_results), hovermode="x unified",
                         legend=dict(orientation="h", yanchor="bottom", y=1.02))
    fig_qd.update_xaxes(title_text="Time (seconds)", row=len(all_results), col=1)
    st.plotly_chart(fig_qd, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    💡 <strong>Key pattern:</strong> In the 1-core chart, every spike corresponds to a heavy request arrival.
    The spike height tells you how many light requests got stuck behind it. In multi-core charts,
    the same heavy requests arrive but the spikes are much smaller — the other cores absorb the load.
    </div>
    """, unsafe_allow_html=True)


# ── TAB 4: Gantt Timeline ───────────────────────────────────────────────────

with tab4:
    st.markdown("""
    ### Request Processing Timeline

    This Gantt chart shows exactly what each CPU core is doing over a 2-second window.
    Each horizontal block is one request being processed.

    | Color | Meaning |
    |-------|---------|
    | 🔵 Blue blocks | Light requests (~20ms) — these should be fast |
    | 🔴 Red blocks | Heavy requests (~500ms) — these cause HOL blocking |

    **What to notice:** In single-core pods, a red block *fills the entire row* for 500ms,
    leaving no room for blue blocks. In multi-core pods, blue blocks continue on other rows
    (cores) even while a red block is running.

    **Hover** over any block to see its exact service time and wait time.
    """)

    gantt_scen = st.selectbox("Select configuration to inspect", list(all_results.keys()))
    gd = all_results[gantt_scen]
    gcfg = gd["config"]
    gcpp, gnp = gcfg["cores_per_pod"], gcfg["num_pods"]
    gps = min(4, gnp)
    gcs = gps * gcpp
    gw = 2.0
    gt_s = sim_time * 0.15
    for tt in np.arange(gt_s, gt_s+60, 0.5):
        wr = [r for r in gd["records"] if r["service_start_date"]<tt+gw
              and (r["service_start_date"]+r["service_time"])>tt and r["node"]<=gps]
        if sum(1 for r in wr if r["service_time"]>0.1) >= 2:
            gt_s = tt
            break
    gt_e = gt_s + gw

    gantt_rows = []
    for r in gd["records"]:
        se = r["service_start_date"]+r["service_time"]
        if se < gt_s or r["service_start_date"] > gt_e or r["node"] > gps:
            continue
        ci = (r["node"]-1)*gcpp + (r["server_id"]-1)
        if ci >= gcs:
            continue
        gantt_rows.append({"Core":f"P{r['node']}:C{r['server_id']}",
                           "Start":max(r["service_start_date"],gt_s),
                           "End":min(se,gt_e),
                           "Type":"Heavy" if r["service_time"]>0.1 else "Light",
                           "svc":f"{r['service_time']*1000:.0f}ms",
                           "wait":f"{r['waiting_time']*1000:.0f}ms"})
    if gantt_rows:
        df_g = pd.DataFrame(gantt_rows)
        core_labels = sorted(df_g["Core"].unique(),
                             key=lambda x: (int(x.split(":")[0][1:]), int(x.split(":")[1][1:])))
        fig_g = go.Figure()
        for _, row in df_g.iterrows():
            heavy = row["Type"]=="Heavy"
            fig_g.add_trace(go.Bar(
                x=[row["End"]-row["Start"]], y=[row["Core"]], base=[row["Start"]],
                orientation="h",
                marker=dict(color="#e74c3c" if heavy else "#3498db",
                            opacity=0.95 if heavy else 0.6,
                            line=dict(color="white", width=0.5)),
                showlegend=False,
                hovertemplate=f"{row['Core']} — {row['Type']}<br>"
                              f"Service: {row['svc']}<br>Wait: {row['wait']}<extra></extra>"))
        for i in range(1, gps):
            fig_g.add_hline(y=i*gcpp-0.5, line_color="black", line_width=2)
        fig_g.add_trace(go.Bar(x=[0],y=[core_labels[0]],marker_color="#3498db",opacity=0.6,
                               name="Light (~20ms)",showlegend=True,visible="legendonly"))
        fig_g.add_trace(go.Bar(x=[0],y=[core_labels[0]],marker_color="#e74c3c",opacity=0.95,
                               name="Heavy (~500ms)",showlegend=True,visible="legendonly"))
        fig_g.update_layout(xaxis_title="Time (seconds)",
                           yaxis=dict(categoryorder="array",categoryarray=core_labels[::-1]),
                           height=max(300, gcs*35+100), xaxis_range=[gt_s,gt_e],
                           barmode="overlay", showlegend=True)
        st.plotly_chart(fig_g, use_container_width=True)
    else:
        st.info("No records in this time window. Try a different scenario.")


# ── TAB 5: Raw Data ─────────────────────────────────────────────────────────

with tab5:
    st.markdown("""
    ### Full Results Table

    All numbers in one place. Use this to compare exact percentiles across configurations
    or to copy data into your own analysis.
    """)

    rows_t = []
    for name, data in all_results.items():
        s, cfg = data["stats"], data["config"]
        rows_t.append({"Configuration": name, "Pods": cfg["num_pods"],
                        "Cores/Pod": cfg["cores_per_pod"],
                        "Requests": f"{s['count']:,}",
                        "Mean": f"{s['mean']:.1f}", "Median": f"{s['median']:.1f}",
                        "P90": f"{s['p90']:.1f}", "P95": f"{s['p95']:.1f}",
                        "P99": f"{s['p99']:.1f}", "P99.9": f"{s['p999']:.1f}",
                        "Max": f"{s['max']:.1f}", "Std Dev": f"{s['std']:.1f}",
                        "Mean Wait": f"{s['mean_wait']:.1f}"})
    st.dataframe(pd.DataFrame(rows_t), use_container_width=True, hide_index=True)

    # Improvement table
    first_key = list(all_results.keys())[0]
    if "1-core" in first_key:
        st.markdown("### Improvement vs Single-Core Baseline")
        st.markdown("How much tail latency drops when you increase cores per pod — "
                    "keeping total compute constant.")
        bl = all_results[first_key]["stats"]
        imp = [{"Config": n, **{f"{p} ↓": f"{(1-d['stats'][k]/bl[k])*100:.1f}%"
                for p,k in [("P90","p90"),("P95","p95"),("P99","p99"),("P99.9","p999"),("Mean","mean")]}}
               for n, d in all_results.items()]
        st.dataframe(pd.DataFrame(imp), use_container_width=True, hide_index=True)

    # Theory vs simulation
    st.markdown("### Theory Check — Pollaczek-Khinchine Formula")
    st.markdown("The P-K formula predicts mean response time for M/G/1 queues (single-core pods). "
                "If theory and simulation agree, the model is sound.")
    avg_s = avg_svc / 1000.0
    es2 = (1-heavy_prob)*(light_time_ms/1000)**2 + heavy_prob*(heavy_time_ms/1000)**2
    lam = utilization / avg_s
    wq_th = (lam * es2) / (2 * (1 - utilization))
    w_th = wq_th + avg_s
    fs = all_results[first_key]["stats"]
    theory_df = pd.DataFrame([
        {"Metric": "CV² (variance ratio)", "Theory": f"{cv2:.2f}", "Simulated": "—"},
        {"Metric": "Mean Wait (Wq)", "Theory": f"{wq_th*1000:.1f}ms", "Simulated": f"{fs['mean_wait']:.1f}ms"},
        {"Metric": "Mean Response (W)", "Theory": f"{w_th*1000:.1f}ms", "Simulated": f"{fs['mean']:.1f}ms"},
        {"Metric": "Utilization (ρ)", "Theory": f"{utilization:.2f}", "Simulated": "—"},
    ])
    st.dataframe(theory_df, use_container_width=True, hide_index=True)

    if abs(w_th*1000 - fs['mean']) / fs['mean'] < 0.15:
        st.success("✅ Theory and simulation agree within 15% — the model is validated.")
    else:
        st.warning("⚠️ Theory and simulation diverge — this can happen at very high utilization "
                   "or with short simulation times.")


# =============================================================================
# Footer
# =============================================================================

st.markdown("---")
col_f1, col_f2 = st.columns([2, 1])
with col_f1:
    st.markdown("""
    <div style="color:#888; font-size:0.85em;">
        Built with <strong>Ciw</strong> (discrete event simulation) •
        <strong>Plotly</strong> (interactive charts) •
        <strong>Streamlit</strong><br>
        Queuing model: M/G/1 vs M/G/c
    </div>
    """, unsafe_allow_html=True)
with col_f2:
    st.page_link("pages/2_📐_Queuing_Theory.py", label="📖 Read the full theory & guide →")
