"""Generate presentation-specific figures from the notebook outputs.

Produces in ``outputs/presentation/`` :
    - ``fig_leadlag_network.png``     : directed lead-lag graph (proxy network)
    - ``fig_entropy_bars.png``        : Shannon entropies ranked
    - ``fig_corr_heatmap.png``        : annotated correlation heatmap
    - ``fig_dtw_distance_heatmap.png``: annotated DTW distance heatmap
    - ``fig_sensitivity_radius.png``  : sensitivity of DTW distance & lag to the
                                        Sakoe-Chiba radius on Gold→Silver
    - ``fig_prices_normalised.png``   : all 8 series rebased to 100

Run::

    python scripts/build_presentation_figs.py
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
PRES = os.path.join(OUT, "presentation")
os.makedirs(PRES, exist_ok=True)

CORR = pd.read_csv(os.path.join(OUT, "03_corr.csv"), index_col=0)
DIST = pd.read_csv(os.path.join(OUT, "04_dtw_distance.csv"), index_col=0)
LAG = pd.read_csv(os.path.join(OUT, "04_lag_matrix.csv"), index_col=0)
ENT = pd.read_csv(os.path.join(OUT, "06_entropy.csv"), index_col=0).iloc[:, 0]
PRICES = pd.read_csv(os.path.join(OUT, "01_prices.csv"), index_col=0, parse_dates=True)

ASSETS = list(CORR.columns)
CLUSTER = {
    "Gold": "Metals", "Silver": "Metals",
    "Oil": "Energy",
    "EURUSD": "FX", "JPYUSD": "FX",
    "DXY": "USD/Rates", "UST10Y": "USD/Rates",
    "Bund10Y": "EUR/Rates",
}
COLOR = {
    "Metals": "#E0A800",
    "Energy": "#444444",
    "FX": "#1F77B4",
    "USD/Rates": "#2CA02C",
    "EUR/Rates": "#D62728",
}


# --------------------------------------------------------------------------- 1
def fig_prices_normalised() -> None:
    base = PRICES.iloc[0]
    norm = 100.0 * PRICES.divide(base)
    fig, ax = plt.subplots(figsize=(11, 5))
    for col in norm.columns:
        ax.plot(norm.index, norm[col], lw=1.2, label=col,
                color=COLOR[CLUSTER[col]], alpha=0.85)
    ax.set_title("Prices rebased to 100 at 2020-01-01  •  8 assets, 5 years",
                 fontsize=12, weight="bold")
    ax.set_ylabel("Index (base 100)")
    ax.axhline(100, color="grey", lw=0.6, ls="--")
    ax.legend(ncol=4, fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(PRES, "fig_prices_normalised.png"), dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- 2
def fig_corr_heatmap() -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(CORR.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(ASSETS)))
    ax.set_yticks(range(len(ASSETS)))
    ax.set_xticklabels(ASSETS, rotation=45, ha="right")
    ax.set_yticklabels(ASSETS)
    for i in range(len(ASSETS)):
        for j in range(len(ASSETS)):
            v = CORR.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if abs(v) > 0.55 else "black", fontsize=8)
    ax.set_title("Pearson correlation — daily log-returns", weight="bold")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(PRES, "fig_corr_heatmap.png"), dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- 3
def fig_dtw_distance_heatmap() -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(DIST.values, cmap="viridis_r")
    ax.set_xticks(range(len(ASSETS)))
    ax.set_yticks(range(len(ASSETS)))
    ax.set_xticklabels(ASSETS, rotation=45, ha="right")
    ax.set_yticklabels(ASSETS)
    for i in range(len(ASSETS)):
        for j in range(len(ASSETS)):
            v = DIST.values[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    color="white" if v > 40 else "black", fontsize=8)
    ax.set_title("DTW distance matrix  (smaller = closer dynamics)",
                 weight="bold")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(PRES, "fig_dtw_distance_heatmap.png"), dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- 4
def fig_leadlag_network() -> None:
    """Directed graph where an arrow A -> B means A leads B (lag > threshold)."""
    G = nx.DiGraph()
    for a in ASSETS:
        G.add_node(a, cluster=CLUSTER[a])

    threshold = 3.0  # days
    for i, a in enumerate(ASSETS):
        for j, b in enumerate(ASSETS):
            if i == j:
                continue
            lag = LAG.loc[a, b]
            if lag > threshold:
                G.add_edge(a, b, weight=float(lag))

    fig, ax = plt.subplots(figsize=(13, 8))
    pos = nx.spring_layout(G, seed=42, k=1.8, iterations=200)

    for cluster_name, color in COLOR.items():
        nodes = [n for n, d in G.nodes(data=True) if d["cluster"] == cluster_name]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=color,
                               node_size=2200, alpha=0.9, ax=ax,
                               label=cluster_name, edgecolors="black", linewidths=1.5)

    weights = np.array([d["weight"] for _, _, d in G.edges(data=True)])
    widths = 0.5 + 3.0 * (weights - weights.min()) / (np.ptp(weights) + 1e-9)
    nx.draw_networkx_edges(
        G, pos, width=widths, alpha=0.7,
        edge_color=weights, edge_cmap=plt.cm.plasma,
        arrowsize=22, arrowstyle="-|>",
        connectionstyle="arc3,rad=0.08",
        node_size=2200, ax=ax,
    )
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", ax=ax)

    edge_labels = {(u, v): f"{d['weight']:.1f}d" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7,
                                 ax=ax, bbox=dict(boxstyle="round,pad=0.2",
                                                  fc="white", ec="none", alpha=0.7))
    ax.set_title(
        f"Proxy lead-lag network  •  edge A→B means A leads B by ≥ {threshold:.0f} days  "
        f"(label = mean lag from DTW path)",
        weight="bold", fontsize=12,
    )
    ax.legend(loc="center left", fontsize=11, frameon=True,
              bbox_to_anchor=(1.01, 0.5), title="Cluster",
              title_fontsize=11)
    ax.axis("off")
    ax.margins(0.15)
    fig.tight_layout()
    fig.savefig(os.path.join(PRES, "fig_leadlag_network.png"), dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- 5
def fig_entropy_bars() -> None:
    e = ENT.sort_values()
    colors = [COLOR[CLUSTER[a]] for a in e.index]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(e.index, e.values, color=colors, edgecolor="black")
    for bar, val in zip(bars, e.values):
        ax.text(val + 0.0005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    ax.set_xlim(e.min() - 0.005, e.max() + 0.01)
    ax.invert_yaxis()
    ax.set_xlabel("Shannon entropy H of similarity profile  (low = stronger transmitter)")
    ax.set_title("Information-theoretic leadership ranking", weight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PRES, "fig_entropy_bars.png"), dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- 6
def fig_sensitivity_radius() -> None:
    """Compute DTW distance & inferred lag for Gold→Silver as a function of
    the Sakoe-Chiba radius. Replicates Task 5's tuning rationale visually.
    """
    from dtaidistance import dtw

    a = PRICES["Gold"].values.astype(float)
    b = PRICES["Silver"].values.astype(float)
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()

    radii = [3, 5, 8, 12, 16, 20, 26, 32, 40, 60, 80, 120]
    dists, lags = [], []
    for r in radii:
        d = dtw.distance(a, b, window=r)
        path = dtw.warping_path(a, b, window=r)
        arr = np.asarray(path, dtype=float)
        dists.append(d)
        lags.append(float(np.mean(arr[:, 1] - arr[:, 0])))

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    color1 = "#1f77b4"
    ax1.plot(radii, dists, "o-", color=color1, lw=2, label="DTW distance")
    ax1.set_xlabel("Sakoe-Chiba radius  (max allowed warping, in days)")
    ax1.set_ylabel("DTW distance", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    color2 = "#d62728"
    ax2.plot(radii, lags, "s--", color=color2, lw=2, label="Mean path lag (days)")
    ax2.set_ylabel("Inferred lag  Gold → Silver  (days)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.axvline(26, color="black", lw=1, ls=":", alpha=0.7)
    ax1.text(26, ax1.get_ylim()[1] * 0.95, "Optuna best = 26",
             rotation=90, va="top", ha="right", fontsize=9, color="black")

    fig.suptitle("Sensitivity analysis — Sakoe-Chiba radius  (Gold → Silver pair)",
                 weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PRES, "fig_sensitivity_radius.png"), dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- 7
def fig_leader_summary() -> None:
    """Bar plot: for each asset, how many other assets it leads (mean lag > 3)."""
    threshold = 3.0
    led = (LAG > threshold).sum(axis=1)
    follows = (LAG > threshold).sum(axis=0)
    df = pd.DataFrame({"# Followers (leads)": led, "# Leaders (follows)": follows})
    df = df.sort_values("# Followers (leads)", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    df.plot(kind="barh", ax=ax, color=["#2ca02c", "#d62728"], edgecolor="black")
    ax.set_title(f"Leadership balance  (threshold = {threshold:.0f}-day mean lag)",
                 weight="bold")
    ax.set_xlabel("Number of asset pairs")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PRES, "fig_leader_summary.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    print("→ writing to", PRES)
    fig_prices_normalised();          print("  ✓ fig_prices_normalised")
    fig_corr_heatmap();                print("  ✓ fig_corr_heatmap")
    fig_dtw_distance_heatmap();        print("  ✓ fig_dtw_distance_heatmap")
    fig_leadlag_network();             print("  ✓ fig_leadlag_network")
    fig_entropy_bars();                print("  ✓ fig_entropy_bars")
    fig_leader_summary();              print("  ✓ fig_leader_summary")
    try:
        fig_sensitivity_radius();      print("  ✓ fig_sensitivity_radius")
    except Exception as exc:
        print(f"  ! skip sensitivity ({exc})")
    print("done.")
