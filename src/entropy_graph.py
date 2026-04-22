"""EntropyTransferGraph — turn DTW distances into an information-flow network.

Pipeline
--------
1. S = exp(-λ D)              similarity from DTW distances
2. p_i = S_i / Σ_j S_ij        row-normalised distribution per asset
3. H(i) = -Σ p_ij log p_ij      Shannon entropy per asset
4. M_ij = 1 - JSD(p_i, p_j)    Jensen-Shannon-based transfer matrix
5. Δ = 1 - M  →  embed via MDS or Spectral Embedding
6. Plot as a weighted network (networkx)

JSD is symmetric, bounded in [0, log 2] (with log base e), and a metric in
its square-root form. Ref: https://en.wikipedia.org/wiki/Jensen-Shannon_divergence
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


def _shannon(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (natural log) in [0, log 2]."""
    m = 0.5 * (p + q)
    def _kl(a, b):
        mask = (a > 0) & (b > 0)
        return float((a[mask] * np.log(a[mask] / b[mask])).sum())
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


@dataclass
class EntropyTransferGraph:
    distance_matrix: pd.DataFrame
    lam: Optional[float] = None  # if None, auto-set so median(S) ≈ 0.5
    similarity_: Optional[pd.DataFrame] = field(default=None, init=False)
    transfer_matrix_: Optional[pd.DataFrame] = field(default=None, init=False)
    entropy_: Optional[pd.Series] = field(default=None, init=False)
    embedding_: Optional[pd.DataFrame] = field(default=None, init=False)

    # ----------------------------------------------------------- embeddings
    def compute_embeddings(self, method: str = "mds", n_components: int = 2) -> dict:
        D = self.distance_matrix.values.astype(float)
        cols = list(self.distance_matrix.columns)

        # 1. λ choice
        if self.lam is None:
            offdiag = D[~np.eye(len(D), dtype=bool)]
            med = np.median(offdiag) if offdiag.size else 1.0
            self.lam = float(np.log(2) / med) if med > 0 else 1.0

        # 2. similarity
        S = np.exp(-self.lam * D)
        np.fill_diagonal(S, 1.0)
        self.similarity_ = pd.DataFrame(S, index=cols, columns=cols)

        # 3. row-normalised distributions
        P = S / S.sum(axis=1, keepdims=True)

        # 4. Shannon entropy per asset
        self.entropy_ = pd.Series([_shannon(P[i]) for i in range(len(cols))], index=cols, name="H")

        # 5. transfer matrix M_ij = 1 - JSD(p_i, p_j)
        n = len(cols)
        M = np.ones((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                jsd = _jsd(P[i], P[j])
                M[i, j] = M[j, i] = 1.0 - jsd
        self.transfer_matrix_ = pd.DataFrame(M, index=cols, columns=cols)

        # 6. embed Δ = 1 - M
        Delta = 1.0 - M
        np.fill_diagonal(Delta, 0.0)
        Delta = np.clip(Delta, 0.0, None)

        if method == "mds":
            from sklearn.manifold import MDS

            emb = MDS(
                n_components=n_components,
                dissimilarity="precomputed",
                normalized_stress="auto",
                random_state=0,
            ).fit_transform(Delta)
        elif method == "spectral":
            from sklearn.manifold import SpectralEmbedding

            # SpectralEmbedding expects an *affinity* (similarity) matrix
            emb = SpectralEmbedding(
                n_components=n_components,
                affinity="precomputed",
                random_state=0,
            ).fit_transform(M)
        else:
            raise ValueError(f"Unknown embedding method: {method}")

        self.embedding_ = pd.DataFrame(
            emb, index=cols, columns=[f"dim{i+1}" for i in range(n_components)]
        )
        return {
            "similarity": self.similarity_,
            "transfer": self.transfer_matrix_,
            "entropy": self.entropy_,
            "embedding": self.embedding_,
            "lambda": self.lam,
        }

    # ---------------------------------------------------------------- plot
    def plot_graph(self, threshold: float = 0.5, ax=None):
        """Network graph: nodes = assets, edges weighted by transfer matrix."""
        import matplotlib.pyplot as plt
        import networkx as nx

        if self.transfer_matrix_ is None or self.embedding_ is None:
            self.compute_embeddings()

        M = self.transfer_matrix_
        G = nx.Graph()
        for c in M.columns:
            G.add_node(c)
        for i, a in enumerate(M.columns):
            for b in M.columns[i + 1 :]:
                w = float(M.loc[a, b])
                if w >= threshold:
                    G.add_edge(a, b, weight=w)

        # eigenvector centrality for node sizing (fallback to degree)
        try:
            cent = nx.eigenvector_centrality_numpy(G, weight="weight")
        except Exception:
            cent = dict(G.degree(weight="weight"))

        pos = {c: self.embedding_.loc[c, ["dim1", "dim2"]].values for c in M.columns}
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
        sizes = [800 + 2000 * cent.get(c, 0) for c in G.nodes]
        weights = [G[u][v]["weight"] for u, v in G.edges]
        nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color="#4a90e2", alpha=0.9, ax=ax)
        nx.draw_networkx_edges(G, pos, width=[3 * w for w in weights], alpha=0.5, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
        ax.set_title("Entropy Transfer Graph")
        ax.set_axis_off()
        return ax, G
