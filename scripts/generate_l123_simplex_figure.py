#!/usr/bin/env python3
"""Generate an illustrative L1/L2/L3 simplex trajectory figure.

This script creates a conceptual visualization (not a protocol-grade benchmark)
for report narration:
- L1: random walk / non-coherent drift
- L2: damped oscillation toward center
- L3: sustained coherent rotation
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _normalize_simplex(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 1e-6, None)
    return x / x.sum(axis=1, keepdims=True)


def _build_state(theta: np.ndarray, amplitude: np.ndarray, noise_sigma: float, rng: np.random.Generator) -> np.ndarray:
    phases = np.array([0.0, -2.0 * np.pi / 3.0, 2.0 * np.pi / 3.0])
    base = 1.0 / 3.0 + amplitude[:, None] * np.cos(theta[:, None] + phases[None, :])
    if noise_sigma > 0.0:
        base = base + rng.normal(0.0, noise_sigma, size=base.shape)
    return _normalize_simplex(base)


def simulate_l1(n_steps: int, rng: np.random.Generator) -> np.ndarray:
    theta = np.cumsum(rng.normal(0.0, 0.48, size=n_steps))
    amp = np.clip(0.045 + rng.normal(0.0, 0.025, size=n_steps), 0.0, 0.11)
    return _build_state(theta, amp, noise_sigma=0.028, rng=rng)


def simulate_l2(n_steps: int, rng: np.random.Generator) -> np.ndarray:
    theta = np.linspace(0.0, 9.0 * np.pi, n_steps)
    amp = 0.22 * np.exp(-np.linspace(0.0, 2.7, n_steps)) + 0.015
    return _build_state(theta, amp, noise_sigma=0.010, rng=rng)


def simulate_l3(n_steps: int, rng: np.random.Generator) -> np.ndarray:
    theta = np.linspace(0.0, 12.0 * np.pi, n_steps)
    amp = np.full(n_steps, 0.175) + 0.004 * np.sin(np.linspace(0.0, 3.0 * np.pi, n_steps))
    return _build_state(theta, amp, noise_sigma=0.007, rng=rng)


def simplex_vertices() -> np.ndarray:
    # strategy order: aggressive, defensive, balanced
    return np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, np.sqrt(3.0) / 2.0],
        ]
    )


def project_simplex(states: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    return states @ vertices


def draw_panel(ax: plt.Axes, states: np.ndarray, title: str) -> None:
    verts = simplex_vertices()
    coords = project_simplex(states, verts)

    tri = np.vstack([verts, verts[0]])
    ax.plot(tri[:, 0], tri[:, 1], color="#5f6f5f", linewidth=1.2)

    ax.text(verts[0, 0] - 0.04, verts[0, 1] - 0.04, "A", fontsize=10)
    ax.text(verts[1, 0] + 0.02, verts[1, 1] - 0.04, "D", fontsize=10)
    ax.text(verts[2, 0] - 0.01, verts[2, 1] + 0.03, "B", fontsize=10)

    t = np.linspace(0.0, 1.0, len(coords))
    ax.scatter(coords[:, 0], coords[:, 1], c=t, cmap="viridis", s=9, alpha=0.85)
    ax.plot(coords[:, 0], coords[:, 1], color="#2f3f2f", linewidth=0.9, alpha=0.55)

    ax.scatter(coords[0, 0], coords[0, 1], color="#2e7d32", s=42, label="start", zorder=5)
    ax.scatter(coords[-1, 0], coords[-1, 1], color="#c62828", s=52, marker="x", label="end", zorder=6)

    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.10, np.sqrt(3.0) / 2.0 + 0.10)
    ax.axis("off")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "docs" / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(20260415)
    n_steps = 260

    l1 = simulate_l1(n_steps, rng)
    l2 = simulate_l2(n_steps, rng)
    l3 = simulate_l3(n_steps, rng)

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4), constrained_layout=True)
    draw_panel(axes[0], l1, "L1: random drift")
    draw_panel(axes[1], l2, "L2: damped oscillation")
    draw_panel(axes[2], l3, "L3: coherent rotation")

    fig.suptitle("Illustrative simplex trajectories for L1/L2/L3", fontsize=13, y=1.03)

    png_path = out_dir / "l123_simplex_simulation.png"
    svg_path = out_dir / "l123_simplex_simulation.svg"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print(png_path)
    print(svg_path)


if __name__ == "__main__":
    main()
