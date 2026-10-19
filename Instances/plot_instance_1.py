from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Circle, Wedge
import numpy as np


# ============================================================
# Helper functions
# ============================================================

def load_instance(json_path: str | Path) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_xy_array(coord_dict: dict) -> np.ndarray:
    """
    Convert coordinate dictionary of the form
    {"0": [x0,y0], "1": [x1,y1], ...}
    to a sorted numpy array [[x0,y0],[x1,y1],...].
    """
    items = sorted(((int(k), v) for k, v in coord_dict.items()), key=lambda t: t[0])
    return np.array([xy for _, xy in items], dtype=float)


def rescale(values: Iterable[float], out_min: float, out_max: float, use_sqrt: bool = True) -> np.ndarray:
    """
    Rescale numeric values to a display interval.
    sqrt scaling is often visually better for capacities / quantities.
    """
    values = np.asarray(list(values), dtype=float)
    if use_sqrt:
        values = np.sqrt(values)

    vmin = float(values.min())
    vmax = float(values.max())

    if np.isclose(vmin, vmax):
        return np.full_like(values, (out_min + out_max) / 2.0)

    return out_min + (values - vmin) / (vmax - vmin) * (out_max - out_min)


def draw_pie(
    ax,
    center: Tuple[float, float],
    fractions,
    radius,
    colors,
    edgecolor="0.25",
    linewidth=0.7,
    zorder=5,
):
    """
    Draw a pie chart centered at 'center' using Wedge patches.
    """
    total = sum(fractions)
    if total <= 0:
        return

    start_angle = 90.0
    for frac, color in zip(fractions, colors):
        theta = 360.0 * frac / total
        wedge = Wedge(
            center,
            r=radius,
            theta1=start_angle,
            theta2=start_angle + theta,
            facecolor=color,
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=zorder,
        )
        ax.add_patch(wedge)
        start_angle += theta

    # outer boundary for crisp appearance
    ax.add_patch(
        Circle(
            center,
            radius=radius,
            facecolor="none",
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=zorder + 0.1,
        )
    )


def add_instance_info_box(ax, meta: dict):
    p = meta["Parameters"]
    cap = meta["Capacities"]

    info_text = (
        f"Instance: {meta.get('Instance Name', 'n/a')}\n"
        f"Regime: {meta.get('Structural Regime', 'n/a')}\n"
        f"Size class: {meta.get('Size Class', 'n/a')}\n"
        f"City size: {p['city_size_x']:.1f} km × {p['city_size_y']:.1f} km\n"
        f"Grid cell size: {p['grid_cell_size']:.1f} km\n"
        f"Waste density: {p['waste_gen_density']:,} t/km²/year\n"
        f"Generated waste: {cap['Total waste generation (t/year)']:,} t/year\n"
        f"Incinerator radii: {p['incinerator_radius_min']:.0f}–{p['incinerator_radius_max']:.0f} km\n"
        f"Landfill radii: {p['landfill_radius_min']:.0f}–{p['landfill_radius_max']:.0f} km\n"
        f"Cement-kiln radii: {p['cement_radius_min']:.0f}–{p['cement_radius_max']:.0f} km"
    )

    ax.text(
        0.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.92, edgecolor="0.55"),
    )


def add_main_legend(ax):
    handles = [
        Rectangle((0, 0), 1, 1, fill=False, edgecolor="black", linewidth=1.2, label="City boundary"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#d62728", markeredgecolor="0.25",
               markersize=10, label="Generation spots (pie size ∝ waste)"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#ff9f1c", markeredgecolor="0.25",
               markersize=9, label="Transfer stations (size ∝ capacity)"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#2ca02c", markeredgecolor="0.25",
               markersize=10, label="Incinerators (size ∝ capacity)"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#d94841", markeredgecolor="0.25",
               markersize=9, label="Landfills (size ∝ capacity)"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor="#9467bd", markeredgecolor="0.25",
               markersize=10, label="Cement kilns (size ∝ clinker demand)"),
    ]
    leg = ax.legend(handles=handles, loc="lower left", fontsize=8.5, frameon=True, framealpha=0.95)
    return leg


def add_composition_legend(ax):
    handles = [
        Rectangle((0, 0), 1, 1, facecolor="#d62728", edgecolor="0.25", label="High-moisture waste"),
        Rectangle((0, 0), 1, 1, facecolor="#1f77b4", edgecolor="0.25", label="Medium-moisture waste"),
    ]
    leg = ax.legend(
        handles=handles,
        loc="lower right",
        fontsize=8.5,
        frameon=True,
        framealpha=0.95,
        title="Generation composition",
    )
    return leg


# ============================================================
# Main plotting function
# ============================================================

def plot_instance_network(
    json_path: str | Path,
    output_stem: str | Path | None = None,
    add_city_inset: bool = True,
    show_labels: bool = False,
    figsize=(11.5, 10.0),
):
    raw = load_instance(json_path)
    meta = raw["metadata"]
    data = raw["data"]
    p = meta["Parameters"]

    # ----------------------------
    # Coordinates
    # ----------------------------
    G = to_xy_array(data["G_coords"])
    S = to_xy_array(data["S_coords"])
    I = to_xy_array(data["I_coords"])
    L = to_xy_array(data["L_coords"])
    C = to_xy_array(data["C_coords"])

    # ----------------------------
    # Quantities / capacities
    # ----------------------------
    q_gw = np.asarray(data["Q_gw"], dtype=float)   # shape (|G|, 2) = [high-moisture, medium-moisture]
    q_g_total = q_gw.sum(axis=1)

    q_s = np.asarray(data["Q_s"], dtype=float)     # transfer station capacities
    q_i = np.asarray(data["Q_i"], dtype=float)     # incinerator capacities
    q_l = np.asarray(data["Q_l"], dtype=float)     # landfill capacities
    alpha_c = np.asarray(data["alpha_c"], dtype=float)  # cement plant size proxy

    # ----------------------------
    # Visual scaling
    # ----------------------------
    # Pie radius is in coordinate units [km]
    pie_radii = rescale(q_g_total, out_min=2.2, out_max=4.8, use_sqrt=True)

    # Scatter sizes are in pt²
    s_sizes = rescale(q_s, out_min=80, out_max=220, use_sqrt=True)
    i_sizes = rescale(q_i, out_min=90, out_max=240, use_sqrt=True)
    l_sizes = rescale(q_l, out_min=100, out_max=250, use_sqrt=True)
    c_sizes = rescale(alpha_c, out_min=100, out_max=240, use_sqrt=True)

    # ----------------------------
    # City boundary (assuming city centered at origin)
    # ----------------------------
    city_x = float(p["city_size_x"])
    city_y = float(p["city_size_y"])
    city_rect = (-city_x / 2.0, -city_y / 2.0, city_x, city_y)

    # ----------------------------
    # Axis limits
    # ----------------------------
    max_coord = max(
        np.abs(np.vstack([G, S, I, L, C])).max(),
        p["incinerator_radius_max"],
        p["landfill_radius_max"],
        p["cement_radius_max"],
    )
    pad = 15
    lim = max_coord + pad

    # ----------------------------
    # Figure
    # ----------------------------
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title("Synthetic Waste Management Network", fontsize=15)
    ax.set_xlabel("x-coordinate [km]")
    ax.set_ylabel("y-coordinate [km]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.grid(True, alpha=0.25)

    # ----------------------------
    # Radii / annuli
    # ----------------------------
    radius_specs = [
        (p["incinerator_radius_min"], "I min"),
        (p["incinerator_radius_max"], "I max"),
        (p["landfill_radius_min"], "L min"),
        (p["landfill_radius_max"], "L max"),
        (p["cement_radius_min"], "C min"),
        (p["cement_radius_max"], "C max"),
    ]
    for r, label in radius_specs:
        ax.add_patch(
            Circle((0, 0), r, fill=False, linestyle="--", linewidth=1.0, edgecolor="0.55", zorder=1)
        )
        ax.text(r + 2.2, 2.0 + 0.02 * r, label, fontsize=8, color="0.35")

    # ----------------------------
    # City boundary
    # ----------------------------
    ax.add_patch(
        Rectangle(
            (city_rect[0], city_rect[1]),
            city_rect[2],
            city_rect[3],
            fill=False,
            linewidth=1.6,
            edgecolor="black",
            zorder=3,
        )
    )

    # ----------------------------
    # Generation spots as pie charts
    # ----------------------------
    pie_colors = ["#d62728", "#1f77b4"]  # high-moisture, medium-moisture
    for (x, y), fractions, radius in zip(G, q_gw, pie_radii):
        draw_pie(ax, center=(x, y), fractions=fractions, radius=radius, colors=pie_colors, zorder=5)

    # ----------------------------
    # Transfer stations
    # ----------------------------
    ax.scatter(
        S[:, 0], S[:, 1],
        s=s_sizes,
        marker="s",
        c="#ff9f1c",
        edgecolors="0.25",
        linewidths=0.8,
        zorder=6,
    )

    # ----------------------------
    # Disposal facilities
    # ----------------------------
    ax.scatter(
        I[:, 0], I[:, 1],
        s=i_sizes,
        marker="^",
        c="#2ca02c",
        edgecolors="0.25",
        linewidths=0.8,
        zorder=6,
    )
    ax.scatter(
        L[:, 0], L[:, 1],
        s=l_sizes,
        marker="D",
        c="#d94841",
        edgecolors="0.25",
        linewidths=0.8,
        zorder=6,
    )
    ax.scatter(
        C[:, 0], C[:, 1],
        s=c_sizes,
        marker="X",
        c="#9467bd",
        edgecolors="0.25",
        linewidths=0.8,
        zorder=6,
    )

    # ----------------------------
    # Optional labels
    # ----------------------------
    if show_labels:
        for idx, (x, y) in enumerate(G):
            ax.text(x + 1.0, y + 1.0, f"G{idx+1}", fontsize=8)
        for idx, (x, y) in enumerate(S):
            ax.text(x + 1.0, y + 1.0, f"S{idx+1}", fontsize=8)
        for idx, (x, y) in enumerate(I):
            ax.text(x + 1.0, y + 1.0, f"I{idx+1}", fontsize=8)
        for idx, (x, y) in enumerate(L):
            ax.text(x + 1.0, y + 1.0, f"L{idx+1}", fontsize=8)
        for idx, (x, y) in enumerate(C):
            ax.text(x + 1.0, y + 1.0, f"C{idx+1}", fontsize=8)

    # ----------------------------
    # Legends
    # ----------------------------
    leg1 = add_main_legend(ax)
    ax.add_artist(leg1)
    leg2 = add_composition_legend(ax)
    ax.add_artist(leg2)

    # ----------------------------
    # Instance information box
    # ----------------------------
    add_instance_info_box(ax, meta)

    # ----------------------------
    # Optional city zoom inset
    # ----------------------------
    if add_city_inset:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

        axins = inset_axes(ax, width="34%", height="34%", loc="center right", borderpad=2.0)

        x0, y0, w, h = city_rect
        axins.set_xlim(x0 - 3, x0 + w + 3)
        axins.set_ylim(y0 - 3, y0 + h + 3)
        axins.set_aspect("equal", adjustable="box")
        axins.grid(True, alpha=0.22)
        axins.set_title("City-area zoom", fontsize=9)

        axins.add_patch(
            Rectangle((x0, y0), w, h, fill=False, linewidth=1.3, edgecolor="black", zorder=3)
        )

        for (x, y), fractions, radius in zip(G, q_gw, pie_radii):
            draw_pie(axins, center=(x, y), fractions=fractions, radius=radius * 0.75, colors=pie_colors, zorder=5)

        axins.scatter(
            S[:, 0], S[:, 1],
            s=s_sizes * 0.55,
            marker="s",
            c="#ff9f1c",
            edgecolors="0.25",
            linewidths=0.8,
            zorder=6,
        )

        # often useful to also show incinerators in the zoom
        axins.scatter(
            I[:, 0], I[:, 1],
            s=i_sizes * 0.45,
            marker="^",
            c="#2ca02c",
            edgecolors="0.25",
            linewidths=0.8,
            zorder=6,
        )

        axins.tick_params(labelsize=8)
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.45", lw=0.9)

    # ----------------------------
    # Save figure
    # ----------------------------
    if output_stem is None:
        output_stem = Path(json_path).with_suffix("")
    output_stem = Path(output_stem)

    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return png_path, pdf_path


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser(description="Plot a synthetic waste-management instance from JSON.")
    # parser.add_argument("json_path", type=str, help="Path to instance JSON file")
    # parser.add_argument("--output-stem", type=str, default=None, help="Output path without extension")
    # parser.add_argument("--no-inset", action="store_true", help="Disable city zoom inset")
    # parser.add_argument("--labels", action="store_true", help="Show labels G1,S1,...")
    # args = parser.parse_args()

    instance_sizeclass = "medium"
    instance_regime = "balanced"
    seed = 19
    # print(instance_sizeclass[0].upper())
    
    instance_path = Path(__file__).parent / "Instances_CaseStudy_V3" / instance_sizeclass / instance_regime
    instance_file = instance_path / f"TI_{instance_sizeclass[0].upper()}_{instance_regime}_S0{seed}.json"

    output_path = Path(__file__).parent / "figures" / f"TI_{instance_sizeclass[0].upper()}_{instance_regime}_S0{seed}"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    png_path, pdf_path = plot_instance_network(
        json_path=instance_file,
        output_stem=output_path,
        add_city_inset=True,
        show_labels=False,
    )

    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")