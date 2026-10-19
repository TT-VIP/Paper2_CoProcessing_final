from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Circle, Wedge
from matplotlib.legend_handler import HandlerBase
import numpy as np

#region Helper Functions
# ============================================================
# Helper functions
# ============================================================
class HandlerHalfCircle(HandlerBase):
    """Legend handler that draws a circle split vertically: left red, right blue."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        center = (width / 2 - xdescent, height / 2 - ydescent)
        radius = 0.9 * min(width, height)

        # Left half: red
        left = Wedge(center, radius, 90, 270, facecolor="#d62728", edgecolor="none")

        # Right half: blue
        right = Wedge(center, radius, 270, 450, facecolor="#1f77b4", edgecolor="none")

        # Circular outline
        outline = Circle(center, radius, facecolor="none", edgecolor="0.25", linewidth=0.7)

        artists = [left, right, outline]

        for artist in artists:
            artist.set_transform(trans)

        return artists

def load_instance(json_path: str | Path) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_xy_array(coord_dict: dict) -> np.ndarray:
    items = sorted(((int(k), v) for k, v in coord_dict.items()), key=lambda t: t[0])
    return np.array([xy for _, xy in items], dtype=float)


def rescale(values: Iterable[float], out_min: float, out_max: float, use_sqrt: bool = True) -> np.ndarray:
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
        # f"Instance: {meta.get('Instance Name', 'n/a')}\n"
        f"Seed: {meta.get('Seed', 'n/a')}\n"
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
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.94, edgecolor="0.55"),
    )


def add_main_legend(ax):
    gen_handle = Line2D([0], [0], marker="o", color="none", markerfacecolor="#d62728", markeredgecolor="0.25",
        markersize=10, label="Generation spots (pie size ∝ waste)")

    handles = [
        Rectangle((0, 0), 1, 1, fill=False, edgecolor="black", linewidth=1.2, label="City boundary"),
        gen_handle,
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#ff9f1c", markeredgecolor="0.25",
            #    markersize=9, label="Transfer stations (size ∝ capacity; zoom only)"),
               markersize=9, label="Transfer stations (size ∝ capacity)"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#2ca02c", markeredgecolor="0.25",
               markersize=10, label="Incinerators (size ∝ capacity)"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#d94841", markeredgecolor="0.25",
               markersize=9, label="Landfills (size ∝ capacity)"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor="#9467bd", markeredgecolor="0.25",
               markersize=10, label="Cement kilns (size ∝ clinker demand)"),
    ]
    handler_map = {gen_handle: HandlerHalfCircle()}

    return ax.legend(handles=handles, handler_map=handler_map, loc="lower left", fontsize=8.5, frameon=True, framealpha=0.95)


def add_composition_legend(ax):
    handles = [
        Rectangle((0, 0), 1, 1, facecolor="#d62728", edgecolor="0.25", label="High-moisture waste"),
        Rectangle((0, 0), 1, 1, facecolor="#1f77b4", edgecolor="0.25", label="Medium-moisture waste"),
    ]
    return ax.legend(
        handles=handles,
        loc="lower right",
        fontsize=8.5,
        frameon=True,
        framealpha=0.95,
        title="Generation composition",
    )

def draw_annulus_fill(ax, r_min, r_max, color, alpha=0.20, zorder=0.4):
    """
    Draw a full annulus (ring) centered at the origin.
    """
    ring = Wedge(
        center=(0, 0),
        r=r_max,
        theta1=0,
        theta2=360,
        width=r_max - r_min,
        facecolor=color,
        edgecolor="none",
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(ring)


def draw_overlap_hatch(
    ax,
    r_min,
    r_max,
    color1,
    color2,
    hatch1="///",
    hatch2="\\\\\\",
    alpha=1.0,
    zorder=0.8,
):
    """
    Draw a striped overlap region for two overlapping annuli.
    Uses two wedge patches with opposite hatch directions.
    """
    # optional faint white background to make hatch clearer
    bg = Wedge(
        center=(0, 0),
        r=r_max,
        theta1=0,
        theta2=360,
        width=r_max - r_min,
        facecolor="white",
        edgecolor="none",
        alpha=0.35,
        zorder=zorder,
    )
    ax.add_patch(bg)

    hatch_patch_1 = Wedge(
        center=(0, 0),
        r=r_max,
        theta1=0,
        theta2=360,
        width=r_max - r_min,
        facecolor="none",
        edgecolor=color1,
        hatch=hatch1,
        linewidth=0.0,
        alpha=alpha,
        zorder=zorder + 0.01,
    )
    ax.add_patch(hatch_patch_1)

    hatch_patch_2 = Wedge(
        center=(0, 0),
        r=r_max,
        theta1=0,
        theta2=360,
        width=r_max - r_min,
        facecolor="none",
        edgecolor=color2,
        hatch=hatch2,
        linewidth=0.0,
        alpha=alpha,
        zorder=zorder + 0.02,
    )
    ax.add_patch(hatch_patch_2)
#endregion


#region Plot Function
# ============================================================
# Main plotting function
# ============================================================

def plot_instance_network(
    json_path: str | Path,
    output_stem: str | Path | None = None,
    figsize=(11.5, 10.0),
    show_labels: bool = False,
):
    raw = load_instance(json_path)
    meta = raw["metadata"]
    data = raw["data"]
    p = meta["Parameters"]

    # --------------------------------------------------------
    # Coordinates
    # --------------------------------------------------------
    G = to_xy_array(data["G_coords"])
    S = to_xy_array(data["S_coords"])
    I = to_xy_array(data["I_coords"])
    L = to_xy_array(data["L_coords"])
    C = to_xy_array(data["C_coords"])

    # --------------------------------------------------------
    # Quantities / capacities
    # --------------------------------------------------------
    q_gw = np.asarray(data["Q_gw"], dtype=float)       # [high-moisture, medium-moisture]
    q_g_total = q_gw.sum(axis=1)

    q_s = np.asarray(data["Q_s"], dtype=float)
    q_i = np.asarray(data["Q_i"], dtype=float)
    q_l = np.asarray(data["Q_l"], dtype=float)
    alpha_c = np.asarray(data["alpha_c"], dtype=float)  # size proxy for cement plants

    # --------------------------------------------------------
    # Visual scales
    # --------------------------------------------------------
    pie_radii = rescale(q_g_total, out_min=1.6, out_max=3.2, use_sqrt=True)
    s_sizes = rescale(q_s, out_min=70, out_max=180, use_sqrt=True)
    i_sizes = rescale(q_i, out_min=90, out_max=240, use_sqrt=True)
    l_sizes = rescale(q_l, out_min=100, out_max=250, use_sqrt=True)
    c_sizes = rescale(alpha_c, out_min=100, out_max=240, use_sqrt=True)

    # --------------------------------------------------------
    # City geometry
    # --------------------------------------------------------
    city_x = float(p["city_size_x"])
    city_y = float(p["city_size_y"])
    cell = float(p["grid_cell_size"])

    # Main plot stays in global coordinates centered at origin
    x0_global = -city_x / 2.0
    y0_global = -city_y / 2.0
    city_rect = (x0_global, y0_global, city_x, city_y)

    # --------------------------------------------------------
    # Axis limits
    # --------------------------------------------------------
    max_coord = max(
        np.abs(np.vstack([G, S, I, L, C])).max(),
        p["incinerator_radius_max"],
        p["landfill_radius_max"],
        p["cement_radius_max"],
    )
    lim = max_coord + 18

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title("Synthetic Waste Management Network", fontsize=15)
    ax.set_xlabel("x-coordinate [km]")
    ax.set_ylabel("y-coordinate [km]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.grid(True, alpha=0.25)

    # --------------------------------------------------------
    # Facility annuli
    # --------------------------------------------------------
    # facility colors
    color_I = "#d94841"   # incinerators
    color_L = "#2ca02c"   # landfills
    color_C = "#9467bd"   # cement kilns

    # annulus definitions
    annuli = {
        "I": {"rmin": p["incinerator_radius_min"], "rmax": p["incinerator_radius_max"], "color": color_I},
        "L": {"rmin": p["landfill_radius_min"],    "rmax": p["landfill_radius_max"],    "color": color_L},
        "C": {"rmin": p["cement_radius_min"],      "rmax": p["cement_radius_max"],      "color": color_C},
    }

    # 1) draw light transparent annulus fills
    for key, spec in annuli.items():
        draw_annulus_fill(
            ax,
            r_min=spec["rmin"],
            r_max=spec["rmax"],
            color=spec["color"],
            alpha=0.10,
            zorder=0.3,
        )

    # 2) draw striped overlap regions
    facility_pairs = [("I", "L"), ("I", "C"), ("L", "C")]

    for a, b in facility_pairs:
        r_overlap_min = max(annuli[a]["rmin"], annuli[b]["rmin"])
        r_overlap_max = min(annuli[a]["rmax"], annuli[b]["rmax"])

        if r_overlap_min < r_overlap_max:
            draw_overlap_hatch(
                ax,
                r_min=r_overlap_min,
                r_max=r_overlap_max,
                color1=annuli[a]["color"],
                color2=annuli[b]["color"],
                hatch1="///",
                hatch2="\\\\\\",
                alpha=1.0,
                zorder=0.7,
            )

    # 3) draw black dotted boundary circles on top
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
            Circle(
                (0, 0),
                r,
                fill=False,
                linestyle="--",
                linewidth=1.1,
                edgecolor="black",
                alpha=0.8,
                zorder=1.2,
            )
        )

        # ax.text(
        #     r + 2.2,
        #     2.0 + 0.02 * r,
        #     label,
        #     fontsize=8,
        #     color="0.25",
        #     zorder=1.3,
        # )

    # --------------------------------------------------------
    # Main plot:
    # show city boundary and disposal facilities only
    # NO generation spots and NO transfer stations here
    # --------------------------------------------------------
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

    ax.scatter(
        I[:, 0], I[:, 1],
        s=i_sizes,
        marker="^",
        c="#d94841",
        edgecolors="0.25",
        linewidths=0.8,
        zorder=6,
    )
    ax.scatter(
        L[:, 0], L[:, 1],
        s=l_sizes,
        marker="D",
        c="#2ca02c",
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

    if show_labels:
        for idx, (x, y) in enumerate(I):
            ax.text(x + 1.0, y + 1.0, f"I{idx+1}", fontsize=8)
        for idx, (x, y) in enumerate(L):
            ax.text(x + 1.0, y + 1.0, f"L{idx+1}", fontsize=8)
        for idx, (x, y) in enumerate(C):
            ax.text(x + 1.0, y + 1.0, f"C{idx+1}", fontsize=8)

    # --------------------------------------------------------
    # Legends and info box
    # --------------------------------------------------------
    leg1 = add_main_legend(ax)
    ax.add_artist(leg1)
    leg2 = add_composition_legend(ax)
    ax.add_artist(leg2)
    add_instance_info_box(ax, meta)

    # --------------------------------------------------------
    # Inset in TOP RIGHT
    # local city coordinates from 0..city_size
    # so pie-chart centers are at 5, 15, 25, ...
    # --------------------------------------------------------
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    axins = inset_axes(ax, width="31%", height="34%", loc="upper right", borderpad=2.0)

    # transform city coordinates from global to local coordinates
    G_local = G.copy()
    G_local[:, 0] -= x0_global
    G_local[:, 1] -= y0_global

    S_local = S.copy()
    S_local[:, 0] -= x0_global
    S_local[:, 1] -= y0_global

    axins.set_xlim(0, city_x)
    axins.set_ylim(0, city_y)
    axins.set_aspect("equal", adjustable="box")
    axins.set_title("City-area zoom", fontsize=9)

    # grid lines at cell boundaries: 0, 10, 20, 30, ...
    axins.set_xticks(np.arange(0, city_x + 0.1, cell))
    axins.set_yticks(np.arange(0, city_y + 0.1, cell))
    axins.grid(True, which="major", alpha=0.35, linewidth=0.8)

    axins.add_patch(
        Rectangle((0, 0), city_x, city_y, fill=False, linewidth=1.3, edgecolor="black", zorder=3)
    )

    # generation pies in inset only
    pie_colors = ["#d62728", "#1f77b4"]  # high-moisture, medium-moisture
    for (x, y), fractions, radius in zip(G_local, q_gw, pie_radii):
        draw_pie(axins, center=(x, y), fractions=fractions, radius=radius, colors=pie_colors, zorder=5)

    # transfer stations in inset only
    axins.scatter(
        S_local[:, 0], S_local[:, 1],
        s=s_sizes * 0.75,
        marker="s",
        c="#ff9f1c",
        edgecolors="0.25",
        linewidths=0.8,
        zorder=6,
    )

    axins.tick_params(labelsize=8)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    if output_stem is None:
        output_stem = Path(json_path).with_suffix("")
    output_stem = Path(output_stem)

    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return png_path, pdf_path
#endregion


#region Main Method
# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    import argparse

    # parser = argparse.ArgumentParser(description="Plot a synthetic waste-management instance from JSON.")
    # parser.add_argument("json_path", type=str, help="Path to instance JSON file")
    # parser.add_argument("--output-stem", type=str, default=None, help="Output path without extension")
    # parser.add_argument("--labels", action="store_true", help="Show labels I1,L1,C1,...")
    # args = parser.parse_args()

    instance_sizeclass = "medium"
    instance_regime = "balanced"
    seed = 19
    
    instance_path = Path(__file__).parent / "Instances_CaseStudy_V3" / instance_sizeclass / instance_regime
    instance_file = instance_path / f"TI_{instance_sizeclass[0].upper()}_{instance_regime}_S0{seed}.json"

    output_path = Path(__file__).parent / "figures" / f"TI_{instance_sizeclass[0].upper()}_{instance_regime}_S0{seed}"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    png_path, pdf_path = plot_instance_network(
        json_path=instance_file,
        output_stem=output_path,
        show_labels=False,
    )

    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")
    #endregion