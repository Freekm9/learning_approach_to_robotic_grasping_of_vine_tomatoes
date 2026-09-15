#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot a top-down heatmap of test_crate_reachability.py's logged sweep.

Reads experiments/crate_reachability/reachability_log.csv (the same file
test_crate_reachability.py writes, and visualize_crate_reachability.py reads
for its RViz markers) and saves a static top-down (x, y) figure: one square
per tested ring point, colored by whether it was reachable straight-down,
reachable only via a tilted approach, or unreachable at every yaw/tilt
tried -- the same three-way status test_crate_reachability's --visualize
and visualize_crate_reachability.py use, so the colors mean the same thing
everywhere. Each cell is also labeled with its "reachable/total" count,
since color alone shouldn't have to carry that.

This only reads the CSV -- no rospy/live scene needed, so it works without
Gazebo or MoveIt running, and it's faster to iterate on than relaunching
RViz for every sweep.

Usage:
    rosrun grasp plot_crate_reachability_heatmap.py
    rosrun grasp plot_crate_reachability_heatmap.py --latest-run-only
    rosrun grasp plot_crate_reachability_heatmap.py --out /tmp/heatmap.png --show
"""
import argparse
import os

import matplotlib
import numpy as np
import pandas as pd
import rospkg

# Okabe-Ito colorblind-safe triple, reused for the same three-way status
# test_crate_reachability.py's --visualize / visualize_crate_reachability.py use.
GREEN = "#009E73"   # reachable straight-down (tilt 0, 0) for at least one yaw
AMBER = "#E69F00"   # reachable, but only via a tilted approach
RED = "#D55E00"     # unreachable at every yaw/tilt tried
GRID = "#B0B0B0"


def get_log_path():
    rospack = rospkg.RosPack()
    grasp_pkg_dir = rospack.get_path('grasp')
    catkin_ws_dir = os.path.dirname(os.path.dirname(os.path.dirname(grasp_pkg_dir)))
    return os.path.join(catkin_ws_dir, "experiments", "crate_reachability", "reachability_log.csv")


def load_rows(path, latest_run_only):
    df = pd.read_csv(path, parse_dates=["timestamp"]).reset_index(drop=True)
    if latest_run_only and len(df):
        # A run's point_idx is non-decreasing (it repeats across the several
        # rows logged for one point's yaws, then steps up for the next
        # point), so a run boundary is a *decrease* from the previous row,
        # not every row where point_idx happens to be 0 -- point 0 alone
        # logs one row per yaw, all with point_idx==0.
        is_run_start = df["point_idx"].diff().fillna(-1) < 0
        run_id = is_run_start.cumsum()
        df = df[run_id == run_id.iloc[-1]].reset_index(drop=True)
    return df


def point_status(point_rows):
    if ((point_rows["reachable"]) & (point_rows["tilt_x_deg"].isna()) & (point_rows["tilt_y_deg"].isna())).any():
        return GREEN, "reachable straight-down"
    if point_rows["reachable"].any():
        return AMBER, "reachable, tilted only"
    return RED, "unreachable"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--log-path', default=None, help="Override the default reachability_log.csv location")
    parser.add_argument('--latest-run-only', action='store_true',
                         help="Only plot the most recent sweep instead of every row ever logged")
    parser.add_argument('--out', default=None,
                         help="Output image path (default: alongside the log, reachability_heatmap.png)")
    parser.add_argument('--show', action='store_true', help="Also open an interactive window")
    args = parser.parse_args()

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D

    log_path = args.log_path or get_log_path()
    out_path = args.out or os.path.join(os.path.dirname(log_path), "reachability_heatmap.png")

    df = load_rows(log_path, args.latest_run_only)
    if df.empty:
        raise SystemExit(f"No rows in {log_path}")

    # Auto-size cells from the tested grid spacing rather than assuming one,
    # since --spacing-m/--edge-margin-m aren't recorded in the log.
    # Round to the nearest tenth of a millimeter first so float noise between
    # otherwise-identical coordinates can't collapse the diff to ~0.
    xs = np.sort(np.unique(np.round(df["x"].to_numpy(), 4)))
    ys = np.sort(np.unique(np.round(df["y"].to_numpy(), 4)))
    min_cell = 0.01  # floor, in case a single point is way off the rest of the ring
    cell_w = max(np.min(np.diff(xs)), min_cell) if len(xs) > 1 else 0.05
    cell_h = max(np.min(np.diff(ys)), min_cell) if len(ys) > 1 else 0.05

    fig, ax = plt.subplots(figsize=(7, 6))
    n_reachable_points = 0
    for point_idx, point_rows in df.groupby("point_idx"):
        x, y = point_rows["x"].iloc[0], point_rows["y"].iloc[0]
        color, _ = point_status(point_rows)
        n_ok = int(point_rows["reachable"].sum())
        n_total = len(point_rows)
        if n_ok:
            n_reachable_points += 1

        ax.add_patch(Rectangle((x - cell_w / 2, y - cell_h / 2), cell_w, cell_h,
                                facecolor=color, edgecolor="white", linewidth=1.5))
        ax.text(x, y, f"{n_ok}/{n_total}", ha="center", va="center",
                fontsize=8, color="white", fontweight="bold")

    ax.set_xlim(xs.min() - cell_w, xs.max() + cell_w)
    ax.set_ylim(ys.min() - cell_h, ys.max() + cell_h)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)

    n_points = df["point_idx"].nunique()
    n_reachable_checks = int(df["reachable"].sum())
    ax.set_title(
        f"Crate reachability sweep (top-down)\n"
        f"{n_reachable_points}/{n_points} points reachable at >=1 yaw "
        f"({n_reachable_checks}/{len(df)} individual checks)"
    )

    legend_elems = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=GREEN, markersize=14, label="reachable straight-down"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=AMBER, markersize=14, label="reachable, tilted only"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=RED, markersize=14, label="unreachable"),
    ]
    ax.legend(handles=legend_elems, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Read {len(df)} checks over {n_points} points from {log_path}")
    print(f"Saved heatmap to {out_path}")

    print("\n=== Per-point summary ===")
    for point_idx, point_rows in df.groupby("point_idx"):
        _, status = point_status(point_rows)
        wall = point_rows["nearest_wall"].iloc[0]
        print(f"  point {point_idx:>2} ({point_rows['x'].iloc[0]:.3f}, {point_rows['y'].iloc[0]:.3f}, {wall:>12}): "
              f"{status} ({int(point_rows['reachable'].sum())}/{len(point_rows)})")

    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
