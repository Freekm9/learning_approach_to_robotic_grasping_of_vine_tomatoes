#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E3: grasp reachability against finger length, top-down vs tilted grasp generator,
on realistic grasp points (peduncles segmented from saved whole-crate clouds).

  points  (offline) Stem segmentation (stem_segmentation/segment_stems.py) of each
          whole-crate cloud; grasp points every --step-m along each truss's main stem
          path (not within --end-margin-m of its ends), at the stem's height, with the
          yaw that puts the TCP x (vine) axis along the local stem direction, as
          stem_snap.py does in the pipeline. Also the distance to the nearest crate
          wall. Writes experiments/finger_reachability/points_<timestamp>.json.
  check   (live sim, one finger length) For every point and --yaw-offsets (deviation
          from the stem yaw), both generators choose an orientation exactly as
          simple_pick_point.py does (full pick sequence: pre-grasp, grasp, back-out,
          lift and the straight lines between them; walls only):
            topdown  straight down, else its 180-degree roll (Van den Bent's 4-DOF)
            tilted   straight down, roll, then the tilt table (our 6/7-DOF)
          Then, pipeline-faithful, whether the pre-grasp pose of that orientation (or
          its roll, as _grasp_orientations tries both) is still collision-free with
          the crate-contents height map of the point's own cloud in the scene, carved
          'cylinder' (the corridor only, the pipeline default) and 'gripper' (also the
          gripper volume along the approach; ~crate_contents_carve_mode). The pipeline
          doesn't fall back to another tilt when the pre-grasp is blocked, so neither
          does this. Appends to finger_<ext>cm_<timestamp>.csv.
  sweep   Runs check for every --finger-lengths-cm: launches the Gazebo/MoveIt sim with
          that finger length on its own ROS master (--port), runs check, shuts it down.
  report  (offline) Reachable fraction per finger length, generator and variant
          (walls / cylinder / gripper) with 95% cluster-bootstrap CIs (resampling
          trusses, since points on one stem aren't independent); McNemar per finger
          length on the paired poses; reachability by distance to the nearest wall in
          1 cm bins; the tilts needed; the shortest finger length reaching --target.
          Writes report_*.csv and two figures next to the first CSV.

Usage:
    rosrun grasp finger_reachability.py points            # all clouds in experiments/full_crate_clouds
    rosrun grasp finger_reachability.py sweep --points <dir>/points_*.json
    rosrun grasp finger_reachability.py report <dir>/finger_*.csv

A 0 cm finger length is launched as 1 mm (the extension is a box in franka_hand.xacro
and can't have zero size). Don't run sweep next to another benchmark that measures time.
"""
import argparse
import csv
import glob
import json
import os
import signal
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'stem_segmentation'))

STOCK_TCP_Z_M = 0.1034  # hand -> TCP without extension (franka_robot.xacro)
DEFAULT_FINGERS_CM = [0, 5, 9, 13, 20]
WALL_BIN_EDGES_CM = [0, 1, 2, 3, 4, 5, 7, 10, 100]
VARIANTS = ("walls", "cylinder", "gripper")
CHECK_FIELDS = ["scan", "cluster", "point_id", "x", "y", "z", "wall_dist_m", "yaw_stem_deg", "yaw_offset_deg",
                "finger_ext_m", "method", "reachable_walls", "tilt_x_deg", "tilt_y_deg", "flip",
                "reachable_cylinder", "reachable_gripper"]


def ws_dir():
    import rospkg
    return os.path.dirname(os.path.dirname(os.path.dirname(rospkg.RosPack().get_path('grasp'))))


def out_dir():
    d = os.path.join(ws_dir(), 'experiments', 'finger_reachability')
    os.makedirs(d, exist_ok=True)
    return d


def scene_interior(scene_file):
    """(x_lo, x_hi, y_lo, y_hi, floor_z, rim_z) of the crate in a .scene file."""
    from scipy.spatial.transform import Rotation as R
    from common.crate_contents import crate_interior_from_boxes
    from move_robot.pink_planner import parse_scene_file
    boxes = [(n, c, R.from_matrix(rot), d) for n, c, rot, d in parse_scene_file(scene_file)]
    return crate_interior_from_boxes(boxes)


# --------------------------------------------------------------------------
# points
# --------------------------------------------------------------------------

def stem_samples(nodes, step_m, end_margin_m):
    """(positions, unit xy tangents) every step_m along a polyline of skeleton nodes,
    leaving end_margin_m free at both ends."""
    seg = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    out = []
    for t in np.arange(end_margin_m, s[-1] - end_margin_m + 1e-9, step_m):
        k = min(np.searchsorted(s, t, side='right') - 1, len(seg) - 1)
        f = (t - s[k]) / seg[k] if seg[k] > 0 else 0.0
        pos = nodes[k] + f * (nodes[k + 1] - nodes[k])
        # Direction over ~2 cm around the sample, so one noisy node doesn't set the yaw
        a = nodes[max(k - 1, 0)]
        b = nodes[min(k + 2, len(nodes) - 1)]
        tan = (b - a)[:2]
        if np.linalg.norm(tan) < 1e-6:
            continue
        out.append((pos, tan / np.linalg.norm(tan)))
    return out


def cmd_points(args):
    from pcd_io import load_pcd
    from segment_stems import segment
    interior = scene_interior(args.scene)
    x_lo, x_hi, y_lo, y_hi, floor_z, rim_z = interior
    clouds = args.pcd or sorted(glob.glob(os.path.join(ws_dir(), 'experiments', 'full_crate_clouds', '*.pcd')))
    points = []
    for path in clouds:
        xyz, rgb = load_pcd(path)
        inside = ((xyz[:, 0] > x_lo) & (xyz[:, 0] < x_hi) & (xyz[:, 1] > y_lo) & (xyz[:, 1] < y_hi)
                  & (xyz[:, 2] > floor_z) & (xyz[:, 2] < rim_z + 0.08))
        result = segment(xyz[inside], rgb[inside])
        scan = os.path.splitext(os.path.basename(path))[0]
        n_before = len(points)
        for ci, cluster in enumerate(result.clusters):
            for pos, tan in stem_samples(cluster.nodes[cluster.main_path], args.step_m, args.end_margin_m):
                wall = float(min(pos[0] - x_lo, x_hi - pos[0], pos[1] - y_lo, y_hi - pos[1]))
                points.append(dict(id=len(points), scan=scan, cloud=os.path.abspath(path), cluster=ci,
                                   x=round(float(pos[0]), 4), y=round(float(pos[1]), 4), z=round(float(pos[2]), 4),
                                   yaw_stem_deg=round(float(np.degrees(np.arctan2(tan[1], tan[0])) % 180.0), 1),
                                   wall_dist_m=round(wall, 4)))
        n = len(points) - n_before
        walls = np.array([p["wall_dist_m"] for p in points[n_before:]])
        print(f"{scan}: {len(result.clusters)} stem clusters, {n} grasp points; z "
              f"{min(p['z'] for p in points[n_before:]) if n else float('nan'):.3f}-"
              f"{max(p['z'] for p in points[n_before:]) if n else float('nan'):.3f} m; "
              f"within 3 cm of a wall: {int((walls < 0.03).sum()) if n else 0}")
    path = os.path.join(out_dir(), f"points_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, 'w') as f:
        json.dump(dict(meta=dict(created=time.strftime('%Y-%m-%d %H:%M:%S'), scene=os.path.abspath(args.scene),
                                 interior=list(interior), step_m=args.step_m, end_margin_m=args.end_margin_m,
                                 clouds=[os.path.abspath(c) for c in clouds]), points=points), f, indent=1)
    print(f"\n{len(points)} points: {path}")


# --------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------

def cmd_check(args):
    import rospy
    from pcd_io import load_pcd
    from common.crate_contents import CrateContentsObstacle, crate_interior_from_boxes
    from move_robot.planner_benchmark import Bench, CONTENTS_CORRIDOR_M
    from move_robot.test_crate_reachability import (IKChecker, build_straight_down_pose, flip_approach_roll,
                                                    find_reachable_tilt, get_approach_vec, offset_pose,
                                                    tilt_pose, PRE_GRASP_OFFSET_M)
    rospy.init_node('finger_reachability_check')
    with open(args.points) as f:
        points = json.load(f)["points"][:args.limit_points]
    bench = Bench(args.robot_name)
    ext = bench.tcp_offset_m() - STOCK_TCP_Z_M
    interior = crate_interior_from_boxes(bench.scene_boxes())
    obstacles = {mode: CrateContentsObstacle(bench.scene, None, bench.frame, interior, cell_m=0.02,
                                             carve_radius_m=0.04, corridor_length_m=CONTENTS_CORRIDOR_M,
                                             carve_mode=mode)
                 for mode in ("cylinder", "gripper")}
    ik = IKChecker(args.robot_name)
    clouds = {}

    path = args.out or os.path.join(out_dir(), f"finger_{ext * 100:.1f}cm_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    print(f"Finger extension {ext * 100:.1f} cm; {len(points)} points x yaw offsets {args.yaw_offsets} -> {path}")
    start = time.time()
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CHECK_FIELDS)
        writer.writeheader()
        for i, p in enumerate(points):
            if p["cloud"] not in clouds:
                clouds[p["cloud"]] = load_pcd(p["cloud"])[0]
            xyz = clouds[p["cloud"]]
            for offset in args.yaw_offsets:
                if rospy.is_shutdown():
                    return
                base = build_straight_down_pose(p["x"], p["y"], p["z"], p["yaw_stem_deg"] + offset, bench.frame)
                if ik.check_grasp(base):
                    topdown = (0, 0, False)
                elif ik.check_grasp(flip_approach_roll(base)):
                    topdown = (0, 0, True)
                else:
                    topdown = None
                if topdown is not None:
                    tilted = topdown
                else:
                    ok, dx, dy, flip = find_reachable_tilt(ik, base)
                    tilted = (dx, dy, flip) if ok else None

                contents_ok = {}
                for orientation in {topdown, tilted} - {None}:
                    dx, dy, flip = orientation
                    grasp = flip_approach_roll(base) if flip else tilt_pose(base, dx, dy) if (dx or dy) else base
                    candidates = [grasp] + ([flip_approach_roll(grasp)] if ik.check_grasp(flip_approach_roll(grasp))
                                            else [])
                    approach = get_approach_vec(grasp.pose)
                    o = grasp.pose.orientation
                    for mode, obstacle in obstacles.items():
                        obstacle.add([p["x"], p["y"], p["z"]], approach, xyz=xyz, grasp_quat=[o.x, o.y, o.z, o.w])
                        contents_ok[(orientation, mode)] = any(
                            ik.check_single(offset_pose(c, approach, PRE_GRASP_OFFSET_M)) for c in candidates)
                        obstacle.remove()

                for method, orientation in (("topdown", topdown), ("tilted", tilted)):
                    row = dict(scan=p["scan"], cluster=p["cluster"], point_id=p["id"], x=p["x"], y=p["y"], z=p["z"],
                               wall_dist_m=p["wall_dist_m"], yaw_stem_deg=p["yaw_stem_deg"], yaw_offset_deg=offset,
                               finger_ext_m=round(ext, 4), method=method, reachable_walls=orientation is not None,
                               tilt_x_deg="", tilt_y_deg="", flip="", reachable_cylinder=False,
                               reachable_gripper=False)
                    if orientation is not None:
                        row.update(tilt_x_deg=orientation[0], tilt_y_deg=orientation[1], flip=orientation[2],
                                   reachable_cylinder=contents_ok[(orientation, "cylinder")],
                                   reachable_gripper=contents_ok[(orientation, "gripper")])
                    writer.writerow(row)
                f.flush()
            done = i + 1
            eta = (time.time() - start) / done * (len(points) - done)
            print(f"[{done}/{len(points)}] {p['scan']} c{p['cluster']} ({p['x']:.3f}, {p['y']:.3f}, {p['z']:.3f}) "
                  f"wall {p['wall_dist_m'] * 100:.1f} cm; ETA {eta / 60:.0f} min")
    print(f"\nResults: {path}")


# --------------------------------------------------------------------------
# sweep
# --------------------------------------------------------------------------

def wait_for_service(env, service, timeout_s):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        out = subprocess.run(['rosservice', 'list'], env=env, capture_output=True, text=True).stdout
        if service in out.split():
            return True
        time.sleep(3)
    return False


def cmd_sweep(args):
    env = dict(os.environ, ROS_MASTER_URI=f'http://localhost:{args.port}',
               GAZEBO_MASTER_URI=f'http://localhost:{args.port + 34}')
    busy = subprocess.run(['rosnode', 'list'], env=env, capture_output=True, text=True)
    if busy.returncode == 0 and busy.stdout.strip():
        raise SystemExit(f"Something is already running on ROS master port {args.port}; pick another --port")
    log_dir = os.path.join(out_dir(), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    for cm in args.finger_lengths_cm:
        ext = max(cm / 100.0, 0.001)
        tag = f"{cm:g}cm"
        print(f"=== finger extension {tag}: launching sim on port {args.port}")
        launch_log = open(os.path.join(log_dir, f"sim_{tag}.log"), 'w')
        launch = subprocess.Popen(
            ['roslaunch', '-p', str(args.port), 'panda_moveit_config_vine_tomato', 'gazebo_moveit.launch',
             f'robot_name:={args.robot_name}', f'arm_id:={args.robot_name}', f'finger_extension_m:={ext}',
             f'tcp_xyz:=0 0 {STOCK_TCP_Z_M + ext:.4f}', f'scene_file:={args.scene_file}', 'rviz:=false',
             'headless:=true'],
            env=env, stdout=launch_log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            if not wait_for_service(env, f'/{args.robot_name}/compute_ik', 240):
                raise SystemExit(f"move_group did not come up for {tag}; see {launch_log.name}")
            time.sleep(5)  # planning scene publisher
            subprocess.run(['rosnode', 'kill', f'/{args.robot_name}/gazebo_gui'], env=env, capture_output=True)
            out = os.path.join(out_dir(), f"finger_{tag}_{time.strftime('%Y%m%d_%H%M%S')}.csv")
            cmd = [sys.executable, os.path.abspath(__file__), 'check', '--points', args.points, '--out', out,
                   '--robot-name', args.robot_name, '--yaw-offsets'] + [str(v) for v in args.yaw_offsets]
            if args.limit_points:
                cmd += ['--limit-points', str(args.limit_points)]
            with open(os.path.join(log_dir, f"check_{tag}.log"), 'w') as check_log:
                result = subprocess.run(cmd, env=env, stdout=check_log, stderr=subprocess.STDOUT)
            print(f"    check exit {result.returncode}: {out}")
        finally:
            os.killpg(launch.pid, signal.SIGINT)
            try:
                launch.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(launch.pid, signal.SIGKILL)
            launch_log.close()
            time.sleep(5)
    print("Sweep done; run: rosrun grasp finger_reachability.py report " + os.path.join(out_dir(), 'finger_*.csv'))


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def cluster_bootstrap(df, col, n_boot=2000, seed=0):
    """95% CI of the mean of col, resampling trusses (scan, cluster) with replacement."""
    rng = np.random.default_rng(seed)
    groups = [g[col].to_numpy(dtype=float) for _, g in df.groupby(["scan", "cluster"])]
    if not groups:
        return np.nan, np.nan
    sums = np.array([g.sum() for g in groups])
    counts = np.array([len(g) for g in groups])
    idx = rng.integers(0, len(groups), size=(n_boot, len(groups)))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def cmd_report(args):
    import pandas as pd
    from scipy.stats import binomtest
    d = pd.concat([pd.read_csv(p) for p in args.results], ignore_index=True)
    for v in VARIANTS:
        d[f"reachable_{v}"] = d[f"reachable_{v}"].astype(str).str.lower() == "true"
    d["finger_cm"] = (d["finger_ext_m"] * 100).round(0)
    d.loc[d["finger_cm"] <= 0.1, "finger_cm"] = 0.0
    base = os.path.dirname(os.path.abspath(args.results[0]))

    rows = []
    for (cm, method), g in d.groupby(["finger_cm", "method"]):
        for v in VARIANTS:
            lo, hi = cluster_bootstrap(g, f"reachable_{v}")
            rows.append(dict(finger_cm=cm, method=method, variant=v, poses=len(g),
                             trusses=g.groupby(["scan", "cluster"]).ngroups,
                             reachable=g[f"reachable_{v}"].mean(), ci_lo=lo, ci_hi=hi))
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(base, "report_summary.csv"), index=False)

    print("Reachable fraction of grasp poses (95% cluster-bootstrap CI over trusses):")
    table = summary.assign(cell=lambda s: s.apply(
        lambda r: f"{r.reachable * 100:5.1f}% [{r.ci_lo * 100:.0f}-{r.ci_hi * 100:.0f}]", axis=1))
    print(table.pivot_table(index=["variant", "finger_cm"], columns="method", values="cell", aggfunc="first")
          .to_string())

    print("\nTilted vs topdown on the same poses (McNemar, exact):")
    key = ["scan", "point_id", "yaw_offset_deg", "finger_cm"]
    for v in VARIANTS:
        paired = d.pivot_table(index=key, columns="method", values=f"reachable_{v}", aggfunc="first").dropna()
        for cm, g in paired.groupby(level="finger_cm"):
            only_t = int((g["tilted"].astype(bool) & ~g["topdown"].astype(bool)).sum())
            only_d = int((~g["tilted"].astype(bool) & g["topdown"].astype(bool)).sum())
            p = binomtest(only_t, only_t + only_d).pvalue if only_t + only_d else 1.0
            print(f"  {v:8s} {cm:4.0f} cm: tilted-only {only_t:4d}, topdown-only {only_d:4d}, p={p:.3g}")

    print(f"\nShortest finger length with reachability >= {args.target * 100:.0f}%:")
    for (v, method), g in summary.groupby(["variant", "method"]):
        ok = g[g["reachable"] >= args.target]["finger_cm"]
        print(f"  {v:8s} {method:8s}: {f'{ok.min():.0f} cm' if len(ok) else 'none of the tested lengths'}")

    edges = np.array(WALL_BIN_EDGES_CM) / 100.0
    labels = [f"{a}-{b}" if b < 100 else f">{a}" for a, b in zip(WALL_BIN_EDGES_CM[:-1], WALL_BIN_EDGES_CM[1:])]
    d["wall_bin_cm"] = pd.cut(d["wall_dist_m"], edges, labels=labels, right=False)
    walls = (d.groupby(["finger_cm", "method", "wall_bin_cm"], observed=True)[f"reachable_{args.variant}"]
             .agg(["mean", "size"]).reset_index())
    walls.to_csv(os.path.join(base, "report_by_wall_distance.csv"), index=False)
    print(f"\nReachable ({args.variant}) by distance to the nearest wall (cm), poses in brackets:")
    walls["cell"] = walls.apply(lambda r: f"{r['mean'] * 100:3.0f}% ({r['size']})", axis=1)
    print(walls.pivot_table(index=["finger_cm", "method"], columns="wall_bin_cm", values="cell",
                            aggfunc="first", observed=True).to_string())

    tilts = d[(d["method"] == "tilted") & d["reachable_walls"] & ((d["tilt_x_deg"].fillna(0) != 0) |
                                                                  (d["tilt_y_deg"].fillna(0) != 0))]
    print("\nTilts the tilted generator needed (x/y deg, poses):")
    if len(tilts):
        print(tilts.groupby(["finger_cm", "tilt_x_deg", "tilt_y_deg"]).size().to_string())
    else:
        print("  none")

    plot(summary, walls, args.variant, base)


def plot(summary, walls, variant, base):
    """Small multiples (one panel per variant) of reachability against finger length,
    plus reachability by wall distance for the chosen variant. Two series: reference
    palette slots 1-2, each also with its own marker and line style and a direct label,
    so identity never rests on colour alone."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    style = {"topdown": dict(color="#2a78d6", marker="o", ls="-", label="Top-down (4-DOF)"),
             "tilted": dict(color="#eb6834", marker="s", ls="--", label="Tilted (6/7-DOF)")}
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    titles = {"walls": "Crate walls only", "cylinder": "+ trusses, corridor carve",
              "gripper": "+ trusses, gripper carve"}

    def tidy(ax):
        ax.set_facecolor("#fcfcfb")
        ax.grid(axis="y", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(muted)
        ax.tick_params(colors=muted, labelsize=9)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True, facecolor="#fcfcfb")
    for ax, v in zip(axes, VARIANTS):
        tidy(ax)
        for method, s in style.items():
            g = summary[(summary["variant"] == v) & (summary["method"] == method)].sort_values("finger_cm")
            if g.empty:
                continue
            y = g["reachable"] * 100
            ax.errorbar(g["finger_cm"], y, yerr=[y - g["ci_lo"] * 100, g["ci_hi"] * 100 - y], color=s["color"],
                        marker=s["marker"], ms=6, ls=s["ls"], lw=2, capsize=3, elinewidth=1,
                        markeredgecolor="#fcfcfb", markeredgewidth=1.5, label=s["label"])
            # Label at the shortest finger, where the curves are furthest apart: tilted
            # (never below top-down) above its point, top-down below
            above = method == "tilted"
            ax.annotate(s["label"].split(" (")[0], (g["finger_cm"].iloc[0], y.iloc[0]),
                        xytext=(8, 9 if above else -11), textcoords="offset points", va="center",
                        fontsize=8, color=ink)
        ax.set_title(titles[v], fontsize=10, color=ink, loc="left")
        ax.set_xlabel("Finger extension (cm)", fontsize=9, color=muted)
        ax.set_ylim(0, 105)
        ax.set_xticks(sorted(summary["finger_cm"].unique()))
    axes[0].set_ylabel("Reachable grasp poses (%)", fontsize=9, color=muted)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    path1 = os.path.join(base, "reachability_vs_finger.png")
    fig.savefig(path1, dpi=200)

    fingers = sorted(walls["finger_cm"].unique())
    fig, axes = plt.subplots(1, len(fingers), figsize=(2.6 * len(fingers), 3.2), sharey=True, facecolor="#fcfcfb",
                             squeeze=False)
    for ax, cm in zip(axes[0], fingers):
        tidy(ax)
        bins = list(walls["wall_bin_cm"].cat.categories) if hasattr(walls["wall_bin_cm"], "cat") else None
        for k, (method, s) in enumerate(style.items()):
            g = walls[(walls["finger_cm"] == cm) & (walls["method"] == method)]
            xs = [bins.index(b) + (k - 0.5) * 0.38 for b in g["wall_bin_cm"]] if bins else range(len(g))
            ax.bar(xs, g["mean"] * 100, width=0.36, color=s["color"], edgecolor="#fcfcfb", linewidth=2,
                   hatch="///" if method == "tilted" else None, label=s["label"])
        if bins:
            ax.set_xticks(range(len(bins)))
            ax.set_xticklabels(bins, rotation=45, fontsize=8)
        ax.set_title(f"{cm:.0f} cm fingers", fontsize=10, color=ink, loc="left")
        ax.set_xlabel("Distance to wall (cm)", fontsize=9, color=muted)
        ax.set_ylim(0, 105)
    axes[0][0].set_ylabel(f"Reachable, {titles[variant].lower()} (%)", fontsize=9, color=muted)
    handles, names = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, fontsize=8, loc="upper right", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path2 = os.path.join(base, "reachability_by_wall_distance.png")
    fig.savefig(path2, dpi=200)
    print(f"\nFigures: {path1}\n         {path2}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    default_scene = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE)))),
                                 'panda_moveit_config_vine_tomato', 'config', 'samla_crate.scene')

    p = sub.add_parser('points', help="Grasp points from whole-crate clouds (offline)")
    p.add_argument('--pcd', nargs='+', default=None, help="Clouds (default: experiments/full_crate_clouds/*.pcd)")
    p.add_argument('--scene', default=default_scene, help="Crate .scene file for the interior/wall distances")
    p.add_argument('--step-m', type=float, default=0.02, help="Spacing of grasp points along a stem")
    p.add_argument('--end-margin-m', type=float, default=0.02, help="Keep this far from the stem's ends")

    c = sub.add_parser('check', help="Reachability of every point at the sim's finger length")
    c.add_argument('--points', required=True)
    c.add_argument('--robot-name', default='panda')
    c.add_argument('--yaw-offsets', type=float, nargs='+', default=[0, 15, -15, 30, -30],
                   help="Grasp yaw deviation from the stem direction (deg)")
    c.add_argument('--limit-points', type=int, default=None)
    c.add_argument('--out', default=None)

    s = sub.add_parser('sweep', help="check at every finger length, one sim launch each")
    s.add_argument('--points', required=True)
    s.add_argument('--finger-lengths-cm', type=float, nargs='+', default=DEFAULT_FINGERS_CM)
    s.add_argument('--port', type=int, default=11314, help="ROS master port for the sims (Gazebo: port + 34)")
    s.add_argument('--robot-name', default='panda')
    s.add_argument('--scene-file', default='samla_crate.scene')
    s.add_argument('--yaw-offsets', type=float, nargs='+', default=[0, 15, -15, 30, -30])
    s.add_argument('--limit-points', type=int, default=None)

    r = sub.add_parser('report', help="Tables and figures from check CSVs (offline)")
    r.add_argument('results', nargs='+')
    r.add_argument('--target', type=float, default=0.95, help="Reachability for 'shortest finger length'")
    r.add_argument('--variant', choices=VARIANTS, default='gripper', help="Variant for the wall-distance breakdown")

    import rospy
    args = parser.parse_args(rospy.myargv()[1:])
    {'points': cmd_points, 'check': cmd_check, 'sweep': cmd_sweep, 'report': cmd_report}[args.cmd](args)


if __name__ == '__main__':
    main()
