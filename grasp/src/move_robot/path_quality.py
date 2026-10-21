#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Path-quality metrics for the planner comparison (planner_benchmark.py and the report
figures). Pure numpy, no ROS, so the saved paths can be re-analysed offline.

Joint paths are (N, 7) arrays of waypoints; Cartesian paths (N, 3) TCP positions.
Geometric metrics (length, turning, deviation, reversals) describe the path itself and
do not depend on how it is timed; jerk and execution time come from the time-
parameterised trajectory, so they also reflect the retimer (the same one for every
planner)."""
import itertools

import numpy as np


def densify(waypoints, max_step):
    """Linear interpolation between waypoints so no joint moves more than max_step (rad)
    per step: the joint-space path the controller follows between waypoints, needed to
    measure the Cartesian path (a straight joint segment is curved in Cartesian space)."""
    q = np.asarray(waypoints, dtype=float)
    out = [q[0]]
    for a, b in zip(q[:-1], q[1:]):
        n = max(int(np.ceil(np.max(np.abs(b - a)) / max_step)), 1)
        for k in range(1, n + 1):
            out.append(a + (b - a) * k / n)
    return np.array(out)


def polyline_length(points):
    p = np.asarray(points, dtype=float)
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))) if len(p) > 1 else 0.0


def resample(points, n):
    """n points equally spaced by arc length along the polyline."""
    p = np.asarray(points, dtype=float)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-12:
        return np.repeat(p[:1], n, axis=0)
    targets = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(targets, s, p[:, d]) for d in range(p.shape[1])], axis=1)


def turning_angle(points, min_segment=1e-4):
    """Total turning angle (rad) of a polyline: the sum of the angles between consecutive
    segments, skipping segments shorter than min_segment. 0 for a straight line; adding
    collinear points does not change it. A geometric smoothness measure."""
    p = np.asarray(points, dtype=float)
    d = np.diff(p, axis=0)
    norms = np.linalg.norm(d, axis=1)
    d = d[norms > min_segment] / norms[norms > min_segment, None]
    if len(d) < 2:
        return 0.0
    cos = np.clip(np.sum(d[:-1] * d[1:], axis=1), -1.0, 1.0)
    return float(np.sum(np.arccos(cos)))


def max_deviation(points):
    """Largest distance (same unit as points) from the straight segment start -> end."""
    p = np.asarray(points, dtype=float)
    a, b = p[0], p[-1]
    ab = b - a
    if ab @ ab < 1e-12:
        return float(np.max(np.linalg.norm(p - a, axis=1)))
    t = np.clip((p - a) @ ab / (ab @ ab), 0.0, 1.0)
    return float(np.max(np.linalg.norm(p - (a + t[:, None] * ab), axis=1)))


def joint_reversals(waypoints, hysteresis=0.02):
    """Number of direction reversals summed over the joints: a joint counts a reversal
    when it moves at least `hysteresis` rad back after moving the other way. A straight
    joint-space path has 0; every reversal is motion that is later undone."""
    q = np.asarray(waypoints, dtype=float)
    total = 0
    for x in q.T:
        direction, extreme = 0, x[0]
        for v in x[1:]:
            if direction == 0:
                if abs(v - x[0]) >= hysteresis:
                    direction, extreme = (1 if v > x[0] else -1), v
            elif direction * (v - extreme) > 0:
                extreme = v
            elif abs(v - extreme) >= hysteresis:
                total, direction, extreme = total + 1, -direction, v
    return int(total)


def wasted_joint_motion(waypoints):
    """Joint travel beyond the net displacement (rad), summed over the joints: 0 when
    every joint moves monotonically from start to goal."""
    q = np.asarray(waypoints, dtype=float)
    travel = np.sum(np.abs(np.diff(q, axis=0)), axis=0)
    return float(np.sum(travel - np.abs(q[-1] - q[0])))


def jerk_rms(times, accelerations):
    """Time-weighted RMS of the joint jerk vector (rad/s^3) of a timed trajectory, from
    finite differences of the accelerations. Depends on the retimer as well as the path."""
    t = np.asarray(times, dtype=float)
    a = np.asarray(accelerations, dtype=float)
    dt = np.diff(t)
    ok = dt > 1e-6
    if not ok.any() or a.size == 0:
        return float('nan')
    jerk = np.diff(a, axis=0)[ok] / dt[ok, None]
    return float(np.sqrt(np.sum(np.sum(jerk ** 2, axis=1) * dt[ok]) / np.sum(dt[ok])))


def box_distance(points, centers, half_extents):
    """Distance from every point to the nearest axis-aligned box (0 inside a box).
    points (N, 3), centers / half_extents (M, 3). Returns (N,)."""
    p = np.asarray(points, dtype=float)[:, None, :]
    c = np.asarray(centers, dtype=float)[None, :, :]
    h = np.asarray(half_extents, dtype=float)[None, :, :]
    outside = np.maximum(np.abs(p - c) - h, 0.0)
    return np.min(np.linalg.norm(outside, axis=2), axis=1)


def path_spread(paths, n=50):
    """Consistency of repeated plans for the same query: the mean, over all pairs of
    paths, of the mean distance between corresponding points after resampling both to
    n points by arc length. 0 when every repeat gives the same path."""
    if len(paths) < 2:
        return float('nan')
    r = [resample(p, n) for p in paths]
    return float(np.mean([np.mean(np.linalg.norm(a - b, axis=1)) for a, b in itertools.combinations(r, 2)]))
