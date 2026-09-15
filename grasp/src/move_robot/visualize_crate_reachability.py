#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Publish the results of test_crate_reachability.py as RViz markers, so you
can actually look at where the tested points land relative to the crate
walls instead of reading numbers off the terminal.

Reads experiments/crate_reachability/reachability_log.csv (the same file
test_crate_reachability.py writes) and publishes a visualization_msgs/
MarkerArray -- one sphere per tested (x, y) point (green: reachable
straight-down for at least one yaw; yellow: reachable but only via a tilted
approach; red: unreachable at every yaw/tilt tried), a small text label with
the "reachable/total" count, and a short arrow per (point, yaw) row showing
that specific tested orientation, colored the same way.

This only reads the CSV -- it doesn't touch the planning scene or move the
arm, so it's fine to run any time after (or instead of) a real sweep, even
against a stale/old log.

Usage:
    rosrun grasp visualize_crate_reachability.py
    # then in RViz: Add -> By display type -> MarkerArray -> Topic: /crate_reachability_markers

    # only the latest run (by point_idx reset) instead of every row ever logged:
    rosrun grasp visualize_crate_reachability.py --latest-run-only
"""
import argparse
import csv
import os

import numpy as np
import rospy
import rospkg

from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

GREEN = ColorRGBA(0.2, 0.8, 0.2, 0.9)
YELLOW = ColorRGBA(0.95, 0.85, 0.1, 0.9)
RED = ColorRGBA(0.9, 0.15, 0.15, 0.85)
WHITE = ColorRGBA(1.0, 1.0, 1.0, 1.0)

ARROW_LENGTH_M = 0.03
POINT_RADIUS_M = 0.012


def get_log_path():
    rospack = rospkg.RosPack()
    grasp_pkg_dir = rospack.get_path('grasp')
    catkin_ws_dir = os.path.dirname(os.path.dirname(os.path.dirname(grasp_pkg_dir)))
    return os.path.join(catkin_ws_dir, "experiments", "crate_reachability", "reachability_log.csv")


def load_rows(path, latest_run_only):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row['x'] = float(row['x'])
        row['y'] = float(row['y'])
        row['z'] = float(row['z'])
        row['yaw_deg'] = float(row['yaw_deg'])
        row['reachable'] = row['reachable'] == 'True'
        row['tilt_x_deg'] = None if row['tilt_x_deg'] in ('', 'None') else float(row['tilt_x_deg'])
        row['tilt_y_deg'] = None if row['tilt_y_deg'] in ('', 'None') else float(row['tilt_y_deg'])
        row['point_idx'] = int(row['point_idx'])

    if latest_run_only and rows:
        # A run's point_idx is non-decreasing (it repeats across the several
        # rows logged for one point's yaws, then steps up for the next
        # point), so a run boundary is a *decrease* from the previous row,
        # not every row where point_idx happens to be 0 -- point 0 alone
        # logs one row per yaw, all with point_idx==0.
        run_starts = [0] + [i for i in range(1, len(rows)) if rows[i]['point_idx'] < rows[i - 1]['point_idx']]
        rows = rows[run_starts[-1]:]
    return rows


def point_color(point_rows):
    if any(r['reachable'] and r['tilt_x_deg'] == 0 and r['tilt_y_deg'] == 0 for r in point_rows):
        return GREEN
    if any(r['reachable'] for r in point_rows):
        return YELLOW
    return RED


def build_markers(rows, frame_id):
    markers = MarkerArray()
    mid = 0

    by_point = {}
    for r in rows:
        by_point.setdefault(r['point_idx'], []).append(r)

    for point_idx, point_rows in sorted(by_point.items()):
        x, y, z = point_rows[0]['x'], point_rows[0]['y'], point_rows[0]['z']
        color = point_color(point_rows)
        n_reachable = sum(r['reachable'] for r in point_rows)

        sphere = Marker()
        sphere.header.frame_id = frame_id
        sphere.ns = "crate_reachability_points"
        sphere.id = mid; mid += 1
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.pose.position = Point(x, y, z)
        sphere.pose.orientation.w = 1.0
        sphere.scale.x = sphere.scale.y = sphere.scale.z = POINT_RADIUS_M * 2
        sphere.color = color
        markers.markers.append(sphere)

        label = Marker()
        label.header.frame_id = frame_id
        label.ns = "crate_reachability_labels"
        label.id = mid; mid += 1
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position = Point(x, y, z + 0.03)
        label.pose.orientation.w = 1.0
        label.scale.z = 0.015
        label.color = WHITE
        label.text = f"{point_idx}: {n_reachable}/{len(point_rows)}\n{point_rows[0]['nearest_wall']}"
        markers.markers.append(label)

        for r in point_rows:
            arrow = Marker()
            arrow.header.frame_id = frame_id
            arrow.ns = "crate_reachability_yaws"
            arrow.id = mid; mid += 1
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            dx = ARROW_LENGTH_M * np.cos(np.radians(r['yaw_deg']))
            dy = ARROW_LENGTH_M * np.sin(np.radians(r['yaw_deg']))
            arrow.points = [Point(x, y, z), Point(x + dx, y + dy, z)]
            arrow.scale.x = 0.003  # shaft diameter
            arrow.scale.y = 0.006  # head diameter
            arrow.color = GREEN if r['reachable'] else RED
            markers.markers.append(arrow)

    return markers


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--robot-name', default='panda')
    parser.add_argument('--topic', default='/crate_reachability_markers')
    parser.add_argument('--latest-run-only', action='store_true',
                         help="Only show the most recent sweep instead of every row ever logged")
    parser.add_argument('--rate-hz', type=float, default=1.0,
                         help="Republish rate, so a MarkerArray display added after this node starts still picks it up")
    args = parser.parse_args(rospy.myargv()[1:])

    rospy.init_node('visualize_crate_reachability')
    frame_id = rospy.get_param(f'/{args.robot_name}/planning_frame',
                                rospy.get_param('/planning_frame', args.robot_name + '_link0'))

    log_path = get_log_path()
    pub = rospy.Publisher(args.topic, MarkerArray, queue_size=1, latch=True)

    rate = rospy.Rate(args.rate_hz)
    last_mtime = None
    while not rospy.is_shutdown():
        mtime = os.path.getmtime(log_path) if os.path.exists(log_path) else None
        if mtime is not None and mtime != last_mtime:
            rows = load_rows(log_path, args.latest_run_only)
            markers = build_markers(rows, frame_id)
            pub.publish(markers)
            rospy.loginfo(f"Published {len(rows)} logged checks ({len(set(r['point_idx'] for r in rows))} points) "
                           f"from {log_path} to {args.topic} (frame: {frame_id})")
            last_mtime = mtime
        rate.sleep()


if __name__ == '__main__':
    main()
