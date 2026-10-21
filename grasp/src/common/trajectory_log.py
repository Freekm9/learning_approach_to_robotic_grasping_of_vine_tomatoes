"""Log of every trajectory the robot executes, so a path can be inspected, compared with
what the robot actually did, and repeated later (move_robot/replay_trajectory.py).

One JSON record per executed move, appended to
experiments/trajectories/trajectories_<YYYYMMDD>.jsonl.gz (one gzip member per record,
so a crash loses at most the move in progress). A record holds:
  time, movement, kind        when, which pipeline move (pre_grasp, grasp_descent, ...)
                              and how it was made: 'planned' (a planner), 'cartesian'
                              (straight line), 'go' (move_group.go: no plan available,
                              only the target and the executed states) or 'replay'
  planner, pipeline, tag      planning configuration, and ~trajectory_log_tag (re-read
                              live; e.g. the E4 runner's run/attempt), video clip
  joint_names, start          joint names and the measured joints when the move started
  trajectory                  the trajectory sent to the controller: positions,
                              velocities, accelerations, time_from_start (planned/cartesian)
  target                      goal joints of a 'go' move
  executed                    measured joint states during execution, at most --rate_hz
                              (t from the start of the move, positions)
  success                     what execute()/go() returned
  scene                       crate-contents height map in the planning scene during the
                              move ((x, y, top) columns, floor z, cell size), if any
Size: roughly 5-10 kB per planned move and 1-2 kB per second of executed states,
compressed. Logging never raises into the caller: a failed write only prints a warning.
"""
import contextlib
import gzip
import json
import os
import threading
import time

import rospy
from sensor_msgs.msg import JointState


def _round(values, digits=5):
    return [round(float(v), digits) for v in values]


def trajectory_to_dict(robot_trajectory):
    jt = robot_trajectory.joint_trajectory
    return dict(joint_names=list(jt.joint_names),
                positions=[_round(p.positions) for p in jt.points],
                velocities=[_round(p.velocities) for p in jt.points] if all(len(p.velocities) for p in jt.points) else None,
                accelerations=[_round(p.accelerations) for p in jt.points]
                if all(len(p.accelerations) for p in jt.points) else None,
                time_from_start=[round(p.time_from_start.to_sec(), 4) for p in jt.points])


def dict_to_trajectory(d, speed=1.0):
    """RobotTrajectory from a logged trajectory dict, slowed down by 1/speed (speed <= 1:
    times stretched, velocities and accelerations scaled to match)."""
    from moveit_msgs.msg import RobotTrajectory
    from trajectory_msgs.msg import JointTrajectoryPoint
    traj = RobotTrajectory()
    traj.joint_trajectory.joint_names = list(d["joint_names"])
    for i, q in enumerate(d["positions"]):
        p = JointTrajectoryPoint(positions=list(q))
        if d.get("velocities"):
            p.velocities = [v * speed for v in d["velocities"][i]]
        if d.get("accelerations"):
            p.accelerations = [a * speed * speed for a in d["accelerations"][i]]
        p.time_from_start = rospy.Duration.from_sec(d["time_from_start"][i] / speed)
        traj.joint_trajectory.points.append(p)
    return traj


def read_log(path):
    """All records of a log file (a truncated last record is skipped)."""
    records = []
    try:
        with gzip.open(path, 'rt') as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except ValueError:
                    pass
    except (EOFError, OSError):
        pass  # file still being written / last member truncated
    return records


class TrajectoryLog(object):
    def __init__(self, log_dir, joint_states_topic='joint_states', rate_hz=100.0, enabled=True):
        self.log_dir = log_dir
        self.enabled = enabled
        self.min_dt = 1.0 / rate_hz
        self.scene = None  # set by the caller while a crate-contents obstacle is in the scene
        self._lock = threading.Lock()
        self._recording = None
        self._latest = None
        if enabled:
            os.makedirs(log_dir, exist_ok=True)
            self._sub = rospy.Subscriber(joint_states_topic, JointState, self._joint_states_cb, queue_size=50)

    def path(self):
        return os.path.join(self.log_dir, f"trajectories_{time.strftime('%Y%m%d')}.jsonl.gz")

    def _joint_states_cb(self, msg):
        with self._lock:
            self._latest = msg
            rec = self._recording
            if rec is None:
                return
            t = msg.header.stamp.to_sec() if msg.header.stamp else time.time()
            if rec["executed"]["t"] and t - rec["_t_last"] < self.min_dt:
                return
            rec["_t_last"] = t
            if not rec["executed"]["names"]:
                rec["executed"]["names"] = list(msg.name)
            rec["executed"]["t"].append(round(t - rec["_t0"], 4))
            rec["executed"]["positions"].append(_round(msg.position))

    @contextlib.contextmanager
    def recording(self, movement, kind, plan=None, target=None, **info):
        """Records the joint states while the body runs; the body sets rec['success'].
        Writes the record afterwards, also if the body raises."""
        if not self.enabled:
            yield {}
            return
        rec = dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), movement=movement or "", kind=kind,
                   tag=rospy.get_param('~trajectory_log_tag', ''), success=None, **info)
        try:
            rec["trajectory"] = trajectory_to_dict(plan) if plan is not None else None
            rec["target"] = dict(target) if target is not None else None
            rec["scene"] = self.scene
            with self._lock:
                latest = self._latest
            rec["start"] = dict(zip(latest.name, _round(latest.position))) if latest is not None else None
        except Exception as e:  # logging must never stop the move
            rospy.logwarn(f"Trajectory log: could not prepare record ({e})")
        rec["executed"] = dict(names=[], t=[], positions=[])
        rec["_t0"], rec["_t_last"] = rospy.get_time() or time.time(), 0.0
        with self._lock:
            self._recording = rec
        try:
            yield rec
        finally:
            with self._lock:
                self._recording = None
            self._write(rec)

    def _write(self, rec):
        rec = {k: v for k, v in rec.items() if not k.startswith('_')}
        try:
            with gzip.open(self.path(), 'at') as f:
                f.write(json.dumps(rec) + "\n")
        except Exception as e:
            rospy.logwarn(f"Trajectory log: could not write record ({e})")
