#!/usr/bin/env python3
"""Recorded session -> portable training file. Runs on the vehicle computer (needs ROS and the estimator package).

    source /opt/ros/jazzy/setup.bash
    source ~/f1tenth_ws/install/setup.bash
    source ~/f1tenth_ws/final_project/test/f1tenth_state_estimation/install/setup.bash
    python3 make_dataset.py <session directory> data/<name>.npz

The raw sensors are re-filtered offline with the lidar-inertial-wheel EKF (no map, no output extrapolation),
so the result does not depend on which estimator ran during recording. Output, 50 Hz at the IMU stamps:
  t, vx, vy, r   velocity of the rear-axle centre in body axes [m/s], yaw rate [rad/s]
  w              wheel peripheral speed from ERPM [m/s]
  current        motor current [A];  servo  commanded servo position (0..1)
  clean          sample usable for training (moving, not lifted or carried, lidar present and matching)
"""
import sys
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from f1tenth_state_estimation_impl.lidar_filter import LidarInertialFilter

G, ERPM_PER_MPS = 9.80665, 4308.2
stamp = lambda h: h.stamp.sec + h.stamp.nanosec * 1e-9


def read(bag):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag, storage_id='sqlite3'), rosbag2_py.ConverterOptions('', ''))
    wanted = ('/scan', '/sensors/imu/raw', '/sensors/core', '/sensors/servo_position_command')
    types = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types() if t.name in wanted}
    events, servo = [], []
    while reader.has_next():
        topic, data, rx = reader.read_next()
        if topic not in types:
            continue
        m = deserialize_message(data, types[topic])
        if topic == '/scan':
            events.append((stamp(m.header), 2, len(events), (np.asarray(m.ranges, np.float32), m.angle_min,
                                                             m.angle_increment, m.time_increment)))
        elif topic == '/sensors/imu/raw':
            a, w = m.linear_acceleration, m.angular_velocity
            events.append((stamp(m.header), 0, len(events), (np.deg2rad(w.z), -a.y * G, a.x * G, a.z)))
        elif topic == '/sensors/core':
            events.append((stamp(m.header), 1, len(events), (m.state.speed / ERPM_PER_MPS, m.state.current_motor)))
        else:
            servo.append((rx * 1e-9, m.data))
    return events, np.array(servo)


def main(session, out):
    events, servo = read(session.rstrip('/') + '/bag')
    f = LidarInertialFilter(use_map=False)
    events = sorted((t + (f.c['scan_time_offset'] if k == 2 else 0), k, i, d) for t, k, i, d in events)
    rows, wheel, current, last_scan = [], 0.0, 0.0, -np.inf
    for t, kind, _, d in events:
        if kind == 0:
            f.imu(t, *d[:3])
            rows.append([t, f.x[3], f.x[4], f.u[0] - f.x[5], wheel, current, f.stat['icp_fail'], d[3], t - last_scan])
        elif kind == 1:
            wheel, current = d
            f.wheel(t, wheel)
        else:
            f.scan(t, *d)
            last_scan = t
    a = np.array(rows)
    t, vx, vy, w, cur = a[:, 0], a[:, 1], a[:, 2], a[:, 4], a[:, 5]
    dt = float(np.median(np.diff(t))); k = int(round(1 / dt))
    box = lambda x, n: np.convolve(x.astype(float), np.ones(n), 'same') > 0
    az = np.convolve(a[:, 7], np.ones(k) / k, 'same')
    speed = np.hypot(vx, vy)
    lifted = box(np.abs(az - np.median(a[:, 7])) > 0.08, 4 * k + 1)                 # tilted for a second: picked up
    lost = box((np.r_[0, np.diff(a[:, 6])] > 0) | (a[:, 8] > 0.3), 2 * k + 1)        # scan matching failed or no lidar
    pushed = (speed > 0.3) & (np.abs(w) < 0.05) & (np.abs(cur) < 0.5)               # moved by hand
    carried = box(np.convolve(pushed.astype(float), np.ones(k) / k, 'same') > 0.9, 2 * k + 1)
    clean = ((np.abs(w) > 0.2) | (speed > 0.3)) & ~lifted & ~lost & ~carried
    np.savez_compressed(out, t=t, vx=vx, vy=vy, r=a[:, 3], w=w, current=cur,
                        servo=np.interp(t, servo[:, 0], servo[:, 1]), clean=clean)
    print(f'{out}: {len(t)} samples ({t[-1] - t[0]:.0f} s), clean moving {clean.sum() * dt:.0f} s, filter {f.stat}')


if __name__ == '__main__':
    main(*sys.argv[1:3])
