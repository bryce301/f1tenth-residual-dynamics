#!/usr/bin/env python3
"""Recorded session -> portable training file. Runs on the vehicle computer (needs ROS and the estimator package).

    source /opt/ros/jazzy/setup.bash
    source ~/f1tenth_ws/install/setup.bash
    source ~/f1tenth_ws/final_project/test/f1tenth_state_estimation/install/setup.bash
    python3 make_dataset.py <session directory> data/<name>.npz [--segments 1 3]

The raw sensors are re-filtered offline with the lidar-inertial-wheel EKF (no map, no output extrapolation),
so the result does not depend on which estimator ran during recording. Output, 50 Hz at the IMU stamps:
  t, vx, vy, r   velocity of the rear-axle centre in body axes [m/s], yaw rate [rad/s]
  w              wheel peripheral speed from ERPM [m/s]
  current        motor current [A];  servo  commanded servo position (0..1)
  clean          sample usable for training: moving, not lifted or carried, lidar present and matching,
                 away from impacts and wall contact, and inside the selected recorder segments (all when none given)

Picked up: vertical acceleration offset above 0.08 g (1 s mean) with less than 3 m/s^2 horizontal
acceleration, or above 0.25 g; 2 s around it are dropped. (Before 2026-11-25 evening the horizontal
condition was missing and sustained drifts were dropped as "lifted".)

Impacts and wall contact (checked by hand against map position on the 2026-11-25 drift recordings):
  impact   horizontal IMU acceleration above COLLISION_MPS2; samples from 0.5 s before to 1.5 s after are dropped
  contact  a lidar return closer than CONTACT_M to the car outline; samples within 0.5 s are dropped
"""
import argparse
import json
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from f1tenth_state_estimation_impl.lidar_filter import LidarInertialFilter

G, ERPM_PER_MPS = 9.80665, 4308.2
COLLISION_MPS2 = 15.0      # hard drifting reaches 12-13 m/s^2 on this IMU; impacts peak at 17-140
CONTACT_M = 0.03
OUTLINE = (-0.12, 0.45, -0.15, 0.15)   # rear, front, right, left of the body from the rear axle [m] (Traxxas Slash 4x4)
stamp = lambda h: h.stamp.sec + h.stamp.nanosec * 1e-9


def outline_gap(ranges, angle_min, increment, laser_x):
    """Distance from the car outline to the closest valid lidar return."""
    r = np.asarray(ranges, float)
    a = angle_min + increment * np.arange(len(r))
    ok = np.isfinite(r) & (r > 0.05)
    x, y = r[ok] * np.cos(a[ok]) + laser_x, r[ok] * np.sin(a[ok])
    rear, front, right, left = OUTLINE
    return float(np.hypot(np.maximum(np.maximum(rear - x, x - front), 0), np.maximum(np.maximum(right - y, y - left), 0)).min(initial=np.inf))


def around(mask, before, after):
    """True within `before` samples before and `after` samples after any True sample."""
    out = np.zeros_like(mask)
    for i in np.flatnonzero(mask):
        out[max(i - before, 0):i + after + 1] = True
    return out


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


def main(session, out, segments):
    events, servo = read(session.rstrip('/') + '/bag')
    f = LidarInertialFilter(use_map=False)
    events = sorted((t + (f.c['scan_time_offset'] if k == 2 else 0), k, i, d) for t, k, i, d in events)
    rows, wheel, current, last_scan, gap = [], 0.0, 0.0, -np.inf, np.inf
    for t, kind, _, d in events:
        if kind == 0:
            f.imu(t, *d[:3])
            rows.append([t, f.x[3], f.x[4], f.u[0] - f.x[5], wheel, current, f.stat['icp_fail'], d[3], t - last_scan,
                         np.hypot(d[1], d[2]), gap])
        elif kind == 1:
            wheel, current = d
            f.wheel(t, wheel)
        else:
            f.scan(t, *d)
            last_scan, gap = t, outline_gap(*d[:3], f.c['laser_x'])
    a = np.array(rows)
    t, vx, vy, w, cur = a[:, 0], a[:, 1], a[:, 2], a[:, 4], a[:, 5]
    dt = float(np.median(np.diff(t))); k = int(round(1 / dt))
    mean = lambda x: np.convolve(x, np.ones(k), 'same') / np.convolve(np.ones_like(x), np.ones(k), 'same')   # 1 s
    tilt = np.abs(mean(a[:, 7]) - np.median(a[:, 7]))                                 # vertical accel offset [g]
    speed = np.hypot(vx, vy)
    # Picked up: tilted for a second without the horizontal acceleration of driving. A sustained drift
    # (5-6 m/s^2) shifts the vertical reading by up to 0.13 g, so a small tilt alone is not enough.
    lifted = around(((tilt > 0.08) & (mean(a[:, 9]) < 3.0)) | (tilt > 0.25), 2 * k, 2 * k)
    lost = around((np.r_[0, np.diff(a[:, 6])] > 0) | (a[:, 8] > 0.3), k, k)          # scan matching failed or no lidar
    pushed = (speed > 0.3) & (np.abs(w) < 0.05) & (np.abs(cur) < 0.5)               # moved by hand
    carried = around(np.convolve(pushed.astype(float), np.ones(k) / k, 'same') > 0.9, k, k)
    impact = a[:, 9] > COLLISION_MPS2
    contact = a[:, 10] < CONTACT_M
    moving = (np.abs(w) > 0.2) | (speed > 0.3)
    inside = np.ones_like(moving)
    if segments:
        marks = {g['id']: (g['start_s'], g['end_s']) for g in json.load(open(session.rstrip('/') + '/session.json'))['segments']}
        inside = np.any([(t >= marks[i][0]) & (t <= marks[i][1]) for i in segments], axis=0)
    reasons = dict(lifted=lifted, lidar_lost=lost, carried=carried, impact=around(impact, k // 2, 3 * k // 2),
                   wall_contact=around(contact, k // 2, k // 2))
    clean = moving & inside & ~np.any(list(reasons.values()), axis=0)
    removed = {name: round(float((moving & inside & mask).sum() * dt), 1) for name, mask in reasons.items()}
    np.savez_compressed(out, t=t, vx=vx, vy=vy, r=a[:, 3], w=w, current=cur,
                        servo=np.interp(t, servo[:, 0], servo[:, 1]), clean=clean)
    starts = lambda m: t[np.flatnonzero(np.diff(np.r_[0, around(m, k // 2, k // 2).astype(int)]) > 0)] - t[0]
    print(f'{out}: {len(t)} samples ({t[-1] - t[0]:.0f} s), clean moving {clean.sum() * dt:.0f} s; '
          f'{len(starts(impact))} impacts at {np.round(starts(impact), 1).tolist()} s; '
          f'{len(starts(contact))} wall-contact episodes ({contact.sum() * dt:.1f} s); '
          f'moving inside the segments {(moving & inside).sum() * dt:.0f} s, removed by rule [s] {removed}; filter {f.stat}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('session'); parser.add_argument('out')
    parser.add_argument('--segments', type=int, nargs='*', default=[], help='recorder segment ids to keep')
    args = parser.parse_args()
    main(args.session, args.out, args.segments)
