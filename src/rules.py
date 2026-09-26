from __future__ import annotations

import numpy as np

STOP_SPEED = 2.0
STOPPED_DURATION = 10.0
STOPPED_DRIFT_PX = 15.0
NEAR_MISS_DECEL = 35.0
NEAR_MISS_PROXIMITY_SCALE = 2.5
ACCIDENT_IOU = 0.15
ACCIDENT_SPEED = 3.0
WRONG_WAY_ANGLE = 2.35
TURN_ANGLE_LOW = 0.9
TURN_ANGLE_HIGH = 2.5
UTURN_ANGLE = 2.6
LANE_LATERAL_PX = 35.0
CONGESTION_SPEED = 8.0
CONGESTION_MIN_VEHICLES = 2
STOPLINE_BAND_PX = 60.0
REDLIGHT_BAND_PX = 120.0


def _heading(vx, vy):
    return float(np.arctan2(vy, vx))


def _iou(a, b):
    xx1 = max(a[0], b[0])
    yy1 = max(a[1], b[1])
    xx2 = min(a[2], b[2])
    yy2 = min(a[3], b[3])
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _deceleration(track, window=0.6):
    if len(track.history) < 3:
        return 0.0
    t_now = track.history[-1][0]
    idx_mid = len(track.history) - 1
    while idx_mid > 0 and t_now - track.history[idx_mid][0] < window / 2.0:
        idx_mid -= 1
    idx_old = idx_mid
    while idx_old > 0 and t_now - track.history[idx_old][0] < window:
        idx_old -= 1

    def spd(i0, i1):
        t0, b0 = track.history[i0]
        t1, b1 = track.history[i1]
        dt = max(t1 - t0, 1e-3)
        c0 = ((b0[0] + b0[2]) / 2.0, (b0[1] + b0[3]) / 2.0)
        c1 = ((b1[0] + b1[2]) / 2.0, (b1[1] + b1[3]) / 2.0)
        d = ((c1[0] - c0[0]) ** 2 + (c1[1] - c0[1]) ** 2) ** 0.5
        return d / dt

    s_old = spd(idx_old, idx_mid)
    s_new = spd(idx_mid, len(track.history) - 1)
    return max(0.0, s_old - s_new)


def track_flags(track, scene):
    flags = dict(
        wrong_way=False,
        stopped_vehicle=False,
        jaywalking=False,
        illegal_u_turn=False,
        illegal_turn=False,
        solid_line_crossing=False,
    )
    cx, cy = track.centroid()
    speed, (vx, vy) = track.speed(1.0)

    if track.cls == "pedestrian":
        if scene.on_road(cx, cy) and not scene.on_crossing(cx, cy):
            flags["jaywalking"] = True
        return flags

    if track.cls != "vehicle":
        return flags

    if speed < STOP_SPEED:
        t_now = track.history[-1][0]
        t_start = t_now
        for t, b in reversed(track.history):
            c = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
            d = ((c[0] - cx) ** 2 + (c[1] - cy) ** 2) ** 0.5
            if d > STOPPED_DRIFT_PX:
                break
            t_start = t
        if t_now - t_start >= STOPPED_DURATION:
            flags["stopped_vehicle"] = True

    if scene.calibrated and speed > 3.0:
        heading = _heading(vx, vy)
        nearest = scene.nearest_flow_angle(heading)
        if nearest is not None:
            diff = scene.angle_diff(heading, nearest)
            if diff > WRONG_WAY_ANGLE:
                flags["wrong_way"] = True
            elif diff < TURN_ANGLE_LOW and len(track.history) > 6:
                perp = nearest + np.pi / 2.0
                px, py = np.cos(perp), np.sin(perp)
                t1, b1 = track.history[-1]
                idx = len(track.history) - 1
                while idx > 0 and t1 - track.history[idx][0] < 2.0:
                    idx -= 1
                t0, b0 = track.history[idx]
                c0 = ((b0[0] + b0[2]) / 2.0, (b0[1] + b0[3]) / 2.0)
                lateral = (cx - c0[0]) * px + (cy - c0[1]) * py
                if abs(lateral) > LANE_LATERAL_PX:
                    flags["solid_line_crossing"] = True

    if speed > 2.0 and len(track.history) > 10:
        t1, b1 = track.history[-1]
        idx = len(track.history) - 1
        while idx > 0 and t1 - track.history[idx][0] < 2.5:
            idx -= 1
        t0, b0 = track.history[idx]
        c0 = ((b0[0] + b0[2]) / 2.0, (b0[1] + b0[3]) / 2.0)
        d0x, d0y = cx - c0[0], cy - c0[1]
        v0speed = (d0x ** 2 + d0y ** 2) ** 0.5
        if v0speed > 5.0:
            h0 = _heading(d0x, d0y)
            h1 = _heading(vx, vy)
            diff = abs(((h1 - h0 + np.pi) % (2 * np.pi)) - np.pi)
            if diff > UTURN_ANGLE:
                flags["illegal_u_turn"] = True
            elif TURN_ANGLE_LOW < diff < TURN_ANGLE_HIGH:
                flags["illegal_turn"] = True

    return flags


def pair_flags(tracks):
    accident = False
    near_miss = False
    active = [t for t in tracks if t.cls in ("vehicle", "pedestrian")]
    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            a, b = active[i], active[j]
            sa, _ = a.speed(0.5)
            sb, _ = b.speed(0.5)
            iou = _iou(a.bbox, b.bbox)
            if iou > ACCIDENT_IOU and sa < ACCIDENT_SPEED and sb < ACCIDENT_SPEED:
                accident = True
                continue
            ca, cb = a.centroid(), b.centroid()
            dist = ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5
            scale = max(a.bbox[2] - a.bbox[0], b.bbox[2] - b.bbox[0], 1.0)
            dec_a, dec_b = _deceleration(a), _deceleration(b)
            if dist < scale * NEAR_MISS_PROXIMITY_SCALE and iou < ACCIDENT_IOU:
                if dec_a > NEAR_MISS_DECEL or dec_b > NEAR_MISS_DECEL:
                    near_miss = True
    return accident, near_miss


def crossing_flag(tracks, scene):
    peds = [
        t
        for t in tracks
        if t.cls == "pedestrian" and t.time_since_update == 0 and scene.on_crossing(*t.centroid())
    ]
    if not peds:
        return False
    for tr in tracks:
        if tr.cls != "vehicle" or tr.time_since_update != 0:
            continue
        speed, _ = tr.speed(0.6)
        if speed > STOP_SPEED and scene.on_crossing(*tr.centroid()):
            return True
    return False


def congestion_flag(tracks, scene):
    if not scene.flow_clusters:
        return False
    for angle in scene.flow_clusters:
        speeds = []
        for tr in tracks:
            if tr.cls != "vehicle" or tr.time_since_update != 0:
                continue
            speed, (vx, vy) = tr.speed(0.8)
            if speed < 0.5:
                continue
            heading = _heading(vx, vy)
            if scene.angle_diff(heading, angle) < 0.6:
                speeds.append(speed)
        if len(speeds) >= CONGESTION_MIN_VEHICLES and float(np.mean(speeds)) < CONGESTION_SPEED:
            return True
    return False


def stopline_flags(tracks, scene, tl_state):
    if scene.stop_line_y is None:
        return False, False
    red_light = False
    stop_line = False
    forward = scene.stop_line_forward
    for tr in tracks:
        if tr.cls != "vehicle" or tr.time_since_update != 0:
            continue
        cx, cy = tr.centroid()
        speed, _ = tr.speed(0.6)
        before_line = (cy - scene.stop_line_y) * forward < 0
        dist_to_line = abs(cy - scene.stop_line_y)
        if tl_state == "red":
            if (not before_line) and speed > STOP_SPEED and dist_to_line < REDLIGHT_BAND_PX:
                red_light = True
            if (not before_line) and speed < STOP_SPEED and dist_to_line < STOPLINE_BAND_PX:
                stop_line = True
    return red_light, stop_line
