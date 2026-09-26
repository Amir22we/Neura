import numpy as np

from src.tracker import Tracker
from src.geometry import SceneModel
from src.rules import track_flags, pair_flags, crossing_flag, congestion_flag, stopline_flags
from src.segments import flags_to_segments
from src.signals import TrafficLightState

WIDTH, HEIGHT = 640, 480
FPS = 25.0

tracker = Tracker()
scene = SceneModel(WIDTH, HEIGHT)
tls = TrafficLightState()

flags_over_time = {k: [] for k in [
    "wrong_way", "stopped_vehicle", "jaywalking", "illegal_u_turn",
    "illegal_turn", "solid_line_crossing", "accident", "near_miss",
    "failure_to_yield", "congestion", "red_light", "stop_line",
]}
times = []

for frame_idx in range(400):
    t = frame_idx / FPS
    boxes = []
    classes = []

    x = 100 + frame_idx * 1.5
    boxes.append((x, 300, x + 40, 340))
    classes.append("vehicle")

    x2 = 500 - frame_idx * 1.5
    boxes.append((x2, 300, x2 + 40, 340))
    classes.append("vehicle")

    if frame_idx > 150:
        boxes.append((300, 100 + max(0, frame_idx - 150) * 0.5, 320, 180 + max(0, frame_idx - 150) * 0.5))
        classes.append("pedestrian")

    if frame_idx > 250:
        boxes.append((450, 350, 470, 370))
        classes.append("vehicle")

    tracks = tracker.update(boxes, classes, t)
    for tr in tracks:
        scene.observe(tr)

    if frame_idx == 100:
        scene.calibrate()

    active = [tr for tr in tracks if tr.time_since_update == 0]
    frame_flags = {k: False for k in flags_over_time}

    for tr in active:
        for k, v in track_flags(tr, scene).items():
            frame_flags[k] = frame_flags[k] or v

    accident, near_miss = pair_flags(active)
    frame_flags["accident"] = accident
    frame_flags["near_miss"] = near_miss
    frame_flags["failure_to_yield"] = crossing_flag(active, scene)
    frame_flags["congestion"] = congestion_flag(active, scene)
    red_light, stop_line = stopline_flags(active, scene, tls.state)
    frame_flags["red_light"] = red_light
    frame_flags["stop_line"] = stop_line

    times.append(t)
    for k in flags_over_time:
        flags_over_time[k].append(frame_flags[k])

events = []
for k, flags in flags_over_time.items():
    events.extend(flags_to_segments(flags, times, k))

print("tracks created:", tracker.next_id - 1)
print("events found:", len(events))
for e in events:
    print(e)
print("SMOKE TEST OK")
