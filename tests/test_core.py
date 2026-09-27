import unittest

from evaluate import validate
from run_submission import clean_events
from src.geometry import SceneModel
from src.segments import clip_to_duration, flags_to_segments
from src.tracker import Tracker, iou


class TrackerTests(unittest.TestCase):
    def test_tracker_keeps_id_for_moving_box(self):
        tracker = Tracker()
        first = tracker.update([(0, 0, 20, 20)], ["vehicle"], 0.0)
        second = tracker.update([(2, 0, 22, 20)], ["vehicle"], 0.1)

        self.assertEqual(first[0].id, second[0].id)
        self.assertGreater(second[0].speed()[0], 0.0)

    def test_tracker_drops_stale_tracks(self):
        tracker = Tracker(max_age=0.5)
        tracker.update([(0, 0, 20, 20)], ["vehicle"], 0.0)
        tracks = tracker.update([], [], 0.6)

        self.assertEqual(tracks, [])

    def test_iou_handles_non_overlapping_boxes(self):
        self.assertEqual(iou((0, 0, 1, 1), (2, 2, 3, 3)), 0.0)


class SegmentTests(unittest.TestCase):
    def test_segments_merge_short_gap_and_use_end_time(self):
        flags = [False, True, True, False, True, True]
        times = [0.0, 1.0, 2.0, 2.4, 3.0, 4.0]

        result = flags_to_segments(
            flags, times, "accident", min_dur=0.5, merge_gap=1.0, end_time=5.0
        )

        self.assertEqual(result, [[1.0, 5.0, "accident"]])

    def test_clip_to_duration_removes_empty_segments(self):
        result = clip_to_duration([[-1.0, 0.0, "x"], [1.0, 8.0, "x"]], 5.0)

        self.assertEqual(result, [[1.0, 5.0, "x"]])


class HarnessTests(unittest.TestCase):
    def test_clean_events_drops_invalid_and_overlapping_events(self):
        events = [
            [0, 2, "accident"],
            [1, 3, "accident"],
            [-1, 1, "near_miss"],
            [2, 8, "wrong_way"],
            [0, 1, "not_official"],
        ]

        cleaned, problems = clean_events(
            events,
            ["accident", "near_miss", "wrong_way"],
            duration=5.0,
        )

        self.assertEqual(cleaned, [[0.0, 2.0, "accident"], [2.0, 5.0, "wrong_way"]])
        self.assertEqual(len(problems), 3)

    def test_prediction_format_is_valid(self):
        prediction = {
            "team": "tests",
            "videos": {
                "clip.mp4": {
                    "events": [[0.0, 1.0, "accident"]],
                    "risk": [[0.0, 0.2], [0.5, 0.8]],
                }
            },
        }

        errors, warnings = validate(prediction)

        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])


class GeometryTests(unittest.TestCase):
    def test_scene_bounds_are_safe(self):
        scene = SceneModel(10, 10)

        self.assertFalse(scene.on_road(-1, 4))
        self.assertFalse(scene.on_crossing(10, 4))


if __name__ == "__main__":
    unittest.main()
