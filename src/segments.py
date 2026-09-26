from __future__ import annotations


def flags_to_segments(
    flags,
    times,
    label,
    min_dur=0.5,
    merge_gap=1.0,
    end_time=None,
):
    """Convert sampled boolean flags into non-overlapping time segments."""
    segments = []
    start = None
    last_true_t = None
    for f, t in zip(flags, times):
        if f:
            if start is None:
                start = t
            last_true_t = t
        else:
            if start is not None and last_true_t is not None and (t - last_true_t) > merge_gap:
                end = t
                if end - start >= min_dur:
                    segments.append([start, end, label])
                start = None
                last_true_t = None
    if start is not None and last_true_t is not None:
        end = end_time if end_time is not None else last_true_t
        if end - start >= min_dur:
            segments.append([start, end, label])
    return segments


def clip_to_duration(segments, duration):
    out = []
    for s, e, label in segments:
        s = max(0.0, min(s, duration))
        e = max(0.0, min(e, duration))
        if e > s:
            out.append([s, e, label])
    return out
