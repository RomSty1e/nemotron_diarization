from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    speaker: int
    score: float


def probabilities_to_segments(probs, frame_duration, threshold=0.5,
                              min_speech_duration=0.10, max_gap_duration=0.10,
                              audio_duration=None):
    """Merge gaps, then filter duration independently per speaker; retain overlap.

    Accepts a NumPy array or a CPU tensor. Segment intervals are half-open.
    """
    probs = np.asarray(probs, dtype=np.float32)
    if probs.ndim != 2 or probs.shape[1] == 0:
        raise ValueError("Expected [T, speakers]")
    if not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1)):
        raise ValueError("Probabilities must be finite and within [0, 1]")
    if not math.isfinite(frame_duration) or frame_duration <= 0:
        raise ValueError("frame_duration must be positive")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be within [0, 1]")
    if any(not math.isfinite(x) or x < 0 for x in (min_speech_duration, max_gap_duration)):
        raise ValueError("Durations must be finite and nonnegative")
    if audio_duration is not None and (not math.isfinite(audio_duration) or audio_duration <= 0):
        raise ValueError("audio_duration must be positive")
    limit = len(probs) * frame_duration if audio_duration is None else audio_duration
    min_frames = max(1, math.ceil(min_speech_duration / frame_duration - 1e-9))
    max_gap_frames = math.floor(max_gap_duration / frame_duration + 1e-9)
    segments = []
    for speaker in range(probs.shape[1]):
        active = probs[:, speaker] >= threshold
        changes = np.diff(np.concatenate(([False], active, [False])).astype(np.int8))
        regions = list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))
        merged = []
        for start, end in regions:
            if merged and start - merged[-1][1] <= max_gap_frames:
                merged[-1][1] = int(end)
            else:
                merged.append([int(start), int(end)])
        for start, end in merged:
            begin, finish = start * frame_duration, min(end * frame_duration, limit)
            if end - start < min_frames or finish <= begin or finish - begin + 1e-9 < min_speech_duration:
                continue
            segments.append(Segment(begin, finish, speaker, float(probs[start:end, speaker].mean())))
    return sorted(segments, key=lambda s: (s.start, s.end, s.speaker))
