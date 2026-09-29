from pathlib import Path


def write_rttm(segments, recording_id, output_path):
    if not recording_id or any(c.isspace() for c in recording_id):
        raise ValueError("recording_id must be nonempty and contain no whitespace")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for s in segments:
        if s.start < 0 or s.end <= s.start:
            raise ValueError("Invalid segment bounds")
        lines.append(f"SPEAKER {recording_id} 1 {s.start:.3f} {s.end-s.start:.3f} <NA> <NA> speaker_{s.speaker} <NA> <NA>\n")
    path.write_text("".join(lines), encoding="utf-8")
