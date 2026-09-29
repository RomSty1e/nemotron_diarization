import argparse
from pathlib import Path
from inference.postprocess import probabilities_to_segments
from inference.rttm import write_rttm


def main():
    p = argparse.ArgumentParser(description="Saved probabilities -> RTTM, without loading NeMo")
    p.add_argument("--probs", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--min-speech-duration", type=float, default=0.10)
    p.add_argument("--max-gap-duration", type=float, default=0.10)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if Path(args.output).exists() and not args.overwrite:
        raise FileExistsError("Output exists; use --overwrite")
    import torch
    data = torch.load(args.probs, map_location="cpu", weights_only=True)
    segments = probabilities_to_segments(
        data["probs"].numpy(), data["frame_duration"], threshold=args.threshold,
        min_speech_duration=args.min_speech_duration, max_gap_duration=args.max_gap_duration,
        audio_duration=data.get("audio_duration"),
    )
    write_rttm(segments, data["recording_id"], args.output)
    print(f"Saved {args.output}; segments={len(segments)}")


if __name__ == "__main__":
    main()
