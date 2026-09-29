import argparse
import json
from pathlib import Path

from inference.audio import inspect_audio
from inference.model import load_model, configure_streaming, frame_duration
from inference.offline import infer_probabilities
from inference.postprocess import probabilities_to_segments
from inference.rttm import write_rttm


def main():
    parser = argparse.ArgumentParser(description="WAV -> Nemotron probabilities -> RTTM")
    parser.add_argument("--config", default="configs/offline.yaml")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output-dir", default="outputs/test")
    parser.add_argument("--recording-id")
    parser.add_argument("--device", help="Override config, e.g. cuda:0 or cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    import torch
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(args.config)
    if args.device:
        cfg.model.device = args.device
    path, duration = inspect_audio(args.audio)
    recording_id = args.recording_id or path.stem
    if any(c.isspace() for c in recording_id) or "/" in recording_id or "\\" in recording_id:
        raise ValueError("Use a recording ID without spaces or path separators")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    probs_path, rttm_path = out / f"{recording_id}.probs.pt", out / f"{recording_id}.rttm"
    meta_path = out / f"{recording_id}.run.json"
    if not args.overwrite and any(p.exists() for p in (probs_path, rttm_path, meta_path)):
        raise FileExistsError("Output exists. Use another output directory or --overwrite")
    print("Loading model...", flush=True)
    model = load_model(cfg.model.path, cfg.model.device, bool(cfg.model.get("strict", False)))
    configure_streaming(model, OmegaConf.to_container(cfg.inference, resolve=True))
    step = frame_duration(model)
    print(f"Running inference; frame duration={step:.3f}s", flush=True)
    probs = infer_probabilities(model, path)
    print(f"Probabilities: {tuple(probs.shape)}, audio={duration:.3f}s", flush=True)
    if abs(len(probs) * step - duration) > max(1.0, 2 * step):
        raise RuntimeError("Probability timeline differs from audio duration. Check NeMo output resolution before writing RTTM.")
    metadata = {
        "recording_id": recording_id, "audio_path": str(path), "audio_duration": duration,
        "frame_duration": step, "torch_version": str(torch.__version__),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    revision_path = Path(__file__).resolve().parents[1] / "third_party/Speech.commit"
    metadata["nemo_commit"] = revision_path.read_text().strip() if revision_path.exists() else "unknown"
    torch.save({"probs": probs, **metadata}, probs_path)
    segments = probabilities_to_segments(probs.numpy(), step, audio_duration=duration, **dict(cfg.postprocess))
    write_rttm(segments, recording_id, rttm_path)
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {probs_path}\nSaved {rttm_path}\nSegments: {len(segments)}")


if __name__ == "__main__":
    main()
