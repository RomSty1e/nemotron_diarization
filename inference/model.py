from pathlib import Path


def load_model(model_path, device="cuda", strict=False):
    import torch
    from nemo.collections.asr.models import SortformerEncLabelModel

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Check driver and PyTorch; use --device cpu for diagnostics.")
    path = Path(model_path).expanduser()
    if path.suffix == ".nemo":
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint missing: {path}. Run bash scripts/download_model.sh")
        model = SortformerEncLabelModel.restore_from(
            restore_path=str(path.resolve()), map_location="cpu", strict=strict
        )
    else:
        model = SortformerEncLabelModel.from_pretrained(model_path)
    return model.to(device).eval()


def configure_streaming(model, config):
    if not getattr(model, "streaming_mode", False):
        raise RuntimeError("Expected a streaming Nemotron checkpoint; streaming_mode is False.")
    for key in ("spkcache_len", "fifo_len", "chunk_len", "chunk_right_context", "spkcache_update_period"):
        value = config[key]
        if type(value) is not int or value < 0:
            raise ValueError(f"Invalid {key}: {value}")
        setattr(model.sortformer_modules, key, value)
    model._check_streaming_parameters()


def frame_duration(model):
    # NeMo's timestamp conversion uses this factor in units of 10 ms.
    factor = getattr(model, "output_subsampling_factor", None)
    if not isinstance(factor, int) or factor < 1:
        raise RuntimeError("Missing/invalid output_subsampling_factor; verify the NeMo revision.")
    return factor * 0.010
