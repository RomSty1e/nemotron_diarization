def infer_probabilities(model, audio_path):
    """File inference using NeMo's internal streaming engine; not a live step API."""
    import torch

    with torch.inference_mode():
        result = model.diarize(audio=[str(audio_path)], batch_size=1, include_tensor_outputs=True)
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError("Unexpected NeMo diarize return format")
    _, predictions = result
    if len(predictions) != 1:
        raise RuntimeError("Expected one prediction for one recording")
    probs = predictions[0]
    if probs.ndim == 3 and probs.shape[0] == 1:
        probs = probs[0]
    if probs.ndim != 2 or probs.shape[0] == 0:
        raise RuntimeError(f"Invalid probability shape: {tuple(probs.shape)}")
    probs = probs.detach().cpu().float()
    if not torch.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
        raise RuntimeError("Expected finite probabilities in [0, 1]")
    return probs
