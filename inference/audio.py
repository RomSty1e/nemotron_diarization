from pathlib import Path


def inspect_audio(path):
    import soundfile as sf

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    info = sf.info(path)
    if info.channels != 1 or info.samplerate != 16000:
        raise ValueError(f"Expected mono 16000 Hz; got {info.channels} channels / {info.samplerate} Hz. Convert with ffmpeg (README).")
    if info.frames <= 0:
        raise ValueError("Empty audio")
    return path, info.frames / info.samplerate
