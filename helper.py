import io
import json
import os
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment

DEFAULT_SR = 16000


def convert_to_wav_bytes(input_audio_path):
    """
    Converts any audio file to WAV and returns the WAV bytes.

    Parameters
    ----------
    input_audio_path : str
        Path to input audio file.

    Returns
    -------
    bytes
        WAV audio content in bytes
    """

    audio = AudioSegment.from_file(input_audio_path)
    wav_buffer = io.BytesIO()
    audio.export(wav_buffer, format="wav")
    wav_buffer.seek(0)
    
    return wav_buffer.read()


def ensure_wav(input_path: Path, work_dir: Path) -> Path:
    if input_path.suffix.lower() == ".wav":
        return input_path

    work_dir.mkdir(parents=True, exist_ok=True)
    output_path = work_dir / "input.wav"
    audio = AudioSegment.from_file(str(input_path))
    audio.export(output_path, format="wav")
    return output_path


def load_mono(path: Path, sr: int = DEFAULT_SR) -> Tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, always_2d=True)
    mono = audio.mean(axis=1).astype(np.float32)
    if sample_rate != sr:
        mono = librosa.resample(mono, orig_sr=sample_rate, target_sr=sr).astype(np.float32)
        sample_rate = sr
    return mono, sample_rate


def save_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr)


def load_json(path: Path, default):
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_env_value(key: str, env_path: Path) -> str | None:
    if key in os.environ and os.environ[key].strip():
        return os.environ[key].strip()
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        k, v = raw.split("=", 1)
        if k.strip() == key:
            return v.strip().strip("\"'")
    return None