import numpy as np
import librosa
from typing import List
from utils.audio_adjustment import adjust_audio_duration

def time_stretch_audio(audio: np.ndarray, sr: int, target_duration: float) -> np.ndarray:
    """Adjust audio duration using pause-aware time-stretching with bounds checking."""
    if len(audio) == 0 or target_duration <= 0:
        return audio
        
    original_duration = len(audio) / sr
    if original_duration <= 0:
        return audio
        
    rate = original_duration / target_duration
    
    # Don't stretch if it would sound completely garbled (e.g. > 2.5x speed or < 0.4x speed)
    if 0.4 <= rate <= 2.5:
        print(f"      [AudioOps] Time-stretch to {target_duration:.2f}s (rate: {rate:.2f}x)")
        try:
            return adjust_audio_duration(audio.astype(np.float32), sr, target_duration)
        except Exception as e:
            print(f"      [ERROR] Audio adjustment failed: {e}")
            raise e
    else:
        # Instead of failing silently or bleeding, raise an error to trigger an LLM rewrite
        raise ValueError(f"Time stretch bounds exceeded! Audio is {original_duration:.2f}s, target is {target_duration:.2f}s (rate {rate:.2f}x).")

def overlay_audio(base: np.ndarray, overlay: np.ndarray, start_sample: int) -> np.ndarray:
    """Safely overlay generated audio onto a base track at a specific sample index."""
    if overlay.size == 0:
        return base
    end_sample = start_sample + len(overlay)
    if end_sample > len(base):
        base = np.pad(base, (0, end_sample - len(base)))
    base[start_sample:end_sample] += overlay
    return base

def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Standardizes sample rates."""
    if orig_sr == target_sr:
        return audio
    return librosa.resample(audio.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr)

def mix_audio_tracks(tracks: List[np.ndarray]) -> np.ndarray:
    """Mixes multiple tracks down to a single normalized mono track."""
    if not tracks:
        return np.array([], dtype=np.float32)

    max_len = max(len(t) for t in tracks)
    mixed = np.zeros(max_len, dtype=np.float32)
    for t in tracks:
        if len(t) < max_len:
            t = np.pad(t, (0, max_len - len(t)))
        mixed += t

    peak = np.max(np.abs(mixed)) if mixed.size else 0.0
    if peak > 1.0:
        mixed = mixed / peak
    return mixed