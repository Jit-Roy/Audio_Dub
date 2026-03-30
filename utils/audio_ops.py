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
    clamped_rate = max(0.4, min(rate, 2.5))
    
    clamped_target_duration = original_duration / clamped_rate
    
    if rate != clamped_rate:
        print(f"      [AudioOps] Warning: Rate {rate:.2f}x exceeds bounds. Clamping to {clamped_rate:.2f}x.")
    else:
        print(f"      [AudioOps] Time-stretch to {target_duration:.2f}s (rate: {rate:.2f}x)")
        
    try:
        stretched = adjust_audio_duration(audio.astype(np.float32), sr, clamped_target_duration)
        
        # Pad or truncate to match exactly the target duration
        target_length = int(target_duration * sr)
        if len(stretched) < target_length:
            stretched = np.pad(stretched, (0, target_length - len(stretched)))
        elif len(stretched) > target_length:
            stretched = stretched[:target_length]
            
        return stretched
    except Exception as e:
        print(f"      [ERROR] Audio adjustment failed: {e}")
        raise e

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