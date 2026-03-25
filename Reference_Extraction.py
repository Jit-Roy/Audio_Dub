import numpy as np
import soundfile as sf
import librosa
from typing import List, Dict, Tuple
from pathlib import Path

def get_tts_reference(
    speaker_audio_path: str | Path,
    segments: List[Dict],
    output_ref_path: str | Path,
    target_duration: float = 7.0,
    max_duration: float = 12.0,
    sr: int = 16000
) -> Tuple[str, str]:
    """
    Extracts approximately `target_duration` seconds of continuous speech and its corresponding text
    from the speaker track to use as a reference for TTS, avoiding overloading the TTS model.
    """
    if not segments:
        return str(speaker_audio_path), ""
        
    speaker_audio_path = Path(speaker_audio_path)
    output_ref_path = Path(output_ref_path)
    
    # Load the full track
    audio_data, file_sr = sf.read(str(speaker_audio_path), always_2d=True)
    audio_data = audio_data.mean(axis=1).astype(np.float32)
    if file_sr != sr:
        audio_data = librosa.resample(audio_data, orig_sr=file_sr, target_sr=sr).astype(np.float32)
    
    best_text_parts = []
    best_audio_clips = []
    accumulated_duration = 0.0
    
    # Strategy 1: Find a single segment that is perfectly sized (3s to max_duration)
    for seg in segments:
        start = float(seg["start"])
        end = float(seg["end"])
        dur = end - start
        text = seg.get("source_text", seg.get("text", "")).strip()
        
        if text and (3.0 <= dur <= max_duration) and len(text) < 300:
            start_idx = int(start * sr)
            end_idx = int(end * sr)
            clip = audio_data[start_idx:end_idx]
            
            output_ref_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_ref_path), clip, sr)
            return str(output_ref_path), text

    # Strategy 2: Accumulate smaller segments until we hit a decent length
    for seg in segments:
        start = float(seg["start"])
        end = float(seg["end"])
        text = seg.get("source_text", seg.get("text", "")).strip()
        dur = end - start
        
        if not text:
            continue
            
        # Avoid appending a huge chunk if we already have some audio
        if accumulated_duration > 0 and (accumulated_duration + dur) > max_duration:
            break
            
        # If this chunk is excessively long on its own and we have no audio, just take it and break
        # but only if we truly can't avoid it. Hopefully it's not a 1-minute chunk!
        if accumulated_duration == 0 and dur > max_duration:
            start_idx = int(start * sr)
            end_idx = int(end * sr)
            clip = audio_data[start_idx:end_idx]
            best_audio_clips.append(clip)
            best_text_parts.append(text)
            break
            
        start_idx = int(start * sr)
        end_idx = int(end * sr)
        clip = audio_data[start_idx:end_idx]
        
        best_audio_clips.append(clip)
        best_text_parts.append(text)
        
        silence = np.zeros(int(0.2 * sr), dtype=np.float32)
        best_audio_clips.append(silence)
        
        accumulated_duration += dur
        if accumulated_duration >= (target_duration - 1.0):
            break
            
    if not best_audio_clips:
        return str(speaker_audio_path), " ".join([s.get("source_text", s.get("text", "")).strip() for s in segments][:3])
        
    merged_audio = np.concatenate(best_audio_clips, axis=0)
    merged_text = " ".join(best_text_parts).strip()
    
    # Failsafe: if the text is ridiculously long from a giant segment, truncate the text string
    # (Not ideal for TTS alignment, but prevents CUDA crash)
    if len(merged_text) > 400:
        merged_text = merged_text[:400]
    
    output_ref_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_ref_path), merged_audio, sr)
    
    return str(output_ref_path), merged_text
