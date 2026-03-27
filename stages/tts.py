import time
import numpy as np
from core.pipeline import PipelineStage, PipelineContext
from core.config import config
from core.cache import CacheManager
from modules.Qwen3tts import load_tts_model, generate_voice_clone
from modules.Reference_Extraction import get_tts_reference
from modules.translator import translate_fragment
from utils.helper import load_mono, save_wav
from utils.audio_ops import time_stretch_audio, overlay_audio, resample_audio, mix_audio_tracks

class TTSAndMixStage(PipelineStage):
    @property
    def name(self) -> str:
        return "TTSAndMix"
        
    def execute(self, context: PipelineContext) -> None:
        tts_model = load_tts_model()
        cache = CacheManager(config.temp_dir, context.input_file)
        
        dubbed_tracks = []
        
        for session in context.speaker_sessions:
            if not session.segments:
                continue
                
            tts_cache_key = f"08_tts/tts_{session.name}.wav"
            tts_cache_path = cache.get_path(tts_cache_key)
            
            # Fetch Reference Audio
            ref_cache_key = f"08_tts/ref_audio/{session.name}_ref_7s.wav"
            ref_audio_path = cache.get_path(ref_cache_key)
            ref_audio_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Need to pass dictionary format to legacy reference extractor for now
            legacy_segments = [{"start": s.start, "end": s.end, "text": s.text} for s in session.segments]
            ref_audio_file, ref_text = get_tts_reference(
                speaker_audio_path=session.audio_path,
                segments=legacy_segments,
                output_ref_path=ref_audio_path,
                target_duration=7.0
            )
            session.reference_audio_path = ref_audio_file
            session.reference_text = ref_text
            
            # Synthesize or load cached speaker track
            if tts_cache_path.exists():
                print(f"  TTS -> {session.name} (cached total track)")
                dubbed_audio, _ = load_mono(tts_cache_path, sr=config.default_sr)
            else:
                print(f"  TTS -> {session.name}")
                base_audio, _ = load_mono(session.audio_path, sr=config.default_sr)
                dubbed_audio = np.zeros_like(base_audio)
                
                for seg in session.segments:
                    if not seg.translated_text or seg.failed:
                        continue
                        
                    segment_cache_key = f"08_tts/tts_segments/{session.name}/segment_{seg.index:04d}_stretched.wav"
                    segment_cache_path = cache.get_path(segment_cache_key)
                    
                    if segment_cache_path.exists():
                        cached_audio, _ = load_mono(segment_cache_path, sr=config.default_sr)
                        dubbed_audio = overlay_audio(dubbed_audio, cached_audio, int(seg.start * config.default_sr))
                        continue
                        
                    print(f"    [TTS] Generating segment {seg.index} ({seg.start:.2f}-{seg.end:.2f}s)")
                    
                    max_retries = 3
                    tts_success = False
                    last_error = None
                    
                    for attempt in range(max_retries):
                        tts_start = time.time()
                        try:
                            # 1. Generate Voice Clone
                            wavs, tts_sr = generate_voice_clone(
                                text=seg.translated_text,
                                language=config.target_language,
                                ref_audio=session.reference_audio_path,
                                ref_text=session.reference_text,
                                model=tts_model,
                            )
                            if not wavs:
                                raise ValueError("Empty wavs list returned from TTS generator.")
                                
                            tts_audio = wavs[0]
                            if isinstance(tts_audio, np.ndarray) and tts_audio.ndim > 1:
                                tts_audio = tts_audio.mean(axis=1)

                            print(f"      [TTS] Generation time: {time.time() - tts_start:.2f}s")

                            # 2. Resample
                            tts_audio = resample_audio(tts_audio, tts_sr, config.default_sr)
                            
                            # 3. Time Stretch (raises ValueError if bounds exceeded)
                            tts_audio = time_stretch_audio(tts_audio, config.default_sr, seg.duration)

                            # 4. Overlay onto track
                            start_sample = int(seg.start * config.default_sr)
                            dubbed_audio = overlay_audio(dubbed_audio, tts_audio, start_sample)
                            
                            # Cache the individual stretched segment
                            segment_cache_path.parent.mkdir(parents=True, exist_ok=True)
                            save_wav(segment_cache_path, tts_audio, config.default_sr)

                            tts_success = True
                            break  # Exit retry loop on success

                        except Exception as e:
                            last_error = e
                            print(f"    [Error] Attempt {attempt+1}/{max_retries} failed for segment {seg.index}: {e}")
                            if attempt < max_retries - 1:
                                print(f"    [LLM Fallback] Generating a new translation to fit {seg.duration:.2f}s window...")
                                try:
                                    seg.translated_text = translate_fragment(
                                        text_fragment=seg.text, 
                                        target_language=config.target_language, 
                                        target_duration=seg.duration, 
                                        target_chars=seg.target_chars
                                    )
                                except Exception as llm_e:
                                    print(f"      [LLM Fallback Error]: {llm_e}")
                                    
                    if not tts_success:
                        raise RuntimeError(f"Segment {seg.index} could not be successfully generated after {max_retries} attempts. Final error: {last_error}")
                
                # Assume if >90% succeeded, saving partial mix is worth it for cache
                save_wav(tts_cache_path, dubbed_audio, config.default_sr)
                
            dubbed_tracks.append(dubbed_audio)

        print("  Mixing all tracks (Music + Dubbed Speakers)...")
        tracks = []
        if context.music_path and context.music_path.exists():
            music_audio, _ = load_mono(context.music_path, sr=config.default_sr)
            tracks.append(music_audio)
            
        tracks.extend(dubbed_tracks)
        final_mix = mix_audio_tracks(tracks)
        
        final_path = config.output_dir / "final_mix.wav"
        save_wav(final_path, final_mix, config.default_sr)
        print(f"  Final mix successfully saved to: {final_path}")
