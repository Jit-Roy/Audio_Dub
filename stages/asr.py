from core.pipeline import PipelineStage, PipelineContext
from core.config import config
from core.cache import CacheManager
from core.models import Segment
from modules.ASR import transcribe_audio, WHISPER_ID

class ASRStage(PipelineStage):
    @property
    def name(self) -> str:
        return "ASR"
        
    def execute(self, context: PipelineContext) -> None:
        if not context.speaker_sessions:
            raise RuntimeError("No speaker sessions found. Run DiarizationStage first.")
            
        cache = CacheManager(config.temp_dir, context.input_file)
        
        for session in context.speaker_sessions:
            asr_cache_key = f"06_asr/asr_{session.name}.json"
            
            if cache.exists(asr_cache_key):
                print(f"  ASR -> {session.name} (cached)")
                result = cache.load_json(asr_cache_key, {})
            else:
                print(f"  ASR -> {session.name}")
                result = transcribe_audio(
                    session.audio_path,
                    whisper_model_id=WHISPER_ID,
                    device=None,
                )
                cache.save_json(asr_cache_key, result)
                
            # Hydrate Segment models from dicts
            segments_data = result.get("segments", [])
            for idx, seg_dict in enumerate(segments_data, start=1):
                text = seg_dict.get("text", "").strip()
                if not text:
                    continue  # Ignore empty segments
                    
                segment = Segment(
                    index=idx,
                    start=seg_dict.get("start", 0.0),
                    end=seg_dict.get("end", 0.0),
                    text=text
                )
                session.segments.append(segment)
                
            print(f"  => Hydrated {len(session.segments)} text segments for {session.name}.")