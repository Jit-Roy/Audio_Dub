from pathlib import Path
from core.pipeline import PipelineStage, PipelineContext
from core.config import config
from core.cache import CacheManager
from modules.Speech_Overlap import detect_overlaps

class OverlapDetectionStage(PipelineStage):
    @property
    def name(self) -> str:
        return "OverlapDetection"
        
    def execute(self, context: PipelineContext) -> None:
        if not context.vocal_path or not context.vocal_path.exists():
            raise RuntimeError("Vocal path missing from context. Run VocalSeparationStage first.")
            
        if not config.hf_token:
            raise RuntimeError(
                "Missing Hugging Face token for overlap detection. "
                "Set hf_token in .env or pass it via config."
            )
            
        cache = CacheManager(config.temp_dir, context.input_file)
        overlaps_cache_key = "02_overlap_detection/overlaps.json"
        
        if cache.exists(overlaps_cache_key):
            print(f"  Using cached overlaps.")
            context.overlaps = cache.load_json(overlaps_cache_key, [])
        else:
            print("  Detecting overlapping speech...")
            overlaps = detect_overlaps(
                hf_token=config.hf_token,
                audio_file=str(context.vocal_path),
                plot=False,
            )
            cache.save_json(overlaps_cache_key, overlaps)
            context.overlaps = overlaps
            
        print(f"  Found {len(context.overlaps)} overlapping segments.")
