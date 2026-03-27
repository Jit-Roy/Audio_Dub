from pathlib import Path
from core.pipeline import PipelineStage, PipelineContext
from core.config import config
from core.cache import CacheManager
from modules.Speaker_Separation import separate_speakers

class SeparationStage(PipelineStage):
    @property
    def name(self) -> str:
        return "OverlapSeparation"
        
    def should_run(self, context: PipelineContext) -> bool:
        # Explicit conditionality: only run if we detected overlaps
        return len(context.overlaps) > 0
        
    def execute(self, context: PipelineContext) -> None:
        if not context.vocal_path or not context.vocal_path.exists():
            raise RuntimeError("Vocal path missing from context.")
            
        cache = CacheManager(config.temp_dir, context.input_file)
        separation_done_key = "04_separation/separation_done.json"
        separation_dir = config.dir_separation
        
        if cache.exists(separation_done_key):
            print("  Using cached separation outputs...")
        else:
            print("  Separating overlapping speakers...")
            separate_speakers(
                audio_file=str(context.vocal_path),
                output_dir=str(separation_dir),
                segments=context.overlaps,
                model=config.speaker_separation_model,
                device=config.separation_device,
            )
            cache.save_json(separation_done_key, {"status": "ok", "count": len(context.overlaps)})
