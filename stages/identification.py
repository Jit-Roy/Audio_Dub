from typing import Dict, List
from pathlib import Path
from core.pipeline import PipelineStage, PipelineContext
from core.config import config
from core.cache import CacheManager
from modules.Speaker_Identification import match_and_merge_speaker

def collect_separation_outputs(separation_dir: Path) -> Dict[int, List[Path]]:
    mapping: Dict[int, List[Path]] = {}
    for segment_dir in separation_dir.glob("segment*"):
        if not segment_dir.is_dir():
            continue
        name = segment_dir.name
        if not name.startswith("segment"):
            continue
        try:
            idx = int(name.replace("segment", ""))
        except ValueError:
            continue
        voices = sorted(segment_dir.glob("voice*.wav"))
        if voices:
            mapping[idx] = voices
    return mapping

class IdentificationStage(PipelineStage):
    @property
    def name(self) -> str:
        return "SpeakerIdentification"
        
    def should_run(self, context: PipelineContext) -> bool:
        # Explicit conditionality: only run if we detected overlaps
        return len(context.overlaps) > 0
        
    def execute(self, context: PipelineContext) -> None:
        cache = CacheManager(config.temp_dir, context.input_file)
        match_done_key = "05_identification/match_done.json"
        
        if cache.exists(match_done_key):
            print("  Using cached speaker matches...")
        else:
            print("  Matching separated voices to diarization speakers...")
            separation_dir = config.dir_separation
            diarization_dir = config.dir_diarization_base
            
            separation_mapping = collect_separation_outputs(separation_dir)
            for seg_idx, voices in separation_mapping.items():
                # seg_idx from standard file naming is 1-based usually, adjusting logic
                # Ensure we don't index out of bounds
                if seg_idx - 1 >= len(context.overlaps):
                    continue
                segment = context.overlaps[seg_idx - 1]
                
                for voice_path in voices:
                    result = match_and_merge_speaker(
                        input_audio_path=str(voice_path),
                        segment=segment,
                        diarization_dir=str(diarization_dir),
                        threshold=config.match_threshold,
                    )
                    print(result)
            cache.save_json(match_done_key, {"status": "ok", "segments_matched": len(separation_mapping)})
