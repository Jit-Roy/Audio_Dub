from pathlib import Path
from core.pipeline import PipelineStage, PipelineContext
from core.config import config
from core.models import SpeakerSession
from modules.Speaker_Diarization import perform_diarization_and_extract

def find_speaker_audio(diarization_dir: Path) -> list:
    """Helper to locate all generated speaker audio tracks."""
    speakers = []
    for spk_dir in sorted(diarization_dir.glob("speaker_*")):
        if not spk_dir.is_dir():
            continue
        merged = spk_dir / f"{spk_dir.name}_combined_merged.wav"
        combined = spk_dir / f"{spk_dir.name}_combined.wav"
        if merged.exists():
            speakers.append((spk_dir.name, merged))
        elif combined.exists():
            speakers.append((spk_dir.name, combined))
    return speakers

class DiarizationStage(PipelineStage):
    @property
    def name(self) -> str:
        return "Diarization"
        
    def execute(self, context: PipelineContext) -> None:
        if not context.vocal_path or not context.vocal_path.exists():
            raise RuntimeError("Vocal path missing from context.")
            
        diarization_dir = config.dir_diarization_base
        speaker_files = find_speaker_audio(diarization_dir)
        
        if speaker_files:
            print("  Using cached diarization outputs...")
        else:
            print("  Running diarization...")
            perform_diarization_and_extract(
                audio_path=str(context.vocal_path),
                output_base_dir=str(diarization_dir),
                remove_segments=context.overlaps if context.overlaps else None,
            )
            speaker_files = find_speaker_audio(diarization_dir)
            
        if not speaker_files:
            raise RuntimeError("No speaker audio files found after diarization.")
            
        # Hydrate speaker sessions into the PipelineContext
        context.speaker_sessions = []
        for name, path in speaker_files:
            session = SpeakerSession(name=name, audio_path=str(path))
            context.speaker_sessions.append(session)
            
        print(f"  Initialized {len(context.speaker_sessions)} speaker sessions.")
