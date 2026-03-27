from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional, Tuple
from core.models import SpeakerSession, Segment

class PipelineContext:
    """Carries strictly typed state between pipeline stages."""
    def __init__(self, input_file: str | Path):
        self.input_file = Path(input_file)
        
        # Strongly typed fields, not a generic dict
        self.vocal_path: Optional[Path] = None
        self.music_path: Optional[Path] = None
        self.overlaps: List[Tuple[float, float]] = []
        self.speaker_sessions: List[SpeakerSession] = []
        
    @property
    def failed_segments(self) -> List[Segment]:
        """Helper to surface all failed segments across all sessions."""
        return [
            seg for session in self.speaker_sessions 
            for seg in session.segments if seg.failed
        ]

class PipelineStage(ABC):
    """Base class for all pipeline stages."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
        
    def should_run(self, context: PipelineContext) -> bool:
        """Determines if this stage should execute given the current context."""
        return True
        
    @abstractmethod
    def execute(self, context: PipelineContext) -> None:
        """Execute the stage's logic."""
        pass

class DubbingPipeline:
    """Orchestrates execution of the various dubbing stages."""
    def __init__(self, stages: list[PipelineStage]):
        self.stages = stages
        
    def run(self, context: PipelineContext, start_from: str = None):
        skip_mode = bool(start_from)
        
        for stage in self.stages:
            if skip_mode:
                if stage.name == start_from:
                    skip_mode = False
                else:
                    print(f"Skipping stage: {stage.name} (seeking {start_from})")
                    continue
                    
            if not stage.should_run(context):
                print(f"[{stage.name}] Condition not met, dynamically skipping.")
                continue
                    
            print(f"\n[{stage.name}] Starting...")
            try:
                stage.execute(context)
                print(f"[{stage.name}] Completed successfully.")
            except Exception as e:
                print(f"[{stage.name}] FAILED: {str(e)}")
                # Potentially log failures, save error state
                raise e
