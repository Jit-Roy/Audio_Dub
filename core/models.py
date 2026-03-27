from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class Segment:
    """A segment of speech spoken by a single speaker."""
    index: int
    start: float
    end: float
    text: str = ""
    translated_text: str = ""
    
    # Error tracking
    failed: bool = False
    error_message: Optional[str] = None
    
    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)
    
    @property
    def target_chars(self) -> int:
        """Target length for translation in characters, assuming 5 chars/sec."""
        return int(self.duration * 5)
        
@dataclass
class SpeakerSession:
    """Stores all data related to a single speaker."""
    name: str  # "speaker_0", "speaker_1"
    audio_path: str
    segments: List[Segment] = field(default_factory=list)
    reference_audio_path: str = ""
    reference_text: str = ""
