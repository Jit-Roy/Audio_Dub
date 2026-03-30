from dataclasses import dataclass, field
from typing import Optional, List
from core.config import config

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
        """Target length for translation in characters, dynamically adjusted by target language."""
        lang = config.target_language.lower()
        if any(l in lang for l in ["english", "spanish", "french", "german", "italian", "portuguese", "russian"]):
            density = 14
        elif "korean" in lang:
            density = 6
        else:
            density = 5  # Default for Chinese/Japanese/logographic
        return int(self.duration * density)
        
@dataclass
class SpeakerSession:
    """Stores all data related to a single speaker."""
    name: str  # "speaker_0", "speaker_1"
    audio_path: str
    segments: List[Segment] = field(default_factory=list)
    reference_audio_path: str = ""
    reference_text: str = ""
