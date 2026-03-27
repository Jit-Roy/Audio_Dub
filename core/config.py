import os
from pathlib import Path
from dataclasses import dataclass, field

@dataclass
class AppConfig:
    # Directories
    temp_dir: Path = Path("temp")
    
    # Global Stage-Based Cache Paths
    dir_vocal_separation: Path = field(init=False)
    dir_overlap_detection: Path = field(init=False)
    dir_diarization: Path = field(init=False)
    dir_diarization_nemo: Path = field(init=False)
    dir_diarization_base: Path = field(init=False)
    dir_separation: Path = field(init=False)
    dir_identification: Path = field(init=False)
    dir_ident_embeddings: Path = field(init=False)
    dir_asr: Path = field(init=False)
    dir_translation: Path = field(init=False)
    dir_tts: Path = field(init=False)
    
    output_dir: Path = Path.cwd()
    
    # Audio constants
    default_sr: int = 16000
    
    # Models
    separator_model: str = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
    speaker_separation_model: str = "lichenda/wsj0_2mix_skim_noncausal"
    target_language: str = "Chinese"
    
    # Execution
    separation_device: str = "cuda"
    match_threshold: float = 0.60
    
    # Secrets
    hf_token: str = field(default_factory=lambda: os.getenv("hf_token", ""))
    
    # LLM Settings
    llm_provider: str = "qwen"
    genai_key: str = field(default_factory=lambda: os.getenv("GENAI_API_KEY", ""))

    def __post_init__(self):
        # Initialize paths
        self.dir_vocal_separation = self.temp_dir / "01_vocal_separation"
        self.dir_overlap_detection = self.temp_dir / "02_overlap_detection"
        self.dir_diarization = self.temp_dir / "03_diarization"
        self.dir_diarization_nemo = self.dir_diarization / "nemo_output"
        self.dir_diarization_base = self.dir_diarization / "base_segments"
        self.dir_separation = self.temp_dir / "04_separation"
        self.dir_identification = self.temp_dir / "05_identification"
        self.dir_ident_embeddings = self.dir_identification / "embeddings"
        self.dir_asr = self.temp_dir / "06_asr"
        self.dir_translation = self.temp_dir / "07_translation"
        self.dir_tts = self.temp_dir / "08_tts"

        # Create all directories
        directories = [
            self.temp_dir,
            self.dir_vocal_separation,
            self.dir_overlap_detection,
            self.dir_diarization, self.dir_diarization_nemo, self.dir_diarization_base,
            self.dir_separation,
            self.dir_identification, self.dir_ident_embeddings,
            self.dir_asr,
            self.dir_translation,
            self.dir_tts
        ]
        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)

config = AppConfig()
