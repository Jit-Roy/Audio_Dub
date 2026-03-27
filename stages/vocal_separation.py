from pathlib import Path
from core.pipeline import PipelineStage, PipelineContext
from core.config import config
from core.cache import CacheManager
from modules.Vocal_Music_Separation import vocal_music_separator
from utils.helper import ensure_wav

class VocalSeparationStage(PipelineStage):
    @property
    def name(self) -> str:
        return "VocalMusicSeparation"
        
    def execute(self, context: PipelineContext) -> None:
        input_path = Path(context.input_file)
        wav_path = ensure_wav(input_path, config.temp_dir)
        
        vocal_path = config.dir_vocal_separation / "vocal.wav"
        music_path = config.dir_vocal_separation / "music.wav"
        
        cache = CacheManager(config.temp_dir, input_path)
        
        # In a real caching system, we'd check hash of the input file instead of just existence
        if vocal_path.exists() and music_path.exists():
            print("  Vocal and Music tracks found in cache.")
        else:
            print("  Separating vocals from background music...")
            vocal_music_separator(
                str(wav_path),
                vocal_dir=str(config.dir_vocal_separation),
                music_dir=str(config.dir_vocal_separation),
                model_name=config.separator_model,
            )
            
        if not vocal_path.exists():
            raise FileNotFoundError(f"Separation failed, missing: {vocal_path}")
            
        # Store results in typed context object
        context.vocal_path = vocal_path
        context.music_path = music_path
