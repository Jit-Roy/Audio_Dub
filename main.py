import argparse
from pathlib import Path
from core.pipeline import DubbingPipeline, PipelineContext
from core.config import config
from stages.vocal_separation import VocalSeparationStage
from stages.overlap_detection import OverlapDetectionStage
from stages.diarization import DiarizationStage
from stages.separation import SeparationStage
from stages.identification import IdentificationStage
from stages.asr import ASRStage
from stages.translation import TranslationStage
from stages.tts import TTSAndMixStage

def main() -> None:
    parser = argparse.ArgumentParser(description="Movie dubbing pipeline")
    parser.add_argument("--input-audio", required=True, help="Path to input mp3 or wav")
    parser.add_argument("--target-language", default="Chinese", help="Target translation and TTS language")
    parser.add_argument("--hf-token", default="", help="Hugging Face token for overlap detection")
    parser.add_argument("--temp-dir", default=str(config.temp_dir), help="Temporary directory for outputs")
    parser.add_argument("--start-from", default=None, help="Stage to resume from (e.g. 'ASR')")
    parser.add_argument("--llm-provider", default="qwen", choices=["qwen", "gemma"], help="Which LLM to use for translation")
    parser.add_argument("--genai-key", default="", help="Google API key for Gemma-3 via genai")
    
    args = parser.parse_args()

    input_path = Path(args.input_audio).resolve()

    # 1. Hydrate the global config object from CLI args
    if args.hf_token:
        config.hf_token = args.hf_token
    if args.target_language:
        config.target_language = args.target_language
    if args.llm_provider:
        config.llm_provider = args.llm_provider
    if args.genai_key:
        config.genai_key = args.genai_key
    config.__post_init__()

    # 2. Construct the logical pipeline
    pipeline = DubbingPipeline([
        VocalSeparationStage(),
        OverlapDetectionStage(),
        DiarizationStage(),
        SeparationStage(),
        IdentificationStage(),
        ASRStage(),
        TranslationStage(),
        TTSAndMixStage()
    ])

    # 3. Create context and run
    context = PipelineContext(input_file=input_path)
    
    try:
        pipeline.run(context, start_from=args.start_from)
        
        if context.failed_segments:
            print(f"\n[Warning] Pipeline completed but {len(context.failed_segments)} segments failed TTS/Translation.")
            for seg in context.failed_segments:
                print(f"  - Seg {seg.index}: {seg.error_message}")
    except Exception as e:
        print(f"\n[Fatal Error] Pipeline aborted: {e}")

if __name__ == "__main__":
    main()
