from core.pipeline import PipelineStage, PipelineContext
from core.config import config
from core.cache import CacheManager
from modules.Qwen3llm import translate_fragment

class TranslationStage(PipelineStage):
    @property
    def name(self) -> str:
        return "Translation"
        
    def execute(self, context: PipelineContext) -> None:
        cache = CacheManager(config.temp_dir, context.input_file)
        
        for session in context.speaker_sessions:
            if not session.segments:
                continue
                
            trans_cache_key = f"07_translation/translated_{session.name}.json"
            
            if cache.exists(trans_cache_key):
                print(f"  LLM translation -> {session.name} (cached)")
                cached_data = cache.load_json(trans_cache_key, [])
                
                # Create a quick dict to map cached indices to translations
                cached_map = {item.get("index"): item.get("translated_text", "") for item in cached_data}
                
                # Re-hydrate translations into the existing Segment objects
                for seg in session.segments:
                    if seg.index in cached_map:
                        seg.translated_text = cached_map[seg.index]
            else:
                print(f"  LLM translation -> {session.name}")
                for seg in session.segments:
                    print(f"    [LLM] Segment {seg.index}: {seg.start:.2f}-{seg.end:.2f}s")
                    try:
                        translated_text = translate_fragment(
                            text_fragment=seg.text, 
                            target_language=config.target_language, 
                            target_duration=seg.duration, 
                            target_chars=seg.target_chars
                        )
                        seg.translated_text = translated_text
                    except Exception as e:
                        print(f"    [Error] Translation failed for segment {seg.index}: {e}")
                        seg.failed = True
                        seg.error_message = str(e)
                        
                # Cache the hydrated segments (only successful ones)
                cache.save_json(
                    trans_cache_key, 
                    [{"index": s.index, "translated_text": s.translated_text} for s in session.segments if not s.failed]
                )