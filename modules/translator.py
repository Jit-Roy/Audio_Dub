from core.config import config

def translate_fragment(
    text_fragment: str,
    target_language: str = "Chinese",
    target_duration: float = None,
    target_chars: int = None,
) -> str:
    """
    Routes the translation request to the appropriate LLM provider based on config.
    """
    if config.llm_provider == "gemma":
        from modules.Gemma3llm import translate_fragment as gemma_translate
        return gemma_translate(
            text_fragment=text_fragment, 
            target_language=target_language, 
            target_duration=target_duration, 
            target_chars=target_chars
        )
    else:
        from modules.Qwen3llm import translate_fragment as qwen_translate
        return qwen_translate(
            text_fragment=text_fragment, 
            target_language=target_language, 
            target_duration=target_duration, 
            target_chars=target_chars
        )
