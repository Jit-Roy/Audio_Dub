import time
import torch
from qwen_tts import Qwen3TTSModel

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True


def load_tts_model(device: str | None = None, dtype: torch.dtype | None = None) -> Qwen3TTSModel:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    print(f"[TTS] Loading Qwen3 model on {device} ({dtype})")
    return Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        device_map=device,
        dtype=dtype,
    )


def generate_voice_clone(
    text: str,
    language: str,
    ref_audio: str,
    ref_text: str,
    model: Qwen3TTSModel | None = None,
):
    if model is None:
        model = load_tts_model()
        
    # Qwen3TTS specifically requested downcased full names like "chinese", "english"
    mapped_lang = language.strip().lower()
    
    print(f"[TTS] Generating voice clone ({len(text)} chars) lang={language} (mapped: {mapped_lang})")
    print(f"[TTS] ref_audio={ref_audio} ref_text_len={len(ref_text)}")
    start_time = time.time()
    gen_kwargs = {}
    print(f"[TTS] generate kwargs={gen_kwargs}")
    try:
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=mapped_lang,
            ref_audio=ref_audio,
            ref_text=ref_text,
            **gen_kwargs,
        )
    except RuntimeError as exc:
        err = str(exc).lower()
        if "device-side assert" in err or "cuda error" in err:
            if model is not None and model.device.type == "cuda":
                print("[TTS] CUDA assert detected, retrying on GPU with float32...")
                model = load_tts_model(device="cuda", dtype=torch.float32)
                wavs, sr = model.generate_voice_clone(
                    text=text,
                    language=mapped_lang,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    **gen_kwargs,
                )
            else:
                raise RuntimeError(
                    "CUDA device-side assert during TTS. Restart the process. "
                    "If it persists, try CUDA_LAUNCH_BLOCKING=1 to get the exact kernel location."
                ) from exc
        else:
            raise
    print(f"[TTS] Generation finished in {time.time() - start_time:.2f}s")
    return wavs, sr


if __name__ == "__main__":
    ref_audio = "D:\\Personal Projects\\Movie_Dub\\Sample_English_Audio.mp3"
    ref_text = "This week we got a open weak model which is really really impressive."
    model = load_tts_model()
    wavs, sr = generate_voice_clone(
        text=(
            "Ciao, bellissima! Hai quel look che mi fa girare la testa-"
            "sempre impeccabile. Stasera c'e un nuovo lounge in centro... "
            "ci vai con me? Prometto: niente di noioso, solo stile e buona musica."
        ),
        language="Italian",
        ref_audio=ref_audio,
        ref_text=ref_text,
        model=model,
    )
    print(f"Generated {len(wavs)} clip(s) at {sr} Hz")