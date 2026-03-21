import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
from IPython.display import Audio, display

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16
)

ref_audio = "D:\\Personal Projects\\Movie_Dub\\Sample_English_Audio.mp3"
ref_text  = "This week we got a open weak model which is really really impressive."

wavs, sr = model.generate_voice_clone(
    text="Ciao, bellissima! Hai quel look che mi fa girare la testa—sempre impeccabile. Stasera c’è un nuovo lounge in centro… ci vai con me? Prometto: niente di noioso, solo stile e buona musica.",
    language="Italian",
    ref_audio=ref_audio,
    ref_text=ref_text,
)

display(Audio(wavs[0], rate=sr))