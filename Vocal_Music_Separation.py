import os
import shutil
import tempfile
import logging


def _suppress_separator_loggers():
    for logger_name in ["separator", "mdxc_separator", "common_separator"]:
        log = logging.getLogger(logger_name)
        log.handlers.clear()
        log.setLevel(logging.ERROR)
        log.propagate = False


_suppress_separator_loggers()
    
from audio_separator.separator import Separator
from helper import convert_to_wav_bytes

# - Kim_Vocal_2 is one of the best for clean vocal extraction
# - 'model_bs_roformer_ep_317_sdr_12.9755.ckpt' (state-of-the-art, but larger/slower)
# - 'UVR_MDXNET_KARA_2.onnx' (good for karaoke-style separation)
# - 'MDX23C-8KFFT-InstVoc_HQ.ckpt' (high quality, balanced)

def vocal_music_separator(input_audio_path):
    """
    Separates vocals and music from an input audio file.
    If the file is not WAV, it converts it first using convert_to_wav_bytes.

    Parameters
    ----------
    input_audio_path : str

    Returns
    -------
    dict
        Paths to separated vocal and music files.
    """

    # Target directories
    vocal_dir = "temp/vocal"
    music_dir = "temp/music"

    os.makedirs(vocal_dir, exist_ok=True)
    os.makedirs(music_dir, exist_ok=True)

    # Check if file is WAV
    file_ext = os.path.splitext(input_audio_path)[1].lower()

    if file_ext != ".wav":
        wav_bytes = convert_to_wav_bytes(input_audio_path)
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_file.write(wav_bytes)
        tmp_file.close()
        input_wav_path = tmp_file.name

    else:
        input_wav_path = input_audio_path

    vocal_path = None
    music_path = None

    # Use a hidden temporary directory for the separator's output
    with tempfile.TemporaryDirectory() as output_dir:
        previous_disable_level = logging.root.manager.disable
        logging.disable(logging.INFO)
        try:
            _suppress_separator_loggers()
            separator = Separator(
                output_dir=output_dir,
                output_format="WAV"
            )

            separator.load_model("model_bs_roformer_ep_317_sdr_12.9755.ckpt")
            output_files = separator.separate(input_wav_path)
        finally:
            logging.disable(previous_disable_level)
    
        for file in output_files:
            filename = os.path.basename(file).lower()
            source_path = os.path.join(output_dir, file)
    
            if "vocal" in filename:
                vocal_path = os.path.join(vocal_dir, "vocal.wav")
                shutil.move(source_path, vocal_path)
    
            elif "instrumental" in filename or "music" in filename:
                music_path = os.path.join(music_dir, "music.wav")
                shutil.move(source_path, music_path)

    if file_ext != ".wav":
        os.remove(input_wav_path)

    return {
        "vocal": vocal_path,
        "music": music_path
    }