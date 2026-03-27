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


def vocal_music_separator(
    input_audio_path,
    vocal_dir="temp/vocal",
    music_dir="temp/music",
    model_name="model_bs_roformer_ep_317_sdr_12.9755.ckpt",
):
    """
    Separates vocals and music from a WAV input audio file.

    Parameters
    ----------
    input_audio_path : str
    vocal_dir : str
    music_dir : str
    model_name : str
        Separation model to use. Common options:
        - model_bs_roformer_ep_317_sdr_12.9755.ckpt (state-of-the-art, larger/slower)
        - UVR_MDXNET_KARA_2.onnx (good for karaoke-style separation)
        - MDX23C-8KFFT-InstVoc_HQ.ckpt (high quality, balanced)

    Returns
    -------
    tuple[bytes | None, bytes | None]
        Tuple of (vocal_bytes, music_bytes).

    Raises
    ------
    ValueError
        If the input file is not in WAV format.
    """

    os.makedirs(vocal_dir, exist_ok=True)
    os.makedirs(music_dir, exist_ok=True)

    # Check if file is WAV
    file_ext = os.path.splitext(input_audio_path)[1].lower()

    if file_ext != ".wav":
        raise ValueError("Input file must be in WAV format. Please provide a .wav file.")

    input_wav_path = input_audio_path

    vocal_path = None
    music_path = None
    vocal_bytes = None
    music_bytes = None

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

            separator.load_model(model_name)
            output_files = separator.separate(input_wav_path)
        finally:
            logging.disable(previous_disable_level)
    
        for file in output_files:
            filename = os.path.basename(file).lower()
            source_path = os.path.join(output_dir, file)
    
            if "vocal" in filename:
                vocal_path = os.path.join(vocal_dir, "vocal.wav")
                shutil.move(source_path, vocal_path)
                with open(vocal_path, "rb") as f:
                    vocal_bytes = f.read()
    
            elif "instrumental" in filename or "music" in filename:
                music_path = os.path.join(music_dir, "music.wav")
                shutil.move(source_path, music_path)
                with open(music_path, "rb") as f:
                    music_bytes = f.read()

    return vocal_bytes, music_bytes