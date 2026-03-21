import io
from pydub import AudioSegment


def convert_to_wav_bytes(input_audio_path):
    """
    Converts any audio file to WAV and returns the WAV bytes.

    Parameters
    ----------
    input_audio_path : str
        Path to input audio file.

    Returns
    -------
    bytes
        WAV audio content in bytes
    """

    audio = AudioSegment.from_file(input_audio_path)
    wav_buffer = io.BytesIO()
    audio.export(wav_buffer, format="wav")
    wav_buffer.seek(0)
    
    return wav_buffer.read()