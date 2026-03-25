import os
from typing import Iterable

import soundfile as sf
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.enh_inference import SeparateSpeech


def separate_speakers(
    audio_file: str,
    output_dir: str = ".",
    segments: list[tuple[float, float]] | None = None,
    model: str = "lichenda/wsj0_2mix_skim_noncausal",
    segment_size: float = 2.4,
    hop_size: float = 0.8,
    device: str = "cuda",
) -> list[str]:
    """
    Separate mixed audio into individual speaker wav files.

    Args:
        audio_file:    Path to the mixed input audio file.
        output_dir:    Directory to save separated speaker files. Defaults to current dir.
        segments:      Optional list of (start_sec, end_sec) ranges to process.
                       If provided, outputs are saved under segment1/, segment2/, etc.
        model:         ESPnet model tag. Options:
                         - "lichenda/wsj0_2mix_skim_noncausal"
                         - "espnet/chenda-li-wsj0_2mix_enh_train_enh_conv_tasnet_raw_valid.si_snr.ave"
                         - "espnet/chenda-li-wsj0_2mix_enh_train_enh_rnn_tf_raw_valid.si_snr.ave"
                         - "lichenda/Chenda_Li_wsj0_2mix_enh_dprnn_tasnet"
                         - "espnet/Wangyou_Zhang_wsj0_2mix_enh_dc_crn_mapping_snr_raw"
        segment_size:  Segment size in seconds for inference. Defaults to 2.4.
        hop_size:      Hop size in seconds for inference. Defaults to 0.8.
        device:        Torch device to run inference on ("cuda" or "cpu"). Defaults to "cuda".

    Returns:
        List of output file paths for each separated speaker.
    """

    def _validate_segments(raw_segments: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
        validated = []
        for idx, item in enumerate(raw_segments, start=1):
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError(f"Segment {idx} must be a tuple/list with 2 values: (start, end)")

            start_raw, end_raw = item
            try:
                start = float(start_raw)
                end = float(end_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Segment {idx} contains non-numeric values: {item}") from exc

            if end <= start:
                raise ValueError(f"Segment {idx} must satisfy end > start, got {item}")

            validated.append((start, end))

        if not validated:
            raise ValueError("segments cannot be empty when provided")

        return validated

    cfg = ModelDownloader().download_and_unpack(model)
    separate_speech = SeparateSpeech(
        train_config=cfg["train_config"],
        model_file=cfg["model_file"],
        segment_size=segment_size,
        hop_size=hop_size,
        normalize_segment_scale=False,
        show_progressbar=True,
        ref_channel=None,
        normalize_output_wav=True,
        device=device,
    )
    print(f"✓ Model loaded: {model}")

    mixwav, sr = sf.read(audio_file)
    if mixwav.ndim > 1:
        mixwav = mixwav.mean(axis=1)  # Convert stereo to mono

    print(f"✓ Audio loaded: {audio_file} (sr={sr})")

    duration_sec = len(mixwav) / sr

    os.makedirs(output_dir, exist_ok=True)
    output_paths = []

    if segments is None:
        waves = separate_speech(mixwav[None, ...], fs=sr)
        print(f"✓ Separated into {len(waves)} speaker(s)")

        for i, wave in enumerate(waves, start=1):
            out_path = os.path.join(output_dir, f"speaker{i}.wav")
            sf.write(out_path, wave.squeeze(), sr)
            print(f"  Saved: {out_path}")
            output_paths.append(out_path)
        return output_paths

    validated_segments = _validate_segments(segments)

    for seg_id, (start, end) in enumerate(validated_segments, start=1):
        start_sec = max(0.0, start)
        end_sec = min(duration_sec, end)

        if end_sec <= start_sec:
            print(f"! Skipping segment{seg_id}: clipped range is empty ({start}, {end})")
            continue

        start_idx = int(start_sec * sr)
        end_idx = int(end_sec * sr)
        segment_audio = mixwav[start_idx:end_idx]
        if len(segment_audio) == 0:
            print(f"! Skipping segment{seg_id}: no audio samples after slicing")
            continue

        segment_dir = os.path.join(output_dir, f"segment{seg_id}")
        os.makedirs(segment_dir, exist_ok=True)

        waves = separate_speech(segment_audio[None, ...], fs=sr)
        print(f"✓ Segment {seg_id}: separated into {len(waves)} speaker(s)")

        for voice_id, wave in enumerate(waves, start=1):
            out_path = os.path.join(segment_dir, f"voice{voice_id}.wav")
            sf.write(out_path, wave.squeeze(), sr)
            print(f"  Saved: {out_path}")
            output_paths.append(out_path)

    return output_paths