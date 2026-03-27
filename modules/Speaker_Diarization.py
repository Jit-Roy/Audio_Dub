import json
import wget
from pathlib import Path
from collections import defaultdict
from typing import Iterable
from pydub import AudioSegment
from omegaconf import OmegaConf
from nemo.collections.asr.models import ClusteringDiarizer


def perform_diarization_and_extract(audio_path, output_base_dir, remove_segments=None):
    audio_path = Path(audio_path)
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    nemo_out_dir = output_base_dir / "nemo_output"
    nemo_out_dir.mkdir(exist_ok=True)

    print(f"Loading audio: {audio_path}")
    full_audio = AudioSegment.from_file(str(audio_path))

    diarization_audio = full_audio

    if remove_segments is not None:
        if not isinstance(remove_segments, Iterable):
            raise ValueError("remove_segments must be an iterable of (start, end) tuples")

        duration_sec = len(full_audio) / 1000.0
        intervals = []
        for idx, segment in enumerate(remove_segments, start=1):
            if not isinstance(segment, (list, tuple)) or len(segment) != 2:
                raise ValueError(
                    f"remove_segments item {idx} must be a tuple/list with 2 values: (start, end)"
                )

            start_raw, end_raw = segment
            try:
                start = float(start_raw)
                end = float(end_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"remove_segments item {idx} has non-numeric values: {segment}"
                ) from exc

            if end <= start:
                raise ValueError(
                    f"remove_segments item {idx} must satisfy end > start, got {segment}"
                )

            clipped_start = max(0.0, start)
            clipped_end = min(duration_sec, end)
            if clipped_end > clipped_start:
                intervals.append((clipped_start, clipped_end))

        if intervals:
            intervals.sort(key=lambda item: item[0])
            merged_intervals = [intervals[0]]
            for start, end in intervals[1:]:
                prev_start, prev_end = merged_intervals[-1]
                if start <= prev_end:
                    merged_intervals[-1] = (prev_start, max(prev_end, end))
                else:
                    merged_intervals.append((start, end))

            # Keep original duration unchanged by replacing overlap regions with silence.
            # This preserves RTTM timestamps in the original timeline.
            for start, end in merged_intervals:
                start_ms = int(start * 1000)
                end_ms = int(end * 1000)
                silent_chunk = AudioSegment.silent(
                    duration=max(0, end_ms - start_ms),
                    frame_rate=full_audio.frame_rate,
                ).set_channels(full_audio.channels).set_sample_width(full_audio.sample_width)

                diarization_audio = (
                    diarization_audio[:start_ms] + silent_chunk + diarization_audio[end_ms:]
                )

            print(
                f"Silenced {len(merged_intervals)} overlap segment(s) before diarization "
                f"(original duration preserved)"
            )

    # Convert to mono for NeMo
    mono_audio_path = nemo_out_dir / "mono_input.wav"
    diarization_audio.set_channels(1).export(mono_audio_path, format="wav")

    # Create manifest
    manifest_path = nemo_out_dir / "input_manifest.json"
    manifest_entry = {
        "audio_filepath": str(mono_audio_path),
        "offset": 0,
        "duration": None,
        "label": "infer",
        "text": "-",
        "num_speakers": None,
        "rttm_filepath": None,
        "uem_filepath": None,
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest_entry, f)
        f.write("\n")

    # Download config if needed
    config_url = "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/diar_infer_general.yaml"
    config_path = nemo_out_dir / "diar_infer_general.yaml"

    if not config_path.exists():
        print("Downloading config...")
        wget.download(config_url, str(config_path))

    cfg = OmegaConf.load(str(config_path))

    # Configure diarizer
    cfg.diarizer.manifest_filepath = str(manifest_path)
    cfg.diarizer.out_dir = str(nemo_out_dir)
    cfg.diarizer.oracle_vad = False

    cfg.num_workers = 0
    cfg.batch_size = 1
    cfg.diarizer.num_workers = 0

    cfg.diarizer.vad.model_path = "vad_multilingual_marblenet"
    cfg.diarizer.speaker_embeddings.model_path = "titanet_large"

    print("Running NeMo diarization...")

    diarizer = ClusteringDiarizer(cfg=cfg)
    diarizer.diarize()

    # Read RTTM
    rttm_path = nemo_out_dir / "pred_rttms" / f"{mono_audio_path.stem}.rttm"

    if not rttm_path.exists():
        raise FileNotFoundError("RTTM file not generated")

    speaker_segments = defaultdict(list)

    with open(rttm_path) as f:
        for line in f:
            parts = line.strip().split()
            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]
            speaker_segments[speaker].append((start, duration))

    target_duration_ms = len(full_audio)
    print(f"Using input audio length for speaker timelines: {target_duration_ms} ms")

    # Extract speaker audio
    for speaker, segments in speaker_segments.items():

        speaker_dir = output_base_dir / speaker
        speaker_dir.mkdir(exist_ok=True)

        combined_audio = AudioSegment.silent(
            duration=target_duration_ms,
            frame_rate=diarization_audio.frame_rate,
        ).set_channels(diarization_audio.channels).set_sample_width(diarization_audio.sample_width)
        segments.sort()

        for start, duration in segments:
            start_ms = int(start * 1000)
            end_ms = int((start + duration) * 1000)
            clip_start = max(0, start_ms)
            clip_end = min(end_ms, len(diarization_audio), target_duration_ms)

            if clip_end <= clip_start:
                continue

            combined_audio = (
                combined_audio[:clip_start]
                + diarization_audio[clip_start:clip_end]
                + combined_audio[clip_end:]
            )

        output_file = speaker_dir / f"{speaker}_combined.wav"
        combined_audio.export(output_file, format="wav")

        print(f"{speaker} exported → {output_file}")