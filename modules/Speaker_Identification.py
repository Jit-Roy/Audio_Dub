import torch
import torchaudio
import torch.nn.functional as F
from pathlib import Path
from speechbrain.inference import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

# -----------------------------
# Load ECAPA-TDNN model (global)
# -----------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb",
    run_opts={"device": device},
    local_strategy=LocalStrategy.COPY,
)

# -----------------------------
# Audio utilities
# -----------------------------

def load_audio(path, target_sr=16000):
    signal, sr = torchaudio.load(path)
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)
    if sr != target_sr:
        signal = torchaudio.functional.resample(signal, sr, target_sr)
    return signal, target_sr


def chunk_signal(signal, sr=16000, chunk_sec=3.0):
    chunk_len = int(sr * chunk_sec)
    chunks = [signal[:, i:i+chunk_len] for i in range(0, signal.shape[-1], chunk_len)]
    chunks = [c for c in chunks if c.shape[-1] >= sr * 0.5]
    return chunks if chunks else [signal]


def voiced_chunks(signal, sr=16000, chunk_sec=2.0, hop_sec=1.0, amp_threshold=0.01, min_voiced_ratio=0.20):
    chunk_len = int(sr * chunk_sec)
    hop_len = int(sr * hop_sec)
    chunks = []

    for i in range(0, max(1, signal.shape[-1] - chunk_len + 1), hop_len):
        c = signal[:, i:i + chunk_len]
        if c.shape[-1] < int(sr * 0.5):
            continue

        voiced_ratio = float((c.abs() > amp_threshold).sum().item()) / c.numel()
        if voiced_ratio >= min_voiced_ratio:
            chunks.append(c)

    if not chunks and signal.shape[-1] >= int(sr * 0.5):
        chunks = chunk_signal(signal, sr=sr, chunk_sec=2.0)

    return chunks


def preprocess_for_matching(signal, amp_threshold=0.01):
    # Remove near-silent samples, then normalize level for robust embedding comparison.
    mono = signal.squeeze(0)
    voiced_mask = mono.abs() > amp_threshold

    if voiced_mask.any():
        processed = mono[voiced_mask].unsqueeze(0)
    else:
        processed = signal

    peak = processed.abs().max()
    if peak > 0:
        processed = processed / peak

    return processed


@torch.no_grad()
def embed_signal(signal):
    emb = classifier.encode_batch(signal)
    emb = emb.squeeze()
    if emb.dim() == 0:
        emb = emb.unsqueeze(0)
    return F.normalize(emb, dim=0)


def embed_long_audio(path):
    signal, sr = load_audio(path)
    chunks = chunk_signal(signal, sr)
    embeddings = [embed_signal(c) for c in chunks]
    mean_emb = torch.stack(embeddings).mean(dim=0)
    return F.normalize(mean_emb, dim=0)


# -----------------------------
# Core Function
# -----------------------------

def match_and_merge_speaker(
    input_audio_path: str,
    segment: tuple,
    diarization_dir: str,
    threshold: float = 0.60,
):
    """
    Matches input audio to a speaker in the diarization database,
    then overlays it at the given segment timestamp in that speaker's combined file.

    Parameters
    ----------
    input_audio_path : str
        Path to the dubbed/translated audio segment.
    segment : tuple
        (start_time, end_time) in seconds.
    diarization_dir : str
        Root folder containing speaker_0, speaker_1, ... subdirectories.
    threshold : float
        Minimum cosine similarity to consider a match.

    Returns
    -------
    dict with keys:
        - matched_speaker: str or None
        - score: float
        - output_path: str or None
        - status: str
    """

    start_sec, end_sec = segment

    # ── Validate segment ─────────────────────────────────────────────────────────
    if start_sec < 0 or end_sec <= start_sec:
        return {"matched_speaker": None, "score": 0.0, "output_path": None,
                "status": f"Invalid segment: start={start_sec}, end={end_sec}"}

    input_audio_path = Path(input_audio_path)
    diarization_dir  = Path(diarization_dir)

    if not input_audio_path.exists():
        return {"matched_speaker": None, "score": 0.0, "output_path": None,
                "status": f"Input audio not found: {input_audio_path}"}

    if not diarization_dir.exists():
        return {"matched_speaker": None, "score": 0.0, "output_path": None,
                "status": f"Diarization directory not found: {diarization_dir}"}

    # ── Discover speaker directories ─────────────────────────────────────────────
    speaker_dirs = sorted(
        [d for d in diarization_dir.iterdir()
         if d.is_dir() and d.name.startswith("speaker_")],
        key=lambda d: int(d.name.split("_")[1])
    )

    if not speaker_dirs:
        return {"matched_speaker": None, "score": 0.0, "output_path": None,
                "status": "No speaker directories found in diarization folder"}

    # ── Embed input audio ────────────────────────────────────────────────────────
    try:
        query_signal, query_sr = load_audio(str(input_audio_path), target_sr=16000)
        query_signal = preprocess_for_matching(query_signal)
        query_chunks = voiced_chunks(query_signal, sr=query_sr)
        query_embs = [embed_signal(c) for c in query_chunks]

        if not query_embs:
            return {
                "matched_speaker": None,
                "score": 0.0,
                "output_path": None,
                "status": "Input audio has no usable voiced chunks for speaker matching",
            }
    except Exception as e:
        return {"matched_speaker": None, "score": 0.0, "output_path": None,
                "status": f"Failed to embed input audio: {e}"}

    # ── Build speaker embeddings & find best match ───────────────────────────────
    scores = {}

    for spk_dir in speaker_dirs:
        combined_wav = spk_dir / f"{spk_dir.name}_combined.wav"

        if not combined_wav.exists():
            print(f"  [WARN] No combined wav found for {spk_dir.name}, skipping.")
            continue

        try:
            spk_signal, spk_sr = load_audio(str(combined_wav), target_sr=16000)
            spk_signal = preprocess_for_matching(spk_signal)
            spk_chunks = voiced_chunks(spk_signal, sr=spk_sr)

            if not spk_chunks:
                print(f"  [WARN] No usable voiced chunks for {spk_dir.name}, skipping.")
                continue

            spk_embs = [embed_signal(c) for c in spk_chunks]

            pair_scores = []
            for q_emb in query_embs:
                for s_emb in spk_embs:
                    pair_scores.append(
                        F.cosine_similarity(q_emb.unsqueeze(0), s_emb.unsqueeze(0)).item()
                    )

            if not pair_scores:
                print(f"  [WARN] Could not compute pairwise scores for {spk_dir.name}, skipping.")
                continue

            pair_scores.sort(reverse=True)
            top_k = pair_scores[: min(5, len(pair_scores))]
            best_sim = pair_scores[0]
            mean_topk_sim = sum(top_k) / len(top_k)
            # Blend best local match and top-k stability.
            sim = 0.7 * best_sim + 0.3 * mean_topk_sim

            scores[spk_dir.name] = (sim, combined_wav, spk_dir, best_sim, mean_topk_sim)
        except Exception as e:
            print(f"  [WARN] Could not embed {spk_dir.name}: {e}")
            continue

    if not scores:
        return {"matched_speaker": None, "score": 0.0, "output_path": None,
                "status": "Could not embed any speaker in the database"}

    # ── Pick best speaker ────────────────────────────────────────────────────────
    best_speaker = max(scores, key=lambda k: scores[k][0])
    best_score, combined_wav, spk_dir, best_local, best_topk = scores[best_speaker]

    print(f"\nSimilarity Scores:")
    for spk, (sim, _, _, local_best, topk_mean) in sorted(scores.items(), key=lambda x: x[1][0], reverse=True):
        print(f"  {spk:15s}  blended={sim:.4f}  best={local_best:.4f}  topk_mean={topk_mean:.4f}")

    if best_score < threshold:
        return {"matched_speaker": None, "score": best_score, "output_path": None,
                "status": f"No match above threshold {threshold}. Best was {best_speaker} @ {best_score:.4f}"}

    print(
        f"\nMATCH -> {best_speaker}  "
        f"(blended={best_score:.4f}, best={best_local:.4f}, topk_mean={best_topk:.4f})"
    )

    # ── Load input audio (the dubbed segment) ────────────────────────────────────
    try:
        input_signal, input_sr = torchaudio.load(str(input_audio_path))
        if input_signal.shape[0] > 1:
            input_signal = input_signal.mean(dim=0, keepdim=True)
    except Exception as e:
        return {"matched_speaker": best_speaker, "score": best_score, "output_path": None,
                "status": f"Failed to load input audio: {e}"}

    sample_rate = input_sr

    # ── Load base audio (merged file if it exists, else original combined) ───────
    merged_wav_path = spk_dir / f"{best_speaker}_combined_merged.wav"
    base_wav_path   = merged_wav_path if merged_wav_path.exists() else combined_wav

    try:
        base_signal, base_sr = torchaudio.load(str(base_wav_path))
        if base_sr != sample_rate:
            base_signal = torchaudio.functional.resample(base_signal, base_sr, sample_rate)
    except Exception as e:
        return {"matched_speaker": best_speaker, "score": best_score, "output_path": None,
                "status": f"Failed to load base audio: {e}"}

    # ── Match channel count ──────────────────────────────────────────────────────
    if base_signal.shape[0] != input_signal.shape[0]:
        if base_signal.shape[0] == 1:
            base_signal = base_signal.expand(input_signal.shape[0], -1).clone()
        else:
            input_signal = input_signal.expand(base_signal.shape[0], -1).clone()

    # ── Compute sample indices for the segment ───────────────────────────────────
    start_sample = int(start_sec * sample_rate)
    end_sample   = int(end_sec   * sample_rate)
    input_len    = input_signal.shape[-1]
    segment_len  = end_sample - start_sample

    # ── Edge case: input audio longer than segment → trim ────────────────────────
    if input_len > segment_len:
        print(f"  [WARN] Input audio ({input_len} samples) longer than segment "
              f"({segment_len} samples). Trimming.")
        input_signal = input_signal[:, :segment_len]
        input_len    = segment_len

    # ── Edge case: base audio shorter than segment end → pad ─────────────────────
    if base_signal.shape[-1] < end_sample:
        pad_len = end_sample - base_signal.shape[-1]
        print(f"  [WARN] Base audio shorter than segment end. Padding {pad_len} samples.")
        base_signal = F.pad(base_signal, (0, pad_len))

    # ── Overlay: silence the base at the segment, then add input ─────────────────
    merged = base_signal.clone()
    merged[:, start_sample : start_sample + input_len] = 0
    merged[:, start_sample : start_sample + input_len] += input_signal

    # ── Save merged output ───────────────────────────────────────────────────────
    try:
        torchaudio.save(str(merged_wav_path), merged, sample_rate)
        print(f"  Saved → {merged_wav_path}")
    except Exception as e:
        return {"matched_speaker": best_speaker, "score": best_score, "output_path": None,
                "status": f"Failed to save merged audio: {e}"}

    return {
        "matched_speaker": best_speaker,
        "score": best_score,
        "output_path": str(merged_wav_path),
        "status": "success"
    }