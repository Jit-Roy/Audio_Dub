import torch
import torch.serialization
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.torch_version import TorchVersion
from pyannote.audio.core.task import Specifications
from pyannote.core import SlidingWindow
from pyannote.audio import Model, Audio
from pyannote.audio.pipelines import OverlappedSpeechDetection


def detect_overlaps(
    hf_token: str,
    audio_file: str,
    min_duration_on: float = 0.1,
    min_duration_off: float = 0.1,
    plot: bool = True,
) -> list[tuple[float, float]]:

    # ── Config ────────────────────────────────────────────────
    HF_TOKEN   = hf_token
    AUDIO_FILE = audio_file
    HYPER_PARAMETERS = {"min_duration_on": min_duration_on, "min_duration_off": min_duration_off}

    # ── Patch torch.load (once) ───────────────────────────────
    true_load = torch.load.__wrapped__ if hasattr(torch.load, "__wrapped__") else torch.load
    if not getattr(torch.load, "_pyannote_patched", False):
        def _patched(f, map_location=None, pickle_module=None, weights_only=True, mmap=None, **kw):
            return true_load(f, map_location=map_location, pickle_module=pickle_module, weights_only=False, **kw)
        _patched._pyannote_patched = True
        torch.load = _patched
        print("✓ torch.load patched (once)")
    else:
        print("⚠ Patch already applied — skipping")

    # ── Register safe globals ─────────────────────────────────
    torch.serialization.add_safe_globals([TorchVersion, Specifications, SlidingWindow])

    # ── Load model & pipeline ─────────────────────────────────
    model    = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=HF_TOKEN)
    pipeline = OverlappedSpeechDetection(segmentation=model)
    pipeline.instantiate(HYPER_PARAMETERS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)
    print(f"✓ Model loaded — running on: {device}")

    # ── Inference ─────────────────────────────────────────────
    print(f"\nProcessing: {AUDIO_FILE} ...")
    osd_output = pipeline(AUDIO_FILE)
    overlap_segments = [
        (round(seg.start, 3), round(seg.end, 3))
        for seg in osd_output.get_timeline().support()
    ]

    # ── Print report ──────────────────────────────────────────
    total = sum(e - s for s, e in overlap_segments)
    print(f"\n{'='*50}")
    print(f"  Detected {len(overlap_segments)} overlapping speech region(s)")
    print(f"{'='*50}")
    for i, (s, e) in enumerate(overlap_segments):
        print(f"  [{i+1:>3}]  {s:.3f}s  →  {e:.3f}s    ({e-s:.3f}s)")
    print(f"{'='*50}")
    print(f"  Total overlap : {total:.3f}s")
    print(f"\nAs list: {overlap_segments}")

    # ── Plot (optional) ───────────────────────────────────────
    if plot:
        io = Audio(sample_rate=16000, mono=True)
        waveform, sr = io({"audio": AUDIO_FILE, "channel": 0})
        waveform  = waveform.squeeze().numpy()
        duration  = len(waveform) / sr
        time_axis = np.linspace(0, duration, len(waveform))

        fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=True)
        fig.suptitle("Overlapped Speech Detection", fontsize=14, fontweight="bold")

        axes[0].plot(time_axis, waveform, color="#1565C0", linewidth=0.4, alpha=0.8)
        axes[0].set_ylabel("Amplitude")
        axes[0].set_title("Waveform")
        axes[0].set_ylim(-1.05, 1.05)
        for s, e in overlap_segments:
            axes[0].axvspan(s, e, alpha=0.35, color="#E53935", zorder=2)

        axes[1].set_ylabel("Overlap")
        axes[1].set_title("Detected Overlap Regions")
        axes[1].set_ylim(0, 1)
        axes[1].set_yticks([])
        axes[1].set_facecolor("#F5F5F5")
        for s, e in overlap_segments:
            axes[1].axvspan(s, e, alpha=0.85, color="#E53935", zorder=2)
            if (e - s) > 0.3:
                axes[1].text((s+e)/2, 0.5, f"{e-s:.2f}s",
                             ha="center", va="center",
                             fontsize=7, color="white", fontweight="bold")

        axes[1].set_xlabel("Time (seconds)")
        fig.legend(handles=[mpatches.Patch(color="#E53935", alpha=0.7, label="Overlapping speech")],
                   loc="upper right", fontsize=10)
        plt.tight_layout()
        plt.show()

    return overlap_segments