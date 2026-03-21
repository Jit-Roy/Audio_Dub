import os
import json
import re
import nltk
import logging
import warnings
from pathlib import Path

import torch
import torchaudio
import faster_whisper
from ctc_forced_aligner import (
    load_alignment_model,
    generate_emissions,
    preprocess_text,
    get_alignments,
    get_spans,
    postprocess_results,
)
from nemo.collections.asr.models import SortformerEncLabelModel

# Suppress warnings and logging
os.environ["NEMO_LOG_LEVEL"] = "ERROR"
warnings.filterwarnings("ignore")
logging.getLogger("nemo").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# Configuration
AUDIO_FILE = Path(r"D:\Personal Projects\Movie_Dub\Audio_Samples\Podcast.mp3")
OUTPUT_TXT = Path(r"D:\Personal Projects\Movie_Dub\diarization_results3.txt")
OUTPUT_SRT = Path(r"D:\Personal Projects\Movie_Dub\diarization_results3.srt")
TARGET_SR = 16000

# Language mappings
langs_to_iso = {
    "af": "afr", "am": "amh", "ar": "ara", "as": "asm", "az": "aze",
    "ba": "bak", "be": "bel", "bg": "bul", "bn": "ben", "bo": "tib",
    "br": "bre", "bs": "bos", "ca": "cat", "cs": "cze", "cy": "wel",
    "da": "dan", "de": "ger", "el": "gre", "en": "eng", "es": "spa",
    "et": "est", "eu": "baq", "fa": "per", "fi": "fin", "fo": "fao",
    "fr": "fre", "gl": "glg", "gu": "guj", "ha": "hau", "haw": "haw",
    "he": "heb", "hi": "hin", "hr": "hrv", "ht": "hat", "hu": "hun",
    "hy": "arm", "id": "ind", "is": "ice", "it": "ita", "ja": "jpn",
    "jw": "jav", "ka": "geo", "kk": "kaz", "km": "khm", "kn": "kan",
    "ko": "kor", "la": "lat", "lb": "ltz", "ln": "lin", "lo": "lao",
    "lt": "lit", "lv": "lav", "mg": "mlg", "mi": "mao", "mk": "mac",
    "ml": "mal", "mn": "mon", "mr": "mar", "ms": "may", "mt": "mlt",
    "my": "bur", "ne": "nep", "nl": "dut", "nn": "nno", "no": "nor",
    "oc": "oci", "pa": "pan", "pl": "pol", "ps": "pus", "pt": "por",
    "ro": "rum", "ru": "rus", "sa": "san", "sd": "snd", "si": "sin",
    "sk": "slo", "sl": "slv", "sn": "sna", "so": "som", "sq": "alb",
    "sr": "srp", "su": "sun", "sv": "swe", "sw": "swa", "ta": "tam",
    "te": "tel", "tg": "tgk", "th": "tha", "tk": "tuk", "tl": "tgl",
    "tr": "tur", "tt": "tat", "uk": "ukr", "ur": "urd", "uz": "uzb",
    "vi": "vie", "yi": "yid", "yo": "yor", "yue": "yue", "zh": "chi",
}

punct_model_langs = ["en", "fr", "de", "es", "it", "nl", "pt", "bg", "pl", "cs", "sk", "sl"]
sentence_ending_punctuations = ".?!"


def find_numeral_symbol_tokens(tokenizer):
    """Find numeral/symbol tokens"""
    numeral_symbol_tokens = [-1]
    for token, token_id in tokenizer.get_vocab().items():
        if any(c in "0123456789%$£" for c in token):
            numeral_symbol_tokens.append(token_id)
    return numeral_symbol_tokens


def get_word_ts_anchor(s, e, option="start"):
    if option == "end":
        return e
    elif option == "mid":
        return (s + e) / 2
    return s


def get_words_speaker_mapping(wrd_ts, spk_ts, word_anchor_option="start"):
    """Map words to speakers with nested segment awareness"""
    wrd_spk_mapping = []
    
    for wrd_dict in wrd_ts:
        ws, we, wrd = (
            int(wrd_dict["start"] * 1000),
            int(wrd_dict["end"] * 1000),
            wrd_dict["text"],
        )
        
        assigned_speaker = spk_ts[0][2]  # Default to first speaker
        
        # Strategy 1: Check if word START falls in any segment
        # Prefer smaller (more specific) segments
        start_segments = [(e - s, sp) for s, e, sp in spk_ts if s <= ws <= e]
        if start_segments:
            start_segments.sort()  # Sort by size (smallest first)
            assigned_speaker = start_segments[0][1]
        else:
            # Strategy 2: Check if word MIDPOINT falls in any segment
            midpoint = (ws + we) // 2
            mid_segments = [(e - s, sp) for s, e, sp in spk_ts if s <= midpoint <= e]
            if mid_segments:
                mid_segments.sort()  # Sort by size
                assigned_speaker = mid_segments[0][1]
            else:
                # Strategy 3: Find segment with maximum overlap percentage
                best_overlap_pct = 0
                best_segment_size = float('inf')
                word_duration = we - ws if we > ws else 1
                
                for s, e, sp in spk_ts:
                    overlap_start = max(ws, s)
                    overlap_end = min(we, e)
                    overlap = max(0, overlap_end - overlap_start)
                    overlap_pct = overlap / word_duration
                    segment_size = e - s
                    
                    if overlap_pct > best_overlap_pct or (
                        overlap_pct == best_overlap_pct and overlap_pct > 0 and segment_size < best_segment_size
                    ):
                        best_overlap_pct = overlap_pct
                        best_segment_size = segment_size
                        assigned_speaker = sp
                
                # Strategy 4: If no overlap, assign to nearest segment
                if best_overlap_pct == 0:
                    min_distance = float('inf')
                    for s, e, sp in spk_ts:
                        if we < s:
                            distance = s - we
                        else:
                            distance = ws - e
                        
                        if distance < min_distance:
                            min_distance = distance
                            assigned_speaker = sp
        
        wrd_spk_mapping.append(
            {"word": wrd, "start_time": ws, "end_time": we, "speaker": assigned_speaker}
        )
    return wrd_spk_mapping


def get_first_word_idx_of_sentence(word_idx, word_list, speaker_list, max_words):
    is_sentence_end = (
        lambda x: x >= 0 and word_list[x][-1] in sentence_ending_punctuations
    )
    left_idx = word_idx
    while (
        left_idx > 0
        and word_idx - left_idx < max_words
        and speaker_list[left_idx - 1] == speaker_list[left_idx]
        and not is_sentence_end(left_idx - 1)
    ):
        left_idx -= 1
    return left_idx if left_idx == 0 or is_sentence_end(left_idx - 1) else -1


def get_last_word_idx_of_sentence(word_idx, word_list, max_words):
    is_sentence_end = (
        lambda x: x >= 0 and word_list[x][-1] in sentence_ending_punctuations
    )
    right_idx = word_idx
    while (
        right_idx < len(word_list) - 1
        and right_idx - word_idx < max_words
        and not is_sentence_end(right_idx)
    ):
        right_idx += 1
    return (
        right_idx
        if right_idx == len(word_list) - 1 or is_sentence_end(right_idx)
        else -1
    )


def get_realigned_ws_mapping_with_punctuation(word_speaker_mapping, max_words_in_sentence=50):
    """Realign words based on punctuation and sentence boundaries"""
    is_sentence_end = (
        lambda x: x >= 0
        and word_speaker_mapping[x]["word"][-1] in sentence_ending_punctuations
    )
    wsp_len = len(word_speaker_mapping)

    words_list, speaker_list = [], []
    for line_dict in word_speaker_mapping:
        words_list.append(line_dict["word"])
        speaker_list.append(line_dict["speaker"])

    k = 0
    while k < len(word_speaker_mapping):
        # Preserve speaker boundaries as detected by diarization model
        # Don't force majority voting across speaker changes
        k += 1

    realigned_list = []
    for k, line_dict in enumerate(word_speaker_mapping):
        new_dict = line_dict.copy()
        new_dict["speaker"] = speaker_list[k]
        realigned_list.append(new_dict)

    return realigned_list


def get_sentences_speaker_mapping(word_speaker_mapping, spk_ts):
    """Group words into sentences by speaker"""
    sentence_checker = nltk.tokenize.PunktSentenceTokenizer().text_contains_sentbreak
    s, e, spk = spk_ts[0]
    prev_spk = spk

    snts = []
    snt = {"speaker": f"Speaker {spk}", "start_time": s, "end_time": e, "text": ""}

    for wrd_dict in word_speaker_mapping:
        wrd, spk = wrd_dict["word"], wrd_dict["speaker"]
        s, e = wrd_dict["start_time"], wrd_dict["end_time"]
        if spk != prev_spk or sentence_checker(snt["text"] + " " + wrd):
            snts.append(snt)
            snt = {
                "speaker": f"Speaker {spk}",
                "start_time": s,
                "end_time": e,
                "text": "",
            }
        else:
            snt["end_time"] = e
        snt["text"] += wrd + " "
        prev_spk = spk

    snts.append(snt)
    return snts


def format_timestamp(milliseconds, always_include_hours=False, decimal_marker="."):
    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000
    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000
    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"


def get_speaker_aware_transcript(sentences_speaker_mapping, f):
    """Write speaker-aware transcript"""
    previous_speaker = sentences_speaker_mapping[0]["speaker"]
    f.write(f"{previous_speaker}: ")

    for sentence_dict in sentences_speaker_mapping:
        speaker = sentence_dict["speaker"]
        sentence = sentence_dict["text"]
        if speaker != previous_speaker:
            f.write(f"\n\n{speaker}: ")
            previous_speaker = speaker
        f.write(sentence + " ")


def write_srt(sentences_speaker_mapping, f):
    """Write SRT subtitle file"""
    for i, segment in enumerate(sentences_speaker_mapping, start=1):
        print(
            f"{i}\n"
            f"{format_timestamp(segment['start_time'], always_include_hours=True, decimal_marker=',')} --> "
            f"{format_timestamp(segment['end_time'], always_include_hours=True, decimal_marker=',')}\n"
            f"{segment['speaker']}: {segment['text'].strip().replace('-->', '->')}\n",
            file=f,
            flush=True,
        )


def main():
    if not AUDIO_FILE.exists():
        raise FileNotFoundError(f"Audio file not found: {AUDIO_FILE}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Processing: {AUDIO_FILE}\n")

    # Step 1: Transcribe with Whisper
    print("[1/4] Transcribing audio with Whisper...")
    whisper_model = faster_whisper.WhisperModel("small", device=device, compute_type="float16")
    whisper_pipeline = faster_whisper.BatchedInferencePipeline(whisper_model)
    
    audio_waveform = faster_whisper.decode_audio(str(AUDIO_FILE))
    suppress_tokens = find_numeral_symbol_tokens(whisper_model.hf_tokenizer)
    
    transcript_segments, info = whisper_pipeline.transcribe(
        audio_waveform,
        language=None,
        suppress_tokens=suppress_tokens,
        batch_size=8,
        without_timestamps=True,
    )
    
    full_transcript = "".join(segment.text for segment in transcript_segments)
    print(f"Transcript: {full_transcript[:80]}...\n")
    
    del whisper_model, whisper_pipeline
    torch.cuda.empty_cache()

    # Step 2: Forced alignment
    print("[2/4] Performing forced alignment...")
    alignment_model, alignment_tokenizer = load_alignment_model(
        device,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    audio_tensor = (
        torch.from_numpy(audio_waveform)
        .to(alignment_model.dtype)
        .to(alignment_model.device)
    )

    emissions, stride = generate_emissions(alignment_model, audio_tensor, batch_size=8)
    del alignment_model
    torch.cuda.empty_cache()

    tokens_starred, text_starred = preprocess_text(
        full_transcript,
        romanize=True,
        language=langs_to_iso.get(info.language, "eng"),
    )

    segments, scores, blank_token = get_alignments(emissions, tokens_starred, alignment_tokenizer)
    spans = get_spans(tokens_starred, segments, blank_token)
    word_timestamps = postprocess_results(text_starred, spans, stride, scores)
    print(f"Aligned {len(word_timestamps)} words\n")

    # Step 3: Speaker diarization with Sortformer
    print("[3/4] Running Sortformer speaker diarization...")
    
    # Load Sortformer model
    diar_model = SortformerEncLabelModel.from_pretrained("nvidia/diar_sortformer_4spk-v1")
    diar_model = diar_model.to(device)
    diar_model.eval()
    
    # Prepare audio as numpy array for diarization
    # Sortformer accepts: str (file path) or np.ndarray
    audio_for_diar = audio_tensor.cpu().float().numpy()
    if audio_for_diar.ndim == 2:
        audio_for_diar = audio_for_diar[0]  # Remove batch dimension if present
    
    print(f"Audio shape for diarization: {audio_for_diar.shape}")
    
    # Run diarization with numpy array + sample rate
    # Output format: list of lists with strings like "2.800 5.360 speaker_0"
    predicted_segments = diar_model.diarize(audio=audio_for_diar, sample_rate=TARGET_SR, batch_size=1)
    
    print(f"Diarization segments detected: {len(predicted_segments[0]) if predicted_segments else 0}\n")

    # Step 4: Map speakers to words/sentences
    print("[4/4] Mapping speakers to sentences...\n")
    
    # Convert Sortformer output to speaker timeline (in milliseconds)
    # Output format: list of lists with strings like "2.800 5.360 speaker_0"
    speaker_ts = []
    for segment_str in predicted_segments[0]:  # predicted_segments is list of lists
        parts = segment_str.split()
        if len(parts) >= 3:
            start_sec = float(parts[0])
            end_sec = float(parts[1])
            speaker_label = parts[2]  # e.g., "speaker_0" or "speaker_1"
            speaker_id = int(speaker_label.split("_")[-1])
            speaker_ts.append([int(start_sec * 1000), int(end_sec * 1000), speaker_id])
    
    # IMPORTANT: Sort speaker_ts by start time, not by speaker ID!
    # Sortformer output groups segments by speaker, but we need chronological order
    speaker_ts.sort(key=lambda x: x[0])
    
    # Debug: check what speakers we detected
    unique_speakers = set(s[2] for s in speaker_ts)
    print(f"Unique speaker IDs in segments: {sorted(unique_speakers)}")
    print(f"Total segments: {len(speaker_ts)}\n")

    # Map words to speakers
    wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

    # Optional punctuation restoration
    if info.language in punct_model_langs:
        try:
            from deepmultilingualpunctuation import PunctuationModel
            print("Restoring punctuation...")
            punct_model = PunctuationModel(model="kredor/punctuate-all")
            words_list = [w["word"] for w in wsm]
            labled_words = punct_model.predict(words_list)

            for word_dict, (word, punct, *_) in zip(wsm, labled_words):
                if punct in ".?!" and word_dict["word"] and word_dict["word"][-1] not in ".,;:!?":
                    word_dict["word"] += punct

            wsm = get_realigned_ws_mapping_with_punctuation(wsm)
        except (ImportError, TypeError) as e:
            print(f"Punctuation restoration skipped: {e}")
    
    ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

    # Write outputs
    print(f"Writing output to {OUTPUT_TXT} and {OUTPUT_SRT}...")
    
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        get_speaker_aware_transcript(ssm, f)

    with open(OUTPUT_SRT, "w", encoding="utf-8") as f:
        write_srt(ssm, f)

    # Print statistics
    print("\n" + "=" * 70)
    print("SPEAKER DIARIZATION COMPLETE (Sortformer)")
    print("=" * 70)
    
    speakers = {}
    for sentence in ssm:
        spk = sentence["speaker"]
        if spk not in speakers:
            speakers[spk] = {"count": 0, "duration": 0}
        speakers[spk]["count"] += 1
        speakers[spk]["duration"] += (sentence["end_time"] - sentence["start_time"]) / 1000

    print(f"Total speakers: {len(speakers)}")
    for spk in sorted(speakers.keys()):
        stats = speakers[spk]
        print(f"  {spk}: {stats['duration']:.2f} sec ({stats['count']} sentences)")
    
    print("=" * 70)
    print(f"Results saved to:\n  {OUTPUT_TXT}\n  {OUTPUT_SRT}")


if __name__ == "__main__":
    main()
