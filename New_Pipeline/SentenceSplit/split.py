#!/usr/bin/env python3
"""
Audio Sentence Splitter — WhisperX-based pipeline
==================================================
Requirements (install before running):
    pip install whisperx
    # WhisperX bundles its own CTC aligner; no separate MFA install needed.
    # For GPU acceleration: pip install torch torchvision torchaudio
    # ffmpeg must be on your PATH

Why WhisperX instead of MFA?
    MFA is a *forced* aligner: it must place every transcript word somewhere
    in the audio, even across stutters, long pauses, and disfluent stretches.
    When the speaker repeats or skips words, MFA distorts surrounding timestamps
    to compensate — producing word times that are seconds off the ground truth.

    WhisperX first *transcribes* the audio with Whisper (so it knows exactly
    what was said and when, including stutters), then applies a CTC phoneme
    aligner only on short, confident segments. Timestamps are therefore
    anchored to what the speaker actually said, not to a clean reference text.

Usage:
    python split_audio_by_sentences.py \
        --audio   my_speech.wav \
        --transcript  my_transcript.txt \
        --out_dir output_segments/
"""

import argparse
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

# Suppress Windows Error Reporting dialog boxes and hard-error popups for this
# process and all subprocesses it spawns.  Without this, a crashing child
# process (e.g. a DLL fault inside ffmpeg or a WhisperX dependency) triggers
# WerFault.exe and shows a modal "application has stopped working" dialog that
# blocks execution and cannot be dismissed programmatically.
#
# SEM_FAILCRITICALERRORS  (0x0001) — send hard-error messages to the calling
#                                    process rather than showing a dialog
# SEM_NOGPFAULTERRORBOX   (0x0002) — disable the WER crash dialog (GPF box)
# SEM_NOOPENFILEERRORBOX  (0x8000) — suppress "file not found" error boxes
#
# SetErrorMode is inherited by child processes, so this covers ffmpeg and every
# other subprocess launched via subprocess.run / ProcessPoolExecutor workers.
if sys.platform == "win32":
    import ctypes
    ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)


# -----------------------------------------------------------------------------
# 1.  WhisperX alignment
# -----------------------------------------------------------------------------

def run_whisperx(audio_path: Path, device: str = "cuda",
                 language: str = "en",
                 compute_type: str = None) -> list[dict]:
    """
    Transcribe and align audio with WhisperX.
    Returns a flat list of word-level dicts:
        [{"word": str, "start": float, "end": float}, ...]

    WhisperX pipeline:
      1. Whisper transcribes the audio into timed segments with word timestamps.
      2. A CTC phoneme model (wav2vec2 / MMS) refines each word boundary.

    Because Whisper first decides *what* was said (including disfluencies) and
    only then aligns, timestamps stay accurate even across stuttered passages.
    Words that MFA would distort (false starts, repetitions) are simply
    labelled as separate word entries here with their actual spoken positions.

    compute_type controls the numerical precision used by CTranslate2:
      - "float16"  fast, GPU only  (WhisperX default, crashes on CPU)
      - "int8"     fast, works on CPU and GPU  (recommended for CPU)
      - "float32"  slowest, most compatible
    If None, this function picks "int8" for CPU and "float16" for CUDA.
    """
    import whisperx

    # Verify CUDA is actually available; fall back to CPU with a warning.
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        print("[WHISPERX] WARNING: CUDA requested but not available -- falling back to CPU.")
        device = "cpu"

    # Auto-select compute type if not specified.
    # float16 is the WhisperX default but is not supported on CPU-only builds;
    # it raises "Requested float16 compute type, but the target device or
    # backend do not support efficient float16 computation."
    if compute_type is None:
        compute_type = "float16" if device == "cuda" else "int8"

    print(f"[WHISPERX] Loading model (device={device}, compute_type={compute_type}) ...")
    model = whisperx.load_model("base.en", device, language=language,
                                compute_type=compute_type)

    print("[WHISPERX] Transcribing ...")
    audio  = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, batch_size=16)

    print("[WHISPERX] Aligning word timestamps ...")
    align_model, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    result = whisperx.align(
        result["segments"], align_model, metadata, audio, device,
        return_char_alignments=False,
    )

    # Flatten all segments into a single word list, dropping entries with no
    # timestamp (WhisperX marks those with missing start/end keys).
    words = []
    for seg in result["segments"]:
        for w in seg.get("words", []):
            if "start" in w and "end" in w:
                words.append({
                    "word":  w["word"].strip(),
                    "start": w["start"],
                    "end":   w["end"],
                })
    return words


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Sentence boundary detection
# ─────────────────────────────────────────────────────────────────────────────

def split_transcript_into_sentences(raw_text: str) -> list[str]:
    """
    Split the original (punctuated) transcript into sentences / phrases.
    Strategy:
      1. Try to use nltk sent_tokenize (best quality).
      2. Fall back to a regex approach if nltk isn't installed.
    Returns a list of sentence strings (whitespace-normalised, no leading/
    trailing spaces).
    """
    try:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        sentences = nltk.sent_tokenize(raw_text)
    except ImportError:
        # Regex fallback: split on . ! ? followed by whitespace/end
        sentences = re.split(r'(?<=[.!?])\s+', raw_text.strip())

    # Also split on em-dash / long pauses marked with " — " or "..."
    result = []
    for s in sentences:
        parts = re.split(r'\s*(?:—|\.{3})\s*', s)
        result.extend(p.strip() for p in parts if p.strip())

    return result


def normalise_word(w: str) -> str:
    """Lowercase, strip punctuation — used for matching."""
    return re.sub(r"[^a-z']", "", w.lower())


def words_match(mfa_word: str, transcript_word: str) -> bool:
    """
    Return True if an MFA token is a plausible match for a transcript token.
    Handles minor phonetic differences and clitic splits by requiring either
    an exact normalised match or a common-prefix match of ≥4 chars (or the
    full shorter word if it's short).
    """
    a, b = normalise_word(mfa_word), normalise_word(transcript_word)
    if not a or not b:
        return False
    if a == b:
        return True
    min_len = min(len(a), len(b))
    prefix  = max(4, min_len)          # require full word if it's short
    return a[:prefix] == b[:prefix]


def align_sentences_to_words(sentences: list[str],
                             words: list[dict]) -> list[dict]:
    """
    Match each sentence to a contiguous span of MFA-aligned words, tolerating
    stutter repetitions and false-start fragments that appear in the audio but
    not in the written transcript.

    Algorithm
    ---------
    1.  Build two flat token lists:
          • transcript_tokens  — one entry per word in all sentences combined,
                                 labelled with its sentence index
          • mfa_tokens         — non-silence words from the TextGrid

    2.  Run difflib.SequenceMatcher between the two token lists.
        SequenceMatcher finds the longest common subsequence, which means:
          • Stutter fragments in mfa_tokens (not in the transcript) become
            *insertions* and are simply skipped — they no longer derail the
            pointer.
          • Transcript words that MFA split or misheard become *deletions*;
            the surrounding matches still anchor the sentence boundaries
            correctly.

    3.  From the matching blocks, record the first and last mfa_token index
        that matched each sentence, then read off start/end timestamps.
    """
    from difflib import SequenceMatcher

    # ── Build flat token sequences ────────────────────────────────────────────
    # transcript side: (sentence_index, normalised_word)
    t_tokens: list[tuple[int, str]] = []
    for si, sent in enumerate(sentences):
        for tok in sent.split():
            nw = normalise_word(tok)
            if nw:
                t_tokens.append((si, nw))

    # MFA side: only real (non-silence) words
    real_words = [w for w in words if w["word"]]
    m_norms    = [normalise_word(w["word"]) for w in real_words]

    if not t_tokens or not real_words:
        raise ValueError("No tokens to align.")

    # ── Sequence match ────────────────────────────────────────────────────────
    # We compare normalised strings so SequenceMatcher can judge equality.
    t_norms = [tok for _, tok in t_tokens]

    # junk heuristic: very short MFA tokens (≤2 chars) are likely stutter
    # fragments — treat them as junk so the matcher skips them automatically.
    def is_junk(w: str) -> bool:
        return len(w) <= 2

    sm = SequenceMatcher(is_junk, t_norms, m_norms, autojunk=False)

    # matching_blocks: list of (t_start, m_start, length) triples where
    # t_norms[t_start:t_start+length] == m_norms[m_start:m_start+length]
    # We use words_match for a softer comparison, so post-filter below.
    blocks = sm.get_matching_blocks()   # last block is always (len_a, len_b, 0)

    # ── Map matched positions → sentence boundaries ───────────────────────────
    # For every (t_idx, m_idx) pair that the DP says matched, record which
    # sentence owns t_idx and which mfa word is at m_idx.
    word_starts: dict[int, int] = {}   # sent_idx → earliest matched mfa index
    word_ends:   dict[int, int] = {}   # sent_idx → latest  matched mfa index

    for t_start, m_start, length in blocks:
        for offset in range(length):
            ti = t_start + offset
            mi = m_start + offset
            # soft re-check: SequenceMatcher matched on exact normalised
            # strings; words_match handles prefix tolerance for short words
            if not words_match(real_words[mi]["word"], t_tokens[ti][1]):
                continue
            si = t_tokens[ti][0]
            if si not in word_starts or mi < word_starts[si]:
                word_starts[si] = mi
            if si not in word_ends or mi > word_ends[si]:
                word_ends[si] = mi

    # ── Handle sentences that got zero direct matches ─────────────────────────
    # This happens when a whole sentence is badly stuttered and SequenceMatcher
    # skipped all its tokens.  Interpolate boundaries from neighbouring
    # sentences so we don't silently drop segments.
    all_si = list(range(len(sentences)))
    for si in all_si:
        if si in word_starts:
            continue

        # Find the nearest matched sentence before and after
        prev_end   = word_ends.get(si - 1)
        next_start = word_starts.get(si + 1)

        if prev_end is not None and next_start is not None:
            # Place the missed sentence in the gap between neighbours
            mid = (prev_end + next_start) // 2
            word_starts[si] = prev_end  + 1 if prev_end  + 1 <= mid else mid
            word_ends[si]   = next_start - 1 if next_start - 1 >= mid else mid
        elif prev_end is not None:
            word_starts[si] = word_ends[si] = prev_end + 1 \
                if prev_end + 1 < len(real_words) else prev_end
        elif next_start is not None:
            word_starts[si] = word_ends[si] = max(0, next_start - 1)
        else:
            print(f"[WARN] Could not align sentence {si} (no neighbours either): "
                  f"{sentences[si][:60]!r}")
            continue

        print(f"[WARN] Sentence {si} had no direct matches (heavy stutter?); "
              f"boundary interpolated from neighbours.")

    # -- Build final segment list ----------------------------------------------

    segments = []
    for si, sent in enumerate(sentences):
        if si not in word_starts:
            print(f"[WARN] Dropping sentence {si} -- could not locate in audio: "
                  f"{sent[:60]!r}")
            continue
        wi_start   = word_starts[si]
        wi_end     = word_ends[si]
        # Guard against inverted boundaries from interpolation edge cases
        if wi_start > wi_end:
            wi_start, wi_end = wi_end, wi_start
        start_time = real_words[wi_start]["start"]
        end_time   = real_words[wi_end]["end"]
        segments.append({"text": sent, "start": start_time, "end": end_time})

    return segments


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Pause-gap fallback segmentation (no punctuation)
# ─────────────────────────────────────────────────────────────────────────────

def segment_by_pauses(words: list[dict], min_pause: float = 0.4) -> list[dict]:
    """
    Fallback: group words into segments separated by silences >= min_pause.
    Returns the same format as align_sentences_to_words.
    """
    real_words = [w for w in words if w["word"]]
    if not real_words:
        return []

    segments = []
    seg_words = [real_words[0]]

    for prev, curr in zip(real_words, real_words[1:]):
        gap = curr["start"] - prev["end"]
        if gap >= min_pause:
            segments.append({
                "text":  " ".join(w["word"] for w in seg_words),
                "start": seg_words[0]["start"],
                "end":   seg_words[-1]["end"],
            })
            seg_words = [curr]
        else:
            seg_words.append(curr)

    if seg_words:
        segments.append({
            "text":  " ".join(w["word"] for w in seg_words),
            "start": seg_words[0]["start"],
            "end":   seg_words[-1]["end"],
        })

    return segments


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Audio cutting with ffmpeg
# ─────────────────────────────────────────────────────────────────────────────

def cut_segment(audio_path: Path, start: float, end: float,
                out_path: Path, pad_ms: int = 50) -> None:
    """
    Cut [start, end] from audio_path, add pad_ms silence on each side.

    CRITICAL: -ss must come AFTER -i (output-side seek, not input seek).
    Placing -ss before -i causes ffmpeg to seek to the nearest *keyframe*,
    which can be seconds before the target — so every segment silently starts
    from the same keyframe position and sounds identical.

    We also never use -c copy: stream-copy inherits the keyframe misalignment
    and produces wrong cuts even when the seek timestamp looks right.
    Re-encoding to PCM WAV is lossless, sample-accurate, and universally
    readable.
    """
    pad_s    = pad_ms / 1000.0
    t_start  = max(0.0, start - pad_s)
    duration = (end + pad_s) - t_start

    cmd = [
        "ffmpeg", "-y",
        "-hide_banner",           # suppress version/build info banner
        "-loglevel", "error",     # silence warnings; only print actual errors
        "-i",  str(audio_path),   # input declared FIRST
        "-ss", f"{t_start:.6f}",  # seek declared AFTER -i  ==>  output seek
        "-t",  f"{duration:.6f}",
        "-acodec", "pcm_s16le",   # re-encode: lossless PCM, no keyframe snapping
        str(out_path),
    ]
    # On Windows, CREATE_NO_WINDOW prevents ffmpeg from flashing a console
    # window when this script is itself called from a GUI or another program
    # with no visible terminal.  On non-Windows platforms the flag is 0 (no-op).
    creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    result = subprocess.run(cmd, capture_output=True, text=True,
                            creationflags=creation_flags)
    if result.returncode != 0:
        print(f"[ffmpeg error] {result.stderr}")


def cut_all_segments(audio_path: Path, segments: list[dict],
                     out_dir: Path, pad_ms: int = 50,
                     workers: int = 0) -> list[Path]:
    """
    Cut all segments and return list of output paths.

    Each ffmpeg call writes to a distinct file with no shared state, so
    this is safe to parallelise with a process pool.

    workers=0 (default) uses os.cpu_count(); workers=1 disables parallelism.
    """
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    out_dir.mkdir(parents=True, exist_ok=True)
    ext       = audio_path.suffix
    n_workers = workers if workers > 0 else os.cpu_count() or 1

    # Build the full list of (index, segment, output_path) up front so we can
    # restore the original order after futures complete out of order.
    tasks = []
    for i, seg in enumerate(segments, 1):
        fname = out_dir / f"segment_{i:04d}{ext}"
        tasks.append((i, seg, fname))

    print(f"\n[CUT] Writing {len(tasks)} segment(s) to {out_dir}/ "
          f"({n_workers} worker(s))")

    # Map index -> path so we can return results in the original order.
    results: dict[int, Path] = {}

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        future_to_idx = {
            pool.submit(cut_segment, audio_path, seg["start"], seg["end"],
                        fname, pad_ms): i
            for i, seg, fname in tasks
        }
        for future in as_completed(future_to_idx):
            i = future_to_idx[future]
            future.result()  # re-raises any exception from the worker
            seg, fname = segments[i - 1], tasks[i - 1][2]
            results[i] = fname
            preview = seg["text"][:70].replace("\n", " ")
            print(f"  [{i:04d}] {seg['start']:.2f}s - {seg['end']:.2f}s  {preview!r}")

    return [results[i] for i in sorted(results)]


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Save manifest
# ─────────────────────────────────────────────────────────────────────────────

def save_manifest(segments: list[dict], out_paths: list[Path],
                  manifest_path: Path) -> None:
    entries = []
    for seg, path in zip(segments, out_paths):
        entries.append({
            "file":  path.name,
            "start": round(seg["start"], 4),
            "end":   round(seg["end"],   4),
            "text":  seg["text"],
        })
    manifest_path.write_text(
        json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[MANIFEST] Saved to {manifest_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Split audio by sentence using WhisperX alignment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples
            --------
            # Basic usage (CPU)
            python split_audio_by_sentences.py \\
                --audio speech.wav --transcript speech.txt --out_dir out/

            # GPU (much faster for long files)
            python split_audio_by_sentences.py \\
                --audio speech.wav --transcript speech.txt --out_dir out/ \\
                --device cuda

            # Pause-based splitting
            python split_audio_by_sentences.py \\
                --audio speech.wav --transcript speech.txt --out_dir out/ \\
                --split_mode pause --min_pause 0.35
        """)
    )
    p.add_argument("--audio",      required=True,  help="Input audio file")
    p.add_argument("--transcript", required=True,  help="Plain-text transcript file")
    p.add_argument("--out_dir",    required=True,  help="Output directory for segments")
    p.add_argument("--device",     default="cuda", help="'cuda' (default) or 'cpu'; falls back to cpu if CUDA unavailable")
    p.add_argument("--language",   default="en",   help="ISO language code (default: en)")
    p.add_argument("--compute_type", default=None,
                   help="float16 (GPU only), int8 (CPU/GPU), float32. Default: auto")
    p.add_argument("--split_mode", default="sentence",
                   choices=["sentence", "pause"],
                   help="'sentence' uses punctuation; 'pause' uses silence gaps")
    p.add_argument("--min_pause",  type=float, default=0.4,
                   help="Min silence gap (s) for pause-based splitting (default: 0.4)")
    p.add_argument("--pad_ms",     type=int,   default=50,
                   help="Silence padding in ms added to each side of a cut (default: 50)")
    p.add_argument("--workers",    type=int,   default=0,
                   help="Parallel ffmpeg workers for cutting (0 = cpu_count, 1 = serial)")
    return p.parse_args()
def run(audio_path: Path,
        transcript_path: Path,
        out_dir: Path,
        device: str = "cuda",
        language: str = "en",
        compute_type: str = None,
        split_mode: str = "sentence",
        min_pause: float = 0.4,
        pad_ms: int = 50,
        workers: int = 0) -> dict:
    """
    Importable entry point for the full split pipeline.

    Returns a dict with:
        {
            "words":      list[dict]   -- WhisperX word-level timestamps
                                          [{"word", "start", "end"}, ...]
            "segments":   list[dict]   -- sentence segments
                                          [{"text", "start", "end"}, ...]
            "audio_files": list[Path]  -- cut audio file paths (same order as segments)
            "out_dir":    Path         -- resolved output directory
        }
    """
    audio_path      = Path(audio_path).resolve()
    transcript_path = Path(transcript_path).resolve()
    out_dir         = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript file not found: {transcript_path}")

    raw_text = transcript_path.read_text(encoding="utf-8").strip()

    # -- Step 1: Transcribe + align with WhisperX --------------------------------
    words = run_whisperx(audio_path, device=device, language=language,
                         compute_type=compute_type)
    print(f"[WHISPERX] Got {len(words)} aligned word(s).")

    # -- DEBUG: dump every word interval ----------------------------------------
    debug_tsv = out_dir / "debug_word_timestamps.tsv"
    with open(debug_tsv, "w", encoding="utf-8") as _fh:
        _fh.write("start\tend\tword\n")
        for _w in words:
            _fh.write(f"{_w['start']:.4f}\t{_w['end']:.4f}\t{_w['word']}\n")
    print(f"[DEBUG] Word timestamps -> {debug_tsv}")
    # ---------------------------------------------------------------------------

    # -- Step 2: Segment --------------------------------------------------------
    if split_mode == "sentence":
        print("[SPLIT] Splitting transcript into sentences ...")
        sentences = split_transcript_into_sentences(raw_text)
        print(f"[SPLIT] {len(sentences)} sentences found.")
        segments = align_sentences_to_words(sentences, words)
    else:
        print(f"[SPLIT] Splitting by pause gaps >= {min_pause}s ...")
        segments = segment_by_pauses(words, min_pause)

    print(f"[SPLIT] {len(segments)} segments to cut.")
    if not segments:
        raise RuntimeError("No segments were produced. Check transcript alignment.")

    # -- Step 3: Cut audio ------------------------------------------------------
    seg_dir    = out_dir / "segments"
    audio_files = cut_all_segments(audio_path, segments, seg_dir, pad_ms,
                                   workers=workers)
    save_manifest(segments, audio_files, out_dir / "manifest.json")
    print(f"\n[SPLIT] Done. {len(audio_files)} audio segment(s) in: {seg_dir}/")

    return {
        "words":       words,
        "segments":    segments,
        "audio_files": audio_files,
        "out_dir":     out_dir,
    }


def main():
    args = parse_args()
    try:
        run(
            audio_path    = args.audio,
            transcript_path = args.transcript,
            out_dir       = args.out_dir,
            device        = args.device,
            language      = args.language,
            compute_type  = args.compute_type,
            split_mode    = args.split_mode,
            min_pause     = args.min_pause,
            pad_ms        = args.pad_ms,
            workers       = args.workers,
        )
    except (FileNotFoundError, RuntimeError) as e:
        sys.exit(f"[ERROR] {e}")


if __name__ == "__main__":
    main()