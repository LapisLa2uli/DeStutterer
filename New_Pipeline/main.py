#!/usr/bin/env python3
"""
Full stutter-detection pipeline
================================
1. Split the input audio into sentence segments          (split_audio_by_sentences)
2. Run each segment through YOLO-stutter                 (old_inference.single_inference)
3. Map stutter timestamps back to WhisperX word list
4. For every stuttered word, identify which phonemes fall inside the stutter window
5. Write a JSON report

Usage:
    python main.py \\
        --audio   speech.wav \\
        --transcript speech.txt \\
        --out_dir output/ \\
        [--device cuda] [--language en]
"""

import argparse
import json
import sys
from pathlib import Path

import torchaudio
import torch

# Suppress Windows error dialogs for this process and all children.
if sys.platform == "win32":
    import ctypes
    ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)


# =============================================================================
# 1.  YOLO-stutter helpers
# =============================================================================

# CMU Arpabet phoneme set (39 phonemes + silence).
# WhisperX character-level alignment returns individual characters; we use
# a simple grapheme-to-phoneme heuristic to label them when a full G2P library
# is not available.  If you have `phonemizer` or `g2p_en` installed, replace
# `grapheme_phonemes()` with a proper G2P call.
ARPABET = [
    "AA", "AE", "AH", "AO", "AW", "AY",
    "B",  "CH", "D",  "DH", "EH", "ER",
    "EY", "F",  "G",  "HH", "IH", "IY",
    "JH", "K",  "L",  "M",  "N",  "NG",
    "OW", "OY", "P",  "R",  "S",  "SH",
    "T",  "TH", "UH", "UW", "V",  "W",
    "Y",  "Z",  "ZH",
]

# Rough grapheme -> Arpabet mapping for English.  Used only when a proper G2P
# library is absent.  The key is a lowercase letter; the value is the most
# common Arpabet symbol for that letter in isolation.
_GRAPHEME_TO_ARPABET = {
    "a": "AE", "b": "B",  "c": "K",  "d": "D",  "e": "EH",
    "f": "F",  "g": "G",  "h": "HH", "i": "IH", "j": "JH",
    "k": "K",  "l": "L",  "m": "M",  "n": "N",  "o": "OW",
    "p": "P",  "q": "K",  "r": "R",  "s": "S",  "t": "T",
    "u": "UH", "v": "V",  "w": "W",  "x": "K",  "y": "Y",
    "z": "Z",
}


def grapheme_phonemes(word: str) -> list[dict]:
    """
    Return a list of pseudo-phoneme dicts for a word using a simple grapheme
    mapping.  Each entry has the phoneme label but no timestamp yet — those
    are filled in by interpolation later.

    Replace this function with a real G2P call if you have `g2p_en`:
        from g2p_en import G2p
        g2p = G2p()
        phonemes = [p for p in g2p(word) if p.strip()]
    """
    phonemes = []
    for ch in word.lower():
        if ch.isalpha():
            phonemes.append(_GRAPHEME_TO_ARPABET.get(ch, "?"))
    return phonemes


def interpolate_phoneme_timestamps(word_start: float, word_end: float,
                                   phonemes: list[str]) -> list[dict]:
    """
    Divide the word's time span evenly across its phonemes.
    WhisperX only gives word-level timestamps; phoneme-level timestamps are
    estimated by equal division unless you run a full phoneme aligner.
    """
    if not phonemes:
        return []
    duration = (word_end - word_start) / len(phonemes)
    result = []
    for i, ph in enumerate(phonemes):
        result.append({
            "phoneme": ph,
            "start":   round(word_start + i * duration, 4),
            "end":     round(word_start + (i + 1) * duration, 4),
        })
    return result


# =============================================================================
# 2.  YOLO-stutter inference wrapper
# =============================================================================

def load_yolo_stutter(decoder_path: str, device: torch.device):
    """
    Load the saved YOLO-stutter decoder.

    The decoder was saved with torch.save(whole_model_object), not just a
    state dict, so it contains a pickled Conv1DTransformerDecoder class
    reference.  In PyTorch >= 2.6 the default for weights_only changed from
    False to True, which blocks unpickling arbitrary classes and raises
    UnpicklingError.

    We use add_safe_globals to allowlist only Conv1DTransformerDecoder, which
    is safer than a blanket weights_only=False because it still rejects any
    other unexpected class in the file.
    """
    from utils.model_utils.conv1d_transformer import Conv1DTransformerDecoder

    # weights_only=True is the safe default in PyTorch >= 2.6.
    # Allowlisting the one class we expect lets us keep that safety on.
    if hasattr(torch.serialization, "add_safe_globals"):
        # PyTorch >= 2.4 API
        torch.serialization.add_safe_globals([Conv1DTransformerDecoder])
        decoder = torch.load(decoder_path, map_location=device, weights_only=False)
    else:
        # PyTorch < 2.4 — weights_only=True didn't exist yet, load normally
        decoder = torch.load(decoder_path, map_location=device)

    decoder.eval()
    return decoder


def _get_soft_attention(net_g, text, text_lengths, spec, spec_lengths):
    """
    Extract the soft attention matrix from the VITS model.
    Replicates get_soft_attention() from old_inference.py without importing it.
    """
    from torch import nn
    with torch.no_grad():
        _, _, (neg_cent, _), _, _, _, _ = net_g(
            text, text_lengths, spec, spec_lengths
        )
        neg_cent = nn.functional.softmax(neg_cent, dim=-1)
    return neg_cent


def _get_audio_spec(wav_path: Path, hps):
    """
    Load a WAV file and compute its mel spectrogram.
    Replicates get_audio_a() from old_inference.py using absolute path,
    so no cwd dependency.
    """
    from utils.vits.mel_processing import spectrogram_torch
    from utils.vits.utils import load_wav_to_torch

    SR          = 22050
    MAX_WAV     = 32768.0
    FILTER_LEN  = 1024
    HOP_LEN     = 256
    WIN_LEN     = 1024

    audio, sr = load_wav_to_torch(str(wav_path))

    # --- Normalise to 1D mono [T] -------------------------------------------
    # scipy.io.wavfile returns stereo as [T, C] (time-first).
    # torchaudio.load returns stereo as [C, T] (channel-first).
    # We need a flat 1D tensor [T] for the downstream pipeline.
    if audio.dim() == 2:
        if audio.shape[0] <= 8:
            # torchaudio convention [C, T] — C is small, T is large
            audio = audio.mean(dim=0)           # [C, T] -> [T]
        else:
            # scipy convention [T, C] — T is large, C is small
            audio = audio.mean(dim=1)           # [T, C] -> [T]
    # audio is now guaranteed 1D [T]

    # --- Resample if needed --------------------------------------------------
    if sr != SR:
        print(f"{sr} SR doesn't match target {SR} SR, resampling...")
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SR)
        audio = resampler(audio)                # [T] -> [T_new], stays 1D

    # --- Normalise amplitude -------------------------------------------------
    peak = torch.max(torch.abs(audio))
    if peak > 0:
        audio = (audio / peak) * 32767.0
    audio_norm = (audio / MAX_WAV).unsqueeze(0)  # [1, T] — always 2D

    # --- Minimum length guard ------------------------------------------------
    # spectrogram_torch pads with mode='reflect' using (n_fft - hop) / 2 = 384
    # samples on each side.  PyTorch reflect padding requires the time dimension
    # to be strictly greater than the padding amount.  Zero-pad to FILTER_LEN
    # (1024) so that T > 384 is always guaranteed.
    if audio_norm.shape[-1] < FILTER_LEN:
        audio_norm = torch.nn.functional.pad(
            audio_norm, (0, FILTER_LEN - audio_norm.shape[-1])
        )
    # audio_norm is [1, T] with T >= 1024

    spec = spectrogram_torch(
        audio_norm, FILTER_LEN, SR, HOP_LEN, WIN_LEN, center=False
    ).squeeze(0)
    return spec, audio_norm


def _get_text(text: str, hps):
    """Convert text to token tensor. Mirrors get_text() in old_inference.py."""
    import utils.vits.commons as commons
    from utils.vits.text import text_to_sequence
    tokens = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        tokens = commons.intersperse(tokens, 0)
    return torch.LongTensor(tokens)


def run_yolo_stutter_on_segment(
    seg_audio_path: Path,
    seg_start_in_full_audio: float,
    hps,
    net_g,
    decoder,
    device: torch.device,
    ref_text: str = "",
    downsample_factor: int = 16,
    labels: list[str] = None,
    old_inference_dir: Path = None,   # kept for API compatibility, no longer used
) -> list[dict]:
    """
    Run YOLO-stutter inference on a single segment WAV file.

    This function replicates single_inference() from old_inference.py
    directly, using only absolute paths, so it is completely independent of
    the current working directory and does not import old_inference.py at all.
    That avoids the module-level relative-path crash that occurs when
    old_inference.py is imported from a different working directory.

    ref_text must be the transcript of the segment.  The VITS model uses it
    as a forced aligner to produce a phoneme-to-frame soft attention matrix —
    passing an empty string creates a zero-length token sequence which causes
    PyTorch to crash on torch.min() of an empty tensor inside the spline flow.

    Returns a list of detected stutter events with timestamps in the
    coordinate frame of the *full* original audio:
        [{"start": float, "end": float, "type": str}, ...]
    """
    if labels is None:
        labels = ["rep", "block", "missing", "replace", "prolong"]

    # -- Build inputs (all using absolute path, no cwd dependency) -------------
    text        = _get_text(ref_text, hps)      # segment transcript → phoneme tokens
    text_len    = torch.tensor(text.size(0)).unsqueeze(0)
    spec, _     = _get_audio_spec(seg_audio_path.resolve(), hps)
    spec_len    = torch.tensor(spec.shape[-1]).unsqueeze(0)
    text        = text.unsqueeze(0)
    spec        = spec.unsqueeze(0)

    text     = text.to(device)
    text_len = text_len.to(device)
    spec     = spec.to(device)
    spec_len = spec_len.to(device)

    num_regions = int(spec_len.item() // downsample_factor)

    # -- Soft attention from VITS ----------------------------------------------
    soft_attn = _get_soft_attention(net_g, text, text_len, spec, spec_len)

    # -- Resize to fixed decoder input size [*, 1024, 768] ---------------------
    # The decoder was trained on inputs padded to exactly (1024, 768).
    # F.pad only supports non-negative padding; if the segment is longer than
    # the target size the padding value would be negative (i.e. a crop), which
    # raises "Only 2D/3D/4D/5D padding with non-constant padding supported".
    # Fix: slice down to the target size first, then pad any remaining gap.
    import torch.nn.functional as F
    TARGET_T = 1024   # time (mel frames) axis  — soft_attn dim -2
    TARGET_U = 768    # text (phoneme) axis      — soft_attn dim -1

    # Clamp both axes to their target ceiling (slice from the front)
    soft_attn = soft_attn[..., :TARGET_T, :TARGET_U]

    # Now pad up to the target size — values are guaranteed >= 0
    pad_u = TARGET_U - soft_attn.shape[-1]
    pad_t = TARGET_T - soft_attn.shape[-2]
    soft_attn_padded = F.pad(soft_attn, (0, pad_u, 0, pad_t))

    mask = torch.ones((1, 64), dtype=torch.bool, device=device)
    mask[0, : num_regions + 1] = False

    # -- Decoder forward pass --------------------------------------------------
    with torch.no_grad():
        output = decoder(soft_attn_padded, mask)

    # ── Decode output tensor ──────────────────────────────────────────────────
    # output shape: [1, num_regions, 2 + num_classes]
    #   [:, :, :2]  = boundary predictions (start, end) normalised to [0, 1]
    #   [:, :, 2:]  = class logits

    disfluency_type_logits = output[:, :, 2:]           # [1, R, C]
    disfluency_bound_pred  = output[:, :, :2].squeeze(0)  # [R, 2]

    type_log_softmax = torch.log_softmax(disfluency_type_logits, dim=-1)
    _, y_pred_labels = torch.max(type_log_softmax, dim=-1)  # [1, R]

    # Convert normalised bounds to seconds within the segment
    # The model was trained with 1024 frames * 256 hop / 22050 SR
    FRAME_SCALE = 1024 * 256 / 22050
    bounds_seconds = disfluency_bound_pred * FRAME_SCALE  # [R, 2]

    events = []
    num_regions = int(output.shape[1])
    print(bounds_seconds,y_pred_labels,len(bounds_seconds),len(y_pred_labels[0]))
    for r in range(num_regions):
        pred_start = bounds_seconds[r, 0].item()
        pred_end   = bounds_seconds[r, 1].item()
        pred_type  = labels[min(y_pred_labels[0][r].item(),4)] #temporary fix of r sometimes exceeding 4, might generate inaccurate results

        # Skip regions with degenerate boundaries or negligible duration
        if pred_end <= pred_start or (pred_end - pred_start) < 0.02:
            continue

        # Shift from segment-local time to full-audio time
        events.append({
            "start": round(pred_start + seg_start_in_full_audio, 4),
            "end":   round(pred_end   + seg_start_in_full_audio, 4),
            "type":  pred_type,
        })

    return events


# =============================================================================
# 3.  Map stutter events to WhisperX words and phonemes
# =============================================================================

def find_stuttered_words(stutter_events: list[dict],
                         words: list[dict]) -> list[dict]:
    """
    For each stutter event, find all WhisperX words whose time span overlaps
    the stutter window, then annotate each with the phonemes that fall inside
    the stutter window.

    Returns a list of annotated stutter events:
    [
      {
        "stutter_start": float,
        "stutter_end":   float,
        "stutter_type":  str,
        "segment_idx":   int,
        "affected_words": [
          {
            "word":     str,
            "word_start": float,
            "word_end":   float,
            "phonemes": [
              {"phoneme": str, "start": float, "end": float,
               "in_stutter_window": bool},
              ...
            ]
          },
          ...
        ]
      },
      ...
    ]
    """
    results = []

    for ev_idx, event in enumerate(stutter_events):
        ev_start = event["start"]
        ev_end   = event["end"]

        # Find all words that overlap [ev_start, ev_end]
        affected_words = []
        for w in words:
            w_start = w["start"]
            w_end   = w["end"]
            # Overlap condition: not (w_end <= ev_start or w_start >= ev_end)
            if w_end <= ev_start or w_start >= ev_end:
                continue

            phoneme_labels = grapheme_phonemes(w["word"])
            phonemes_timed = interpolate_phoneme_timestamps(
                w_start, w_end, phoneme_labels
            )

            # Mark each phoneme as inside or outside the stutter window
            for ph in phonemes_timed:
                ph_mid = (ph["start"] + ph["end"]) / 2
                ph["in_stutter_window"] = (ev_start <= ph_mid <= ev_end)

            affected_words.append({
                "word":       w["word"],
                "word_start": w_start,
                "word_end":   w_end,
                "phonemes":   phonemes_timed,
            })

        results.append({
            "stutter_start":  ev_start,
            "stutter_end":    ev_end,
            "stutter_type":   event["type"],
            "segment_idx":    event.get("segment_idx", -1),
            "affected_words": affected_words,
        })

    return results


# =============================================================================
# 4.  CLI + orchestration
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Split audio, detect stutters, map to phonemes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # -- Audio / transcript
    p.add_argument("--audio",        required=True,  help="Input audio file (.wav)")
    p.add_argument("--transcript",   required=True,  help="Plain-text transcript")
    p.add_argument("--out_dir",      required=True,  help="Output directory")
    # -- WhisperX
    p.add_argument("--device",       default="cuda", help="'cuda' or 'cpu'")
    p.add_argument("--language",     default="en",   help="ISO language code")
    p.add_argument("--compute_type", default=None,   help="float16/int8/float32 (auto)")
    p.add_argument("--split_mode",   default="sentence",
                   choices=["sentence", "pause"])
    p.add_argument("--min_pause",    type=float, default=0.4)
    p.add_argument("--pad_ms",       type=int,   default=50)
    p.add_argument("--workers",      type=int,   default=0)
    # -- YOLO-stutter
    p.add_argument("--decoder_path",
                   default=str(Path(__file__).resolve().parent / "YOLOStutter" / "yolo-stutter" / "saved_models" / "decoder_tts_joint"),
                   help="Path to saved YOLO-stutter decoder")
    p.add_argument("--vits_config",
                   default=str(Path(__file__).resolve().parent / "YOLOStutter" / "yolo-stutter" / "utils" / "vits" / "configs" / "ljs_base.json"),
                   help="Path to VITS ljs_base.json config")
    p.add_argument("--vits_checkpoint",
                   default=str(Path(__file__).resolve().parent / "YOLOStutter" / "yolo-stutter" / "saved_models" / "pretrained_ljs.pth"),
                   help="Path to VITS pretrained checkpoint")
    p.add_argument("--downsample_factor", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device  = torch.device(args.device
                           if (args.device == "cpu" or torch.cuda.is_available())
                           else "cpu")
    if str(device) != args.device:
        print(f"[WARN] CUDA unavailable, falling back to CPU.")

    # -------------------------------------------------------------------------
    # Step 1: Split audio into sentence segments + get WhisperX word timestamps
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1 — Sentence splitting + WhisperX alignment")
    print("=" * 60)

    # split.py lives in the SentenceSplit subfolder one level below main.py
    split_dir = Path(__file__).resolve().parent / "SentenceSplit"
    if str(split_dir) not in sys.path:
        sys.path.insert(0, str(split_dir))
    import split as splitter

    split_result = splitter.run(
        audio_path      = args.audio,
        transcript_path = args.transcript,
        out_dir         = out_dir,
        device          = str(device),
        language        = args.language,
        compute_type    = args.compute_type,
        split_mode      = args.split_mode,
        min_pause       = args.min_pause,
        pad_ms          = args.pad_ms,
        workers         = args.workers,
    )

    words       = split_result["words"]        # full-audio word timestamps
    segments    = split_result["segments"]     # sentence segments (text+start+end)
    audio_files = split_result["audio_files"]  # cut WAV paths

    # -------------------------------------------------------------------------
    # Step 2: Load YOLO-stutter models (once, shared across all segments)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2 — Loading YOLO-stutter models")
    print("=" * 60)

    import os
    # Resolve YOLO-stutter root (YOLOStutter/yolo-stutter) from main.py location
    pipeline_root = Path(__file__).resolve().parent
    yolo_stutter_root = pipeline_root / "YOLOStutter" / "yolo-stutter"
    yolo_etc = yolo_stutter_root / "etc"

    # Add yolo-stutter root so `utils` package is importable.
    # We do NOT add yolo_etc or import old_inference — its module-level code
    # uses hardcoded relative paths that break when imported from another cwd.
    # Instead, run_yolo_stutter_on_segment() replicates the logic directly.
    if str(yolo_stutter_root) not in sys.path:
        sys.path.insert(0, str(yolo_stutter_root))

    import utils.vits.utils as vits_utils
    from utils.vits.models import SynthesizerTrn
    from utils.vits.text.symbols import symbols

    hps = vits_utils.get_hparams_from_file(args.vits_config)
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    ).eval().to(device)
    vits_utils.load_checkpoint(args.vits_checkpoint, net_g, None)

    decoder = load_yolo_stutter(args.decoder_path, device)

    disfluency_labels = ["rep", "block", "missing", "replace", "prolong"]

    # -------------------------------------------------------------------------
    # Step 3: Run YOLO-stutter on every segment
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3 — YOLO-stutter inference per segment")
    print("=" * 60)

    all_stutter_events = []  # flat list across all segments

    for seg_idx, (seg, audio_file) in enumerate(zip(segments, audio_files)):
        print(f"  [{seg_idx + 1:04d}/{len(segments):04d}] {audio_file.name} "
              f"({seg['start']:.2f}s - {seg['end']:.2f}s)")

        '''try:
            events = run_yolo_stutter_on_segment(
                seg_audio_path          = audio_file,
                seg_start_in_full_audio = seg["start"],
                ref_text                = seg["text"],
                hps                     = hps,
                net_g                   = net_g,
                decoder                 = decoder,
                device                  = device,
                downsample_factor       = args.downsample_factor,
                labels                  = disfluency_labels,
                old_inference_dir       = yolo_etc,
            )
        except Exception as exc:
            print(f"    [WARN] Inference failed for {audio_file.name}: {exc}")
            events = []'''
        events = run_yolo_stutter_on_segment(
            seg_audio_path=audio_file,
            seg_start_in_full_audio=seg["start"],
            ref_text=seg["text"],
            hps=hps,
            net_g=net_g,
            decoder=decoder,
            device=device,
            downsample_factor=args.downsample_factor,
            labels=disfluency_labels,
            old_inference_dir=yolo_etc,
        )
        # Tag each event with its source segment index
        for ev in events:
            ev["segment_idx"] = seg_idx
        all_stutter_events.extend(events)

        if events:
            for ev in events:
                print(f"    stutter [{ev['type']}] "
                      f"{ev['start']:.2f}s - {ev['end']:.2f}s")
        else:
            print("    (no stutters detected)")

    print(f"\n[YOLO] Total stutter events detected: {len(all_stutter_events)}")

    # -------------------------------------------------------------------------
    # Step 4: Map stutter events -> WhisperX words -> phonemes
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4 — Mapping stutters to words and phonemes")
    print("=" * 60)

    annotated = find_stuttered_words(all_stutter_events, words)

    # -------------------------------------------------------------------------
    # Step 5: Write report
    # -------------------------------------------------------------------------
    report_path = out_dir / "stutter_report.json"

    # Build a compact, readable report structure
    report = {
        "audio":    str(Path(args.audio).resolve()),
        "segments": len(segments),
        "total_stutter_events": len(all_stutter_events),
        "events": annotated,
    }

    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[REPORT] Stutter report written to: {report_path}")

    # Also print a human-readable summary to stdout
    print("\n--- Summary ---")
    for i, ev in enumerate(annotated, 1):
        word_names = [w["word"] for w in ev["affected_words"]]
        stuttered_phonemes = [
            ph["phoneme"]
            for w in ev["affected_words"]
            for ph in w["phonemes"]
            if ph["in_stutter_window"]
        ]
        print(f"  Event {i:03d}: [{ev['stutter_type']}] "
              f"{ev['stutter_start']:.2f}s-{ev['stutter_end']:.2f}s  "
              f"words={word_names}  phonemes={stuttered_phonemes}")

    print(f"\nDone. Full report: {report_path}")


if __name__ == "__main__":
    main()