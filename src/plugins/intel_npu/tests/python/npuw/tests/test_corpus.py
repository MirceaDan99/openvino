# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared test corpus for Kokoro audio-quality tests.

This module is the **single source-of-truth** for test texts.
Both the CI pytest suite (`test_audio_quality_sweep.py`) and the local
development sweep (`samples/kokoro/validate_nan_sweep.py`) should import
from here rather than duplicating strings.

Organisation:
    TEXTS              — list of (id, text) tuples for English (lang_code="a").
    CATEGORIES         — human-readable category labels keyed by text id prefix.
    DEFAULT_VOICE      — default voice used by the parametrized tests.
    DEFAULT_LANGUAGE   — default Kokoro language matching DEFAULT_VOICE.
    MCD_THRESHOLDS     — per-category MCD thresholds (dB).
    get_mcd_threshold  — lookup helper: test_id → threshold.
"""

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_VOICE = "af_heart"
DEFAULT_LANGUAGE = "en-us"
MCD_THRESHOLD_DEFAULT = 6.0  # dB — fallback for categories without an explicit entry
NAN_THRESHOLD = 0  # any NaN in output is a failure
TIMEOUT = 600  # seconds — default timeout for NPU inference

# fmt: off
# ── Per-category MCD thresholds (dB) ─────────────────────────────────────────
# Short / minimal texts produce very little audio, making MCD noisy → lenient.
# Long texts should be very stable → strict.
MCD_THRESHOLDS: dict[str, float] = {
    "minimal":         9.0,   # 1-2 tokens — MCD extremely noisy
    "short":           7.0,   # 1-4 words
    "repetitive":      7.0,   # repetitive syllables
    "sentence":        6.0,   # short sentences
    "numbers":         5.0,   # numbers / special content
    "volume":          5.0,   # whisper / loud
    "punctuation":     6.0,   # heavy punctuation
    "phonetic":        8.0,   # foreign phoneme approximations
    "pauses":          6.0,   # pause-heavy
    "medium":          5.0,   # medium sentences
    "technical":       5.0,   # technical jargon
    "multisentence":   4.0,   # multi-sentence
    "paragraph":       4.0,   # long paragraphs
    "long":            4.0,   # very long text — most stable
}


def get_mcd_threshold(test_id: str) -> float:
    """Return the MCD threshold for a test_id based on its category prefix."""
    category = test_id.rsplit("_", 1)[0]
    return MCD_THRESHOLDS.get(category, MCD_THRESHOLD_DEFAULT)


# ── Category descriptions (for reporting) ────────────────────────────────────
CATEGORIES = {
    "short":       "Very short inputs (1-4 words)",
    "sentence":    "Short sentences",
    "medium":      "Medium sentences with varied phonemes",
    "numbers":     "Numbers and special content",
    "paragraph":   "Long paragraphs (stress test)",
    "repetitive":  "Repetitive patterns",
    "volume":      "Whisper-quiet and loud content",
    "technical":   "Technical jargon",
    "multisentence": "Multi-sentence paragraphs",
    "punctuation": "Special characters and punctuation",
    "long":        "Very long text (max stress)",
    "phonetic":    "Foreign phoneme patterns (English phonetic approximation)",
    "minimal":     "Minimal token counts",
    "pauses":      "Silence/pause heavy",
}

# ── Test texts ────────────────────────────────────────────────────────────────
# Each entry is (test_id: str, text: str).
# test_id follows the pattern "<category>_<seq>" for easy filtering with
# pytest -k, e.g.:  pytest -k "short" or pytest -k "paragraph"
TEXTS: list[tuple[str, str]] = [
    # --- Very short (1-4 words) ---
    ("short_01", "Hello."),
    ("short_02", "Yes!"),
    ("short_03", "No, absolutely not."),
    ("short_04", "Goodbye forever."),

    # --- Short sentences ---
    ("sentence_01", "The quick brown fox jumps over the lazy dog."),
    ("sentence_02", "She sells seashells by the seashore."),
    ("sentence_03", "How much wood would a woodchuck chuck?"),
    ("sentence_04", "Peter Piper picked a peck of pickled peppers."),
    ("sentence_05", "Red lorry, yellow lorry, red lorry, yellow lorry."),

    # --- Medium sentences with varied phonemes ---
    ("medium_01", "The synthesis of speech from text requires careful handling of prosody and intonation."),
    ("medium_02", "Neural network models can generate surprisingly natural-sounding audio output."),
    ("medium_03", "OpenVINO accelerates deep learning inference across Intel hardware platforms."),
    ("medium_04", "The fundamental frequency of a voice determines its perceived pitch."),
    ("medium_05", "Acoustic features extracted from spectrograms drive modern speech generation."),

    # --- Numbers and special content ---
    ("numbers_01", "The temperature today is 72 degrees Fahrenheit, or about 22 Celsius."),
    ("numbers_02", "Call me at 555-0123 or email john@example.com for details."),
    ("numbers_03", "The year 2025 marks the 50th anniversary of the founding."),
    ("numbers_04", "Mix 2.5 cups of flour with 1/3 cup of sugar and 3 eggs."),
    ("numbers_05", "The stock price rose 14.7% to $127.50 per share."),

    # --- Long paragraphs (stress test) ---
    ("paragraph_01",
     "In the realm of artificial intelligence, text to speech synthesis has undergone "
     "remarkable transformation. Early systems relied on concatenative approaches, "
     "stitching together pre-recorded speech units. Modern neural approaches generate "
     "waveforms directly from text, producing natural and expressive speech."),

    ("paragraph_02",
     "The architecture of modern TTS systems typically involves several stages. First, "
     "a text encoder processes the input characters or phonemes. Then, an acoustic model "
     "predicts mel spectrogram frames. Finally, a vocoder converts these spectrograms "
     "into audible waveforms. Each stage can introduce numerical precision challenges."),

    ("paragraph_03",
     "When running inference on neural processing units, floating point precision becomes "
     "critical. The NPU typically operates in float16 or bfloat16 to maximize throughput. "
     "However, this reduced precision can cause issues when values are very small, near "
     "zero, or when operations like division encounter degenerate inputs."),

    # --- Repetitive patterns (may stress specific frequency ranges) ---
    ("repetitive_01", "Ba ba ba ba ba ba ba ba ba ba ba ba ba ba ba ba."),
    ("repetitive_02", "La la la la la la la la la la la la la la la la."),
    ("repetitive_03", "Do re mi fa sol la ti do, do ti la sol fa mi re do."),

    # --- Whisper-quiet and loud content ---
    # Kokoro supports stress modifiers: [word](-1) lowers, [word](+2) raises stress.
    ("volume_01", "[Shh](-2), [whisper](-1) [very](-1) [quietly](-2) [now](-1), [nobody](-1) [should](-2) [hear](-2) [us](-2)."),
    ("volume_02", "[ATTENTION](+2)! [THIS](+2) [IS](+2) [AN](+2) [EMERGENCY](+2) [BROADCAST](+2) [SYSTEM](+2) [TEST](+2)!"),

    # --- Technical jargon ---
    ("technical_01", "The LSTM network processes sequential data through forget gates and cell states."),
    ("technical_02", "Quantization reduces model weights from float32 to int8 with minimal accuracy loss."),
    ("technical_03", "The inverse short-time Fourier transform reconstructs time-domain signals from spectrograms."),

    # --- Multi-sentence paragraphs ---
    ("multisentence_01",
     "The sun set behind the mountains. A cool breeze swept through the valley. "
     "Birds returned to their nests as darkness crept across the landscape. "
     "In the distance, a wolf howled at the rising moon."),

    ("multisentence_02",
     "Machine learning has revolutionized many fields. Computer vision can now identify "
     "objects with superhuman accuracy. Natural language processing enables machines to "
     "understand and generate text. Speech synthesis brings these advances to the audio domain."),

    # --- Special characters and punctuation ---
    ("punctuation_01", "Wait... what? No! Really? Yes, I'm sure. Absolutely, positively, one hundred percent."),
    ("punctuation_02", "First: prepare the ingredients. Second: mix thoroughly. Third: bake at 350 degrees."),
    ("punctuation_03", "He said, 'To be, or not to be,' and the audience was silent."),

    # --- Very long text (max stress) ---
    ("long_01",
     "This is a comprehensive test of the text to speech pipeline designed to push the "
     "limits of the system. The purpose is to generate a substantial amount of audio "
     "output that exercises all components of the neural network, including the encoder, "
     "the decoder, the duration predictor, and the vocoder. By processing a large block "
     "of text in a single pass, we can observe whether numerical instabilities accumulate "
     "over time, potentially leading to NaN values or other artifacts in the output audio. "
     "The specific concern is that very small floating point values, when processed through "
     "the FFT and inverse FFT stages on the NPU, may be rounded to zero due to float16 "
     "precision limitations. When both the real and imaginary components of a complex number "
     "become zero, the angle computation, which is implemented as divide followed by arctan, "
     "produces a NaN that then propagates through the rest of the computation graph."),

    # --- Foreign phoneme patterns (English phonetic approximation) ---
    ("phonetic_01", "Konnichiwa, watashi wa robotto desu. Hajimemashite, douzo yoroshiku."),
    ("phonetic_02", "Bonjour, je suis un systeme de synthese vocale. Comment allez-vous aujourd'hui?"),
    ("phonetic_03", "Guten Tag, ich bin ein Sprachsynthesesystem. Wie geht es Ihnen heute?"),

    # --- Minimal token counts ---
    ("minimal_01", "A."),
    ("minimal_02", "I."),
    ("minimal_03", "Oh?"),

    # --- Silence/pause heavy ---
    ("pauses_01", "One... two... three... four... five... six... seven... eight... nine... ten."),
    ("pauses_02", "Ready, set, go! And, stop. Again. Ready, set, go! And, stop."),
]
# fmt: off
# Convenience: plain text list for backward compatibility with sweep script
TEXT_STRINGS: list[str] = [text for _, text in TEXTS]
TEXT_IDS: list[str] = [tid for tid, _ in TEXTS]
