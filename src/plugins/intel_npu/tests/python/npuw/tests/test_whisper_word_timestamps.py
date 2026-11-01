# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino_genai import WhisperPipeline
import os
import pytest

from utils import (
    SAMPLE_RATE,
    default_whisper_config,
    get_whisper_audio_path,
    get_whisper_model_root,
    read_wav_for_whisper,
)

# Word timestamps depend on the cross-attention shapes, which do vary across
# model sizes, so covering different model sizes and precisions here.
# large-v3 was tried too but dropped: it hit ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
# on CI agents (32 decoder layers vs tiny's 4 pushes host memory too far for
# this test to run alongside everything else in the same CI job).
whisper_word_timestamps_models = [
    "whisper-tiny_fp16_dyn_stateful",
    "whisper-base_int8_dyn_stateful",
]


def get_word_timestamps_model_names():
    if 'NPUW_TEST_WHISPER_MODELS' in os.environ:
        return str(os.environ['NPUW_TEST_WHISPER_MODELS']).split(',')
    return whisper_word_timestamps_models


def generate_whisper_word_timestamps(pipe, raw_speech):
    config = pipe.get_generation_config()
    config.language = "<|en|>"
    config.task = "transcribe"
    config.return_timestamps = True
    config.word_timestamps = True
    result = pipe.generate(raw_speech, config)

    assert len(result.words) > 0, "Whisper returned no word-level timestamps"
    return result


def describe_word(word):
    return f"{word.word!r} [{word.start_ts:.3f}, {word.end_ts:.3f}]"


def compare_texts_with_assert(expected, actual):
    assert len(expected.texts) == len(actual.texts), (
        f"Transcript segment count mismatch: CPU produced {len(expected.texts)}, "
        f"NPU produced {len(actual.texts)}"
    )

    for idx, (exp_text, act_text) in enumerate(zip(expected.texts, actual.texts)):
        assert exp_text == act_text, (
            f"Transcript mismatch at segment {idx}:\n  CPU: {exp_text!r}\n  NPU: {act_text!r}"
        )


def compare_word_timestamps_with_assert(expected_words, actual_words, ts_tolerance=0.08):
    assert len(expected_words) == len(actual_words), (
        f"Word count mismatch: CPU produced {len(expected_words)} words, "
        f"NPU produced {len(actual_words)}.\n"
        f"  CPU: {[w.word for w in expected_words]}\n"
        f"  NPU: {[w.word for w in actual_words]}"
    )

    for idx, (exp_word, act_word) in enumerate(zip(expected_words, actual_words)):
        assert exp_word.word == act_word.word, (
            f"Word mismatch at index {idx}: CPU {describe_word(exp_word)} "
            f"vs NPU {describe_word(act_word)}"
        )
        assert exp_word.start_ts - act_word.start_ts == pytest.approx(0.0, abs=ts_tolerance), (
            f"start_ts drift at index {idx} exceeds {ts_tolerance}s: "
            f"CPU {describe_word(exp_word)} vs NPU {describe_word(act_word)}"
        )
        assert exp_word.end_ts - act_word.end_ts == pytest.approx(0.0, abs=ts_tolerance), (
            f"end_ts drift at index {idx} exceeds {ts_tolerance}s: "
            f"CPU {describe_word(exp_word)} vs NPU {describe_word(act_word)}"
        )


def assert_word_timestamps_are_sane(words, audio_duration_s, device):
    """Absolute invariants any word timestamp must satisfy.

    The CPU-vs-NPU comparison above only proves the two devices agree - it would
    still pass if both regressed identically. These checks encode what a word
    timestamp actually means, independent of the other device.
    """
    previous_end = 0.0

    for idx, word in enumerate(words):
        where = f"{device} word {idx} {describe_word(word)}"

        assert word.start_ts <= word.end_ts, f"{where}: start_ts is after end_ts"
        assert word.start_ts >= 0.0, f"{where}: negative start_ts"
        assert word.end_ts <= audio_duration_s, (
            f"{where}: end_ts runs past the {audio_duration_s:.3f}s audio"
        )
        assert word.start_ts >= previous_end, (
            f"{where}: starts before the previous word ended ({previous_end:.3f}s) - "
            f"word timestamps should be monotonically non-decreasing"
        )

        previous_end = word.end_ts


@pytest.mark.parametrize("model_name", get_word_timestamps_model_names())
def test_whisper_word_timestamps_npu_vs_cpu(model_name):
    model_root = get_whisper_model_root()
    model_path = os.path.join(model_root, model_name)
    audio_path = get_whisper_audio_path()

    if not os.path.isdir(model_path):
        pytest.skip(f"Whisper model path doesn't exist: {model_path}")
    if not os.path.isfile(audio_path):
        pytest.skip(f"Whisper audio path doesn't exist: {audio_path}")

    raw_speech = read_wav_for_whisper(audio_path)
    audio_duration_s = len(raw_speech) / SAMPLE_RATE

    # word_timestamps must be passed at construction time: GenAI decides whether to
    # decompose cross-attention SDPA (needed to extract per-word timestamps) when the
    # pipeline is built, not when generate() is called.
    cpu_pipe = WhisperPipeline(model_path, "CPU", word_timestamps=True)
    expected = generate_whisper_word_timestamps(cpu_pipe, raw_speech)
    del cpu_pipe

    npu_pipe = WhisperPipeline(model_path, "NPU", word_timestamps=True, **default_whisper_config)
    actual = generate_whisper_word_timestamps(npu_pipe, raw_speech)
    del npu_pipe

    assert_word_timestamps_are_sane(expected.words, audio_duration_s, "CPU")
    assert_word_timestamps_are_sane(actual.words, audio_duration_s, "NPU")

    compare_texts_with_assert(expected, actual)
    compare_word_timestamps_with_assert(expected.words, actual.words)
