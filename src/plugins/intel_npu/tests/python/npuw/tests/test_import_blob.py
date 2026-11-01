# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino_genai import LLMPipeline, WhisperPipeline
import os
import pytest
import time
import shutil
import stat
from threading import Thread
from utils import (
    default_whisper_config,
    get_model_dir,
    get_whisper_audio_path,
    get_whisper_model_root,
    read_wav_for_whisper,
    get_agent_ram_gb,
)

generate_hint_config = ["BEST_PERF", "FAST_COMPILE"]
blob_with_weights = [True, False]
attention_blob_with_weights = [False]
is_cache_mode = [True, False]
default_models_list = [
    "Llama-3.2-3B-Instruct_int4_sym_group-1_dyn_stateful",
    "Llama-3.2-3B-Instruct_int4_sym_group128_dyn_stateful",
    "Qwen2.5-1.5B-Instruct_int4_sym_group128_dyn_stateful",
    "TinyLlama-1.1B-Chat-v1.0_int4_sym_group128_dyn_stateful"
]
parallel_models_list = [
    "Qwen2.5-Coder-0.5B-Instruct_fp16_dyn_stateful",
    "Qwen3-0.6B_int4_sym_group128_dyn_stateful"
]
default_attention_model = "Phi-3.5-mini-instruct_int4_sym_group-1_dyn_stateful"
default_moe_model = "gpt-oss-20b_int4_sym_group-1_dyn_stateful"
attention_prefill_chunk_size = 32
attention_max_prompt_len = 256
attention_prompt = (
    "Explain key value cache reuse in transformer attention during autoregressive decoding. " * 6).strip()
attention_configs = [
    pytest.param({"NPUW_LLM_PREFILL_ATTENTION_HINT": "DYNAMIC",
                  "NPUW_DEVICES": "NPU,GPU",
                  "NPUW_LLM_PREFILL_CHUNK_SIZE": str(attention_prefill_chunk_size),
                  "MAX_PROMPT_LEN": attention_max_prompt_len}, id="dynamic"),
    pytest.param({"NPUW_LLM_PREFILL_ATTENTION_HINT": "PYRAMID",
                  "NPUW_LLM_GENERATE_PYRAMID": "YES",
                  "NPUW_LLM_PREFILL_CHUNK_SIZE": str(attention_prefill_chunk_size),
                  "MAX_PROMPT_LEN": attention_max_prompt_len}, id="pyramid"),
    pytest.param({"NPUW_LLM_PREFILL_ATTENTION_HINT": "HFA",
                  "NPUW_LLM_PREFILL_CHUNK_SIZE": str(attention_prefill_chunk_size),
                  "MAX_PROMPT_LEN": attention_max_prompt_len}, id="hfa"),
]
default_config = {"NPU_BYPASS_UMD_CACHING": "YES", "NPUW_DEVICES": "NPU", "MAX_PROMPT_LEN": 64}

default_whisper_models_list = [
    "whisper-tiny_fp16_dyn_stateful"
]

temp_path = ".\\tmp_blobs"


def remove_readonly(func, path, _):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def get_model_names():
    if 'NPUW_TEST_MODELS' in os.environ:
        return str(os.environ['NPUW_TEST_MODELS']).split(',')
    return default_models_list


def get_parallel_model_names():
    if 'NPUW_TEST_MODELS' in os.environ:
        return str(os.environ['NPUW_TEST_MODELS']).split(',')
    return parallel_models_list


def get_attention_model_name():
    return str(os.environ.get("NPUW_TEST_ATTN_MODEL", default_attention_model))


def get_moe_model_name():
    return str(os.environ.get("NPUW_TEST_MOE_MODEL", default_moe_model))


def is_platform_NPU3720():
    return 'NPUW_TEST_PLATFORM' in os.environ and \
        'NPU3720' in str(os.environ['NPUW_TEST_PLATFORM'])


def is_platform_NPU4000_or_older():
    return 'NPUW_TEST_PLATFORM' in os.environ and \
        any(abr in str(os.environ['NPUW_TEST_PLATFORM']) for abr in ('NPU3720', 'NPU4000'))


_agent_ram_gb = get_agent_ram_gb()
# Note: low RAM agents (< 30GB) fail the GPT-OSS import tests with "Tensor was not initialized" or import timeout.
# Other models may be affected as well, under investigation.
skip_gpt_oss_low_ram = pytest.mark.skipif(
    _agent_ram_gb is not None and _agent_ram_gb < 30,
    reason=f"GPT-OSS import blob tests need >= 30GB RAM (agent has {_agent_ram_gb}GB)",
)


def get_whisper_model_names():
    if 'NPUW_TEST_MODELS' in os.environ:
        return str(os.environ['NPUW_TEST_MODELS']).split(',')
    return default_whisper_models_list


def get_whisper_weights_size(model_path):
    weights_files = [
        os.path.join(model_path, file)
        for file in os.listdir(model_path)
        if file.endswith("model.bin")
    ]
    return sum(os.path.getsize(weights_file) for weights_file in weights_files)


def get_any_whisper_weights_path(model_path):
    for filename in ["openvino_encoder_model.bin", "openvino_decoder_model.bin", "openvino_model.bin"]:
        candidate = os.path.join(model_path, filename)
        if os.path.isfile(candidate):
            return candidate
    return None


def generate_whisper_text(pipe, raw_speech):
    config = pipe.get_generation_config()
    config.language = "<|en|>"
    config.task = "transcribe"
    config.return_timestamps = False
    result = pipe.generate(raw_speech, config)

    assert len(result.texts) > 0, "Whisper returned no texts"
    return result.texts[0]


@pytest.mark.parametrize("with_weights", blob_with_weights, ids=["with_weights", "weightless"])
@pytest.mark.parametrize("generate_hint", generate_hint_config)
@pytest.mark.parametrize("model_name", get_model_names())
def test_import_aot(with_weights, generate_hint, model_name):
    prompt = 'What is OpenVINO?'


@pytest.fixture(scope="session")
def validate_environment():
    try:
        model_dir = get_model_dir("LLM")
        assert os.path.exists(model_dir), f"Model directory does not exist: {model_dir}"
    except Exception as e:
        pytest.skip(f"Environment validation failed: {e}")

    yield


def ensure_temp_dir():
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)


def cleanup_temp_dir():
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path, onerror=remove_readonly)


def get_model_path(model_name):
    model_path = get_model_dir(f"LLM/{model_name}")
    if not os.path.exists(model_path):
        pytest.skip(f"Model is not available in the package: {model_name}")
    return model_path


def run_import_aot_roundtrip(model_name, with_weights, cfg_overrides=None, prompt='What is OpenVINO?', max_new_tokens=30):
    ensure_temp_dir()

    blob_path = os.path.join(temp_path, "compiled_model.blob")
    model_path = get_model_path(model_name)
    weights_path = os.path.join(model_path, "openvino_model.bin")
    weights_size = os.path.getsize(weights_path)

    cfg = {"EXPORT_BLOB": "YES", "BLOB_PATH": blob_path}
    cfg |= default_config
    if with_weights:
        cfg |= {"CACHE_MODE": "OPTIMIZE_SPEED"}
    if cfg_overrides:
        cfg |= cfg_overrides

    npu_pipe = LLMPipeline(model_path, "NPU", **cfg)
    ref_out = npu_pipe.generate(prompt, max_new_tokens=max_new_tokens)
    del npu_pipe

    import_cfg = {"BLOB_PATH": blob_path, "WEIGHTS_PATH": weights_path}
    if with_weights:
        import_cfg.pop("WEIGHTS_PATH")

    start_time = time.time()
    npu_pipe = LLMPipeline(model_path, "NPU", **import_cfg)
    elapsed_time = time.time() - start_time
    if with_weights:
        assert elapsed_time < 10, "Import time exceeded 10 seconds"
    else:
        assert elapsed_time < 60, "Import time exceeded 60 seconds"

    actual_out = npu_pipe.generate(prompt, max_new_tokens=max_new_tokens)
    del npu_pipe
    assert ref_out == actual_out

    blob_size = os.path.getsize(blob_path)
    if with_weights:
        assert blob_size < weights_size * 2, "Blob size is much bigger than weights size!"
    else:
        assert blob_size < weights_size, "Blob size is bigger than weights size!"


def run_import_cache_dir_roundtrip(model_name, with_weights, cfg_overrides=None, prompt='What is OpenVINO?', max_new_tokens=30, import_timeout=None):
    ensure_temp_dir()

    model_path = get_model_path(model_name)
    weights_size = os.path.getsize(os.path.join(model_path, "openvino_model.bin"))

    cfg = {"CACHE_DIR": str(temp_path)}
    cfg |= default_config
    if with_weights:
        cfg |= {"CACHE_MODE": "OPTIMIZE_SPEED"}
    if cfg_overrides:
        cfg |= cfg_overrides

    npu_pipe = LLMPipeline(model_path, "NPU", **cfg)
    ref_out = npu_pipe.generate(prompt, max_new_tokens=max_new_tokens)
    del npu_pipe

    blobs = [file for file in os.listdir(temp_path) if file.endswith(".blob")]
    assert len(blobs) > 0, "Couldn't cache the blob"

    start_time = time.time()
    npu_pipe = LLMPipeline(model_path, "NPU", **cfg)
    elapsed_time = time.time() - start_time
    if with_weights:
        limit = import_timeout if import_timeout is not None else 10
        assert elapsed_time < limit, f"Import time exceeded {limit} seconds"
    else:
        limit = import_timeout if import_timeout is not None else 60
        assert elapsed_time < limit, f"Import time exceeded {limit} seconds"

    actual_out = npu_pipe.generate(prompt, max_new_tokens=max_new_tokens)
    del npu_pipe
    assert ref_out == actual_out

    blobs = [file for file in os.listdir(temp_path) if file.endswith(".blob")]
    assert len(blobs) > 0, "Couldn't import the blob"

    blob_size = os.path.getsize(os.path.join(temp_path, blobs[0]))
    if with_weights:
        assert blob_size < weights_size * 2, "Blob size is much bigger than weights size!"
    else:
        assert blob_size < weights_size, "Blob size is bigger than weights size!"


@pytest.mark.parametrize("with_weights", blob_with_weights, ids=["with_weights", "weightless"])
@pytest.mark.parametrize("generate_hint", generate_hint_config)
@pytest.mark.parametrize("model_name", get_model_names())
def test_import_aot(with_weights, generate_hint, model_name, validate_environment):
    run_import_aot_roundtrip(model_name, with_weights, {"GENERATE_HINT": generate_hint})
    cleanup_temp_dir()


@pytest.mark.parametrize("with_weights", blob_with_weights, ids=["with_weights", "weightless"])
@pytest.mark.parametrize("generate_hint", generate_hint_config)
@pytest.mark.parametrize("model_name", get_model_names())
def test_import_cache_dir(with_weights, generate_hint, model_name, validate_environment):
    run_import_cache_dir_roundtrip(model_name, with_weights, {"GENERATE_HINT": generate_hint})
    cleanup_temp_dir()


@pytest.mark.parametrize("with_weights", attention_blob_with_weights, ids=["weightless"])
@pytest.mark.parametrize("attention_cfg", attention_configs)
def test_import_aot_attention_modes(with_weights, attention_cfg, validate_environment):
    run_import_aot_roundtrip(get_attention_model_name(),
                             with_weights,
                             attention_cfg,
                             prompt=attention_prompt,
                             max_new_tokens=16)
    cleanup_temp_dir()


@pytest.mark.parametrize("with_weights", attention_blob_with_weights, ids=["weightless"])
@pytest.mark.parametrize("attention_cfg", attention_configs)
def test_import_cache_dir_attention_modes(with_weights, attention_cfg, validate_environment):
    run_import_cache_dir_roundtrip(get_attention_model_name(),
                                   with_weights,
                                   attention_cfg,
                                   prompt=attention_prompt,
                                   max_new_tokens=16)
    cleanup_temp_dir()


@pytest.mark.skipif(is_platform_NPU4000_or_older(), reason="MoE export/import requires platform newer than NPU4000")
@skip_gpt_oss_low_ram
@pytest.mark.parametrize("with_weights", blob_with_weights, ids=["with_weights", "weightless"])
def test_import_aot_moe_gpt_oss(with_weights, validate_environment):
    cfg = {"GENERATE_HINT": "FAST_COMPILE", "MAX_PROMPT_LEN": 128}
    run_import_aot_roundtrip(get_moe_model_name(), with_weights, cfg, max_new_tokens=8)
    cleanup_temp_dir()


@pytest.mark.skipif(is_platform_NPU4000_or_older(), reason="MoE export/import requires platform newer than NPU4000")
@skip_gpt_oss_low_ram
@pytest.mark.parametrize("with_weights", blob_with_weights, ids=["with_weights", "weightless"])
def test_import_cache_dir_moe_gpt_oss(with_weights, validate_environment):
    cfg = {"GENERATE_HINT": "FAST_COMPILE", "MAX_PROMPT_LEN": 128}
    run_import_cache_dir_roundtrip(get_moe_model_name(), with_weights, cfg, max_new_tokens=8, import_timeout=30)
    cleanup_temp_dir()


@pytest.mark.skipif(is_platform_NPU3720(), reason="not enough memory on NPU3720")
@pytest.mark.parametrize("generate_hint", generate_hint_config)
@pytest.mark.parametrize("model_name", get_parallel_model_names())
def test_import_parallel(generate_hint, model_name, validate_environment):
    prompt = 'What is OpenVINO?'

    ensure_temp_dir()
    model_path = get_model_path(model_name)
    weights_path = os.path.join(model_path, "openvino_model.bin")

    blob_path1 = os.path.join(temp_path, "compiled_model1.blob")
    blob_path2 = os.path.join(temp_path, "compiled_model2.blob")
    blob_path3 = os.path.join(temp_path, "compiled_model3.blob")
    blob_path4 = os.path.join(temp_path, "compiled_model4.blob")

    ref_out1 = None
    ref_out2 = None
    ref_out3 = None
    ref_out4 = None

    # NB: Generate the blobs
    cfg1 = {"EXPORT_BLOB": "YES", "GENERATE_HINT": generate_hint, "BLOB_PATH": blob_path1}
    cfg1 |= default_config
    cfg1["MAX_PROMPT_LEN"] = 64
    cfg2 = {"EXPORT_BLOB": "YES", "GENERATE_HINT": generate_hint, "BLOB_PATH": blob_path2}
    cfg2 |= default_config
    cfg2["MAX_PROMPT_LEN"] = 128
    cfg3 = {"EXPORT_BLOB": "YES", "GENERATE_HINT": generate_hint, "BLOB_PATH": blob_path3}
    cfg3 |= default_config
    cfg3["MAX_PROMPT_LEN"] = 192
    cfg4 = {"EXPORT_BLOB": "YES", "GENERATE_HINT": generate_hint, "BLOB_PATH": blob_path4}
    cfg4 |= default_config
    cfg4["MAX_PROMPT_LEN"] = 256

    npu_pipe = LLMPipeline(model_path, "NPU", **cfg1)
    ref_out1 = npu_pipe.generate(prompt, max_new_tokens=30)
    del npu_pipe

    npu_pipe = LLMPipeline(model_path, "NPU", **cfg2)
    ref_out2 = npu_pipe.generate(prompt, max_new_tokens=30)
    del npu_pipe

    npu_pipe = LLMPipeline(model_path, "NPU", **cfg3)
    ref_out3 = npu_pipe.generate(prompt, max_new_tokens=30)
    del npu_pipe

    npu_pipe = LLMPipeline(model_path, "NPU", **cfg4)
    ref_out4 = npu_pipe.generate(prompt, max_new_tokens=30)
    del npu_pipe

    # Import blobs and check accuracy
    import_cfg = [{"BLOB_PATH": blob_path1, "WEIGHTS_PATH": weights_path},
                  {"BLOB_PATH": blob_path2, "WEIGHTS_PATH": weights_path},
                  {"BLOB_PATH": blob_path3, "WEIGHTS_PATH": weights_path},
                  {"BLOB_PATH": blob_path4, "WEIGHTS_PATH": weights_path}]

    npu_pipe_imported = [None, None, None, None]
    threads = []

    def import_f(model_path_, cfg_, i_):
        npu_pipe_imported[i_] = LLMPipeline(model_path_, "NPU", **cfg_)

    for i, cfg_ in enumerate(import_cfg):
        thread = Thread(target=import_f, args=(model_path, cfg_, i))
        threads.append(thread)
        threads[-1].start()

    for t in threads:
        t.join()

    actual_out1 = npu_pipe_imported[0].generate(prompt, max_new_tokens=30)
    assert ref_out1 == actual_out1
    actual_out2 = npu_pipe_imported[1].generate(prompt, max_new_tokens=30)
    assert ref_out2 == actual_out2
    actual_out3 = npu_pipe_imported[2].generate(prompt, max_new_tokens=30)
    assert ref_out3 == actual_out3
    actual_out4 = npu_pipe_imported[3].generate(prompt, max_new_tokens=30)
    assert ref_out4 == actual_out4
    del npu_pipe_imported

    # Clear temp folder (so test cases won't interfere with each other)
    cleanup_temp_dir()


# Note: Disable for now, as Whisper pipeline uses the same config file for encoder/decoder models but
# for encoder model there's no "BLOB_PATH" support. Need to revisit and separate properties setting
# for Whisper encoder/decoder models
@pytest.mark.skip(reason="Temporary")
@pytest.mark.parametrize("cache_mode", is_cache_mode, ids=["cache_mode_opt_size", "cache_mode_off"])
@pytest.mark.parametrize("model_name", get_whisper_model_names())
def test_import_aot_whisper(cache_mode, model_name):
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)

    model_root = get_whisper_model_root()
    model_path = os.path.join(model_root, model_name)
    audio_path = get_whisper_audio_path()
    blob_path = os.path.join(temp_path, "compiled_model.blob")

    if not os.path.isdir(model_path):
        pytest.skip(f"Whisper model path doesn't exist: {model_path}")
    if not os.path.isfile(audio_path):
        pytest.skip(f"Whisper audio path doesn't exist: {audio_path}")

    raw_speech = read_wav_for_whisper(audio_path)
    # weights_size = get_whisper_weights_size(model_path)

    cfg = {"EXPORT_BLOB": "YES", "BLOB_PATH": blob_path}
    cfg |= default_whisper_config
    if cache_mode:
        cfg |= {"CACHE_MODE": "OPTIMIZE_SIZE"}

    npu_pipe = WhisperPipeline(model_path, "NPU", **cfg)
    ref_out = generate_whisper_text(npu_pipe, raw_speech)
    del npu_pipe

    import_cfg = {"BLOB_PATH": blob_path}
    if not cache_mode:
        weights_path = get_any_whisper_weights_path(model_path)
        if weights_path is None:
            pytest.skip("No suitable Whisper weights file found for WEIGHTS_PATH")
        import_cfg["WEIGHTS_PATH"] = weights_path

    start_time = time.time()
    npu_pipe = WhisperPipeline(model_path, "NPU", **import_cfg)
    elapsed_time = time.time() - start_time
    if cache_mode:
        assert elapsed_time < 10, "Import time exceeded 10 seconds"
    else:
        assert elapsed_time < 60, "Import time exceeded 60 seconds"

    actual_out = generate_whisper_text(npu_pipe, raw_speech)
    del npu_pipe
    assert ref_out == actual_out

    assert os.path.isfile(blob_path), "Couldn't export the blob"
    # blob_size = os.path.getsize(blob_path)
    # if cache_mode:
    #     assert blob_size < weights_size * 2, "Blob size is much bigger than total weights size!"
    # else:
    #     assert blob_size < weights_size, "Blob size is bigger than total weights size!"

    cleanup_temp_dir()


# Note: now WS is disabled for Whisper models in NPUW, so need to update tests from "with_weights" to "weightless"
@pytest.mark.parametrize("cache_mode", is_cache_mode, ids=["cache_mode_opt_size", "cache_mode_off"])
@pytest.mark.parametrize("model_name", get_whisper_model_names())
def test_import_cache_dir_whisper(cache_mode, model_name):
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)

    model_root = get_whisper_model_root()
    model_path = os.path.join(model_root, model_name)
    audio_path = get_whisper_audio_path()

    if not os.path.isdir(model_path):
        pytest.skip(f"Whisper model path doesn't exist: {model_path}")
    if not os.path.isfile(audio_path):
        pytest.skip(f"Whisper audio path doesn't exist: {audio_path}")

    raw_speech = read_wav_for_whisper(audio_path)
    # weights_size = get_whisper_weights_size(model_path)

    cfg = {"CACHE_DIR": str(temp_path)}
    cfg |= default_whisper_config
    if cache_mode:
        cfg |= {"CACHE_MODE": "OPTIMIZE_SIZE"}

    npu_pipe = WhisperPipeline(model_path, "NPU", **cfg)
    ref_out = generate_whisper_text(npu_pipe, raw_speech)
    del npu_pipe

    blobs = [file for file in os.listdir(temp_path) if file.endswith(".blob")]
    assert len(blobs) > 0, "Couldn't cache the blob"

    start_time = time.time()
    npu_pipe = WhisperPipeline(model_path, "NPU", **cfg)
    elapsed_time = time.time() - start_time
    if cache_mode:
        assert elapsed_time < 10, "Import time exceeded 10 seconds"
    else:
        assert elapsed_time < 60, "Import time exceeded 60 seconds"

    actual_out = generate_whisper_text(npu_pipe, raw_speech)
    del npu_pipe
    assert ref_out == actual_out

    blobs = [file for file in os.listdir(temp_path) if file.endswith(".blob")]
    assert len(blobs) > 0, "Couldn't import the blob"

    # Note: Disable size check for now, as Whisper models contains weights (NPUW Weights sharing
    # is disabled, EISW-170416)
    # blob_size = os.path.getsize(blobs[0])
    # if cache_mode:
    #     assert blob_size < weights_size * 2, "Blob size is much bigger than total weights size!"
    # else:
    #     assert blob_size < weights_size, "Blob size is bigger than total weights size!"

    cleanup_temp_dir()
