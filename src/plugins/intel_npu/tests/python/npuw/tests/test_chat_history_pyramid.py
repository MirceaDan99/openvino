# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import gc
import os

import openvino_genai as ov_genai
import pytest

from utils import get_model_dir


default_chat_history_model = "TinyLlama-1.1B-Chat-v1.0_int4_sym_group128_dyn_stateful"

chat_history_prompts = [
    "explain me the concept of polymorphism. please come-up with a comprehensive answer, refer to the actual literature, and reflect the modern look at this problem. illustrate with three representative examples",
    "demonstrate it in Python",
    "now translate it to Java",
    "now lets conclude our session",
]

MAX_PROMPT_LEN = 8192
MAX_NEW_TOKENS = 128

chat_history_config = {
    "NPU_BYPASS_UMD_CACHING": "YES",
    "NPUW_DEVICES": "NPU",
    "MAX_PROMPT_LEN": MAX_PROMPT_LEN,
    "NPUW_LLM_ENABLE_PREFIX_CACHING": "YES",
    "NPUW_LLM_GENERATE_PYRAMID": "YES",
}

chat_generation_config = ov_genai.GenerationConfig()
chat_generation_config.max_new_tokens = MAX_NEW_TOKENS  # Otherwise it will exceed 'MAX'
chat_generation_config.max_length = MAX_PROMPT_LEN + MAX_NEW_TOKENS


def get_chat_history_model_name():
    return str(os.environ.get("NPUW_TEST_CHAT_HISTORY_MODEL", default_chat_history_model))


def get_model_path(model_name):
    model_path = get_model_dir(f"LLM/{model_name}")
    if not os.path.exists(model_path):
        pytest.skip(f"Model is not available in the package: {model_name}")
    return model_path


@pytest.fixture(scope="session")
def validate_environment():
    try:
        model_dir = get_model_dir("LLM")
        assert os.path.exists(model_dir), f"Model directory does not exist: {model_dir}"
    except Exception as exc:
        pytest.skip(f"Environment validation failed: {exc}")

    yield


def test_chat_history_pyramid_prefix_caching_smoke(validate_environment):
    model_path = get_model_path(get_chat_history_model_name())
    chat_history = ov_genai.ChatHistory()
    pipe = ov_genai.LLMPipeline(model_path, "NPU", **chat_history_config)

    responses = []
    try:
        for prompt in chat_history_prompts:
            chat_history.append({"role": "user", "content": prompt})
            result = pipe.generate(chat_history, chat_generation_config)

            answer = result.texts[0].strip()
            assert answer, f"Model returned an empty response for prompt: {prompt!r}"

            responses.append(answer)
            chat_history.append({"role": "assistant", "content": answer})
    finally:
        del pipe
        gc.collect()

    assert len(responses) == len(chat_history_prompts)
