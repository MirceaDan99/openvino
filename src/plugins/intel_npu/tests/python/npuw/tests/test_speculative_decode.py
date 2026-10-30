# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino_genai as ov_genai
import openvino as ov

import os
from shared.utils_text import WwbTextSimilarity
from utils import get_model_dir
import textwrap

import pytest
import gc


def get_npu_llm_properties_for_test():
    config = {}
    config["GENERATE_HINT"] = "BEST_PERF"
    config["MAX_PROMPT_LEN"] = 1024
    return config


models = [("Phi-3-mini-4k-instruct_int4_sym_group-1_dyn_stateful", "Phi-3-mini-FastDraft-50M-int8-sym-ov"),
          ("Phi-3.5-mini-instruct_int4_sym_group-1_dyn_stateful", "Phi-3-mini-FastDraft-50M-int8-sym-ov")]
prompts_and_max_new_tokens = \
    [pytest.param("When a diver points a flashlight upward toward the surface of the water at an angle 20° from the normal, the beam of light",
                  128,
                  id='prompt=diver,max_tokens=128'),
     pytest.param("Best vacation places in Ireland?", 512,
                  id='prompt=vacation_places,max_tokens=512')]

# FIXME: Main model on CPU in Speculative setting can differ in output: CVS-172338
# FIXME: Validation time is now very long, adding (main_device=CPU, draft_device=NPU)
#        and (main_device=NPU, draft_device=CPU) cases will make PR's validation a bottleneck.


@pytest.mark.parametrize("main_model,draft_model", models)
@pytest.mark.parametrize("prompt, max_new_tokens", prompts_and_max_new_tokens)
def test_acc_perf(request, main_model, draft_model, prompt, max_new_tokens):
    platform = os.environ["NPUW_TEST_PLATFORM"]
    if "NPU3720" in platform:
        pytest.skip("Speculative Decoding tests are disabled on NPU3720 due to insufficient"
                    " memory issues.")
    ov_version = ov.get_version()
    ov_version_parts = ov_version.split(".")
    ov_version_year, ov_version_major = int(ov_version_parts[0]), int(ov_version_parts[1])
    # Speculative Decode is not available in older OpenVINO versions:
    if ov_version_year < 2025 or (ov_version_year == 2025 and ov_version_major < 4):
        pytest.skip("OpenVINO version is too old, Speculative Decode is supported only starting from"
                    " OpenVINO 2025.4.0")

    main_model_path = get_model_dir(f"LLM/{main_model}")
    draft_model_path = get_model_dir(f"LLM/{draft_model}")

    # Create reference LLM pipeline for main model on NPU:
    main_config = get_npu_llm_properties_for_test()
    ov_llm_npu_pipe = ov_genai.LLMPipeline(main_model_path, "NPU", **main_config)

    # Create actual Stateful Speculative LLM pipeline for main and draft models on NPU:
    draft_config = get_npu_llm_properties_for_test()

    # W/A for OpenVINO 2025.4, draft kvcache trim is broken here
    if "2025.4.0" in ov_version:
        draft_config["NPUW_LLM_MAX_GENERATION_TOKEN_LEN"] = 2
    ov_draft_model = ov_genai.draft_model(draft_model_path, "NPU", **draft_config)
    ov_speculative_npu_pipe = ov_genai.LLMPipeline(main_model_path, "NPU", **main_config, draft_model=ov_draft_model)

    # Run Stateful LLM pipeline:
    generation_config = ov_genai.GenerationConfig()
    generation_config.do_sample = False
    generation_config.apply_chat_template = True
    generation_config.max_new_tokens = max_new_tokens
    reference = ov_llm_npu_pipe.generate([prompt], generation_config)

    # Run Stateful Speculative LLM pipeline:
    actual = ov_speculative_npu_pipe.generate([prompt], generation_config)

    del ov_llm_npu_pipe, ov_speculative_npu_pipe

    # Accuracy metrics collection:
    reference_text = reference.texts[0]
    actual_text = actual.texts[0]
    similarity_check = WwbTextSimilarity()
    accuracy = similarity_check.evaluate(reference_text, actual_text)

    # Performance metric collection:
    reference_tput = reference.perf_metrics.get_throughput().mean
    actual_tput = actual.perf_metrics.get_throughput().mean
    reference_gen_duration = reference.perf_metrics.get_generate_duration().mean
    actual_gen_duration = actual.perf_metrics.get_generate_duration().mean
    reference_num_generated_tokens = reference.perf_metrics.get_num_generated_tokens()
    actual_num_generated_tokens = actual.perf_metrics.get_num_generated_tokens()
    if (actual_num_generated_tokens != reference_num_generated_tokens):
        print(f"WARNING: Number of generated tokens differ between reference and actual runs")
        print(f"Num generated tokens by reference pipeline = {reference_num_generated_tokens}")
        print(f"Num generated tokens by actual pipeline = {actual_num_generated_tokens}")
        min_generated_tokens = min(actual_num_generated_tokens, reference_num_generated_tokens)
        print(f"Minimal number of generated tokens for fair comparison: {min_generated_tokens}")
        print(
            f"Sum of durations for last actual_num_generated_tokens - min_generated_tokens = {sum(float(x) / 1000.0 for x in actual.perf_metrics.raw_metrics.m_durations[:-(actual_num_generated_tokens - min_generated_tokens)])}")
        print(
            f"Sum of durations for last reference_num_generated_tokens - min_generated_tokens = {sum(float(x) / 1000.0 for x in reference.perf_metrics.raw_metrics.m_durations[:-(reference_num_generated_tokens - min_generated_tokens)])}")
        print(f"Actual generation duration before correction: {actual_gen_duration} ms")
        actual_gen_duration -= sum(float(x) / 1000.0 for x in actual.perf_metrics.raw_metrics.m_durations[:-(
            actual_num_generated_tokens - min_generated_tokens)])
        print(f"Actual generation duration after correction: {actual_gen_duration} ms")
        print(f"Reference generation duration before correction: {reference_gen_duration} ms")
        reference_gen_duration -= sum(float(x) / 1000.0 for x in reference.perf_metrics.raw_metrics.m_durations[:-(
            reference_num_generated_tokens - min_generated_tokens)])
        print(f"Reference generation duration after correction: {reference_gen_duration} ms")
    gen_duration_decrease = (1 - actual_gen_duration / reference_gen_duration) * 100.0

    # Verbosing results:
    test_name = request.node.originalname
    num_generated_tokens = actual.perf_metrics.get_num_generated_tokens()
    num_accepted_tokens = "N/A"
    if actual.extended_perf_metrics:
        num_accepted_tokens = actual.extended_perf_metrics.get_num_accepted_tokens()
    print()
    print()
    print(f"{test_name}:")
    print(f"   Main model: {main_model}")
    print(f"   Draft model: {draft_model}")
    print(f"   Prompt:  \"{prompt}\"")
    print()
    reference_with_indent = textwrap.indent(f'"{reference_text}"', '   ')
    print(f"   Reference :\n{reference_with_indent}")
    print()
    actual_with_indent = textwrap.indent(f'"{actual_text}"', '   ')
    print(f"   Actual :\n{actual_with_indent}")
    print()
    print()
    print(f"   Similarity check: {accuracy:.2f}")
    print()
    print()
    print(f"   Stateful LLM pipeline on NPU generation time:                  {reference_gen_duration:.2f} ms")
    print(f"   Stateful LLM pipeline on NPU throughput:                       {reference_tput:.2f} token/s")
    print(f"   Stateful LLM pipeline on NPU num generated tokens:             {reference_num_generated_tokens} tokens")
    print(f"   Stateful Speculative LLM pipeline on NPU generation time:      {actual_gen_duration:.2f} ms")
    print(f"   Stateful Speculative LLM pipeline on NPU throughput:           {actual_tput:.2f} token/s")
    print(f"   Stateful Speculative LLM pipeline on NPU num generated tokens: {actual_num_generated_tokens} tokens")
    print(f"   Speculative Decode generate duration descrease (normalized):   {gen_duration_decrease:.2f} %")
    print()
    print()
    print(f"   Accepted tokens in Speculative Decode: {num_accepted_tokens}/{num_generated_tokens} tokens")

    # Actual validation:
    assert accuracy >= 0.8
    assert reference_tput < actual_tput, "Speculative Decoding should boost the pipeline throughput!"
    assert gen_duration_decrease >= 25, "Speculative Decoding should decrease generation time on >=25%!"

    gc.collect()
