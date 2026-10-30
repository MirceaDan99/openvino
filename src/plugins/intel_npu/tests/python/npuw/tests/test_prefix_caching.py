# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino_genai as ov_genai
from openvino import get_version
import hashlib
import os
import gc
import pytest
import time
import tempfile
import shutil
from typing import Dict, List, Tuple
from pathlib import Path
from utils import get_model_dir

from conftest import NonCriticalError

# Default test configuration
default_models_list = [
    "Llama-3.2-3B-Instruct_int4_sym_group-1_dyn_stateful",
    "Llama-3.2-3B-Instruct_int4_sym_group128_dyn_stateful",
    "Qwen2.5-1.5B-Instruct_int4_sym_group128_dyn_stateful",
    "TinyLlama-1.1B-Chat-v1.0_int4_sym_group128_dyn_stateful"
]

default_config = {
    "NPU_BYPASS_UMD_CACHING": "YES",
    "NPUW_DEVICES": "NPU",
    "MAX_PROMPT_LEN": 2048,
    "NPUW_LLM_PREFILL_CHUNK_SIZE": "256"
}


def non_critical_assert(condition, message):
    if not condition:
        raise NonCriticalError(message)


def get_model_names():
    """Get test model names from environment or use defaults."""
    if 'NPUW_TEST_MODELS' in os.environ:
        return str(os.environ['NPUW_TEST_MODELS']).split(',')
    return default_models_list


def get_test_prompts():
    """Get test prompts from the prompts directory."""
    import json

    current_dir = Path(__file__).parent
    prompts_dir = current_dir / "prefix_caching" / "prompts"

    if not prompts_dir.exists():
        raise FileNotFoundError(f"Test prompts directory not found: {prompts_dir}")

    prompt_files = list(prompts_dir.glob("*.jsonl"))
    if not prompt_files:
        raise FileNotFoundError(f"No prompt files found in: {prompts_dir}")

    prompts = []
    for prompt_file in sorted(prompt_files):
        with open(prompt_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        if 'prompt' in data:
                            content = data['prompt'].strip()
                            if content:
                                prompts.append([content])
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse JSON in {prompt_file.name}: {e}")
                        continue

    if not prompts:
        raise RuntimeError("No valid prompts found in the prompts directory!")

    return prompts


def create_ov_config(device: str, enable_prefix_caching: bool) -> Dict[str, str]:
    """Create OpenVINO configuration with or without prefix caching."""
    ov_config = default_config.copy()
    if device == 'NPU':
        ov_config["NPUW_LLM_ENABLE_PREFIX_CACHING"] = "YES" if enable_prefix_caching else "NO"
    return ov_config


def run_benchmark(models_path: str, device: str, prompts: List[List[str]],
                  config: ov_genai.GenerationConfig, enable_prefix_caching: bool) -> Dict[int, Dict[str, any]]:
    """Run benchmark with given configuration and return results."""
    cache_status = "ENABLED" if enable_prefix_caching else "DISABLED"
    print(f"\nRunning benchmark with prefix caching {cache_status}")

    ov_config = create_ov_config(device, enable_prefix_caching)

    start_time = time.time()
    pipe = ov_genai.LLMPipeline(models_path, device, **ov_config)
    load_time = time.time() - start_time

    results = {}

    try:
        for i, prompt in enumerate(prompts, start=1):
            print(f"Generating for prompt {i}")
            res = pipe.generate(prompt, config)

            output_text = res.texts[0]
            results_md5 = hashlib.new("md5", output_text.encode(), usedforsecurity=False).hexdigest()

            perf_metrics = res.perf_metrics

            results[i] = {
                'text': output_text,
                'md5': results_md5,
                'output_tokens': perf_metrics.get_num_generated_tokens(),
                'load_time': load_time * 1000,  # Convert to ms for consistency
                'generate_time_mean': perf_metrics.get_generate_duration().mean,
                'generate_time_std': perf_metrics.get_generate_duration().std,
                'tokenization_time_mean': perf_metrics.get_tokenization_duration().mean,
                'tokenization_time_std': perf_metrics.get_tokenization_duration().std,
                'detokenization_time_mean': perf_metrics.get_detokenization_duration().mean,
                'detokenization_time_std': perf_metrics.get_detokenization_duration().std,
                'ttft_mean': perf_metrics.get_ttft().mean,
                'ttft_std': perf_metrics.get_ttft().std,
                'tpot_mean': perf_metrics.get_tpot().mean,
                'tpot_std': perf_metrics.get_tpot().std,
                'throughput_mean': perf_metrics.get_throughput().mean,
                'throughput_std': perf_metrics.get_throughput().std,
            }

            print(f"  Generated Text: {output_text}")
            print(f"  MD5: {results_md5}")
            print(f"  Tokens: {results[i]['output_tokens']}")
            print(f"  TTFT: {results[i]['ttft_mean']:.2f} ms")
            print(f"  TPOT: {results[i]['tpot_mean']:.2f} ms")
            print(f"  Throughput: {results[i]['throughput_mean']:.2f} tokens/s")

            # Clear intermediate results to free memory
            del res
            del perf_metrics

    finally:
        # Explicitly delete the pipeline object and force garbage collection
        del pipe
        gc.collect()
        print(f"Memory cleanup completed for {cache_status} benchmark")

    return results


def validate_prefix_caching_results(results_with_cache: Dict[int, Dict[str, any]],
                                    results_without_cache: Dict[int, Dict[str, any]]) -> Tuple[bool, Dict[str, List[float]]]:
    """
    Validate prefix caching results and return accuracy status and performance metrics.

    Returns:
        Tuple of (accuracy_passed, performance_summary)
        accuracy_passed: True if all MD5 hashes match
        performance_summary: Dictionary with per-round performance improvements
    """
    print(f"\nValidating prefix caching results...")

    accuracy_passed = True
    performance_metrics = {
        'ttft_improvement': [],
        'tpot_improvement': [],
        'throughput_improvement': [],
        'generate_time_improvement': []
    }

    for prompt_idx in results_with_cache.keys():
        # MD5 comparison for accuracy
        md5_with = results_with_cache[prompt_idx]['md5']
        md5_without = results_without_cache[prompt_idx]['md5']
        md5_match = md5_with == md5_without

        if not md5_match:
            accuracy_passed = False
            print(f"  ❌ Prompt {prompt_idx}: MD5 mismatch")
            print(f"     With cache:    {md5_with}")
            print(f"     Without cache: {md5_without}")
        else:
            print(f"  ✅ Prompt {prompt_idx}: MD5 match ({md5_with[:8]}...)")

        # Performance comparison
        with_cache = results_with_cache[prompt_idx]
        without_cache = results_without_cache[prompt_idx]

        # Calculate improvements (positive = better performance with caching)
        ttft_improvement = ((without_cache['ttft_mean'] - with_cache['ttft_mean']) / without_cache['ttft_mean']) * 100
        tpot_improvement = ((without_cache['tpot_mean'] - with_cache['tpot_mean']) / without_cache['tpot_mean']) * 100
        throughput_improvement = (
            (with_cache['throughput_mean'] - without_cache['throughput_mean']) / without_cache['throughput_mean']) * 100
        generate_time_improvement = (
            (without_cache['generate_time_mean'] - with_cache['generate_time_mean']) / without_cache['generate_time_mean']) * 100

        # Record each round's improvement
        performance_metrics['ttft_improvement'].append(ttft_improvement)
        performance_metrics['tpot_improvement'].append(tpot_improvement)
        performance_metrics['throughput_improvement'].append(throughput_improvement)
        performance_metrics['generate_time_improvement'].append(generate_time_improvement)

        print(f"     TTFT: {ttft_improvement:+5.1f}% | TPOT: {tpot_improvement:+5.1f}% | "
              f"Throughput: {throughput_improvement:+5.1f}% | GenTime: {generate_time_improvement:+5.1f}%")

    return accuracy_passed, performance_metrics


# Pytest test configurations
max_new_tokens_list = [128]


@pytest.fixture(scope="function")
def cleanup_memory():
    """Fixture to ensure proper memory cleanup after each test."""
    yield
    # Force garbage collection after each test
    gc.collect()
    print("\nMemory cleanup completed")


@pytest.fixture(scope="session")
def validate_environment():
    """Validate test environment is properly set up."""
    try:
        # Check if at least one test model exists
        model_names = get_model_names()
        found_models = []
        for model_name in model_names:
            model_path = get_model_dir(f"LLM/{model_name}")
            if os.path.exists(model_path):
                found_models.append(model_name)

        assert len(found_models) > 0, f"No test models found in {model_dir}. Available models: {model_names}"
        print(f"Environment validated. Found {len(found_models)} test models.")

    except Exception as e:
        pytest.skip(f"Environment validation failed: {e}")

    yield


@pytest.mark.parametrize("max_new_tokens", max_new_tokens_list)
@pytest.mark.parametrize("model_name", get_model_names())
def test_prefix_caching(max_new_tokens, model_name, validate_environment, cleanup_memory):
    """Test prefix caching functionality - validates accuracy and performance."""
    device = "NPU"

    print(f"\n{'=' * 60}")
    print(f"Testing prefix caching for {model_name}")
    print(f"Max tokens: {max_new_tokens}, Device: {device}")
    print(f"OpenVINO version: {get_version()}")
    print(f"{'=' * 60}")

    model_path = get_model_dir(f"LLM/{model_name}")

    # Verify model exists
    assert os.path.exists(model_path), f"Model path does not exist: {model_path}"

    # Get test prompts
    prompts = get_test_prompts()
    print(f"Testing with {len(prompts)} prompts")

    # Create generation config
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = max_new_tokens

    # Run benchmark without prefix caching
    results_without_cache = run_benchmark(
        model_path, device, prompts, config,
        enable_prefix_caching=False
    )

    # Run benchmark with prefix caching enabled
    results_with_cache = run_benchmark(
        model_path, device, prompts, config,
        enable_prefix_caching=True
    )

    # Validate results
    accuracy_passed, performance_metrics = validate_prefix_caching_results(
        results_with_cache, results_without_cache
    )

    # Assert accuracy - outputs must be identical
    assert accuracy_passed, "Prefix caching produced different outputs - accuracy test failed"

    # Performance assertions (relaxed thresholds for CI stability)
    print(f"\nPerformance validation:")

    # Get per-round improvements
    ttft_improvements = performance_metrics['ttft_improvement']
    throughput_improvements = performance_metrics['throughput_improvement']

    # Print detailed per-round results
    print(f"  TTFT improvements per round: {[f'{x:+5.1f}%' for x in ttft_improvements]}")
    print(f"  Throughput improvements per round: {[f'{x:+5.1f}%' for x in throughput_improvements]}")

    # Use last round's improvement for pass/fail decision
    last_ttft_improvement = ttft_improvements[-1]
    last_throughput_improvement = throughput_improvements[-1]

    print(f"  Last round TTFT improvement: {last_ttft_improvement:+5.1f}%")
    print(f"  Last round throughput improvement: {last_throughput_improvement:+5.1f}%")

    # Performance check is a non-functional check.
    # Any performance deviations should be marked as warnings but not errors.

    # TTFT should generally improve (allow small degradations due to test noise)
    non_critical_assert(last_ttft_improvement > -10.0,
                        f"Last round TTFT degraded too much: {last_ttft_improvement:.1f}%")

    # Throughput should generally improve or stay similar
    non_critical_assert(last_throughput_improvement > -
                        5.0, f"Last round throughput degraded too much: {last_throughput_improvement:.1f}%")

    print(f"\nPrefix caching test passed!")
    print(f"  ✅ Accuracy: All outputs match")
    print(
        f"  ✅ Performance: Last round TTFT {last_ttft_improvement:+5.1f}%, Throughput {last_throughput_improvement:+5.1f}%")


if __name__ == "__main__":
    # Allow running as standalone script for debugging
    import sys
    if len(sys.argv) == 1:
        # Run main test if no args provided
        test_prefix_caching(128, get_model_names()[0], validate_environment, cleanup_memory)
    else:
        print("Use pytest to run the full test suite:")
        print("pytest test_prefix_caching.py -v")
        print("pytest test_prefix_caching.py::test_prefix_caching -v")
