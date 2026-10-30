# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os
import textwrap
from pathlib import Path

import openvino_genai as ov_genai
import pytest
import gc

from shared.utils_text import WwbTextSimilarity
from utils import get_model_dir


PROMPT = "What is OpenVINO?"
MAX_NEW_TOKENS = 256

# Path to the committed golden reference answers (generated once via --generate-references).
REFERENCES_FILE = Path(__file__).parent / "sliding_window_references.json"

# Collects per-test results for the summary table printed at the end of the session.
_session_results = []

# ─── Full test matrix (all packages × all models) ────────────────────────────────────────────────
# Each entry: (model_name, artifactory_subpath, window_size, is_vlm)
# window_size is passed as MAX_PROMPT_LEN to the NPU pipeline to force the
# sliding-window attention code path.
sliding_window_models = [
    # DeepSeek-R1-Distill-Qwen-1.5B — ov253rc1 only (SWA not present in newer packages, so only ov253rc1 is relevant)
    pytest.param(("DeepSeek-R1-Distill-Qwen-1.5B_int4_sym_group128_dyn_stateful", "ov253rc1/LLM", 4096, False), id="deepseek_qwen_1.5b_ov253rc1"),
    # Phi-3-mini across all packages
    pytest.param(("Phi-3-mini-4k-instruct_int4_sym_group-1_dyn_stateful", "ov253rc1/LLM", 2047, False), id="phi3_mini_ov253rc1"),
    pytest.param(("Phi-3-mini-4k-instruct_int4_sym_group-1_dyn_stateful", "ov254rc1/LLM", 2047, False), id="phi3_mini_ov254rc1"),
    pytest.param(("Phi-3-mini-4k-instruct_int4_sym_group-1_dyn_stateful", "ov260rc1/LLM", 2047, False), id="phi3_mini_ov260rc1"),
    pytest.param(("Phi-3-mini-4k-instruct_int4_sym_group-1_dyn_stateful", "ov261rc1/LLM", 2047, False), id="phi3_mini_ov261rc1"),
    pytest.param(("Phi-3-mini-4k-instruct_int4_sym_group-1_dyn_stateful", "ov262/LLM", 2047, False), id="phi3_mini_ov262"),
    # Google-Gemma-2-2B from ov254rc1 onwards (ov253rc1 skipped — known attention_mask shape bug,)
    pytest.param(("Google-Gemma-2-2B-it_int4_sym_group-1_dyn_stateful", "ov254rc1/LLM", 4096, False), id="gemma2_2b_ov254rc1"),
    pytest.param(("Google-Gemma-2-2B-it_int4_sym_group-1_dyn_stateful", "ov260rc1/LLM", 4096, False), id="gemma2_2b_ov260rc1"),
    pytest.param(("Google-Gemma-2-2B-it_int4_sym_group-1_dyn_stateful", "ov261rc1/LLM", 4096, False), id="gemma2_2b_ov261rc1"),
    pytest.param(("Google-Gemma-2-2B-it_int4_sym_group-1_dyn_stateful", "ov262/LLM", 4096, False), id="gemma2_2b_ov262"),
    # Google-Gemma-3-4B: VLM only from ov254rc1 onwards (not present in ov253rc1)
    pytest.param(("Google-Gemma-3-4B-it_int4_sym_group-1_dyn_stateful", "ov254rc1/VLM", 1024, True),  id="gemma3_4b_ov254rc1"),
    pytest.param(("Google-Gemma-3-4B-it_int4_sym_group-1_dyn_stateful", "ov260rc1/VLM", 1024, True),  id="gemma3_4b_ov260rc1"),
    pytest.param(("Google-Gemma-3-4B-it_int4_sym_group-1_dyn_stateful", "ov261rc1/VLM", 1024, True),  id="gemma3_4b_ov261rc1"),
    pytest.param(("Google-Gemma-3-4B-it_int4_sym_group-1_dyn_stateful", "ov262/VLM", 1024, True),  id="gemma3_4b_ov262"),
    # Google-Gemma-4-E4B: ov262 only (SWA framework support introduced in ov262)
    pytest.param(("Google-Gemma-4-E4B-it_int4_sym_group-1_dyn_stateful", "ov262/VLM", 512, True),  id="gemma4_e4b_ov262"),
    # Google-Gemma-4-E2B: available from ov262 only
    pytest.param(("Google-Gemma-4-E2B-it_int4_sym_group-1_dyn_stateful", "ov262/VLM", 512, True),  id="gemma4_e2b_ov262"),
]


@pytest.fixture(scope="session", autouse=True)
def check_references_file():
    """Fail the entire session immediately if the references file is missing,
    before any model downloads are attempted."""
    if not REFERENCES_FILE.exists():
        pytest.fail(
            f"Golden references file not found: {REFERENCES_FILE}."
        )


@pytest.fixture(scope="session", autouse=True)
def print_sliding_window_summary():
    yield
    if not _session_results:
        return

    SNIPPET_LEN = 120
    SEP = "=" * 120
    THIN = "-" * 120

    print("\n\n" + SEP)
    print(" SLIDING WINDOW ACCURACY — SESSION SUMMARY")
    print(f" Prompt: \"{PROMPT}\"  |  max_new_tokens: {MAX_NEW_TOKENS}")
    print(SEP)

    # Group results by model name for readability
    from collections import OrderedDict
    grouped = OrderedDict()
    for r in _session_results:
        grouped.setdefault(r["model"], []).append(r)

    for model, entries in grouped.items():
        print(f"\n  Model : {model}")
        print(f"  Type  : {'VLM' if entries[0]['is_vlm'] else 'LLM'}")
        print(THIN)
        print(f"  {'Package':<12} {'Window':>8} {'Similarity':>12} {'Status':>8}")
        print(THIN)
        for r in entries:
            status = "PASSED" if r["passed"] else "FAILED"
            print(f"  {r['package']:<12} {r['window_size']:>8} {r['similarity']:>12.2f} {status:>8}")
        print()
        for r in entries:
            status = "PASSED" if r["passed"] else "FAILED"
            ref_snippet = r["reference_output"][:SNIPPET_LEN].replace("\n", " ")
            npu_snippet = r["npu_output"][:SNIPPET_LEN].replace("\n", " ")
            print(f"  [{r['package']}] — {status}")
            print(f"    REF : {ref_snippet}...")
            print(f"    NPU : {npu_snippet}...")
        print(THIN)

    print()
    print(SEP)
    passed = sum(1 for r in _session_results if r["passed"])
    failed = len(_session_results) - passed
    print(f" TOTAL: {len(_session_results)}  |  PASSED: {passed}  |  FAILED: {failed}")
    if failed:
        print()
        print(" FAILED TESTS:")
        for r in _session_results:
            if not r["passed"]:
                print(f"   - {r['model']}  [{r['package']}]  similarity={r['similarity']:.2f}")
    print(SEP)


@pytest.fixture
def sliding_window_model(request):
    model_name, prefix, window_size, is_vlm = request.param

    model_path = get_model_dir(prefix) + model_name
    if not Path(model_path).exists() or not Path(model_path).is_dir():
        raise Exception(f"Could not find '{model_name}' at location {model_path}")

    yield str(model_path), window_size, is_vlm


@pytest.mark.parametrize("sliding_window_model", sliding_window_models, indirect=True)
def test_sliding_window_accuracy(request, sliding_window_model):
    references = json.loads(REFERENCES_FILE.read_text(encoding="utf-8"))

    model_path, window_size, is_vlm = sliding_window_model
    model_name = os.path.basename(model_path)

    if model_name not in references:
        pytest.fail(f"No golden reference found for '{model_name}' in {REFERENCES_FILE}.")

    reference_text = references[model_name]

    # Setting MAX_PROMPT_LEN equal to the model's sliding window size forces
    # NPUW into the sliding-window attention code path on NPU.
    npu_config = {
        "GENERATE_HINT": "BEST_PERF",
        "MAX_PROMPT_LEN": window_size,
        "NPU_COMPILER_TYPE": "DRIVER"
    }

    generation_config = ov_genai.GenerationConfig()
    generation_config.do_sample = False
    generation_config.apply_chat_template = True
    generation_config.max_new_tokens = MAX_NEW_TOKENS

    if is_vlm:
        vlm_config = {
            "config": {
                "DEVICE_PROPERTIES": {
                    "NPU": {
                        "GENERATE_HINT": "BEST_PERF",
                        "MAX_PROMPT_LEN": window_size,
                        "NPU_COMPILER_TYPE": "DRIVER"
                    }
                }
            }
        }
        npu_pipe = ov_genai.VLMPipeline(model_path, "NPU", **vlm_config)
        npu_result = npu_pipe.generate(PROMPT, generation_config=generation_config)
    else:
        npu_pipe = ov_genai.LLMPipeline(model_path, "NPU", **npu_config)
        npu_result = npu_pipe.generate([PROMPT], generation_config)

    del npu_pipe

    actual_text = npu_result.texts[0]

    similarity_check = WwbTextSimilarity()
    accuracy = similarity_check.evaluate(reference_text, actual_text)

    # Verbosing results:
    test_name = request.node.originalname
    print()
    print()
    print(f"{test_name}:")
    print(f"   Model:                        {model_path}")
    print(f"   Window size (MAX_PROMPT_LEN): {window_size}")
    print(f"   Prompt:                       \"{PROMPT}\"")
    print()
    reference_with_indent = textwrap.indent(f'"{reference_text}"', '   ')
    print(f"   Golden reference (CPU/ov262):\n{reference_with_indent}")
    print()
    actual_with_indent = textwrap.indent(f'"{actual_text}"', '   ')
    print(f"   NPU actual:\n{actual_with_indent}")
    print()
    print(f"   Similarity: {accuracy:.2f}")
    print()

    # Record result for session summary table.
    _, package, _, _ = request.node.callspec.params["sliding_window_model"]
    _session_results.append({
        "model":            model_name,
        "package":          package,
        "window_size":      window_size,
        "is_vlm":           is_vlm,
        "similarity":       accuracy,
        "passed":           accuracy >= 0.75,
        "reference_output": reference_text,
        "npu_output":       actual_text,
    })

    assert accuracy >= 0.75, (
        f"NPU output diverged from golden reference (similarity={accuracy:.2f}). "
        f"Sliding window may be broken for window_size={window_size}."
    )

    gc.collect()
