# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Parametrized audio-quality sweep for Kokoro TTS on NPUW.

Runs every text in the shared test corpus through OpenVINO GenAI
Text2SpeechPipeline on CPU and NPUW and validates:
    1. No NaN / Inf values in the output audio.
    2. MCD vs. OpenVINO GenAI CPU reference stays below threshold.

The test consumes the pre-exported Kokoro model package. Conversion is covered
by the model-converter/GenAI flow and is intentionally not repeated here.

Usage:
    # Run all sweep tests
    pytest test_audio_quality_sweep.py -v

    # Run only "paragraph" texts
    pytest test_audio_quality_sweep.py -v -k paragraph

    # Run without MCD (faster, no PyTorch reference needed)
    pytest test_audio_quality_sweep.py -v -m "not mcd"

    # Skip if no NPU available
    pytest test_audio_quality_sweep.py -v  # auto-skips via fixture
"""

import os
import multiprocessing
from multiprocessing.connection import Connection
import pytest
import numpy as np
import time
from pathlib import Path

from test_corpus import TEXTS, DEFAULT_VOICE, DEFAULT_LANGUAGE, get_mcd_threshold, TIMEOUT
from utils_audio import calculate_mcd
from utils import get_model_dir


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture(scope="module", autouse=True)
def check_dependencies():
    try:
        import openvino as ov
    except ImportError:
        pytest.skip("openvino not installed")

    core = ov.Core()
    if "NPU" not in core.available_devices:
        pytest.skip("NPU device not available")

    try:
        import openvino_genai
    except ImportError:
        pytest.skip("openvino_genai not installed")


@pytest.fixture(scope="module")
def kokoro_model_dir() -> Path:
    try:
        tts_model_dir = Path(get_model_dir("TTS"))
    except KeyError as e:
        pytest.skip(str(e))

    candidates = [
        tts_model_dir / "Kokoro-82M_fp16_dyn",
        tts_model_dir / "Kokoro-82M_fp32_dyn"
    ]
    for candidate in candidates:
        if (candidate / "openvino_model.xml").is_file() and (candidate / "voices" / f"{DEFAULT_VOICE}.bin").is_file():
            print(f'Model to test: {candidate}')
            return candidate

    pytest.skip(f"Kokoro OpenVINO GenAI export is not available: {candidates}")


def _load_speaker_embedding(pipe, model_dir: Path):
    import openvino as ov

    embedding_path = model_dir / "voices" / f"{DEFAULT_VOICE}.bin"
    data = np.fromfile(embedding_path, dtype=np.float32)
    if data.size == 0:
        raise RuntimeError(f"Speaker embedding file is empty: {embedding_path}")
    return ov.Tensor(data.reshape(pipe.get_speaker_embedding_shape()))


def _load_pipeline(model_dir: Path, device: str):
    """Load the Kokoro OpenVINO GenAI pipeline on the specified device.
    """
    import openvino_genai

    print(f"\nLoading OpenVINO GenAI Kokoro from {model_dir} on device {device}...")
    t0 = time.time()
    pipe = openvino_genai.Text2SpeechPipeline(str(model_dir), device)
    print(f"Loaded Kokoro on {device} in {time.time() - t0:.1f}s")
    embedding = _load_speaker_embedding(pipe, model_dir)

    return pipe, embedding


def _run_pipeline(pipeline, speaker_embedding, text: str) -> tuple[np.ndarray, int]:
    """Run text through OpenVINO GenAI Text2SpeechPipeline."""
    result = pipeline.generate(text, speaker_embedding, language=DEFAULT_LANGUAGE)
    if len(result.speeches) != 1:
        raise RuntimeError(f"Expected one audio chunk, got {len(result.speeches)}")
    audio = np.asarray(result.speeches[0].data).reshape(-1).astype(np.float32)
    return audio, int(result.output_sample_rate)


def _run_inference_worker(model_dir: Path, device: str, connection: Connection) -> None:
    pipeline, embedding = _load_pipeline(model_dir, device)
    try:
        while True:
            try:
                request = connection.recv()
            except EOFError:
                break
            if request is None:
                break
            text, output_path = request
            audio, sample_rate = _run_pipeline(pipeline, embedding, text)
            np.savez(output_path, audio=audio, sample_rate=sample_rate)
            connection.send(output_path)
    finally:
        connection.close()


class _Worker:
    def __init__(self, model_dir: Path, device: str):
        self.model_dir = model_dir
        self.device = device
        self.process = None
        self.connection = None

    def _start(self):
        context = multiprocessing.get_context("spawn")
        parent_connection, child_connection = context.Pipe()
        process = context.Process(
            target=_run_inference_worker,
            args=(self.model_dir, self.device, child_connection),
        )
        try:
            process.start()
        except Exception:
            parent_connection.close()
            raise
        finally:
            child_connection.close()
        self.connection = parent_connection
        self.process = process

    def close(self, graceful: bool = False):
        failed_to_terminate = False
        if self.process is not None:
            if graceful and self.process.is_alive() and self.connection is not None:
                try:
                    self.connection.send(None)
                except (BrokenPipeError, EOFError, OSError):
                    pass
                self.process.join(5)
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(5)
            failed_to_terminate = self.process.is_alive()
            if not failed_to_terminate:
                self.process = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if failed_to_terminate:
            raise RuntimeError(f"{self.device} inference child process did not exit after termination")

    def generate(self, text: str, output_path: Path) -> tuple[np.ndarray, int]:
        if self.process is None or not self.process.is_alive():
            self.close()
            self._start()

        process = self.process
        if process is None or self.connection is None:
            raise RuntimeError(f"{self.device} inference worker failed to start")
        timeout_seconds = int(os.environ.get("NPUW_KOKORO_INFERENCE_TIMEOUT_SEC", str(TIMEOUT)))
        failure = None
        try:
            self.connection.send((text, output_path))
            if not self.connection.poll(timeout_seconds):
                if process.is_alive():
                    failure = f"{self.device} inference child process did not complete within {timeout_seconds} seconds"
                else:
                    failure = f"{self.device} inference child process exited before responding"
            else:
                written_output_path = self.connection.recv()
                if written_output_path is None or written_output_path != output_path:
                    failure = f"{self.device} inference child process did not write the expected output"
        except (BrokenPipeError, EOFError, OSError) as error:
            failure = f"{self.device} inference child process failed: {error}"

        if failure is not None:
            self.close()
            pytest.fail(f"{failure} (exit code {process.exitcode})")

        if not output_path.is_file():
            self.close()
            pytest.fail(f"{self.device} inference child process completed without writing its output")

        with np.load(output_path) as output:
            return output["audio"], int(output["sample_rate"])


@pytest.fixture(scope="module")
def npu_worker(kokoro_model_dir):
    worker = _Worker(kokoro_model_dir, "NPU")
    yield worker
    worker.close(graceful=True)


@pytest.fixture(scope="module")
def cpu_worker(kokoro_model_dir):
    worker = _Worker(kokoro_model_dir, "CPU")
    yield worker
    worker.close(graceful=True)


# ============================================================================
# TESTS — parametrized over the shared corpus
# ============================================================================

@pytest.mark.parametrize("test_id, text", TEXTS, ids=[t[0] for t in TEXTS])
class TestAudioQualitySweep:
    """Parametrized audio quality tests over the full text corpus.

    Each test infers the text once on NPUW, checks for NaN/Inf first,
    then computes MCD against the OpenVINO GenAI CPU reference. This avoids
    reference inference when the output is already broken.
    """

    def test_audio_quality(self, npu_worker, cpu_worker, test_id, text, tmp_path, record_property):
        """Output audio must be free of NaN/Inf and MCD must be below threshold."""
        npuw_audio, npuw_sr = npu_worker.generate(text, tmp_path / f"{test_id}_npuw.npz")

        # --- Stage 1: NaN / Inf check ---
        n_nan = int(np.isnan(npuw_audio).sum())
        n_inf = int(np.isinf(npuw_audio).sum())
        n_samples = len(npuw_audio)
        print(f"  [{test_id}] samples={n_samples:,}  NaN={n_nan}  Inf={n_inf}")

        assert n_nan == 0, (
            f"[{test_id}] Audio contains {n_nan:,} NaN samples "
            f"out of {n_samples:,} ({n_nan / n_samples * 100:.1f}%)"
        )
        assert n_inf == 0, (
            f"[{test_id}] Audio contains {n_inf:,} Inf samples "
            f"out of {n_samples:,}"
        )

        # --- Stage 2: MCD vs. OpenVINO GenAI CPU reference ---
        ref_audio, ref_sr = cpu_worker.generate(text, tmp_path / f"{test_id}_cpu.npz")
        assert npuw_sr == ref_sr, f"Sample-rate mismatch: NPUW={npuw_sr}, CPU={ref_sr}"
        mcd_score = calculate_mcd(ref_audio, npuw_audio, sr=npuw_sr)
        threshold = get_mcd_threshold(test_id)

        record_property("mcd_score", f"{mcd_score:.2f}")
        record_property("mcd_threshold", f"{threshold:.1f}")
        print(f"  [{test_id}] MCD: {mcd_score:.2f} dB  (threshold: {threshold:.1f} dB)")

        if mcd_score >= threshold:
            import scipy.io.wavfile as wavfile
            npuw_path = tmp_path / f"{test_id}_npuw.wav"
            ref_path = tmp_path / f"{test_id}_ref.wav"
            wavfile.write(str(npuw_path), npuw_sr, npuw_audio)
            wavfile.write(str(ref_path), ref_sr, ref_audio)
            pytest.fail(
                f"[{test_id}] MCD {mcd_score:.2f} dB exceeds threshold {threshold:.1f} dB\n"
                f"  text:    \"{text[:80]}\"\n"
                f"  npuw:    {npuw_path}\n"
                f"  ref:     {ref_path}"
            )


# ============================================================================
# STANDALONE RUNNER
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
