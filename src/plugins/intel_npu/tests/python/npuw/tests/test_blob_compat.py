# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Phase-0 blob compatibility (EISW-214507): a blob exported by one OpenVINO
# version must still import and produce the same result on a newer one.
#
# Nothing is uploaded or persisted anywhere: the older-version blob is
# produced fresh every run, in a throwaway venv, and discarded afterward.

import io
import os
import subprocess
import sys
import venv

import numpy as np
import openvino as ov
from pathlib import Path
import pytest

from utils import get_model_dir, preprocess_resnet_image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from gen_blob_compat_reference import generate

DEVICE = "NPU"
OLD_OV_VERSIONS = ["2026.2", "2026.3", "2026.4"]

# Predefined input picture: reused from the existing VLM test assets rather
# than hosting a separate one in Artifactory.
PICTURE = os.path.join(os.path.dirname(__file__), "..", "..",
                       "samples", "images", "ekaterina.png")


def _venv_python(venv_dir):
    bin_dir = "Scripts" if sys.platform == "win32" else "bin"
    exe = "python.exe" if sys.platform == "win32" else "python"
    return os.path.join(venv_dir, bin_dir, exe)


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    """(ov.Model, model_xml, weights_path, input_tensor) for the hardcoded ResNet-50."""
    out = Path(get_model_dir("resnet-50"))

    model_xml = next(out.rglob("*.xml"))
    ov_model = ov.Core().read_model(str(model_xml))
    model_bin = str(model_xml.with_suffix(".bin"))
    return ov_model, str(model_xml), model_bin, preprocess_resnet_image(PICTURE)


@pytest.fixture(scope="module", params=OLD_OV_VERSIONS)
def old_version_output(request, model, tmp_path_factory):
    """(base_version, {"full"/"weightless": (blob_path, reference_path)}),
    produced by compiling the model under base_version in a throwaway venv.

    Parametrized so each base version in OLD_OV_VERSIONS gets its own venv
    and its own entry in the test IDs, per Dmitry's review: the baseline
    OpenVINO version must be an explicit test parameter, not a hardcoded
    single value.
    """
    base_version = request.param
    _, model_xml, _, input_tensor = model
    work = tmp_path_factory.mktemp(f"old_version_{base_version}")

    input_npy = work / "input.npy"
    np.save(input_npy, input_tensor)

    venv_dir = work / "venv"
    venv.create(venv_dir, with_pip=True)
    python = _venv_python(venv_dir)
    subprocess.check_call([python, "-m", "pip", "install", "-q",
                           f"openvino=={base_version}", "numpy"])

    out_dir = work / "out"
    generator = os.path.join(os.path.dirname(__file__), "..",
                             "tools", "gen_blob_compat_reference.py")
    subprocess.check_call([python, generator,
                           "--model", str(model_xml),
                           "--input", str(input_npy),
                           "--out", str(out_dir)])

    return base_version, {
        flavor: (str(out_dir / flavor / "model.blob"), str(out_dir / flavor / "reference.bin"))
        for flavor in ("full", "weightless")
    }


def test_blob_compat_export_is_allowed(model, tmp_path):
    """NPUW_ENSURE_COMPATIBILITY must let a phase-0 clean model export, and
    each blob must self-import bit-exactly under the current OpenVINO
    version -- verified internally by generate() itself, for both flavors."""
    _, model_xml, _, input_tensor = model
    input_npy = tmp_path / "input.npy"
    np.save(input_npy, input_tensor)
    generate(model_xml, str(input_npy), str(tmp_path / "current"))


@pytest.mark.parametrize("weightless", [False, True], ids=["full", "weightless"])
def test_blob_compat_import_older_version(model, old_version_output, weightless):
    """A blob compiled under an old, pinned OpenVINO version (one of
    OLD_OV_VERSIONS) must still import and produce the same result under
    whatever OpenVINO version is running now."""
    _, _, weights_path, input_tensor = model
    base_version, outputs = old_version_output
    flavor = "weightless" if weightless else "full"
    blob_path, ref_path = outputs[flavor]

    with open(blob_path, "rb") as f:
        blob = f.read()

    import_cfg = {}
    if weightless:
        import_cfg["WEIGHTS_PATH"] = weights_path

    core = ov.Core()
    imported = core.import_model(io.BytesIO(blob), DEVICE, import_cfg)
    actual = imported([input_tensor])

    ref = np.fromfile(ref_path, dtype=np.float32).reshape(
        imported.outputs[0].get_shape())
    act = actual[imported.outputs[0]]
    l2 = np.linalg.norm(ref - act)
    print(f"{base_version} -> {ov.get_version()}, L2 diff vs reference: {l2}")
    assert np.array_equal(ref, act), (
        f"output from the {base_version} blob differs under {ov.get_version()}, L2={l2}"
    )
