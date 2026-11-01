#!/usr/bin/env python

#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

# EISW-214507 blob compatibility: given a model and an input tensor, compile
# it, export the blob, and save the reference output -- for both full and
# weightless blobs.  Takes everything as input and touches no network: the
# caller (a pinned-OpenVINO-version venv, or tests/python_tests/test_blob_compat.py)
# owns getting the model/input and doing anything with the result.

import argparse
import io
import os

import numpy as np
import openvino as ov

DEVICE = "NPU"


def generate(model_xml, input_npy, out_dir):
    """Compile model_xml on input_npy, writing {full,weightless}/model.blob
    and reference.bin under out_dir.

    Self-imports each blob immediately after exporting it, in the same
    process/OpenVINO version, to prove it is sound at its own source version
    before anything else (e.g. a different OpenVINO version) tries to import
    it -- isolates a bad blob from a cross-version import failure.
    """
    core = ov.Core()
    model = core.read_model(model_xml)
    inputs = [np.load(input_npy)]
    weights_path = os.path.splitext(model_xml)[0] + ".bin"

    for weightless in (False, True):
        flavor = "weightless" if weightless else "full"
        cfg = {
            "NPU_USE_NPUW": "YES",
            "NPUW_FUNCALL_FOR_ALL": "YES",
            "NPUW_CWAI": "YES",
            "NPUW_ENSURE_COMPATIBILITY": "YES",
            "NPUW_DEVICES": "NPU",
            "CACHE_MODE": "OPTIMIZE_SIZE" if weightless else "OPTIMIZE_SPEED",
        }
        compiled = core.compile_model(model, DEVICE, cfg)
        reference = compiled(inputs)

        blob = compiled.export_model()
        blob = blob.getvalue() if hasattr(blob, "getvalue") else bytes(blob)

        assert b"NPUWORC\0" in blob, (
            f"{ov.get_version()} produced a non-ORC blob; it predates the "
            f"compatibility format and cannot be a valid baseline"
        )

        import_cfg = {}
        if weightless:
            import_cfg["WEIGHTS_PATH"] = weights_path
        imported = core.import_model(io.BytesIO(blob), DEVICE, import_cfg)
        actual = imported(inputs)
        ref = reference[compiled.outputs[0]]
        act = actual[imported.outputs[0]]
        if not np.array_equal(ref, act):
            raise RuntimeError(
                f"{flavor}: self-import mismatch at generation time, "
                f"L2={np.linalg.norm(ref - act)}"
            )
        print(f"{flavor}: self-check OK")

        entry_dir = os.path.join(out_dir, flavor)
        os.makedirs(entry_dir, exist_ok=True)
        with open(os.path.join(entry_dir, "model.blob"), "wb") as f:
            f.write(blob)
        reference[compiled.outputs[0]].tofile(os.path.join(entry_dir, "reference.bin"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to the model .xml")
    parser.add_argument("--input", required=True, help="Path to a .npy input tensor")
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()
    generate(args.model, args.input, args.out)
    print("Done")
