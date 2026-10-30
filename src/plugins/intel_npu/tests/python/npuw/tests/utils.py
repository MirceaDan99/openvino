# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import glob

def get_model_dir(subdir=None):
    """Get model directory from environment variable."""
    model_dir = None
    try:
        model_dir = os.environ['NPUW_PACKAGE_MODEL_DIR']
        if not model_dir:
            raise ValueError("Empty NPUW_PACKAGE_MODEL_DIR provided in the environment!")
    except KeyError:
        raise KeyError("NPUW_PACKAGE_MODEL_DIR environment variable is not set!")

    if subdir:
        model_dir = os.path.join(model_dir, glob.glob(model_dir + '/**/*' + subdir + '*/', recursive=True)[0])

    return model_dir
