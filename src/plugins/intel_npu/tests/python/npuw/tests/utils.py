# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import glob
import os
import subprocess
from pathlib import Path
import re

import pytest

import numpy as np
from PIL import Image

ARTIFACTORY_REPO = "ir-public-models-ir-local"
SAMPLE_RATE = 16000

default_whisper_config = {"NPU_BYPASS_UMD_CACHING": "YES", "NPUW_DEVICES": "NPU"}

# Standard ImageNet preprocessing, matching what optimum-cli's image-classification
# export bakes into the reference pipeline for a plain (non-preprocessing) ResNet-50 IR.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_resnet_image(path, size=224):
    """Load a picture and preprocess it into a ResNet-50 NCHW input tensor."""
    image = Image.open(path).convert("RGB").resize((size, size))
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    pixels = (pixels - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(pixels, (2, 0, 1))[np.newaxis, ...]


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

def get_whisper_model_root():
    for subdir in ["whisper", "Whisper"]:
        candidate = get_model_dir(subdir)
        if os.path.isdir(candidate):
            return candidate


def get_whisper_audio_path():
    if 'NPUW_TEST_WHISPER_AUDIO' in os.environ:
        return str(os.environ['NPUW_TEST_WHISPER_AUDIO'])

    return str(Path(__file__).resolve().parents[2] / "samples" / "whisper" / "media" / "how_are_you_doing_today.wav")


def read_wav_for_whisper(path):
    librosa = pytest.importorskip("librosa")
    raw_speech, _ = librosa.load(path, sr=SAMPLE_RATE)
    return raw_speech.tolist()


def get_agent_ram_gb():
    try:
        return int(os.environ["NPUW_AGENT_RAM_GB"].split()[0])
    except (KeyError, ValueError):
        return None
