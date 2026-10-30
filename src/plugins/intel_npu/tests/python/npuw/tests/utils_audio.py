# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import os
import warnings
from mel_cepstral_distance import compare_audio_files


def load_audio(path, sr=None):
    """
    Load audio from .npy or .wav file.

    Args:
        path (str): Path to the audio file.
        sr (int, optional): Target sample rate for .wav files. Defaults to None (native).

    Returns:
        np.ndarray: 1D audio array.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    if path.endswith('.npy'):
        audio = np.load(path)
        # Ensure 1D
        if audio.ndim > 1:
            audio = audio.flatten()
        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        return audio
    else:
        try:
            import librosa
            audio, _ = librosa.load(path, sr=sr)
            return audio
        except ImportError:
            raise ImportError("librosa is required to load .wav files. Please install it.")


def calculate_mcd(ref_audio, synthesized_audio, sr=24000):
    """
    Calculate Mel-Cepstral Distortion (MCD) between reference and synthesized audio.

    Args:
        ref_audio (np.ndarray): Reference audio signal.
        synthesized_audio (np.ndarray): Synthesized audio signal.
    """
    try:
        import librosa
        import scipy.io.wavfile as wavfile
        import tempfile
    except ImportError:
        raise ImportError("librosa and scipy are required to calculate MCD. Please install them.")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ref_file, \
            tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as synth_file:

        wavfile.write(ref_file.name, sr, ref_audio)
        wavfile.write(synth_file.name, sr, synthesized_audio)

        ref_path = ref_file.name
        synth_path = synth_file.name

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*power of 2.*")
            mcd, penalty = compare_audio_files(ref_path, synth_path)
    finally:
        if os.path.exists(ref_path):
            os.remove(ref_path)
        if os.path.exists(synth_path):
            os.remove(synth_path)

    print(f'MCD: {mcd:.2f}, Penalty: {penalty:.4f}')
    return mcd
