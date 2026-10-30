# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from whowhatbench.tts_similarity import evaluate_tts_similarity


class WwbTtsSimilarity:
    def __init__(self, metric="overall") -> None:
        self.metric = metric

    def evaluate(self, reference_path, actual_path):
        scores = evaluate_tts_similarity(
            target_path=actual_path,
            reference_path=reference_path,
        )
        return getattr(scores, self.metric)
