# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from openvino_genai import TextEmbeddingPipeline
import numpy as np


class WwbTextSimilarity:
    similarity_model_id = "sentence-transformers/all-mpnet-base-v2"

    def __init__(self) -> None:
        trust_remote_code = False
        model_id = WwbTextSimilarity.similarity_model_id
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
        except Exception:
            trust_remote_code = True
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        if hasattr(tokenizer, "pad_token") and tokenizer.pad_token:
            pad_token = tokenizer.pad_token
        else:
            pad_token = tokenizer.eos_token
        self.model = SentenceTransformer(model_id,
                                         tokenizer_kwargs={"pad_token": pad_token},
                                         trust_remote_code=trust_remote_code)

    def evaluate(self, reference, actual):
        embeddings = self.model.encode([reference, actual])
        cos_sim = util.cos_sim(embeddings, embeddings)
        return cos_sim[0, 1].item()


class EmbeddingSimilarity:

    DEFAULT_PASSAGES = [
        "OpenVINO is an open-source toolkit for optimizing and deploying AI inference.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models can be compressed and deployed on edge devices.",
        "Neural networks consist of layers of interconnected nodes that process data.",
        "Quantization reduces model size while preserving accuracy."
    ]

    def __init__(self, max_length=20) -> None:
        self.max_length = max_length

    def _make_config(self, batch_size=1):
        config = TextEmbeddingPipeline.Config()
        config.pooling_type = TextEmbeddingPipeline.PoolingType.MEAN
        config.normalize = True
        config.max_length = self.max_length
        config.pad_to_max_length = True
        config.batch_size = batch_size
        return config

    def evaluate(self, model_path, ref_device="CPU",
                 target_device="CPU", passages=None) -> float:
        if passages is None:
            passages = self.DEFAULT_PASSAGES

        config = self._make_config()

        ref_pipeline = TextEmbeddingPipeline(model_path, ref_device, config)
        ref_embeds = [np.array(ref_pipeline.embed_documents([p])) for p in passages]
        ref_embeds = np.vstack(ref_embeds)

        target_pipeline = TextEmbeddingPipeline(model_path, target_device, config)
        target_embeds = [np.array(target_pipeline.embed_documents([p])) for p in passages]
        target_embeds = np.vstack(target_embeds)

        per_passage = np.diag(cosine_similarity(ref_embeds, target_embeds))
        return float(np.mean(per_passage))
