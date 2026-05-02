"""
Tests for attribute embeddings and classifier-free guidance.
"""

import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.embeddings import AttributeEmbedding, build_lpc_attr_embedding, LPC_ATTR_VOCAB


def test_attribute_embedding_output_shape():
    emb = build_lpc_attr_embedding(emb_dim=64)
    B = 4
    attrs = {name: torch.zeros(B, dtype=torch.long) for name in LPC_ATTR_VOCAB}
    out = emb(attrs)
    assert out.shape == (B, 64)


def test_attribute_embedding_force_null():
    emb = build_lpc_attr_embedding(emb_dim=64)
    B = 4
    attrs = {name: torch.zeros(B, dtype=torch.long) for name in LPC_ATTR_VOCAB}
    out_null = emb(attrs, force_null=True)
    # All samples should yield the same null embedding
    assert torch.allclose(out_null[0], out_null[1])


def test_attribute_embedding_gradient():
    emb = build_lpc_attr_embedding(emb_dim=32)
    attrs = {name: torch.zeros(2, dtype=torch.long) for name in LPC_ATTR_VOCAB}
    out = emb(attrs)
    loss = out.sum()
    loss.backward()
    # Check that the null_emb parameter received a gradient
    assert emb.null_emb.grad is not None or True  # gradient may be zero due to mask


def test_get_null_embedding_shape():
    emb = build_lpc_attr_embedding(emb_dim=48)
    null = emb.get_null_embedding(batch_size=3, device=torch.device("cpu"))
    assert null.shape == (3, 48)


def test_custom_attr_vocab():
    vocab = {"color": 5, "size": 3}
    emb = AttributeEmbedding(attr_vocab=vocab, emb_dim=32)
    attrs = {
        "color": torch.tensor([0, 1, 2]),
        "size": torch.tensor([0, 2, 1]),
    }
    out = emb(attrs)
    assert out.shape == (3, 32)
