"""
Attribute embedding module for classifier-free guidance.

Attributes are represented as discrete tokens (body type, hair style,
outfit, etc.).  Each attribute is mapped to a learned embedding vector.
The vectors are summed to form a single conditioning signal.

Classifier-free guidance (CFG) is supported by:
  - With probability p_uncond, replacing the embedding with a learned
    "null" embedding during training.
  - At inference, computing cond and uncond predictions and interpolating:
      output = uncond + guidance_scale * (cond - uncond)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn


class AttributeEmbedding(nn.Module):
    """
    Converts a dict of discrete attribute indices to a dense embedding.

    Args:
        attr_vocab: {attr_name: vocab_size} mapping.
        emb_dim:    Output embedding dimension.
        p_uncond:   Probability of dropping all attributes (CFG null embed).
    """

    def __init__(
        self,
        attr_vocab: Dict[str, int],
        emb_dim: int = 256,
        p_uncond: float = 0.1,
    ) -> None:
        super().__init__()
        self.attr_names: List[str] = sorted(attr_vocab.keys())
        self.p_uncond = p_uncond
        self.emb_dim = emb_dim

        # One embedding table per attribute
        self.embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(vocab_size + 1, emb_dim)  # +1 for null token
                for name, vocab_size in attr_vocab.items()
            }
        )

        # Shared null embedding vector (used when dropping all attributes)
        self.null_emb = nn.Parameter(torch.zeros(emb_dim))

        # Output projection
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(
        self,
        attrs: Dict[str, torch.Tensor],
        force_null: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            attrs:      {attr_name: (B,) int64 index tensors}.
            force_null: If True, always return the null embedding (for CFG).

        Returns:
            (B, emb_dim) embedding tensor.
        """
        # Determine which samples are "null" (dropped during training)
        batch_size = next(iter(attrs.values())).shape[0]
        device = next(iter(attrs.values())).device

        if force_null:
            drop_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        else:
            drop_mask = torch.rand(batch_size, device=device) < self.p_uncond

        # Sum attribute embeddings
        emb = torch.zeros(batch_size, self.emb_dim, device=device)
        for name in self.attr_names:
            if name in attrs:
                emb = emb + self.embeddings[name](attrs[name])

        # Replace dropped samples with null embedding
        null = self.null_emb.unsqueeze(0).expand(batch_size, -1)
        emb = torch.where(drop_mask.unsqueeze(1), null, emb)

        return self.proj(emb)

    @torch.no_grad()
    def get_null_embedding(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return the null (unconditional) embedding for CFG inference."""
        return self.proj(self.null_emb.unsqueeze(0).expand(batch_size, -1))


# ---------------------------------------------------------------------------
# Default attribute vocabulary for LPC characters
# ---------------------------------------------------------------------------

LPC_ATTR_VOCAB: Dict[str, int] = {
    "body":       4,   # human adult male/female, child, etc.
    "hair":       16,  # hair styles
    "hat":        8,   # hat types (0 = none)
    "outfit":     12,  # tops/full outfits
    "legs":       8,   # leg wear
    "shoes":      6,   # footwear
    "weapon":     10,  # right-hand item (0 = none)
    "shield":     6,   # left-hand item (0 = none)
}


def build_lpc_attr_embedding(emb_dim: int = 256,
                              p_uncond: float = 0.1) -> AttributeEmbedding:
    """Instantiate an AttributeEmbedding using the default LPC vocabulary."""
    return AttributeEmbedding(LPC_ATTR_VOCAB, emb_dim=emb_dim, p_uncond=p_uncond)
