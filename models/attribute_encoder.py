"""Attribute encoder for conditional Flow Matching.

Maps discrete character attributes (colors, body type, clothing types) to a
continuous embedding vector injected into the UNet time embedding.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Attribute vocabularies
# ---------------------------------------------------------------------------

COLOR_VOCAB = [
    "none", "black", "white", "gray", "brown", "red", "pink", "orange",
    "yellow", "green", "teal", "blue", "purple", "gold", "silver", "copper",
]

BODY_TYPE_VOCAB = ["male", "female", "teen", "child", "muscular", "adult"]

HAIR_STYLE_VOCAB = [
    "none", "short", "medium", "long", "ponytail", "braid", "curly",
    "spiked", "bangs", "pigtails", "dreadlocks", "messy", "parted", "bun",
]

TORSO_TYPE_VOCAB = ["bare", "clothes", "jacket", "armour"]

LEGS_TYPE_VOCAB = ["pants", "shorts", "skirt", "dress", "leggings", "armour"]

FEET_TYPE_VOCAB = ["boots", "shoes", "sandals", "armour"]

COLOR_TO_IDX = {c: i for i, c in enumerate(COLOR_VOCAB)}
BODY_TYPE_TO_IDX = {b: i for i, b in enumerate(BODY_TYPE_VOCAB)}
HAIR_STYLE_TO_IDX = {s: i for i, s in enumerate(HAIR_STYLE_VOCAB)}
TORSO_TYPE_TO_IDX = {t: i for i, t in enumerate(TORSO_TYPE_VOCAB)}
LEGS_TYPE_TO_IDX = {l: i for i, l in enumerate(LEGS_TYPE_VOCAB)}
FEET_TYPE_TO_IDX = {f: i for i, f in enumerate(FEET_TYPE_VOCAB)}

NUM_COLORS = len(COLOR_VOCAB)
NUM_BODY_TYPES = len(BODY_TYPE_VOCAB)
NUM_HAIR_STYLES = len(HAIR_STYLE_VOCAB)
NUM_TORSO_TYPES = len(TORSO_TYPE_VOCAB)
NUM_LEGS_TYPES = len(LEGS_TYPE_VOCAB)
NUM_FEET_TYPES = len(FEET_TYPE_VOCAB)

ATTR_FIELDS = [
    "body_type", "hair_style",
    "torso_type", "torso_color",
    "legs_type", "legs_color",
    "feet_type", "feet_color",
]

ATTR_TO_IDX = {
    "body_type": BODY_TYPE_TO_IDX,
    "hair_style": HAIR_STYLE_TO_IDX,
    "torso_type": TORSO_TYPE_TO_IDX,
    "torso_color": COLOR_TO_IDX,
    "legs_type": LEGS_TYPE_TO_IDX,
    "legs_color": COLOR_TO_IDX,
    "feet_type": FEET_TYPE_TO_IDX,
    "feet_color": COLOR_TO_IDX,
}

COLOR_FIELDS = {"torso_color", "legs_color", "feet_color"}


def encode_attributes_batch(
    attrs_list: list[dict],
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """Convert a list of attribute dicts to a dict of index tensors.

    Each attribute value is mapped to its vocabulary index.
    None values map to index 0 (the "none" / default class).
    """
    batch_size = len(attrs_list)
    result: dict[str, torch.Tensor] = {}

    for field in ATTR_FIELDS:
        vocab = ATTR_TO_IDX[field]
        indices = []
        for attrs in attrs_list:
            val = attrs.get(field)
            if val is None:
                indices.append(0)
            else:
                indices.append(vocab.get(val, 0))
        result[field] = torch.tensor(indices, dtype=torch.long, device=device)

    return result


# ---------------------------------------------------------------------------
# Attribute encoder model
# ---------------------------------------------------------------------------

class AttributeEncoder(nn.Module):
    """Encode discrete character attributes into a continuous embedding vector.

    Each attribute field gets a learnable embedding; all embeddings are
    concatenated and projected through an MLP to produce the global conditioning
    vector (attr_cond).  Additionally, each field's embedding is independently
    projected to a token vector, yielding a sequence (attr_tokens) for
    cross-attention in the UNet.
    """

    def __init__(
        self,
        num_colors: int = NUM_COLORS,
        num_body_types: int = NUM_BODY_TYPES,
        num_hair_styles: int = NUM_HAIR_STYLES,
        num_torso_types: int = NUM_TORSO_TYPES,
        num_legs_types: int = NUM_LEGS_TYPES,
        num_feet_types: int = NUM_FEET_TYPES,
        embed_dim: int = 64,
        output_dim: int = 256,
        token_dim: int | None = None,
        use_pos_embed: bool = True,
    ) -> None:
        super().__init__()
        self.color_embed = nn.Embedding(num_colors, embed_dim)
        self.body_type_embed = nn.Embedding(num_body_types, embed_dim)
        self.hair_style_embed = nn.Embedding(num_hair_styles, embed_dim)
        self.torso_type_embed = nn.Embedding(num_torso_types, embed_dim)
        self.legs_type_embed = nn.Embedding(num_legs_types, embed_dim)
        self.feet_type_embed = nn.Embedding(num_feet_types, embed_dim)

        # 5 type + 3 color (torso/legs/feet) = 8
        num_fields = 8

        # --- Global conditioning vector (FiLM) ---
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * num_fields, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

        # --- Per-field token sequence (cross-attention) ---
        token_dim = token_dim or output_dim
        self.token_dim = token_dim
        self.per_field_proj = nn.Sequential(
            nn.Linear(embed_dim, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )
        if use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_fields, token_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = None

    def forward(
        self, attr_indices: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            attr_indices: dict mapping field names to (B,) long tensors.

        Returns:
            (attr_cond, attr_tokens)
              - attr_cond: (B, output_dim) global conditioning vector for FiLM.
              - attr_tokens: (B, 9, token_dim) per-field tokens for cross-attention.
        """
        per_field_embs: list[torch.Tensor] = []
        for field in ATTR_FIELDS:
            idx = attr_indices[field]
            if field in COLOR_FIELDS:
                per_field_embs.append(self.color_embed(idx))
            elif field == "body_type":
                per_field_embs.append(self.body_type_embed(idx))
            elif field == "hair_style":
                per_field_embs.append(self.hair_style_embed(idx))
            elif field == "torso_type":
                per_field_embs.append(self.torso_type_embed(idx))
            elif field == "legs_type":
                per_field_embs.append(self.legs_type_embed(idx))
            elif field == "feet_type":
                per_field_embs.append(self.feet_type_embed(idx))

        # Global vector: concat all → MLP
        concat = torch.cat(per_field_embs, dim=-1)
        attr_cond = self.mlp(concat)  # (B, output_dim)

        # Token sequence: stack (B, 9, embed_dim) → project → (B, 9, token_dim)
        tokens = torch.stack(per_field_embs, dim=1)  # (B, 9, embed_dim)
        attr_tokens = self.per_field_proj(tokens)  # (B, 9, token_dim)
        if self.pos_embed is not None:
            attr_tokens = attr_tokens + self.pos_embed

        return attr_cond, attr_tokens
