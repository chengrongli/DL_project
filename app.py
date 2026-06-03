"""Flask backend for Pixel Sprite Generator UI.

Serves the web UI and provides API endpoints for conditional / random
sprite generation via the trained Flow Matching model.
"""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from flask import Flask, jsonify, render_template, request, send_from_directory

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.flow_unet import UNet
from models.flow_matching import FlowMatching
from models.attribute_encoder import (
    AttributeEncoder,
    encode_attributes_batch,
    ATTR_FIELDS,
    HAIR_STYLE_VOCAB,
    TORSO_TYPE_VOCAB,
    LEGS_TYPE_VOCAB,
    FEET_TYPE_VOCAB,
    COLOR_VOCAB,
    BODY_TYPE_VOCAB,
)
import random

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG_PATH = str(ROOT / "configs" / "random_batch_v2_config.yaml")
CKPT_PATH = str(ROOT / "outputs" / "random_batch_v3" / "checkpoints" / "flow_epoch0100.pth")

# ---------------------------------------------------------------------------
# Load model once at startup
# ---------------------------------------------------------------------------
print("[init] Loading config …")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
attr_cfg = cfg.get("attributes", {})

print("[init] Building UNet …")
unet = UNet(
    in_channels=cfg["model"]["in_channels"],
    model_channels=cfg["model"]["model_channels"],
    out_channels=cfg["model"]["out_channels"],
    channel_mult=tuple(cfg["model"]["channel_mults"]),
    num_res_blocks=cfg["model"]["n_res_blocks"],
    dropout=cfg["model"].get("dropout", 0.1),
    time_emb_dim=cfg["model"]["time_emb_dim"],
    attention_resolutions=tuple(cfg["model"].get("attn_resolutions", [16])),
    attr_cond_dim=attr_cfg.get("output_dim", 256),
    attr_token_dim=attr_cfg.get("token_dim", 256),
    cross_attn_heads=cfg["model"].get("cross_attn_heads", 4),
).to(device)

flow_model = FlowMatching(unet, time_scale=cfg.get("flow", {}).get("time_scale", 999.0)).to(device)
attr_encoder = AttributeEncoder(
    embed_dim=attr_cfg.get("embed_dim", 64),
    output_dim=attr_cfg.get("output_dim", 256),
    token_dim=attr_cfg.get("token_dim", 256),
    use_pos_embed=attr_cfg.get("use_pos_embed", True),
).to(device)

print(f"[init] Loading checkpoint: {CKPT_PATH}")
ckpt = torch.load(CKPT_PATH, map_location=device)
state = ckpt.get("ema_state_dict", ckpt.get("model_state_dict", ckpt))
flow_model.load_state_dict(state)
attr_state = ckpt.get("ema_attr_encoder_state_dict", ckpt.get("attr_encoder_state_dict"))
if attr_state is not None:
    attr_encoder.load_state_dict(attr_state)

flow_model.eval()
attr_encoder.eval()
print("[init] Model ready ✓")

# ---------------------------------------------------------------------------
# Helper: tensor → base64 PNG
# ---------------------------------------------------------------------------

def _random_attrs() -> dict:
    """按训练数据分布随机采样一组属性。"""
    # torso_type 加权：clothes 50%, jacket 25%, bare 15%, armour 10%
    torso_choices = ["clothes"] * 50 + ["jacket"] * 25 + ["bare"] * 15 + ["armour"] * 10
    return {
        "body_type": random.choice(BODY_TYPE_VOCAB),
        "hair_style": random.choice(HAIR_STYLE_VOCAB[1:]),
        "torso_type": random.choice(torso_choices),
        "torso_color": random.choice(COLOR_VOCAB[1:]),
        "legs_type": random.choice(LEGS_TYPE_VOCAB),
        "legs_color": random.choice(COLOR_VOCAB[1:]),
        "feet_type": random.choice(FEET_TYPE_VOCAB),
        "feet_color": random.choice(COLOR_VOCAB[1:]),
    }



def _tensor_to_b64(t: torch.Tensor, upscale: int = 4) -> str:
    """Convert a (C, H, W) tensor in [-1,1] to base64 PNG string."""
    t = t.detach().cpu().clamp(-1, 1)
    t = (t + 1.0) / 2.0
    arr = (t.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    mode = "RGBA" if arr.shape[2] == 4 else "RGB"
    img = Image.fromarray(arr, mode=mode)
    if upscale > 1:
        w, h = img.size
        img = img.resize((w * upscale, h * upscale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder=str(ROOT / "static"), static_url_path="/static")


@app.route("/")
def index():
    return send_from_directory(str(ROOT / "static"), "index.html")


# ---- Vocabularies --------------------------------------------------------

@app.route("/api/vocabs")
def api_vocabs():
    """Return all attribute vocabularies so the frontend can build dropdowns."""
    return jsonify({
        "body_type": BODY_TYPE_VOCAB,
        "hair_style": HAIR_STYLE_VOCAB[1:],  # skip "none"
        "torso_type": TORSO_TYPE_VOCAB,
        "torso_color": COLOR_VOCAB[1:],      # skip "none"
        "legs_type": LEGS_TYPE_VOCAB,
        "legs_color": COLOR_VOCAB[1:],
        "feet_type": FEET_TYPE_VOCAB,
        "feet_color": COLOR_VOCAB[1:],
    })


# ---- Conditional generation -----------------------------------------------

@app.route("/api/generate_conditional", methods=["POST"])
def api_generate_conditional():
    """Generate sprite pair(s) conditioned on user-selected attributes.

    Expects JSON body:
        {
            "attrs": { "body_type": "female", "hair_style": "long", ... },
            "count": 1,           // number of samples (1-8)
            "guidance_scale": 3.0,
            "seed": null | int
        }
    Returns:
        { "images": [ { "front": "<base64>", "back": "<base64>" }, ... ] }
    """
    body = request.get_json(force=True)
    raw_attrs = body.get("attrs", {})
    count = min(max(body.get("count", 1), 1), 8)
    guidance_scale = float(body.get("guidance_scale", 3.0))
    seed = body.get("seed")

    if seed is not None:
        torch.manual_seed(int(seed))
        random.seed(int(seed))

    # 空字段自动填随机值，避免全 "none" 走无条件模式
    def _fill_random(attr_dict):
        defaults = _random_attrs()
        for field in ATTR_FIELDS:
            if not attr_dict.get(field):
                attr_dict[field] = defaults[field]
        return attr_dict

    # Build attribute dicts for the batch
    attrs_batch = [_fill_random(dict(raw_attrs)) for _ in range(count)]
    attr_indices = encode_attributes_batch(attrs_batch, device=device)
    attr_cond, attr_tokens = attr_encoder(attr_indices)

    h = cfg["data"]["image_size"]
    shape = (count, cfg["model"]["out_channels"], h, 2 * h)

    with torch.no_grad():
        samples = flow_model.sample(
            shape,
            steps=60,
            attr_cond=attr_cond,
            attr_tokens=attr_tokens,
            guidance_scale=guidance_scale,
        )

    # Split front/back (each 64x64) from the 64x128 output
    images = []
    for i in range(count):
        sample = samples[i]                    # (4, 64, 128)
        front_t = sample[:, :, :h]             # (4, 64, 64)
        back_t = sample[:, :, h:]              # (4, 64, 64)
        images.append({
            "front": _tensor_to_b64(front_t, upscale=4),
            "back": _tensor_to_b64(back_t, upscale=4),
        })

    return jsonify({"images": images})


# ---- Random (unconditional) generation ------------------------------------

@app.route("/api/generate_random", methods=["POST"])
def api_generate_random():
    """随机生成：为每个样本随机采样属性，走条件生成路径。"""
    body = request.get_json(force=True)
    count = min(max(body.get("count", 4), 1), 8)
    seed = body.get("seed")

    if seed is not None:
        torch.manual_seed(int(seed))

    h = cfg["data"]["image_size"]
    shape = (count, cfg["model"]["out_channels"], h, 2 * h)

    with torch.no_grad():
        samples = flow_model.sample(shape, steps=60)

    images = []
    for i in range(count):
        sample = samples[i]
        front_t = sample[:, :, :h]
        back_t = sample[:, :, h:]
        images.append({
            "front": _tensor_to_b64(front_t, upscale=4),
            "back": _tensor_to_b64(back_t, upscale=4),
        })

    return jsonify({"images": images})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6006, debug=False)
