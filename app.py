"""Flask Web UI for Pixel Sprite Generator.

Mode A: FLUX text-to-sprite generation (placeholder, to be connected).
Mode B: Front-to-back diffusion reconstruction (Task 2).
"""

import base64
import io
import sys
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
import yaml
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, jsonify, render_template, request

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.diffusion import GaussianDiffusion
from models.unet import UNet
from utils.visualization import tensor_to_pil

app = Flask(__name__, static_folder="static", static_url_path="/static")

# ---------------------------------------------------------------------------
# Load Task 2 model at startup
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CFG_PATH = "configs/task2_config.yaml"
CKPT_PATH = "outputs/task2/checkpoints/ckpt_epoch0199.pth"

with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

IMAGE_SIZE = CFG["data"]["image_size"]

print("Loading Task 2 model...")
_task2_unet = UNet(
    in_channels=CFG["model"]["in_channels"],
    out_channels=CFG["model"]["out_channels"],
    model_channels=CFG["model"]["model_channels"],
    channel_mults=tuple(CFG["model"]["channel_mults"]),
    n_res_blocks=CFG["model"]["n_res_blocks"],
    attn_resolutions=tuple(CFG["model"]["attn_resolutions"]),
    time_emb_dim=CFG["model"]["time_emb_dim"],
    cond_emb_dim=0,
    dropout=0.0,
    image_size=IMAGE_SIZE,
).to(DEVICE)

_task2_diffusion = GaussianDiffusion(
    model=_task2_unet,
    timesteps=CFG["diffusion"]["timesteps"],
    schedule=CFG["diffusion"]["schedule"],
    loss_type=CFG["diffusion"]["loss_type"],
).to(DEVICE)

_ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
_task2_diffusion.load_state_dict(_ckpt["diffusion"])
_task2_diffusion.eval()
print(f"Task 2 model loaded on {DEVICE}")


def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _make_placeholder(text: str) -> str:
    img = Image.new("RGB", (256, 256), "#2a2a3e")
    draw = ImageDraw.Draw(img)
    draw.text((128, 128), text, fill="#e94560", anchor="mm")
    img = img.resize((64, 64), Image.NEAREST)
    return _pil_to_b64(img)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/api/generate_flux", methods=["POST"])
def api_generate_flux():
    """Mode A: FLUX text-to-sprite. Currently returns placeholders."""
    data = request.get_json(force=True)
    prompt = data.get("prompt", "a character")
    print(f"[FLUX placeholder] prompt: {prompt}")

    front_b64 = _make_placeholder("FLUX\nfront")
    back_b64 = _make_placeholder("FLUX\nback")
    return jsonify({"front": front_b64, "back": back_b64})


@app.route("/api/generate_back", methods=["POST"])
def api_generate_back():
    """Mode B: Upload front image, generate back via Task 2 diffusion."""
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    try:
        front_pil = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    front_pil = front_pil.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
    front_t = TF.to_tensor(front_pil) * 2.0 - 1.0
    front_t = front_t.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        back_t = _task2_diffusion.ddim_sample(
            shape=(1, 3, IMAGE_SIZE, IMAGE_SIZE),
            device=DEVICE,
            ddim_steps=50,
            eta=0.0,
            cond_image=front_t,
        )

    front_out = tensor_to_pil(front_t[0])
    back_out = tensor_to_pil(back_t[0])

    return jsonify({
        "front": _pil_to_b64(front_out),
        "back": _pil_to_b64(back_out),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
