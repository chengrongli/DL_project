import argparse
import csv
from pathlib import Path
from typing import Iterable

from PIL import Image


def list_front_images(input_dir: Path, front_suffix: str) -> Iterable[Path]:
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        for path in input_dir.glob(ext):
            if path.stem.endswith(front_suffix):
                yield path


def ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGBA":
        bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(bg, image).convert("RGB")
        return image
    return image.convert("RGB")


def make_paired_image(
    front_path: Path,
    back_path: Path,
    out_path: Path,
    single_width: int,
    height: int,
) -> None:
    front = ensure_rgb(Image.open(front_path)).resize((single_width, height), Image.BICUBIC)
    back = ensure_rgb(Image.open(back_path)).resize((single_width, height), Image.BICUBIC)
    paired = Image.new("RGB", (single_width * 2, height), (255, 255, 255))
    paired.paste(front, (0, 0))
    paired.paste(back, (single_width, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    paired.save(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build front/back manifest for HD consistency training.")
    parser.add_argument("--input-dir", required=True, help="Directory containing *_front / *_back images")
    parser.add_argument("--manifest", required=True, help="Output csv path")
    parser.add_argument("--paired-dir", default="", help="Optional output dir for paired images")
    parser.add_argument("--front-suffix", default="_front")
    parser.add_argument("--back-suffix", default="_back")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--single-width", type=int, default=512)
    parser.add_argument(
        "--prompt-template",
        default=(
            "high quality character sheet, full body, front view on left, back view on right, "
            "same character, same outfit details, id:{char_id}"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    manifest_path = Path(args.manifest)
    paired_dir = Path(args.paired_dir) if args.paired_dir else None

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    rows = []
    for front_path in list_front_images(input_dir, args.front_suffix):
        stem_prefix = front_path.stem[: -len(args.front_suffix)]
        back_name = f"{stem_prefix}{args.back_suffix}{front_path.suffix}"
        back_path = front_path.with_name(back_name)
        if not back_path.exists():
            continue

        paired_path = ""
        if paired_dir is not None:
            paired_path = str((paired_dir / f"{stem_prefix}_paired.png").resolve())
            make_paired_image(
                front_path=front_path,
                back_path=back_path,
                out_path=Path(paired_path),
                single_width=args.single_width,
                height=args.height,
            )

        rows.append(
            {
                "char_id": stem_prefix,
                "front_path": str(front_path.resolve()),
                "back_path": str(back_path.resolve()),
                "paired_path": paired_path,
                "prompt": args.prompt_template.format(char_id=stem_prefix),
            }
        )

    if not rows:
        raise RuntimeError("No valid front/back pairs found.")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["char_id", "front_path", "back_path", "paired_path", "prompt"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} pairs to {manifest_path}")


if __name__ == "__main__":
    main()
