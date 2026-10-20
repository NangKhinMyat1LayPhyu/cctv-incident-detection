"""Convert UA-DETRAC to a YOLO dataset (images/, labels/, data.yaml).

Usage:
  python scripts/prepare_detrac.py --images path/to/DETRAC-Images \
      --annotations path/to/DETRAC-Train-Annotations-XML --out data/detrac_yolo
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from incident_recon.detrac import (  # noqa: E402
    CLASS_NAMES,
    convert_sequence,
    read_weather,
    split_by_weather,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True, help="Folder containing MVI_xxxxx frame folders")
    p.add_argument("--annotations", required=True, help="Folder containing MVI_xxxxx.xml files")
    p.add_argument("--out", default="data/detrac_yolo")
    p.add_argument("--every", type=int, default=10, help="Keep every Nth frame")
    p.add_argument("--val-per-weather", type=int, default=3)
    p.add_argument("--limit-videos", type=int, default=0, help="For quick tests; 0 = all")
    args = p.parse_args()

    xml_paths = sorted(Path(args.annotations).glob("*.xml"))
    if not xml_paths:
        sys.exit(f"No XML files found in {args.annotations}")
    if args.limit_videos:
        xml_paths = xml_paths[: args.limit_videos]

    train, val = split_by_weather(xml_paths, args.val_per_weather)
    print(f"{len(xml_paths)} videos -> {len(train)} train, {len(val)} val")

    out = Path(args.out)
    split_record = {"train": [], "val": []}
    for split, paths in (("train", train), ("val", val)):
        for xml_path in paths:
            n_img, n_box = convert_sequence(
                xml_path,
                args.images,
                out / "images" / split,
                out / "labels" / split,
                every=args.every,
            )
            weather = read_weather(xml_path)
            split_record[split].append({"video": xml_path.stem, "weather": weather,
                                        "images": n_img, "boxes": n_box})
            print(f"[{split}] {xml_path.stem} ({weather}): {n_img} images, {n_box} boxes")

    # data.yaml tells Ultralytics where the data is and what the classes are
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n"
    )
    (out / "split.json").write_text(json.dumps(split_record, indent=2))

    for split in ("train", "val"):
        n_img = sum(r["images"] for r in split_record[split])
        n_box = sum(r["boxes"] for r in split_record[split])
        print(f"{split}: {n_img} images, {n_box} boxes")
    print(f"\nDone. Dataset config: {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
