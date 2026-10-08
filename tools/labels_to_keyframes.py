# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Convert edited YOLO labels back into box_tagger keyframes.

Use this after reviewing/fixing ``labels/*.txt`` in an external tool
(CVAT, Label Studio, Roboflow, labelImg): export the corrected YOLO txt
files over ``<output-dir>/labels/``, then rebuild ``keyframes.json`` so a
rerun conditions SAM 2 on the fixed boxes instead of re-tracking blindly.

Example:
    python tools/labels_to_keyframes.py \\
        --labels-dir outputs/cafe_test_boxes/labels \\
        --classes phone,laptop,wallet,bottle,bag,cup \\
        --probe-image sample_dataset/image_sequence/00000.jpg \\
        --output outputs/cafe_test_boxes/keyframes.json

Then rerun hands-free:
    python tools/box_tagger.py --video ... --keyframes \\
        outputs/cafe_test_boxes/keyframes.json --no-interactive ...
"""

import argparse
import json
import os

import cv2


def parse_args():
    p = argparse.ArgumentParser(
        description="YOLO labels dir -> box_tagger keyframes.json")
    p.add_argument("--labels-dir", type=str, required=True,
                   help="dir of <frame>.txt YOLO files")
    p.add_argument("--classes", type=str, required=True,
                   help="comma-separated class names (validates cls ids)")
    p.add_argument("--probe-image", type=str, default=None,
                   help="any frame JPEG to read W/H from "
                        "(else --width/--height required)")
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--output", type=str, required=True,
                   help="output keyframes.json path")
    p.add_argument("--frames", type=int, nargs="*", default=None,
                   help="only include these frame indices "
                        "(default: every non-empty txt)")
    return p.parse_args()


def yolo_to_xyxy(cls, cx, cy, bw, bh, w, h):
    w_px, h_px = bw * w, bh * h
    x1 = cx * w - w_px / 2.0
    y1 = cy * h - h_px / 2.0
    return [x1, y1, x1 + w_px - 1.0, y1 + h_px - 1.0]


def main():
    args = parse_args()
    names = [c.strip() for c in args.classes.split(",") if c.strip()]
    if args.probe_image:
        img = cv2.imread(args.probe_image)
        if img is None:
            raise RuntimeError(f"cannot read {args.probe_image}")
        h, w = img.shape[:2]
    elif args.width and args.height:
        w, h = args.width, args.height
    else:
        raise ValueError("need --probe-image or --width/--height")

    files = sorted(p for p in os.listdir(args.labels_dir)
                   if p.endswith(".txt"))

    def _key(p):
        stem = os.path.splitext(p)[0]
        return int(stem) if stem.isdigit() else stem

    files.sort(key=_key)
    keyframes = []
    for idx, fname in enumerate(files):
        with open(os.path.join(args.labels_dir, fname)) as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]
        if args.frames is not None and idx not in args.frames:
            continue
        if not lines:
            continue
        boxes = []
        for ln in lines:
            cls, cx, cy, bw, bh = ln.split()
            cls = int(float(cls))
            if not 0 <= cls < len(names):
                raise ValueError(f"cls {cls} out of range in {fname}")
            boxes.append({"xyxy": yolo_to_xyxy(
                cls, float(cx), float(cy), float(bw), float(bh), w, h),
                "cls": cls})
        keyframes.append({"frame": idx, "boxes": boxes})
    payload = {"classes": names, "keyframes": keyframes}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".",
                exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"wrote {len(keyframes)} keyframe(s) to {args.output}")


if __name__ == "__main__":
    main()
