# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Draw-once video box tagger with SAM 2 tracking + YOLO txt export.

Workflow:
  1. If ``--video`` is an mp4, extract it to a JPEG image sequence first
     (streaming frame-by-frame, so RAM stays flat). The sequence defaults
     to ``<parent-of-videos>/image_sequence`` (e.g. ``sample_dataset/videos/
     cafe_test.mp4`` -> ``sample_dataset/image_sequence/``). The dir is
     cleared before new frames are written (unless ``--keep-frames``).
  2. Draw box(es) on a single annotation frame (interactive OpenCV window
     via ``cv2.selectROI`` or non-interactive ``--box x1 y1 x2 y2``).
  3. SAM 2 tracks each box through the video
     (``add_new_points_or_box`` + ``propagate_in_video``). Masks are
     converted to pixel boxes immediately (streaming) so only tiny
     box lists are kept in memory, not full mask logits.
  4. Per-frame boxes are written as YOLO txt files
     (one ``.txt`` per frame; empty file = no detection).
  5. Optional preview: live ``cv2.imshow`` stream + saved annotated mp4
     (``--preview`` + ``--preview-video``).

Example (interactive, annotate frame 0 of an mp4):
    python tools/box_tagger.py --video notebooks/videos/bedroom.mp4 \\
        --output-dir outputs/bedroom_boxes

Example (non-interactive, box given in ORIGINAL video pixels, xyxy):
    python tools/box_tagger.py --video notebooks/videos/bedroom.mp4 \\
        --output-dir outputs/bedroom_boxes \\
        --box 100 200 400 500 --ann-frame 0

Example (annotate clearest middle frame, track BOTH directions - for HOI):
    python tools/box_tagger.py --video myvideo.mp4 \\
        --output-dir outputs/myvideo_boxes \\
        --ann-frame 50 --bidirectional --box 100 200 400 500

Example (mp4 -> image_sequence + live preview + saved preview mp4):
    python tools/box_tagger.py --video sample_dataset/videos/cafe_test.mp4 \\
        --output-dir outputs/cafe_test_boxes \\
        --preview --preview-video outputs/cafe_test_boxes/preview.mp4

Output layout:
    <output-dir>/labels/<frame>.txt   # YOLO: "cls cx cy w h" normalized
    <output-dir>/data.yaml             # names/nc for training
    <output-dir>/images/               # only with --save-images
    <preview-mp4>                      # only with --preview-video
"""

import argparse
import os
import shutil

import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2_video_predictor
from sam2.utils.misc import mask_to_box


def parse_args():
    p = argparse.ArgumentParser(
        description="Draw-once SAM 2 video box tagger -> YOLO txt"
    )
    p.add_argument("--video", type=str, required=True,
                   help="mp4 path or JPEG-frame dir (like SAM 2 init_state). "
                        "mp4 input is extracted to a JPEG sequence first.")
    p.add_argument("--output-dir", type=str, required=True,
                   help="output dir for labels/ (+ data.yaml)")
    p.add_argument("--frame-dir", type=str, default=None,
                   help="where to write extracted JPEG frames for mp4 input. "
                        "Default: <parent-of-videos>/image_sequence "
                        "(e.g. sample_dataset/videos/x.mp4 -> "
                        "sample_dataset/image_sequence).")
    p.add_argument("--keep-frames", action="store_true",
                   help="do NOT clear --frame-dir before extracting "
                        "(default clears it first).")
    p.add_argument("--frame-stride", type=int, default=1,
                   help="write every Nth source frame (default 1 = all). "
                        "Streaming, so stride is for speed/label count, "
                        "not RAM.")
    p.add_argument("--jpeg-quality", type=int, default=95,
                   help="JPEG quality 1-100 for extracted frames.")
    p.add_argument("--max-frames", type=int, default=None,
                   help="only use first N frames (mp4: stop extraction early; "
                        "JPEG dir: use first N in sorted order). "
                        "Use for laptop-friendly tests, e.g. 200.")
    p.add_argument("--image-size", type=int, default=None,
                   help="override SAM 2 image_size (default from cfg, 1024). "
                        "Smaller = much less RAM, e.g. 512 (~4x less).")
    p.add_argument("--offload-state-to-cpu", action="store_true",
                   help="also offload SAM 2 state to CPU (slower, less VRAM).")
    p.add_argument("--async-frames", action="store_true",
                   help="lazy-load JPEG frames (less RAM, slower).")
    p.add_argument("--chunk-size", type=int, default=None,
                   help="process N frames at a time (e.g. 200) for hour-long "
                        "videos. Only C frames are ever loaded by SAM 2, so "
                        "RAM stays flat. Requires --ann-frame < chunk-size; "
                        "forward-only (no --bidirectional).")
    p.add_argument("--chunk-overlap", type=int, default=20,
                   help="overlap frames between chunks for seam continuity.")
    p.add_argument("--resume", action="store_true",
                   help="with --chunk-size: skip chunks whose kept labels "
                        "already exist; loads their boxes from txt for "
                        "reprompting the next chunk.")
    p.add_argument("--review-resumed", action="store_true",
                   help="with --resume: still prompt at each resumed chunk "
                        "boundary (Enter=keep old labels, r=redraw and "
                        "re-track this chunk).")
    p.add_argument("--preview", action="store_true",
                   help="show live cv2.imshow stream of tracked boxes.")
    p.add_argument("--preview-video", type=str, default=None,
                   help="save annotated preview mp4 to this path "
                        "(e.g. outputs/x/preview.mp4).")
    p.add_argument("--preview-fps", type=float, default=None,
                   help="fps for live preview + saved preview "
                        "(default: source video fps, else 30).")
    p.add_argument("--sam2_cfg", type=str,
                   default="configs/sam2.1/sam2.1_hiera_s.yaml",
                   help="SAM 2 config (matches tools/vos_inference.py style)")
    p.add_argument("--sam2_checkpoint", type=str,
                   default="./checkpoints/sam2.1_hiera_small.pt",
                   help="SAM 2 checkpoint")
    p.add_argument("--ann-frame", type=int, default=0,
                   help="annotation frame index to draw box(es) on")
    p.add_argument("--box", type=float, nargs=4, action="append",
                   metavar=("X1", "Y1", "X2", "Y2"),
                   help="box in ORIGINAL video pixels (xyxy). "
                        "Repeat for multiple objects. Omit for GUI select. "
                        "Order matches --cls-id order.")
    p.add_argument("--cls-id", type=int, action="append", default=None,
                   help="YOLO class id per --box, in order. Repeat per box; "
                        "a single value broadcasts to all boxes "
                        "(default 0). GUI: used as default prompt.")
    p.add_argument("--classes", type=str, default="person",
                   help="comma-separated class names for data.yaml")
    p.add_argument("--gui-class-order", type=int, nargs="*", default=None,
                   help="pre-declared class per drawn GUI box in order "
                        "(avoids per-box prompts). "
                        "Example: --gui-class-order 0 0 2.")
    p.add_argument("--init-boxes-file", type=str, default=None,
                   help="load initial (box, cls) prompts from JSON instead "
                        "of GUI/--box. Saved automatically to "
                        "<output-dir>/init_boxes.json on every run for "
                        "future automation.")
    p.add_argument("--keyframes", type=str, default=None,
                   help="multi-tag keyframes JSON "
                        "{classes, keyframes:[{frame, boxes:[{xyxy, cls}]}]}. "
                        "Accepts legacy init_boxes.json as one keyframe. "
                        "Chunked mode conditions each chunk on keyframes "
                        "inside it plus a reviewed boundary prompt.")
    p.add_argument("--review-size", type=int, nargs=2, default=[800, 600],
                   metavar=("W", "H"),
                   help="display size for GUI windows (default 800 600). "
                        "Display-only; saved boxes stay full-res.")
    p.add_argument("--no-review-preview", action="store_true",
                   help="boundary review without image window "
                        "(terminal prompt only).")
    p.add_argument("--bidirectional", action="store_true",
                   help="also propagate backwards from --ann-frame "
                        "(use when annotating a middle frame, e.g. HOI)")
    p.add_argument("--score-thresh", type=float, default=0.0,
                   help="mask logit threshold for foreground")
    p.add_argument("--min-area", type=float, default=25.0,
                   help="min mask pixel area to emit a box")
    p.add_argument("--save-images", action="store_true",
                   help="also dump per-frame JPGs to <output-dir>/images/ "
                        "(mp4 input; JPEG-dir input copies frames)")
    p.add_argument("--device", type=str, default="cuda",
                   help="cuda or cpu")
    p.add_argument("--offload-video-to-cpu", action="store_true",
                   help="save GPU memory (slower)")
    p.add_argument("--vos-optimized", action="store_true",
                   help="torch.compile predictor (faster after warmup)")
    p.add_argument("--no-interactive", action="store_true",
                   help="fail instead of opening GUI when --box is missing")
    return p.parse_args()


def is_jpeg_dir(path):
    return isinstance(path, str) and os.path.isdir(path)


def list_frame_basenames(jpeg_dir):
    names = [os.path.splitext(p)[0] for p in os.listdir(jpeg_dir)
             if os.path.splitext(p)[-1] in (".jpg", ".jpeg", ".JPG", ".JPEG")]
    names.sort(key=lambda p: int(os.path.splitext(p)[0]) if os.path.splitext(p)[0].isdigit() else p)
    return names


def default_frame_dir_for_video(video_path):
    """Sibling image_sequence dir: <parent-of-videos>/image_sequence.

    e.g. sample_dataset/videos/cafe_test.mp4 -> sample_dataset/image_sequence.
    Falls back to <video-parent>/<stem>_frames for non-videos layouts.
    """
    video_abs = os.path.abspath(video_path)
    parent = os.path.basename(os.path.dirname(video_abs))
    if parent == "videos":
        return os.path.join(os.path.dirname(os.path.dirname(video_abs)),
                            "image_sequence")
    stem = os.path.splitext(os.path.basename(video_abs))[0]
    return os.path.join(os.path.dirname(video_abs), stem + "_frames")


def clear_jpeg_dir(frame_dir):
    """Remove existing JPEG frames so a fresh extraction starts clean."""
    removed = 0
    for p in os.listdir(frame_dir):
        if os.path.splitext(p)[-1] in (".jpg", ".jpeg", ".JPG", ".JPEG"):
            os.remove(os.path.join(frame_dir, p))
            removed += 1
    return removed


def extract_video_to_frames(video_path, out_dir, stride=1, jpeg_quality=95,
                            clear=True, max_frames=None):
    """Extract mp4 -> JPEG sequence, streaming frame-by-frame (O(1) RAM).

    Returns (num_written, fps). Names are %05d.jpg from 00000 (SAM 2 style).
    Stops early after max_frames kept frames when set.
    """
    if stride < 1:
        raise ValueError("--frame-stride must be >= 1")
    os.makedirs(out_dir, exist_ok=True)
    if clear:
        n = clear_jpeg_dir(out_dir)
        print(f"cleared {n} old frame(s) in {out_dir}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 240:
        fps = 30.0
    encode = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    src_idx, kept = 0, 0
    while True:
        if max_frames is not None and kept >= max_frames:
            break
        ok, img = cap.read()
        if not ok or img is None:
            break
        if src_idx % stride == 0:
            cv2.imwrite(os.path.join(out_dir, f"{kept:05d}.jpg"), img, encode)
            kept += 1
        src_idx += 1
    cap.release()
    if kept == 0:
        raise RuntimeError(f"no frames extracted from {video_path}")
    print(f"extracted {kept} frame(s) (stride {stride}) to {out_dir}")
    return kept, float(fps)


def limit_jpeg_dir_to_first_n(frame_dir, max_frames, output_dir):
    """Copy first N JPEGs to a subset dir so SAM 2 only loads N frames."""
    basenames = list_frame_basenames(frame_dir)
    if len(basenames) <= max_frames:
        return frame_dir, basenames
    sub = os.path.join(output_dir, f"frames_sub_{max_frames}")
    os.makedirs(sub, exist_ok=True)
    clear_jpeg_dir(sub)
    kept_bases = basenames[:max_frames]
    for b in kept_bases:
        for ext in (".jpg", ".jpeg", ".JPG", ".JPEG"):
            src = os.path.join(frame_dir, b + ext)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(sub, b + ".jpg"))
                break
    print(f"limited {len(basenames)} -> {len(kept_bases)} frames in {sub}")
    return sub, kept_bases


def split_into_chunks(num_frames, chunk_size, overlap):
    """Split N frames into overlapping (start, end) windows."""
    if chunk_size is None or chunk_size <= 0:
        raise ValueError("--chunk-size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("--chunk-overlap must be in [0, chunk-size)")
    chunks = []
    start = 0
    while start < num_frames:
        end = min(start + chunk_size, num_frames)
        chunks.append((start, end))
        if end >= num_frames:
            break
        start = end - overlap
    return chunks


def yolo_lines_to_pixel_boxes(lines, video_w, video_h):
    """Invert pixel_boxes_to_yolo_lines; returns (boxes, clss) for resume."""
    boxes, clss = [], []
    for ln in lines:
        parts = ln.strip().split()
        if len(parts) != 5:
            continue
        cls, cx, cy, bw, bh = parts
        cls = int(float(cls))
        cx, cy, bw, bh = float(cx), float(cy), float(bw), float(bh)
        w_px = bw * video_w
        h_px = bh * video_h
        x1 = cx * video_w - w_px / 2.0
        y1 = cy * video_h - h_px / 2.0
        x2 = x1 + w_px - 1.0
        y2 = y1 + h_px - 1.0
        boxes.append([x1, y1, x2, y2])
        clss.append(cls)
    return boxes, clss


def yolo_lines_to_boxes_with_cls(lines, video_w, video_h):
    boxes, clss = yolo_lines_to_pixel_boxes(lines, video_w, video_h)
    return list(zip(boxes, clss))


def find_jpeg_ext(frame_dir, basename):
    for ext in (".jpg", ".jpeg", ".JPG", ".JPEG"):
        if os.path.exists(os.path.join(frame_dir, basename + ext)):
            return ext
    return None


def prepare_chunk_work_dir(full_frame_dir, full_basenames, start, end,
                           work_dir):
    """Copy one chunk's JPEGs as 00000.. so SAM 2 loads only C frames."""
    os.makedirs(work_dir, exist_ok=True)
    clear_jpeg_dir(work_dir)
    for local, g in enumerate(range(start, end)):
        base = full_basenames[g]
        ext = find_jpeg_ext(full_frame_dir, base)
        if ext is None:
            raise RuntimeError(f"missing frame {base} in {full_frame_dir}")
        shutil.copy(os.path.join(full_frame_dir, base + ext),
                    os.path.join(work_dir, f"{local:05d}.jpg"))


def get_reprompt_boxes(boxes_by_frame, search_from, search_to_excl,
                       fallback_boxes):
    """Last non-empty boxes in [search_from, search_to) for next chunk."""
    for g in range(search_to_excl - 1, search_from - 1, -1):
        b = boxes_by_frame.get(g)
        if b:
            return b
    return fallback_boxes


def resolve_box_classes(num_boxes, cls_ids, class_names, default_cls=0):
    """Map --cls-id list to per-box classes (single value broadcasts)."""
    if cls_ids is None or len(cls_ids) == 0:
        clss = [default_cls] * num_boxes
    elif len(cls_ids) == 1:
        clss = list(cls_ids) * num_boxes
    elif len(cls_ids) == num_boxes:
        clss = list(cls_ids)
    else:
        raise ValueError(f"--cls-id count {len(cls_ids)} must be 1 or match "
                         f"--box count {num_boxes}")
    nc = len(class_names)
    for c in clss:
        if not 0 <= c < nc:
            raise ValueError(f"class id {c} out of range for "
                             f"--classes {class_names} (nc={nc})")
    return clss


def prompt_class_for_box(idx, class_names, default_cls):
    listing = " ".join(f"{i}={n}" for i, n in enumerate(class_names))
    while True:
        raw = input(f"kept box {idx} [{listing}] -> class "
                    f"[{default_cls}] (Enter=default): ").strip()
        if raw == "":
            return default_cls
        try:
            c = int(raw)
        except ValueError:
            print(f"invalid '{raw}'; enter 0..{len(class_names)-1}")
            continue
        if 0 <= c < len(class_names):
            return c
        print(f"out of range; enter 0..{len(class_names)-1}")


def save_init_boxes(path, ann_frame, boxes_with_cls, class_names):
    """Persist initial prompts for future automation (always overwritten)."""
    import json
    payload = {
        "ann_frame": int(ann_frame),
        "classes": list(class_names),
        "boxes": [{"xyxy": [float(v) for v in b], "cls": int(c)}
                  for b, c in boxes_with_cls],
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"saved {len(boxes_with_cls)} init box(es) to {path}")


def load_init_boxes(path, class_names):
    """Load [(xyxy, cls)] + ann_frame; validates cls against class_names."""
    import json
    with open(path) as fh:
        payload = json.load(fh)
    ann_frame = int(payload.get("ann_frame", 0))
    items = payload.get("boxes", [])
    if not items:
        raise ValueError(f"no boxes in {path}")
    file_classes = payload.get("classes")
    if file_classes and list(file_classes) != list(class_names):
        print(f"WARNING: {path} classes {file_classes} != "
              f"--classes {class_names}; validating ids only")
    out = []
    for it in items:
        b = [float(v) for v in it["xyxy"]]
        c = int(it["cls"])
        if not 0 <= c < len(class_names):
            raise ValueError(f"class id {c} in {path} out of range")
        if len(b) != 4:
            raise ValueError(f"bad xyxy {b} in {path}")
        out.append((b, c))
    print(f"loaded {len(out)} init box(es) from {path} "
          f"(ann_frame {ann_frame})")
    return ann_frame, out


def load_keyframes_file(path, class_names):
    """Load keyframes; accepts keyframes.json or legacy init_boxes.json.

    Returns (sorted [(frame, [(xyxy, cls)])], file_classes_or_None).
    """
    import json
    with open(path) as fh:
        payload = json.load(fh)
    file_classes = payload.get("classes")
    if file_classes and list(file_classes) != list(class_names):
        print(f"WARNING: {path} classes {file_classes} != "
              f"--classes {class_names}; validating ids only")
    keyframes = []
    if "keyframes" in payload:
        for kf in payload["keyframes"]:
            frame = int(kf["frame"])
            items = []
            for it in kf.get("boxes", []):
                b = [float(v) for v in it["xyxy"]]
                c = int(it["cls"])
                if len(b) != 4:
                    raise ValueError(f"bad xyxy {b} in {path}")
                if not 0 <= c < len(class_names):
                    raise ValueError(f"class id {c} in {path} out of range")
                items.append((b, c))
            if items:
                keyframes.append((frame, items))
    elif "boxes" in payload:
        ann, items = load_init_boxes(path, class_names)
        keyframes.append((ann, items))
    else:
        raise ValueError(f"unrecognized keyframes format in {path}")
    if not keyframes:
        raise ValueError(f"no keyframes in {path}")
    keyframes.sort(key=lambda t: t[0])
    print(f"loaded {len(keyframes)} keyframe(s) from {path} "
          f"(frames {[f for f, _ in keyframes]})")
    return keyframes, file_classes


def save_keyframes(path, class_names, keyframes_by_frame):
    """Write {classes, keyframes:[{frame, boxes}]} sorted by frame."""
    import json
    payload = {
        "classes": list(class_names),
        "keyframes": [
            {"frame": int(f),
             "boxes": [{"xyxy": [float(v) for v in b], "cls": int(c)}
                       for b, c in boxes]}
            for f, boxes in sorted(keyframes_by_frame.items())
            if boxes
        ],
    }
    if not payload["keyframes"]:
        print(f"WARNING: no keyframes to save; leaving {path} untouched")
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"saved {len(payload['keyframes'])} keyframe(s) to {path}")


def review_chunk_boundary(full_frame_dir, global_frame, auto_with_cls,
                          class_names, default_cls, num_frames,
                          review_size=None, no_preview=False):
    """Show overlap frame with auto boxes; accept, redraw, skip, or quit.

    In-window keys (pumped waitKey loop, never blocks the message pump):
    Enter/Space=accept, r=redraw via GUI, s=skip, q/ESC=quit.
    With no_preview, falls back to a terminal prompt (no image window).
    Returns final [(xyxy, cls)] for this boundary (clamped by caller).
    """
    auto_with_cls = list(auto_with_cls)
    listing = " ".join(f"{i}={n}" for i, n in enumerate(class_names))
    if no_preview:
        print(f"frame {global_frame} auto {len(auto_with_cls)} box(es) "
              f"[{listing}]")
        while True:
            raw = input("[Enter]=accept, r=redraw, s=skip, q=quit: "
                        ).strip().lower()
            if raw in ("", "a", "accept"):
                return auto_with_cls
            if raw in ("r", "redraw"):
                return select_boxes_gui(
                    full_frame_dir, global_frame, class_names,
                    default_cls, None, review_size)
            if raw in ("s", "skip"):
                return []
            if raw in ("q", "quit"):
                raise RuntimeError("stopped by user at chunk boundary")
            print("enter empty (accept), r (redraw), s (skip), q (quit)")
    img = read_display_frame(full_frame_dir, global_frame)
    draw_boxes_on_img(img, auto_with_cls, class_names)
    win = (f"boundary {global_frame} auto={len(auto_with_cls)} "
           f"[Enter=accept r=redraw s=skip q=quit]")
    print(f"frame {global_frame} auto {len(auto_with_cls)} box(es) "
          f"[{listing}] -> keys in image window: Enter=accept, r=redraw, "
          f"s=skip, q=quit")
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    apply_window_size(win, review_size)
    try:
        while True:
            cv2.imshow(win, img)
            k = cv2.waitKey(50) & 0xFF
            if k in (13, 32):  # Enter / Space
                return auto_with_cls
            if k in (ord("r"), ord("R")):
                cv2.destroyWindow(win)
                return select_boxes_gui(
                    full_frame_dir, global_frame, class_names,
                    default_cls, None, review_size)
            if k in (ord("s"), ord("S")):
                return []
            if k in (27, ord("q"), ord("Q")):
                raise RuntimeError("stopped by user at chunk boundary")
    finally:
        try:
            cv2.destroyWindow(win)
        except Exception:
            pass


def read_display_frame(video_path, frame_idx):
    """Read one frame at ORIGINAL resolution as BGR (for box drawing)."""
    if is_jpeg_dir(video_path):
        names = [p for p in os.listdir(video_path)
                 if os.path.splitext(p)[-1] in (".jpg", ".jpeg", ".JPG", ".JPEG")]
        names.sort(key=lambda p: int(os.path.splitext(p)[0]) if os.path.splitext(p)[0].isdigit() else p)
        if not 0 <= frame_idx < len(names):
            raise ValueError(f"--ann-frame {frame_idx} out of range "
                             f"(0..{len(names)-1})")
        img = cv2.imread(os.path.join(video_path, names[frame_idx]))
        if img is None:
            raise RuntimeError(f"cannot read frame {names[frame_idx]}")
        return img
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, img = cap.read()
    cap.release()
    if not ok or img is None:
        raise ValueError(f"--ann-frame {frame_idx} out of range for {video_path}")
    return img


def select_boxes_gui(video_path, frame_idx, class_names=None,
                     default_cls=0, gui_class_order=None, win_size=None):
    """Draw boxes, return [(xyxy, cls)] using prompts or pre-declared order."""
    img = read_display_frame(video_path, frame_idx)
    class_names = class_names or ["object"]
    boxes = []
    win = "box_tagger: drag box, ENTER=keep, ESC=done"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    apply_window_size(win, win_size)
    while True:
        # selectROI blocks until ENTER/ESC; zero-area = finished/cancelled
        x, y, w, h = cv2.selectROI(win, img, showCrosshair=True)
        if w <= 0 or h <= 0:
            break
        box = [float(x), float(y), float(x + w), float(y + h)]
        idx = len(boxes) + 1
        if gui_class_order is not None:
            if idx - 1 >= len(gui_class_order):
                cv2.destroyWindow(win)
                raise ValueError(
                    f"--gui-class-order has {len(gui_class_order)} entries "
                    f"but drew box {idx}")
            cls = int(gui_class_order[idx - 1])
            if not 0 <= cls < len(class_names):
                cv2.destroyWindow(win)
                raise ValueError(f"gui class {cls} out of range")
        else:
            # Prompt in terminal (window stays open for next draw).
            listing = " ".join(f"{i}={n}"
                               for i, n in enumerate(class_names))
            print(f"kept box {idx}: {[x, y, x + w, y + h]} "
                  f"(draw another, or ESC to finish)")
            cls = prompt_class_for_box(idx, class_names, default_cls)
        boxes.append((box, cls))
    cv2.destroyWindow(win)
    if not boxes:
        raise RuntimeError("no boxes drawn; pass --box x1 y1 x2 y2 or draw one")
    return boxes


def clamp_box_xyxy(box, width, height):
    x1, y1, x2, y2 = box
    x1 = min(max(x1, 0.0), width - 1.0)
    y1 = min(max(y1, 0.0), height - 1.0)
    x2 = min(max(x2, 0.0), width - 1.0)
    y2 = min(max(y2, 0.0), height - 1.0)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def masks_to_pixel_boxes(masks, score_thresh, min_area):
    """Convert per-object video_res_masks (B,1,H,W) logits to pixel xyxy."""
    if masks is None:
        return []
    if torch.is_tensor(masks):
        logits = masks.detach().cpu()
    else:
        logits = torch.as_tensor(np.asarray(masks))
    if logits.numel() == 0:
        return []
    b = logits.shape[0]
    boxes = []
    for i in range(b):
        m = (logits[i:i + 1] > score_thresh)
        area = int(m.sum().item())
        if area < min_area:
            continue
        xyxy = mask_to_box(m).reshape(-1).tolist()  # inclusive x1,y1,x2,y2
        boxes.append([float(v) for v in xyxy])
    return boxes


def masks_to_boxes_with_cls(masks, obj_ids, obj_id_to_cls, score_thresh,
                            min_area):
    """Convert batch masks to [(xyxy, cls)] preserving obj-id mapping."""
    if masks is None:
        return []
    if torch.is_tensor(masks):
        logits = masks.detach().cpu()
    else:
        logits = torch.as_tensor(np.asarray(masks))
    if logits.numel() == 0:
        return []
    out = []
    for i, oid in enumerate(list(obj_ids)):
        m = (logits[i:i + 1] > score_thresh)
        if int(m.sum().item()) < min_area:
            continue
        xyxy = mask_to_box(m).reshape(-1).tolist()
        out.append(([float(v) for v in xyxy], int(obj_id_to_cls[int(oid)])))
    return out


def split_boxes_and_clss(boxes_with_cls):
    """Split [(xyxy, cls)] into (boxes, clss) for YOLO/preview/reprompt."""
    boxes = [b for b, _ in boxes_with_cls]
    clss = [int(c) for _, c in boxes_with_cls]
    return boxes, clss


def pixel_boxes_to_yolo_lines(pixel_boxes, box_clss, video_w, video_h):
    """Convert pixel xyxy (inclusive) boxes to YOLO normalized lines."""
    if isinstance(box_clss, int):
        clss = [box_clss] * len(pixel_boxes)
    else:
        clss = list(box_clss)
        if len(clss) != len(pixel_boxes):
            raise ValueError("boxes/clss length mismatch")
    lines = []
    for (x1, y1, x2, y2), cls in zip(pixel_boxes, clss):
        # mask_to_box uses inclusive max index; +1 converts to pixel extent
        cx = ((x1 + x2 + 1.0) / 2.0) / video_w
        cy = ((y1 + y2 + 1.0) / 2.0) / video_h
        bw = (x2 - x1 + 1.0) / video_w
        bh = (y2 - y1 + 1.0) / video_h
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        bw = min(max(bw, 0.0), 1.0)
        bh = min(max(bh, 0.0), 1.0)
        if bw <= 0 or bh <= 0:
            continue
        lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def masks_to_yolo_lines(masks, cls_id, score_thresh, min_area):
    """Convert per-object video_res_masks (B,1,H,W) logits to YOLO lines."""
    if masks is None:
        return []
    if torch.is_tensor(masks):
        logits = masks.detach().cpu()
    else:
        logits = torch.as_tensor(np.asarray(masks))
    if logits.numel() == 0:
        return []
    _, _, h, w = logits.shape
    pixel_boxes = masks_to_pixel_boxes(logits, score_thresh, min_area)
    return pixel_boxes_to_yolo_lines(pixel_boxes, cls_id, w, h)


PALETTE = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255),
           (255, 0, 255), (255, 255, 0)]


def apply_window_size(win, review_size):
    """Resize a WINDOW_NORMAL window; display-only, boxes stay full-res."""
    if not review_size:
        return
    try:
        w, h = int(review_size[0]), int(review_size[1])
        if w > 0 and h > 0:
            cv2.resizeWindow(win, w, h)
    except Exception:
        pass


def draw_boxes_on_img(img, boxes_with_cls, class_names=None):
    """Draw [(xyxy, cls)] in place with per-class colors + labels."""
    for oi, item in enumerate(boxes_with_cls):
        if (isinstance(item, (list, tuple)) and len(item) == 2
                and isinstance(item[0], (list, tuple))):
            (x1, y1, x2, y2), cls = item
            cls = int(cls)
        else:
            x1, y1, x2, y2 = item
            cls = 0
        color = PALETTE[cls % len(PALETTE)]
        cv2.rectangle(img, (int(round(x1)), int(round(y1))),
                      (int(round(x2)), int(round(y2))), color, 2)
        if class_names and 0 <= cls < len(class_names):
            tag = f"{class_names[cls]} {oi + 1}"
        else:
            tag = f"cls{cls} obj{oi + 1}"
        cv2.putText(img, tag, (int(round(x1)), max(0, int(round(y1)) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


def render_preview_stream(frame_dir, basenames, boxes_by_frame, video_w,
                          video_h, fps, show_live, save_path,
                          class_names=None, win_size=None):
    """Stream annotated frames: live cv2.imshow and/or saved mp4.

    boxes_by_frame values are [(xyxy, cls)] (multi-class) or legacy [xyxy].
    Color is per-class; label shows class name when available.
    Reads one JPEG at a time (O(1) RAM). Press q/ESC to stop early.
    """
    writer = None
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".",
                    exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (video_w, video_h))
        if not writer.isOpened():
            raise RuntimeError(f"cannot open VideoWriter for {save_path}")
        print(f"saving preview video to {save_path} @ {fps:.2f}fps")
    if show_live:
        cv2.namedWindow("box_tagger preview (q/ESC to quit)",
                        cv2.WINDOW_NORMAL)
        apply_window_size("box_tagger preview (q/ESC to quit)", win_size)
    delay = max(1, int(round(1000.0 / fps))) if fps and fps > 0 else 30
    ext_cache = {}
    try:
        for f, base in enumerate(basenames):
            if base not in ext_cache:
                found = None
                for ext in (".jpg", ".jpeg", ".JPG", ".JPEG"):
                    if os.path.exists(os.path.join(frame_dir, base + ext)):
                        found = ext
                        break
                ext_cache[base] = found
            ext = ext_cache[base]
            img = (cv2.imread(os.path.join(frame_dir, base + ext))
                   if ext else None)
            if img is None:
                print(f"WARNING: cannot read preview frame {base}; skipping")
                continue
            if img.shape[1] != video_w or img.shape[0] != video_h:
                img = cv2.resize(img, (video_w, video_h))
            draw_boxes_on_img(img, boxes_by_frame.get(f, []), class_names)
            if writer is not None:
                writer.write(img)
            if show_live:
                cv2.imshow("box_tagger preview (q/ESC to quit)", img)
                key = cv2.waitKey(delay) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    print("preview stopped by user")
                    break
    finally:
        if writer is not None:
            writer.release()
        if show_live:
            cv2.destroyAllWindows()


@torch.inference_mode()
def run_chunked(args, predictor, full_frame_dir, full_basenames, preview_fps,
                device, label_dir):
    """Process N frames in overlapping chunks; RAM stays O(chunk-size)."""
    import torch as _torch

    num_frames = len(full_basenames)
    if args.bidirectional:
        raise ValueError("--chunk-size is forward-only; drop --bidirectional")
    if not 0 <= args.ann_frame < num_frames:
        raise ValueError(f"--ann-frame {args.ann_frame} out of range "
                         f"(0..{num_frames-1})")
    if args.ann_frame >= args.chunk_size:
        raise ValueError(f"--ann-frame {args.ann_frame} must be < "
                         f"--chunk-size {args.chunk_size} in chunk mode "
                         f"(annotate inside the first chunk)")
    chunks = split_into_chunks(num_frames, args.chunk_size,
                               args.chunk_overlap)
    print(f"chunked mode: {num_frames} frames -> {len(chunks)} chunk(s) "
          f"(size {args.chunk_size}, overlap {args.chunk_overlap})")

    # Video dims from disk (SAM's video_w/h equals original dims).
    first_ext = find_jpeg_ext(full_frame_dir, full_basenames[0])
    probe = cv2.imread(os.path.join(full_frame_dir,
                                    full_basenames[0] + first_ext))
    if probe is None:
        raise RuntimeError(f"cannot read {full_basenames[0]}")
    video_h, video_w = probe.shape[:2]

    os.makedirs(label_dir, exist_ok=True)
    class_names = [c.strip() for c in args.classes.split(",") if c.strip()]
    if not class_names:
        raise ValueError("--classes must list at least one name")
    default_cls = (args.cls_id[0] if args.cls_id else 0)
    keyframes_by_frame = {}
    if args.keyframes:
        kf_list, _ = load_keyframes_file(args.keyframes, class_names)
    else:
        kf_list = []
    if kf_list:
        for kf_frame, kf_boxes in kf_list:
            if not 0 <= kf_frame < num_frames:
                print(f"WARNING: keyframe {kf_frame} out of range "
                      f"(0..{num_frames-1}); skipped")
                continue
            keyframes_by_frame[kf_frame] = [
                (clamp_box_xyxy(b, video_w, video_h), int(c))
                for b, c in kf_boxes]
        first_kf = min(keyframes_by_frame)
        if first_kf != args.ann_frame:
            print(f"using ann_frame {first_kf} from {args.keyframes} "
                  f"(CLI --ann-frame {args.ann_frame} ignored)")
            args.ann_frame = first_kf
        init_boxes_with_cls = list(keyframes_by_frame[first_kf])
    elif args.init_boxes_file:
        file_ann, loaded = load_init_boxes(args.init_boxes_file,
                                           class_names)
        if file_ann != args.ann_frame:
            print(f"using ann_frame {file_ann} from {args.init_boxes_file} "
                  f"(CLI --ann-frame {args.ann_frame} ignored)")
            args.ann_frame = file_ann
        init_boxes_with_cls = [
            (clamp_box_xyxy(b, video_w, video_h), int(c)) for b, c in loaded]
    elif args.box:
        raw_boxes = [clamp_box_xyxy(b, video_w, video_h)
                     for b in args.box]
        clss = resolve_box_classes(len(raw_boxes), args.cls_id,
                                   class_names, default_cls)
        init_boxes_with_cls = list(zip(raw_boxes, clss))
    elif args.no_interactive:
        raise ValueError("no --box given with --no-interactive")
    else:
        print(f"draw box(es) on frame {args.ann_frame} "
              f"(ENTER keeps, ESC finishes)")
        drawn = select_boxes_gui(full_frame_dir, args.ann_frame,
                                 class_names, default_cls,
                                 args.gui_class_order, args.review_size)
        init_boxes_with_cls = [
            (clamp_box_xyxy(b, video_w, video_h), int(c)) for b, c in drawn]
        for _, c in init_boxes_with_cls:
            if not 0 <= c < len(class_names):
                raise ValueError(f"class id {c} out of range")
    print(f"tracking {len(init_boxes_with_cls)} object(s) from frame "
          f"{args.ann_frame}")
    if not init_boxes_with_cls:
        raise RuntimeError("no boxes to track")
    if not 0 <= args.ann_frame < num_frames:
        raise ValueError(f"ann_frame {args.ann_frame} out of range "
                         f"(0..{num_frames-1})")
    if args.ann_frame >= args.chunk_size:
        raise ValueError(f"ann_frame {args.ann_frame} must be < "
                         f"--chunk-size {args.chunk_size} in chunk mode")
    save_init_boxes(os.path.join(args.output_dir, "init_boxes.json"),
                    args.ann_frame, init_boxes_with_cls, class_names)
    if args.ann_frame not in keyframes_by_frame:
        keyframes_by_frame[args.ann_frame] = init_boxes_with_cls
    save_keyframes(os.path.join(args.output_dir, "keyframes.json"),
                   class_names, keyframes_by_frame)

    boxes_by_frame = {}
    work_dir = os.path.join(args.output_dir, "chunk_work")
    autocast = (_torch.autocast(device_type="cuda", dtype=_torch.bfloat16)
                if device == "cuda"
                else _torch.cpu.amp.autocast(enabled=False))

    for ki, (s, e) in enumerate(chunks):
        keep_start = s if ki == 0 else s + args.chunk_overlap
        resume_override = None  # redrawn boundary -> re-track, don't skip
        # --- resume: kept labels already exist -> load, skip SAM ---
        if args.resume:
            missing = [g for g in range(keep_start, e)
                       if not os.path.exists(
                           os.path.join(label_dir,
                                        full_basenames[g] + ".txt"))]
            if not missing:
                for g in range(keep_start, e):
                    with open(os.path.join(
                            label_dir, full_basenames[g] + ".txt")) as fh:
                        lines = [ln for ln in fh.read().splitlines()
                                 if ln.strip()]
                    boxes_by_frame[g] = yolo_lines_to_boxes_with_cls(
                        lines, video_w, video_h)
                print(f"chunk {ki+1}/{len(chunks)} [{s}:{e}] resumed "
                      f"(kept {keep_start}:{e})")
                if (args.review_resumed and ki > 0
                        and not args.no_interactive
                        and s not in keyframes_by_frame):
                    auto = get_reprompt_boxes(
                        boxes_by_frame, s,
                        s + args.chunk_overlap or s + 1,
                        init_boxes_with_cls)
                    auto = [(clamp_box_xyxy(b, video_w, video_h), int(c))
                            for b, c in auto]
                    reviewed = review_chunk_boundary(
                        full_frame_dir, s, auto, class_names,
                        default_cls, num_frames, args.review_size,
                        args.no_review_preview)
                    reviewed = [
                        (clamp_box_xyxy(b, video_w, video_h), int(c))
                        for b, c in reviewed]
                    if reviewed and reviewed != auto:
                        print(f"chunk {ki+1}: boundary redrawn; "
                              f"re-tracking [{s}:{e}]")
                        keyframes_by_frame[s] = reviewed
                        save_keyframes(
                            os.path.join(args.output_dir, "keyframes.json"),
                            class_names, keyframes_by_frame)
                        resume_override = reviewed
                    else:
                        if reviewed:
                            keyframes_by_frame[s] = reviewed
                            save_keyframes(
                                os.path.join(
                                    args.output_dir, "keyframes.json"),
                                class_names, keyframes_by_frame)
                        continue
                else:
                    continue

        prepare_chunk_work_dir(full_frame_dir, full_basenames, s, e,
                               work_dir)
        inference_state = predictor.init_state(
            video_path=work_dir,
            offload_video_to_cpu=args.offload_video_to_cpu,
            offload_state_to_cpu=args.offload_state_to_cpu,
            async_loading_frames=args.async_frames,
        )
        if (inference_state["video_width"] != video_w
                or inference_state["video_height"] != video_h):
            print(f"WARNING: chunk {ki} SAM dims "
                  f"{inference_state['video_width']}x"
                  f"{inference_state['video_height']} != "
                  f"probe {video_w}x{video_h}; using probe for YOLO")
        if ki == 0:
            boundary = [(clamp_box_xyxy(b, video_w, video_h), int(c))
                        for b, c in init_boxes_with_cls]
            keyframes_by_frame[s + (args.ann_frame - s)] = boundary
            cond_global = {args.ann_frame: boundary}
            for kf in sorted(keyframes_by_frame):
                if s < kf < e and kf != args.ann_frame:
                    cond_global[kf] = keyframes_by_frame[kf]
            start_global = min(cond_global)
        else:
            if resume_override is not None:
                boundary = resume_override  # reviewed during resume; recorded
                record = False
            else:
                auto = get_reprompt_boxes(
                    boxes_by_frame, s, s + args.chunk_overlap or s + 1,
                    init_boxes_with_cls)
                if not auto:
                    print(f"WARNING: chunk {ki} has no reprompt boxes; "
                          f"reusing initial boxes (may re-acquire)")
                    auto = init_boxes_with_cls
                auto = [(clamp_box_xyxy(b, video_w, video_h), int(c))
                        for b, c in auto]
                if s in keyframes_by_frame:
                    boundary = keyframes_by_frame[s]
                    print(f"chunk {ki+1}: using filed keyframe at {s}")
                    record = False
                elif args.no_interactive:
                    boundary = auto
                    record = False
                else:
                    boundary = review_chunk_boundary(
                        full_frame_dir, s, auto, class_names, default_cls,
                        num_frames, args.review_size,
                        args.no_review_preview)
                    boundary = [
                        (clamp_box_xyxy(b, video_w, video_h), int(c))
                        for b, c in boundary]
                    record = True
            if boundary:
                keyframes_by_frame[s] = boundary
                cond_global = {s: boundary}
                for kf in sorted(keyframes_by_frame):
                    if s < kf < e:
                        cond_global[kf] = keyframes_by_frame[kf]
            else:
                print(f"chunk {ki+1}: boundary skipped; "
                      f"using interior keyframes only")
                cond_global = {kf: keyframes_by_frame[kf]
                               for kf in sorted(keyframes_by_frame)
                               if s < kf < e}
                record = False
                if not cond_global:
                    print(f"WARNING: chunk {ki} has no prompts; "
                          f"falling back to auto at {s}")
                    keyframes_by_frame[s] = auto
                    cond_global = {s: auto}
                    record = False
            if record:
                save_keyframes(
                    os.path.join(args.output_dir, "keyframes.json"),
                    class_names, keyframes_by_frame)
            start_global = min(cond_global)
        first_prompt = cond_global[min(cond_global)]
        obj_id_to_cls = {i + 1: int(c)
                         for i, (_, c) in enumerate(first_prompt)}
        for kf in sorted(cond_global):
            _, kc = split_boxes_and_clss(cond_global[kf])
            if len(kc) != len(first_prompt):
                print(f"WARNING: keyframe {kf} has {len(kc)} boxes vs "
                      f"{len(first_prompt)} at {min(cond_global)}; "
                      f"aligning by order")
                for i, c in enumerate(kc):
                    obj_id_to_cls[i + 1] = int(c)
        print(f"chunk {ki+1}/{len(chunks)} [{s}:{e}] "
              f"{len(cond_global)} cond frame(s) "
              f"{sorted(cond_global)} ...")
        with autocast:
            for kf in sorted(cond_global):
                local = int(kf - s)
                kf_boxes, _ = split_boxes_and_clss(cond_global[kf])
                for obj_idx, box in enumerate(kf_boxes):
                    predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=local,
                        obj_id=obj_idx + 1,
                        box=np.array(box, dtype=np.float32),
                    )
            for f_local, obj_ids, m in predictor.propagate_in_video(
                inference_state, start_frame_idx=int(start_global - s),
                reverse=False,
            ):
                g = s + int(f_local)
                if g < keep_start or g >= e:
                    continue  # drop overlap dup / out-of-chunk
                if g not in boxes_by_frame:
                    boxes_by_frame[g] = masks_to_boxes_with_cls(
                        m.cpu(), obj_ids, obj_id_to_cls,
                        args.score_thresh, args.min_area)
        # Write kept labels immediately (crash-resume friendly).
        for g in range(keep_start, e):
            boxes, clss = split_boxes_and_clss(
                boxes_by_frame.get(g, []))
            lines = pixel_boxes_to_yolo_lines(boxes, clss, video_w, video_h)
            with open(os.path.join(label_dir,
                                    full_basenames[g] + ".txt"), "w") as fh:
                fh.write("\n".join(lines))
                if lines:
                    fh.write("\n")
        print(f"chunk {ki+1}/{len(chunks)} done "
              f"(kept {keep_start}:{e})")
        del inference_state
        if device == "cuda" and _torch.cuda.is_available():
            _torch.cuda.empty_cache()
    save_keyframes(os.path.join(args.output_dir, "keyframes.json"),
                   class_names, keyframes_by_frame)
    # Cleanup scratch JPEGs (labels already written).
    if os.path.isdir(work_dir):
        clear_jpeg_dir(work_dir)
    return boxes_by_frame, video_w, video_h, num_frames


@torch.inference_mode()
def main():
    args = parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: cuda requested but not available, falling back to cpu")
        device = "cpu"

    # --- Input: mp4 -> extract to JPEG sequence first (streaming) ---
    preview_fps = args.preview_fps
    if is_jpeg_dir(args.video):
        frame_dir = args.video
        if args.max_frames is not None:
            sam2_path, _ = limit_jpeg_dir_to_first_n(
                frame_dir, args.max_frames, args.output_dir)
        else:
            sam2_path = frame_dir
        if preview_fps is None:
            preview_fps = 30.0
    else:
        frame_dir = args.frame_dir or default_frame_dir_for_video(args.video)
        _, src_fps = extract_video_to_frames(
            args.video, frame_dir,
            stride=args.frame_stride,
            jpeg_quality=args.jpeg_quality,
            clear=not args.keep_frames,
            max_frames=args.max_frames,
        )
        if preview_fps is None:
            preview_fps = float(src_fps)
        sam2_path = frame_dir

    overrides = []
    if args.image_size is not None:
        overrides.append(f"++model.image_size={int(args.image_size)}")
        print(f"overriding SAM 2 image_size to {args.image_size}")
    predictor = build_sam2_video_predictor(
        config_file=args.sam2_cfg,
        ckpt_path=args.sam2_checkpoint,
        device=device,
        vos_optimized=args.vos_optimized,
        hydra_overrides_extra=overrides,
    )

    label_dir = os.path.join(args.output_dir, "labels")
    os.makedirs(label_dir, exist_ok=True)

    # --- Chunked streaming path: O(chunk-size) RAM, hour-long safe ---
    if args.chunk_size is not None:
        if is_jpeg_dir(args.video):
            full_frame_dir = frame_dir
            all_bases = list_frame_basenames(frame_dir)
            full_basenames = (all_bases[:args.max_frames]
                              if args.max_frames is not None else all_bases)
        else:
            full_frame_dir = frame_dir  # already extracted (+max_frames)
            full_basenames = list_frame_basenames(frame_dir)
        if not full_basenames:
            raise RuntimeError(f"no frames found in {full_frame_dir}")
        boxes_by_frame, video_w, video_h, num_frames = run_chunked(
            args, predictor, full_frame_dir, full_basenames,
            preview_fps if preview_fps else 30.0, device, label_dir)
        basenames = full_basenames
        # Labels already written per-chunk; count positives for log.
        n_pos = 0
        for f in range(num_frames):
            p = os.path.join(label_dir, basenames[f] + ".txt")
            if os.path.exists(p) and os.path.getsize(p) > 0:
                n_pos += 1
        print(f"wrote {num_frames} txt files to {label_dir} "
              f"({n_pos} with >=1 box)")

        names = [c.strip() for c in args.classes.split(",") if c.strip()]
        with open(os.path.join(args.output_dir, "data.yaml"), "w") as fh:
            fh.write(f"train: {os.path.join(args.output_dir, 'images')}\n")
            fh.write(f"val: {os.path.join(args.output_dir, 'images')}\n")
            fh.write(f"nc: {len(names)}\n")
            fh.write(f"names: {names}\n")

        if args.save_images:
            img_dir = os.path.join(args.output_dir, "images")
            os.makedirs(img_dir, exist_ok=True)
            ok_count = 0
            for b in basenames[:num_frames]:
                ext = find_jpeg_ext(full_frame_dir, b)
                if ext:
                    shutil.copy(os.path.join(full_frame_dir, b + ext),
                                os.path.join(img_dir, b + ".jpg"))
                    ok_count += 1
            print(f"saved {ok_count} frames to {img_dir}")

        if args.preview or args.preview_video:
            class_names = [c.strip() for c in args.classes.split(",")
                           if c.strip()]
            render_preview_stream(full_frame_dir, basenames[:num_frames],
                                  boxes_by_frame, video_w, video_h,
                                  float(preview_fps if preview_fps else 30.0),
                                  show_live=bool(args.preview),
                                  save_path=args.preview_video,
                                  class_names=class_names,
                                  win_size=args.review_size)
        print("done")
        return

    inference_state = predictor.init_state(
        video_path=sam2_path,
        offload_video_to_cpu=args.offload_video_to_cpu,
        offload_state_to_cpu=args.offload_state_to_cpu,
        async_loading_frames=args.async_frames,
    )
    num_frames = inference_state["num_frames"]
    video_h = inference_state["video_height"]
    video_w = inference_state["video_width"]
    print(f"video: {num_frames} frames, {video_w}x{video_h}")

    if not 0 <= args.ann_frame < num_frames:
        raise ValueError(f"--ann-frame {args.ann_frame} out of range "
                         f"(0..{num_frames-1})")

    names = [c.strip() for c in args.classes.split(",") if c.strip()]
    if not names:
        raise ValueError("--classes must list at least one name")
    default_cls = (args.cls_id[0] if args.cls_id else 0)
    keyframes_by_frame = {}
    if args.keyframes:
        kf_list, _ = load_keyframes_file(args.keyframes, names)
        for kf_frame, kf_boxes in kf_list:
            if not 0 <= kf_frame < num_frames:
                print(f"WARNING: keyframe {kf_frame} out of range; skipped")
                continue
            keyframes_by_frame[kf_frame] = [
                (clamp_box_xyxy(b, video_w, video_h), int(c))
                for b, c in kf_boxes]
        if not keyframes_by_frame:
            raise ValueError(f"no in-range keyframes in {args.keyframes}")
        first_kf = min(keyframes_by_frame)
        if first_kf != args.ann_frame:
            print(f"using ann_frame {first_kf} from {args.keyframes} "
                  f"(CLI --ann-frame {args.ann_frame} ignored)")
            args.ann_frame = first_kf
        boxes_with_cls = list(keyframes_by_frame[first_kf])
    elif args.init_boxes_file:
        file_ann, loaded = load_init_boxes(args.init_boxes_file, names)
        if file_ann != args.ann_frame:
            print(f"using ann_frame {file_ann} from {args.init_boxes_file} "
                  f"(CLI --ann-frame {args.ann_frame} ignored)")
            args.ann_frame = file_ann
        boxes_with_cls = [(clamp_box_xyxy(b, video_w, video_h), int(c))
                          for b, c in loaded]
    elif args.box:
        raw_boxes = [clamp_box_xyxy(b, video_w, video_h)
                     for b in args.box]
        clss = resolve_box_classes(len(raw_boxes), args.cls_id, names,
                                   default_cls)
        boxes_with_cls = list(zip(raw_boxes, clss))
    elif args.no_interactive:
        raise ValueError("no --box given with --no-interactive")
    else:
        print(f"draw box(es) on frame {args.ann_frame} "
              f"(ENTER keeps, ESC finishes)")
        drawn = select_boxes_gui(sam2_path, args.ann_frame, names,
                                 default_cls, args.gui_class_order,
                                 args.review_size)
        boxes_with_cls = [(clamp_box_xyxy(b, video_w, video_h), int(c))
                          for b, c in drawn]
    if not 0 <= args.ann_frame < num_frames:
        raise ValueError(f"ann_frame {args.ann_frame} out of range "
                         f"(0..{num_frames-1})")
    print(f"tracking {len(boxes_with_cls)} object(s) from frame "
          f"{args.ann_frame}")
    save_init_boxes(os.path.join(args.output_dir, "init_boxes.json"),
                    args.ann_frame, boxes_with_cls, names)
    if keyframes_by_frame:
        save_keyframes(os.path.join(args.output_dir, "keyframes.json"),
                       names, keyframes_by_frame)
    else:
        save_keyframes(os.path.join(args.output_dir, "keyframes.json"),
                       names, {args.ann_frame: boxes_with_cls})
    boxes, _ = split_boxes_and_clss(boxes_with_cls)
    obj_id_to_cls = {i + 1: int(c) for i, (_, c) in enumerate(boxes_with_cls)}

    autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device == "cuda" else torch.cpu.amp.autocast(enabled=False))
    # Streaming: convert each mask to tiny (box, cls) immediately and
    # discard the full logits, so memory stays O(frames x boxes).
    boxes_by_frame = {}
    with autocast:
        extra_kfs = sorted(f for f in keyframes_by_frame
                           if f != args.ann_frame)
        if extra_kfs:
            print(f"conditioning on {1 + len(extra_kfs)} keyframe(s): "
                  f"{[args.ann_frame] + extra_kfs}")
        for obj_idx, box in enumerate(boxes):
            obj_id = obj_idx + 1  # SAM2 obj ids are 1-based here
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=args.ann_frame,
                obj_id=obj_id,
                box=np.array(box, dtype=np.float32),
            )
        for kf in extra_kfs:
            kf_boxes, kf_clss = split_boxes_and_clss(keyframes_by_frame[kf])
            if len(kf_boxes) != len(boxes):
                print(f"WARNING: keyframe {kf} has {len(kf_boxes)} boxes vs "
                      f"{len(boxes)} at {args.ann_frame}; aligning by order")
            for obj_idx, box in enumerate(kf_boxes):
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=int(kf),
                    obj_id=obj_idx + 1,
                    box=np.array(box, dtype=np.float32),
                )

        # Two passes when bidirectional (ann-frame in both; keep first).
        passes = [(args.ann_frame, False)]
        if args.bidirectional:
            passes.append((args.ann_frame, True))
        for start, reverse in passes:
            direction = "backward" if reverse else "forward"
            print(f"propagating {direction} from frame {start} ...")
            for f, obj_ids, m in predictor.propagate_in_video(
                inference_state, start_frame_idx=start, reverse=reverse
            ):
                if f not in boxes_by_frame:
                    boxes_by_frame[f] = masks_to_boxes_with_cls(
                        m.cpu(), obj_ids, obj_id_to_cls,
                        args.score_thresh, args.min_area)

    if len(boxes_by_frame) != num_frames:
        missing = sorted(set(range(num_frames)) - set(boxes_by_frame))
        print(f"WARNING: {len(missing)} frames without output "
              f"(e.g. {missing[:5]}); writing empty txt for them")

    # SAM input is always a JPEG dir now (mp4 was extracted above).
    # Use sam2_path (subset dir when --max-frames on JPEG input).
    basenames = list_frame_basenames(sam2_path)
    if len(basenames) != num_frames:
        print(f"WARNING: frame_dir has {len(basenames)} jpgs but SAM saw "
              f"{num_frames} frames; using SAM ordering for labels")
        if len(basenames) < num_frames:
            basenames += [f"{i:05d}" for i in
                          range(len(basenames), num_frames)]

    label_dir = os.path.join(args.output_dir, "labels")
    os.makedirs(label_dir, exist_ok=True)
    n_pos = 0
    for f in range(num_frames):
        boxes_f, clss_f = split_boxes_and_clss(boxes_by_frame.get(f, []))
        lines = pixel_boxes_to_yolo_lines(boxes_f, clss_f, video_w, video_h)
        n_pos += 1 if lines else 0
        with open(os.path.join(label_dir, basenames[f] + ".txt"), "w") as fh:
            fh.write("\n".join(lines))
            if lines:
                fh.write("\n")
    print(f"wrote {num_frames} txt files to {label_dir} "
          f"({n_pos} with >=1 box)")

    names = [c.strip() for c in args.classes.split(",") if c.strip()]
    with open(os.path.join(args.output_dir, "data.yaml"), "w") as fh:
        fh.write(f"train: {os.path.join(args.output_dir, 'images')}\n")
        fh.write(f"val: {os.path.join(args.output_dir, 'images')}\n")
        fh.write(f"nc: {len(names)}\n")
        fh.write(f"names: {names}\n")

    if args.save_images:
        img_dir = os.path.join(args.output_dir, "images")
        os.makedirs(img_dir, exist_ok=True)
        ok_count = 0
        for b in basenames[:num_frames]:
            for ext in (".jpg", ".jpeg", ".JPG", ".JPEG"):
                src = os.path.join(sam2_path, b + ext)
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(img_dir, b + ".jpg"))
                    ok_count += 1
                    break
        print(f"saved {ok_count} frames to {img_dir}")

    if args.preview or args.preview_video:
        render_preview_stream(sam2_path, basenames[:num_frames],
                              boxes_by_frame, video_w, video_h,
                              float(preview_fps),
                              show_live=bool(args.preview),
                              save_path=args.preview_video,
                              class_names=names,
                              win_size=args.review_size)

    print("done")


if __name__ == "__main__":
    main()
