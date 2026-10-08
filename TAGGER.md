# Video Box Tagger (SAM 2 → YOLO)

Draw box(es) once on a single frame. SAM 2 tracks them through the video.
Exports per-frame YOLO txt plus an optional annotated preview video.

Script: `tools/box_tagger.py`. Full flag reference: `tools/box_tagger_args.md`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python numpy hydra-core iopath
pip install git+https://github.com/facebookresearch/sam2.git
```

Checkpoints are not committed. Place `.pt` under `checkpoints/`:

```text
checkpoints/sam2.1_hiera_tiny.pt
checkpoints/sam2.1_hiera_small.pt
```

## How it works

```text
1. mp4 → JPEG sequence (streaming, O(1) RAM)
   sample_dataset/videos/cafe_test.mp4 → sample_dataset/image_sequence/
   Clears old JPEGs unless --keep-frames.
2. Annotate one frame (--ann-frame): GUI selectROI or --box.
3. SAM 2 tracks: add_new_points_or_box + propagate_in_video.
   Masks convert to boxes immediately; only tiny box lists kept.
4. Write labels/*.txt (empty file = no detection) + data.yaml.
5. Optional preview: live imshow + saved mp4.
```

## Quickstart

Interactive (draw with mouse, `ENTER` keeps, `ESC` finishes):

```bash
python tools/box_tagger.py --video sample_dataset/videos/cafe_test.mp4 --output-dir outputs/cafe_test_boxes --preview --preview-video outputs/cafe_test_boxes/preview.mp4 --offload-video-to-cpu
```

Non-interactive (box in original pixels, `x1 y1 x2 y2`):

```bash
python tools/box_tagger.py --video sample_dataset/videos/cafe_test.mp4 --output-dir outputs/cafe_test_boxes --box 100 200 400 500 --ann-frame 0 --no-interactive --preview --preview-video outputs/cafe_test_boxes/preview.mp4 --offload-video-to-cpu
```

Middle frame, both directions (e.g. clearest view mid-video):

```bash
python tools/box_tagger.py --video myvideo.mp4 --output-dir outputs/myvideo_boxes --box 100 200 400 500 --ann-frame 50 --bidirectional --no-interactive
```

## Multi-class

Prepare the list first:

```bash
--classes person,car,dog
```

Per-box ids in `--box` order (single value broadcasts):

```bash
--box 100 200 400 500 --box 600 100 700 300 --cls-id 0 --cls-id 2 --classes person,car,dog
```

GUI drawing prompts per box in the terminal:

```text
classes: 0=person 1=car 2=dog
kept box 1 [...] -> class [0] (Enter=default): 2
```

Skip prompts with a pre-declared order:

```bash
--gui-class-order 0 0 2
```

Preview colors are per class and labels show `name + obj#`.

## Laptop mode (avoid freezes)

SAM 2 loads all frames it processes: `1500 × 1024px ≈ 18 GB`.
Keep what SAM loads small:

```bash
python tools/box_tagger.py --video sample_dataset/videos/cafe_test.mp4 --output-dir outputs/cafe_test_small --sam2_cfg configs/sam2.1/sam2.1_hiera_t.yaml --sam2_checkpoint ./checkpoints/sam2.1_hiera_tiny.pt --frame-stride 5 --max-frames 200 --image-size 512 --offload-video-to-cpu --offload-state-to-cpu --preview --preview-video outputs/cafe_test_small/preview.mp4
```

Rules:

- `--frame-stride 5` = fewer labels, faster; not about RAM by itself.
- `--max-frames 200` = test subset first.
- `--image-size 512` = ~4x less RAM than 1024.
- Tiny checkpoint + both `--offload-*-to-cpu` for low VRAM.

## Hour-long videos (chunk streaming)

Process 200 frames at a time; RAM stays flat:

```bash
python tools/box_tagger.py --video sample_dataset/videos/cafe_test.mp4 --output-dir outputs/cafe_test_boxes --sam2_cfg configs/sam2.1/sam2.1_hiera_t.yaml --sam2_checkpoint ./checkpoints/sam2.1_hiera_tiny.pt --frame-stride 1 --chunk-size 200 --chunk-overlap 20 --image-size 512 --offload-video-to-cpu --offload-state-to-cpu --preview --preview-video outputs/cafe_test_boxes/preview.mp4 --resume
```

- Requires `--ann-frame < --chunk-size`, forward-only (no `--bidirectional`).
- Overlap reuses previous chunk's last boxes as prompts; kept output drops the dup.
- `--resume` skips finished chunks and reloads their labels for reprompting.

## Outputs

```text
<output-dir>/labels/<frame>.txt  # YOLO: "cls cx cy w h" normalized, empty = no detection
<output-dir>/data.yaml            # train/val/nc/names
<output-dir>/images/              # only with --save-images
<preview-mp4>                     # only with --preview-video
```
