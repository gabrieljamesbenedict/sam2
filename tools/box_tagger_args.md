# box_tagger.py — Arguments Reference

Source: `box_tagger.py:64` (`parse_args()`).

Draw box(es) once, track with SAM 2, export YOLO txt + optional preview video.

## Input / extraction

```python
--video VIDEO  # required, str
```
mp4 path or JPEG-frame dir. mp4 is extracted to a JPEG sequence first (streaming, `O(1)` RAM).

```python
--frame-dir FRAME_DIR  # default None
```
Where extracted JPEGs go. Default: `<parent-of-videos>/image_sequence`.
Example: `sample_dataset/videos/cafe_test.mp4` -> `sample_dataset/image_sequence/`.

```python
--keep-frames  # store_true, default False
```
Do NOT clear frame dir before extracting. Default clears `*.jpg` first.

```python
--frame-stride INT  # default 1
```
Write every Nth source frame. For speed / fewer labels, not RAM.

```python
--jpeg-quality INT  # default 95
```
JPEG quality 1-100 for extracted frames.

```python
--max-frames INT  # default None, e.g. 200
```
Only use first N frames. mp4: stops extraction early. JPEG dir: copies first N to `<output-dir>/frames_sub_N`. Use for laptop-friendly tests.

```python
--image-size INT  # default None (= cfg, usually 1024), e.g. 512
```
Override SAM 2 image_size. Smaller = much less RAM (~4x less at 512).

```python
--offload-state-to-cpu  # store_true
```
Also offload SAM 2 state to CPU (slower, less VRAM). Pair with `--offload-video-to-cpu` on laptops.

```python
--async-frames  # store_true
```
Lazy-load JPEG frames (less RAM, slower).

```python
--chunk-size INT  # default None, e.g. 200
```
Process N frames at a time for hour-long videos. Only C frames are ever loaded by SAM 2, so RAM stays flat. Requires `--ann-frame < chunk-size`. Forward-only (no `--bidirectional`).

```python
--chunk-overlap INT  # default 20
```
Overlap frames between chunks. First chunk keeps all, later chunks drop the overlapped dup and keep `[start+overlap:end)`. Larger = safer seams.

```python
--resume  # store_true, use with --chunk-size
```
Skip chunks whose kept `labels/*.txt` already exist; loads them back for reprompting the next chunk. Crash-safe reruns.

## Annotation

```python
--ann-frame INT  # default 0
```
Frame index to draw box(es) on (refers to extracted sequence index).

```python
--box X1 Y1 X2 Y2  # float x4, action=append
```
Box in original video pixels, `xyxy`. Repeat flag per object. Omit for GUI `cv2.selectROI` (`ENTER`=keep, `ESC`=done). Order matches `--cls-id` order.

```python
--cls-id INT  # action=append, default [0]
```
Class id per `--box`, in order. Repeat per box; a single value broadcasts to all. Example: `--box A --box B --cls-id 0 --cls-id 2`.

```python
--classes STR  # default "person", e.g. "person,car,dog"
```
Comma-separated names for `data.yaml`. Must satisfy `max(cls-id) < len(classes)`.

```python
--gui-class-order INT...  # default None, e.g. --gui-class-order 0 0 2
```
Pre-declared class per drawn GUI box in order (no prompts). Omit for terminal prompt `kept box N [0=person 1=car] -> class [0]:`.

```python
--no-interactive  # store_true
```
Fail if `--box` is missing instead of opening GUI.

```python
--bidirectional  # store_true
```
Propagate forward + backward from `--ann-frame`. Use for middle-frame annotation (e.g. HOI).

## SAM 2 / filtering

```python
--sam2_cfg STR  # default configs/sam2.1/sam2.1_hiera_s.yaml
--sam2_checkpoint STR  # default ./checkpoints/sam2.1_hiera_small.pt
--device STR  # default cuda (falls back to cpu)
--offload-video-to-cpu  # store_true, saves GPU memory, slower
--vos-optimized  # store_true, torch.compile predictor
--score-thresh FLOAT  # default 0.0, mask logit threshold
--min-area FLOAT  # default 25.0, min mask pixel area to emit a box
```

## Outputs

```python
--output-dir DIR  # required
```
Writes `labels/<frame>.txt` (per-line class) + `data.yaml`.

```python
--save-images  # store_true, also copy frames to <output-dir>/images/
```

Output layout:

```text
<output-dir>/labels/<frame>.txt  # YOLO: "cls cx cy w h" normalized, empty = no detection
<output-dir>/data.yaml            # train/val/nc/names
<output-dir>/images/              # only with --save-images
```

## Preview

```python
--preview  # store_true
```
Show live `cv2.imshow` stream. `q`/`ESC` quits early.

```python
--preview-video PATH  # default None, e.g. outputs/x/preview.mp4
```
Save annotated preview mp4 (`mp4v` codec).

```python
--preview-fps FLOAT  # default None (= source fps, else 30)
```
Fps for live preview + saved video.

## Examples

Laptop-friendly test (200 frames, stride 5, tiny model, 512px, both offloads):

```bash
python tools/box_tagger.py --video sample_dataset/videos/cafe_test.mp4 --output-dir outputs/cafe_test_small --sam2_cfg configs/sam2.1/sam2.1_hiera_t.yaml --sam2_checkpoint ./checkpoints/sam2.1_hiera_tiny.pt --frame-stride 5 --max-frames 200 --image-size 512 --offload-video-to-cpu --offload-state-to-cpu --preview --preview-video outputs/cafe_test_small/preview.mp4
```

Chunked full video, stride 1, hour-long safe (200-frame chunks, overlap 20):

```bash
python tools/box_tagger.py --video sample_dataset/videos/cafe_test.mp4 --output-dir outputs/cafe_test_boxes --sam2_cfg configs/sam2.1/sam2.1_hiera_t.yaml --sam2_checkpoint ./checkpoints/sam2.1_hiera_tiny.pt --frame-stride 1 --chunk-size 200 --chunk-overlap 20 --image-size 512 --offload-video-to-cpu --offload-state-to-cpu --preview --preview-video outputs/cafe_test_boxes/preview.mp4 --resume
```

Interactive full-quality:

```bash
python tools/box_tagger.py --video sample_dataset/videos/cafe_test.mp4 --output-dir outputs/cafe_test_boxes --preview --preview-video outputs/cafe_test_boxes/preview.mp4 --offload-video-to-cpu
```

Non-interactive:

```bash
python tools/box_tagger.py --video sample_dataset/videos/cafe_test.mp4 --output-dir outputs/cafe_test_boxes --box 100 200 400 500 --ann-frame 0 --preview --preview-video outputs/cafe_test_boxes/preview.mp4 --offload-video-to-cpu --no-interactive
```
