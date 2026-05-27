# HW3 — Image Descriptors & Object Tracking

Level II (both groups) of `task3.pdf` — done by Popova Yelyzaveta.

## Setup

```bat
py -m venv .venv
.\.venv\Scripts\activate.bat
pip install -r requirements.txt
```

`opencv-contrib-python` is required (regular `opencv-python` ships without
MOSSE and the `cv2.legacy` namespace).

## Scripts

### `task_1.py` — Group 1: descriptor-based identification

Extends HW2: matches keypoint descriptors between `media/bing.png`
(high-precision reference) and `media/landsat.png` (operational query) and
converts the number of robust matches into an identification probability.

```bat
python task_1.py                          # SIFT, default bing vs landsat
python task_1.py --method orb             # binary descriptor, faster
python task_1.py --method both --no-show  # run SIFT and ORB, save images only
python task_1.py --ref path\to\a.png --query path\to\b.png
```

Outputs (in `outputs/`):
- `task1_matches_sift.png` / `task1_matches_orb.png` — match visualisation.

How the probability is computed (`matches_to_probability`):
- `score = good_matches / sqrt(min(keypoints_ref, keypoints_query))`
- `p = sigmoid((score − 0.5) * 10)`

The normalisation removes dependence on image size and feature density; the
midpoint and slope are chosen so a sparse match (~5 good points over a few
hundred keypoints) maps to ~0.5, and a dense match (~20+) saturates near 1.0.

### `task_2.py` — Group 2: object tracking in video

Compares MeanShift, CamShift, KCF, CSRT, and MOSSE on the same ROI of
`media/Download.mp4`. Each tracker runs from frame 0 with the bbox selected
once; metrics: FPS, frames where the tracker reported success, lost-frame
ratio. The empirical comparison answers the task's
"обґрунтувати та довести ефективність".

```bat
python task_2.py                                  # interactive ROI, all 5
python task_2.py --method csrt                    # one tracker
python task_2.py --bbox 320,180,80,60 --no-show   # headless run
```

Outputs (in `outputs/`):
- `task2_<method>.mp4` — annotated video per tracker.
- `task2_metrics.csv` — one row per tracker (only written in `--method all`).

### Choosing the best tracker

For the shahed-drone clip the object is small, moves fast, has a low-contrast
silhouette against the sky, and changes apparent size as it approaches the
camera. Expected behaviour:

| Tracker   | Strengths                              | Weaknesses on this clip            |
|-----------|----------------------------------------|------------------------------------|
| MeanShift | Simple, fast                           | Fixed bbox size; drifts on sky     |
| CamShift  | Adapts size/orientation                | Sensitive to similar-colour clutter|
| KCF       | Fast, good on rigid objects            | No scale adaptation, fails on occlusion |
| **CSRT**  | Scale + spatial reliability, robust    | Slower than KCF                    |
| MOSSE     | Very fast (~hundreds of FPS)           | No scale, weak on appearance change|

CSRT is the expected winner on quality (lowest `lost_ratio`); MOSSE on FPS. The
csv lets you confirm with numbers from the actual clip rather than vibes.

## Layout

```
HW/Image-Descriptors-Object-Tracking/
├── task_1.py
├── task_2.py
├── README.md
├── requirements.txt
├── task3.pdf
├── media/
│   ├── bing.png          # high-precision DZZ — reference
│   ├── landsat.png       # operational DZZ — query
│   └── Download.mp4      # shahed-drone clip
└── outputs/              # generated artifacts
```
