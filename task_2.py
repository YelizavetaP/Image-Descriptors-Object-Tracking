"""Task 2 (HW3) — multi-tracker comparison for object tracking in video.

Done by Popova Yelyzaveta.

Initialises 5 OpenCV trackers on the same hand-selected ROI of the first frame
and reports per-tracker FPS, number of frames where the tracker reported
success, and the lost-frame ratio. The comparison is the empirical
justification for choosing the best tracker for the shahed-drone clip.

Trackers compared:
    meanshift, camshift   - HSV-histogram backprojection (from Lesson 5)
    kcf                   - Kernelised Correlation Filter
    csrt                  - Channel and Spatial Reliability Tracker
    mosse                 - Minimum Output Sum of Squared Error
"""

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).parent
MEDIA = ROOT / "media"
OUTPUTS = ROOT / "outputs"

ALL_METHODS = ["meanshift", "camshift", "kcf", "csrt", "mosse"]


def make_legacy_tracker(name):
    """Build an OpenCV tracker. The legacy namespace ships with
    opencv-contrib-python; we prefer it because MOSSE lives only there."""
    name = name.lower()
    if hasattr(cv2, "legacy"):
        if name == "kcf":
            return cv2.legacy.TrackerKCF_create()
        if name == "csrt":
            return cv2.legacy.TrackerCSRT_create()
        if name == "mosse":
            return cv2.legacy.TrackerMOSSE_create()
    if name == "kcf" and hasattr(cv2, "TrackerKCF_create"):
        return cv2.TrackerKCF_create()
    if name == "csrt" and hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    raise RuntimeError(
        f"Tracker '{name}' is unavailable — install opencv-contrib-python.")


def init_hsv_histogram(frame, bbox):
    x, y, w, h = bbox
    roi = frame[y:y + h, x:x + w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)),
                       np.array((180., 255., 255.)))
    hist = cv2.calcHist([hsv_roi], [0], mask, [16], [0, 180])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    return hist


def run_tracker(name, video_path, bbox, save_path=None, show=True):
    name = name.lower()
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("cannot read first frame")

    track_window = tuple(int(v) for v in bbox)
    legacy_obj = None
    hist = None

    if name in ("meanshift", "camshift"):
        hist = init_hsv_histogram(frame, track_window)
    else:
        legacy_obj = make_legacy_tracker(name)
        legacy_obj.init(frame, track_window)

    writer = None
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        writer = cv2.VideoWriter(str(save_path), fourcc, fps_in, size)

    frames = 0
    successes = 0
    start = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames += 1
        ok = False
        bbox_drawn = None
        rotated = None

        if name == "meanshift":
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], hist, [0, 180], 1)
            _, track_window = cv2.meanShift(dst, track_window, term_crit)
            x, y, w, h = track_window
            ok = w > 0 and h > 0 and int(dst[y:y + h, x:x + w].sum()) > 1000
            bbox_drawn = (x, y, w, h)
        elif name == "camshift":
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], hist, [0, 180], 1)
            rotated, track_window = cv2.CamShift(dst, track_window, term_crit)
            x, y, w, h = track_window
            ok = w > 0 and h > 0 and int(dst[y:y + h, x:x + w].sum()) > 1000
            bbox_drawn = (x, y, w, h)
        else:
            ok, new_bbox = legacy_obj.update(frame)
            if ok:
                bbox_drawn = tuple(int(v) for v in new_bbox)

        if ok:
            successes += 1
            if rotated is not None:
                pts = np.intp(cv2.boxPoints(rotated))
                cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            elif bbox_drawn is not None:
                x, y, w, h = bbox_drawn
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.putText(frame, "LOST", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.putText(frame, name.upper(),
                    (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if writer is not None:
            writer.write(frame)
        if show:
            cv2.imshow(name, frame)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break

    elapsed = time.perf_counter() - start
    cap.release()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    return {
        "method": name,
        "frames": frames,
        "successes": successes,
        "lost_ratio": (1 - successes / frames) if frames else 1.0,
        "elapsed_s": round(elapsed, 2),
        "fps": round(frames / elapsed, 2) if elapsed > 0 else 0.0,
    }


def parse_bbox(s):
    parts = [int(v) for v in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox must be x,y,w,h")
    return tuple(parts)


def select_roi(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("cannot read first frame for ROI selection")
    bbox = cv2.selectROI("select ROI — Enter to confirm, c to cancel",
                         frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    return tuple(int(v) for v in bbox)


def print_summary(results):
    print(f"\n{'method':<10}{'frames':>8}{'success':>10}"
          f"{'lost':>10}{'fps':>8}")
    for r in results:
        print(f"{r['method']:<10}{r['frames']:>8}{r['successes']:>10}"
              f"{r['lost_ratio'] * 100:>9.1f}%{r['fps']:>8.1f}")


def main():
    ap = argparse.ArgumentParser(
        description="Tracker comparison on a video (HW3 task 2).")
    ap.add_argument("--video", default=str(MEDIA / "Download.mp4"))
    ap.add_argument("--method", choices=ALL_METHODS + ["all"], default="all")
    ap.add_argument("--bbox", type=parse_bbox,
                    help="initial bbox x,y,w,h — skip interactive selectROI")
    ap.add_argument("--no-show", action="store_true",
                    help="run headless (no preview window)")
    ap.add_argument("--no-save", action="store_true",
                    help="do not write annotated MP4s")
    args = ap.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    bbox = args.bbox if args.bbox else select_roi(video_path)
    if bbox[2] <= 0 or bbox[3] <= 0:
        raise SystemExit("empty ROI — aborting")
    print(f"bbox = {bbox}")

    methods = ALL_METHODS if args.method == "all" else [args.method]
    results = []
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    for m in methods:
        save_path = None if args.no_save else (OUTPUTS / f"task2_{m}.mp4")
        try:
            res = run_tracker(m, video_path, bbox,
                              save_path=save_path, show=not args.no_show)
        except Exception as e:
            print(f"[{m}] FAILED: {e}")
            continue
        results.append(res)
        print(f"[{m}]  frames={res['frames']}  success={res['successes']}"
              f"  lost={res['lost_ratio']:.2%}  fps={res['fps']}")

    if len(results) > 1:
        csv_path = OUTPUTS / "task2_metrics.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"\nsummary written: {csv_path}")
        print_summary(results)


if __name__ == "__main__":
    main()
