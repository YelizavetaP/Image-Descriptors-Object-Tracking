"""Task 1 (HW3) — descriptor-based image identification.

Done by Popova Yelyzaveta.

Extends HW2 by matching keypoint descriptors between two remote-sensing views
of the KPI campus and converting the count of robust matches into an
identification probability.

  reference: media/bing.png    (high-precision DZZ — eтalon)
  query:     media/landsat.png (operational DZZ — to identify)

The descriptor (SIFT by default, ORB optional) is computed on the whole image:
since the KPI campus fills the frame, every keypoint belongs to the
identification object.
"""

import argparse
import math
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).parent
MEDIA = ROOT / "media"
OUTPUTS = ROOT / "outputs"


def extract_features(img, method):
    if method == "sift":
        detector = cv2.SIFT_create()
    elif method == "orb":
        detector = cv2.ORB_create(nfeatures=5000)
    else:
        raise ValueError(f"unknown method: {method}")
    kp, des = detector.detectAndCompute(img, None)
    return kp, des


def match_features(des1, des2, method, ratio=0.75):
    if method == "sift":
        index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
    else:
        # ORB descriptors are binary, use LSH
        index_params = dict(algorithm=6, table_number=12,
                            key_size=20, multi_probe_level=2)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    knn = matcher.knnMatch(des1, des2, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def matches_to_probability(n_good, n_keypoints, k=10.0, midpoint=0.5):
    """Logistic squashing of (good_matches / sqrt(min keypoints)).

    Raw count grows with image size and feature density, so normalise by
    sqrt of the smaller keypoint set. ``midpoint`` is the normalised score at
    which probability equals 0.5; ``k`` controls slope steepness.
    """
    if n_keypoints <= 0:
        return 0.0
    score = n_good / math.sqrt(n_keypoints)
    return 1.0 / (1.0 + math.exp(-(score - midpoint) * k))


def visualize(img1, kp1, img2, kp2, good, save_path=None):
    out = cv2.drawMatches(
        img1, kp1, img2, kp2, good, None,
        matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), out)
    return out


def run(ref_path, query_path, method, show=True, save=True):
    ref = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
    query = cv2.imread(str(query_path), cv2.IMREAD_GRAYSCALE)
    if ref is None:
        raise FileNotFoundError(ref_path)
    if query is None:
        raise FileNotFoundError(query_path)

    kp1, des1 = extract_features(ref, method)
    kp2, des2 = extract_features(query, method)

    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        print(f"[{method}] not enough features (ref={len(kp1)}, query={len(kp2)})")
        return None

    good = match_features(des1, des2, method)
    n_kp_min = min(len(kp1), len(kp2))
    prob = matches_to_probability(len(good), n_kp_min)

    print(f"\n=== {method.upper()} ===")
    print(f"keypoints   : ref={len(kp1)}  query={len(kp2)}")
    print(f"good matches: {len(good)}  (Lowe ratio 0.75)")
    print(f"identification probability: {prob:.3f}")

    save_path = (OUTPUTS / f"task1_matches_{method}.png") if save else None
    vis = visualize(ref, kp1, query, kp2, good, save_path=save_path)
    if save_path:
        print(f"saved: {save_path}")

    if show:
        cv2.imshow(f"{method}  matches={len(good)}  p={prob:.2f}", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return {
        "method": method,
        "n_keypoints_ref": len(kp1),
        "n_keypoints_query": len(kp2),
        "n_good_matches": len(good),
        "probability": prob,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Descriptor-based image identification (HW3 task 1).")
    ap.add_argument("--ref", default=str(MEDIA / "bing.png"),
                    help="reference (high-precision) image path")
    ap.add_argument("--query", default=str(MEDIA / "landsat.png"),
                    help="query (operational) image path")
    ap.add_argument("--method", choices=["sift", "orb", "both"], default="sift")
    ap.add_argument("--no-show", action="store_true",
                    help="do not open a preview window")
    ap.add_argument("--no-save", action="store_true",
                    help="do not write the matches PNG")
    args = ap.parse_args()

    methods = ["sift", "orb"] if args.method == "both" else [args.method]
    for m in methods:
        run(Path(args.ref), Path(args.query), method=m,
            show=not args.no_show, save=not args.no_save)


if __name__ == "__main__":
    main()
