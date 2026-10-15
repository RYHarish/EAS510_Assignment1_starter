"""EAS 510 - Project 1 - Phase 2: the V2 rule set.

Rules 1-3 are kept exactly as in `rules.py`. Your job is to ADD Rule 4 that
targets the systematic weakness you diagnosed from results_v1_hard.txt.

Constraint: the rules' `out_of` weights must still sum to 100 so the Final
Score stays on a /100 scale (the format validator enforces "Final Score:
<s>/100" equals the sum of the block's rule scores).

Example split that keeps the original spirit: Rules 1-3 -> 25/25/40 and
Rule 4 -> 10 (or trim the losers more aggressively).
"""

import rules
import cv2
import numpy as np
from functools import lru_cache

#: Import the original rules and append your new one.
RULES = rules.RULES + ("rule4_edges",)

@lru_cache(maxsize=256)
def _features(path):
    img = rules._arr(path)
    if img is None:
      return None
    gray = rules._downscale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 640)
    kps, des = cv2.ORB_create(nfeatures=2000, nlevels=10, scaleFactor=1.2).detectAndCompute(gray, None)
    if des is None or len(kps)< 8:
      return None
    return np.float32([k.pt for k in kps]), des

def _sane(H):
    if H is None or not np.all(np.isfinite(H)):
      return False
    det = abs(np.linalg.det(H[:2,:2]))
    return 0.02 < det < 25 and abs(H[2,0]) < 2e-3 and abs(H[2,1]) < 2e-3

def rule4_edges(target, input_path):
    """TODO: replace with your Phase 2 rule that fixes the V1 weakness.

    Starter returns a no-op so the pipeline still runs before you implement it.
    Parity with rules.py rule dict: rule/name/fired/score/out_of/note/metric.
    """
    out = {
        "rule": 4,
        "name": "ORB",
        "fired": False,
        "score": 0,
        "out_of": 10,
        "note": "Feature Score 0.00 (0 inliers)",
        "metric": 0.0,
    }
    try:
      a, b = _features(input_path), _features(target["path"])
      if a is None or b is None:
        return out
      pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(a[1], b[1], k=2)
      good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
      inliers = 0
      if len(good) >= 8:
        src = a[0][[m.queryIdx for m in good]]
        dst = b[0][[m.trainIdx for m in good]]
        cv2.setRNGSeed(0)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if mask is not None and _sane(H):
          inliers = int(mask.sum())
        out["metric"] = round(min(1.0, inliers / 45), 3)
        out["note"] = f"Feature score {out['metric']:.2f} ({inliers} inliers)"
        if inliers >= 15:
          out["fired"] = True
          out["score"] = int(round(10 * min(1.0, (inliers - 10) / 35)))
    except Exception:
      pass
    return out
