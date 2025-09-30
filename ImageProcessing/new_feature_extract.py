# src/feature_extraction.py
import os
import cv2
import numpy as np
from typing import Tuple, List, Optional

def extract_orb_features_from_bgr_and_mask(
    img_bgr: np.ndarray,
    mask_u8: np.ndarray,
    n_features: int = 2000
) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray], np.ndarray]:
    """
    ORB features on a BGR image constrained by a binary (0/255) mask.
    Returns (keypoints, descriptors, visualization_bgr).
    """
    assert img_bgr is not None and img_bgr.ndim == 3 and img_bgr.shape[2] == 3, "img_bgr must be HxWx3 BGR"
    assert mask_u8 is not None and mask_u8.ndim == 2, "mask_u8 must be HxW 8-bit mask"

    # Grayscale + guarantee mask is 0/255
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = (mask_u8 > 0).astype(np.uint8) * 255

    # ORB tuned a bit for object features
    orb = cv2.ORB_create(
        nfeatures=int(n_features),
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=15,
        patchSize=31,
        fastThreshold=10
    )

    # Detect keypoints only in masked area
    keypoints = orb.detect(gray, mask=mask)
    if keypoints:
        keypoints = sorted(keypoints, key=lambda k: -k.response)[:n_features]

    keypoints, descriptors = orb.compute(gray, keypoints)

    # --- Visualization over a masked blend (object on white bg) ---
    background = np.full_like(img_bgr, 255)
    alpha = (mask.astype(np.float32) / 255.0)[..., None]  # HxWx1
    blended = (img_bgr.astype(np.float32) * alpha + background.astype(np.float32) * (1 - alpha)).astype(np.uint8)

    vis = blended.copy()
    if keypoints:
        for kp in keypoints:
            x, y = map(int, kp.pt)
            size = max(1, int(round(kp.size / 2)))
            ang_rad = (kp.angle if kp.angle is not None else 0.0) * np.pi / 180.0
            dx = int(round(size * np.cos(ang_rad)))
            dy = int(round(size * np.sin(ang_rad)))
            cv2.circle(vis, (x, y), size, (0, 255, 0), 1)
            cv2.line(vis, (x, y), (x + dx, y + dy), (0, 255, 0), 1)

    return keypoints, descriptors, vis


def save_feature_results(
    out_dir: str,
    base: str,
    vis_bgr: np.ndarray,
    descriptors: Optional[np.ndarray],
    keypoints: Optional[List[cv2.KeyPoint]] = None
) -> None:
    """
    Save visualization and (optionally) descriptors + keypoints.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_img = os.path.join(out_dir, f"{base}_features_MASKED.jpg")
    cv2.imwrite(out_img, vis_bgr)

    if descriptors is not None:
        np.save(os.path.join(out_dir, f"{base}_descriptors.npy"), descriptors)

    if keypoints is not None and len(keypoints) > 0:
        # Save compact keypoint info (x, y, size, angle, response, octave, class_id)
        kp_arr = np.array(
            [[kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id] for kp in keypoints],
            dtype=np.float32
        )
        np.save(os.path.join(out_dir, f"{base}_keypoints.npy"), kp_arr)
