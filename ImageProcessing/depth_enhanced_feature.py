

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Depth + Colored Point Cloud from a side-by-side stereo image (NO RESIZING)
— SGBM/WLS tuned + sign-safe disparity + optional blur equalization.
Now with: segmentation + ORB feature extraction on the SEGMENTED rectified-left.
"""

import os
from typing import Tuple, Dict, Optional, List
import numpy as np
import cv2
import open3d as o3d

# ---- segmentation + features (from your modules) ----
from segmentation_fg import segment_foreground
from new_feature_extract import (
    extract_orb_features_from_bgr_and_mask,
    save_feature_results
)

# ========== EDIT THESE ==========
IMAGE_PATH = r"data/capture_images/WIN_20251003_10_08_51_Pro.jpg"
CALIB_NPZ  = r"opencv_stereo_params2.npz"
OUTPUT_DIR = r"data/output/depth_binarysgbm"

PROFILE   = "lowtex"   # "balanced" | "detail" | "lowtex"
NUM_DISP  = 256        # multiple of 16
USE_WLS   = True
DO_HIST_MATCH = True
SMOOTH_PREVIEW = True

# New toggles
MATCH_BLUR     = True   # equalize blur between views
ALLOW_NEGATIVE = False  # let SGBM search a small negative range
AUTO_FLIP_SIGN = True   # auto-flip disparity to positive if needed
MIRROR_RIGHT   = False  # if capture path mirrors the right half
SAVE_ZUP = True

# ORB feature extraction on segmented rectified-left
DO_FEATURES = True
N_FEATURES  = 2000

# --- Sparsity controls for point cloud ---
POINT_STRIDE = 3              # 1=all pixels, 2=every 2nd, 3=every 3rd …
VOXEL_DOWNSAMPLE_M = 0.003    # voxel size in meters for downsample; 0 disables
RANDOM_KEEP_RATIO = 1.0       # random thinning after stride (0..1]

CALIB_T_UNIT = "mm"           # "mm" or "m"
PLY_Z_MIN = 20.0
PLY_Z_MAX = 20000.0

# World-frame to save point clouds in:
#   "camera": X right, Y down, Z forward (OpenCV camera frame)
#   "z_up"  : X right, Y forward, Z up
#   "y_up"  : X right, Y up, Z forward
EXPORT_FRAME = "y_up"
# ===============================

# Feature extraction outputs (from feature_extraction.py)
FEATURE_EXTRACT_DIR = r"data/output/feature_extraction"

# Plane refinement parameters
ENABLE_PLANE_REFINEMENT = True
PLANE_MIN_SAMPLES = 20
PLANE_RESID_THRESH = 1.8
PLANE_BLEND = 0.85
ANCHOR_WEIGHT = 50.0


def load_npz_calibration(npz_path: str) -> Dict[str, np.ndarray | tuple]:
    if not os.path.exists(npz_path):
        raise FileNotFoundError(npz_path)
    p = np.load(npz_path, allow_pickle=False)
    K1 = np.array(p["camera_matrix_1"])
    D1 = np.array(p["dist_coeffs_1"]).reshape(-1, 1)
    K2 = np.array(p["camera_matrix_2"])
    D2 = np.array(p["dist_coeffs_2"]).reshape(-1, 1)
    R  = np.array(p["R"])
    T  = np.array(p["T"]).reshape(3, 1)
    size = tuple(np.array(p["image_size"]).tolist())
    if size[0] < size[1]: size = (size[1], size[0])
    return dict(K1=K1, D1=D1, K2=K2, D2=D2, R=R, T=T, size=size)

def split_sbs(path: str) -> Tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    h, w = img.shape[:2]
    if w % 2 != 0:
        raise ValueError(f"Width must be even for SBS. Got {w}")
    half = w // 2
    L, R = img[:, :half].copy(), img[:, half:].copy()
    return L, R

def rectify_no_resize(left_bgr: np.ndarray, right_bgr: np.ndarray, calib: Dict
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Wc, Hc = calib["size"]
    Hl, Wl = left_bgr.shape[:2]; Hr, Wr = right_bgr.shape[:2]
    if (Wl, Hl) != (Wc, Hc) or (Wr, Hr) != (Wc, Hc):
        raise ValueError(
            f"Input halves size mismatch with calibration.\n"
            f"  Left:  {(Wl, Hl)}  Right: {(Wr, Hr)}  Calib: {(Wc, Hc)}"
        )
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        calib["K1"], calib["D1"], calib["K2"], calib["D2"],
        (Wc, Hc), calib["R"], calib["T"],
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )
    map1L, map2L = cv2.initUndistortRectifyMap(calib["K1"], calib["D1"], R1, P1, (Wc, Hc), cv2.CV_16SC2)
    map1R, map2R = cv2.initUndistortRectifyMap(calib["K2"], calib["D2"], R2, P2, (Wc, Hc), cv2.CV_16SC2)
    rectL = cv2.remap(left_bgr,  map1L, map2L, cv2.INTER_LINEAR)
    rectR = cv2.remap(right_bgr, map1R, map2R, cv2.INTER_LINEAR)
    return rectL, rectR, P1, P2, Q

def get_profile(name: str) -> Dict:
    name = (name or "balanced").lower()
    if name == "detail":
        return dict(mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY, bs=3, uniq=1, p2_mult=48, disp12MaxDiff=2,
                    wls_lambda=80000.0, wls_sigma=1.9, clahe_clip=1.5, clahe_tile=(12, 12))
    elif name == "lowtex":
        return dict(mode=cv2.STEREO_SGBM_MODE_HH, bs=7, uniq=12, p2_mult=48, disp12MaxDiff=1,
                    wls_lambda=32000.0, wls_sigma=1.2, clahe_clip=1.5, clahe_tile=(12, 12))
    else:
        return dict(mode=cv2.STEREO_SGBM_MODE_HH, bs=5, uniq=14, p2_mult=64, disp12MaxDiff=2,
                    wls_lambda=28000.0, wls_sigma=1.2, clahe_clip=1.5, clahe_tile=(12, 12))

def _sharpness_var_laplacian(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def equalize_blur(grayL: np.ndarray, grayR: np.ndarray,
                  max_sigma: float = 2.5, step: float = 0.25, tol_ratio: float = 0.15):
    vL0, vR0 = _sharpness_var_laplacian(grayL), _sharpness_var_laplacian(grayR)
    if abs(vL0 - vR0) / max(vL0, vR0, 1e-6) < tol_ratio:
        return grayL, grayR, (vL0, vR0)
    if vL0 > vR0:
        src, other, is_left = grayL, grayR, True
    else:
        src, other, is_left = grayR, grayL, False
    best = src
    sigma = 0.25
    target = _sharpness_var_laplacian(other) * (1.0 + tol_ratio)
    while sigma <= max_sigma and _sharpness_var_laplacian(best) > target:
        best = cv2.GaussianBlur(src, (0, 0), sigma); sigma += step
    return (best, grayR, (vL0, vR0)) if is_left else (grayL, best, (vL0, vR0))

def build_sgbm(num_disp: int, mode: int, bs: int, uniq: int, p2_mult: int,
               disp12MaxDiff: int, min_disp: int) -> cv2.StereoSGBM:
    nd = int(np.ceil(num_disp / 16.0)) * 16
    bs = bs if bs % 2 == 1 else bs + 1
    cn = 1
    P1 = 8  * cn * bs * bs
    P2 = p2_mult * cn * bs * bs
    return cv2.StereoSGBM_create(
        minDisparity=min_disp, numDisparities=nd, blockSize=bs,
        P1=P1, P2=P2, disp12MaxDiff=disp12MaxDiff, uniquenessRatio=uniq,
        speckleWindowSize=200, speckleRange=2, preFilterCap=63, mode=mode
    )

def match_histogram_to(src_gray: np.ndarray, ref_gray: np.ndarray) -> np.ndarray:
    src = src_gray.ravel(); ref = ref_gray.ravel()
    s_values, bin_idx, s_counts = np.unique(src, return_inverse=True, return_counts=True)
    r_values, r_counts = np.unique(ref, return_counts=True)
    s_quant = np.cumsum(s_counts).astype(np.float64); s_quant /= s_quant[-1]
    r_quant = np.cumsum(r_counts).astype(np.float64); r_quant /= r_quant[-1]
    interp = np.interp(s_quant, r_quant, r_values)
    return interp[bin_idx].reshape(src_gray.shape).astype(np.uint8)

def _dominant_sign(d: np.ndarray) -> float:
    H, W = d.shape
    y0, y1 = int(0.25*H), int(0.75*H); x0, x1 = int(0.25*W), int(0.75*W)
    roi = d[y0:y1, x0:x1]
    v = roi[np.isfinite(roi)]; v = v[np.abs(v) > 0.1]
    if v.size < 500: return 0.0
    med = float(np.median(v)); return 1.0 if med >= 0 else -1.0

def compute_disparity(rectL_bgr: np.ndarray,
                      rectR_bgr: np.ndarray,
                      num_disp: int,
                      prof: Dict,
                      use_wls: bool) -> np.ndarray:
    grayL = cv2.cvtColor(rectL_bgr, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rectR_bgr, cv2.COLOR_BGR2GRAY)
    grayL = cv2.bilateralFilter(grayL, 9, 50, 50)
    grayR = cv2.bilateralFilter(grayR, 9, 50, 50)
    clahe = cv2.createCLAHE(clipLimit=prof["clahe_clip"], tileGridSize=prof["clahe_tile"])
    grayL = clahe.apply(grayL); grayR = clahe.apply(grayR)
    if DO_HIST_MATCH:
        grayR = match_histogram_to(grayR, grayL)
    if MATCH_BLUR:
        grayL, grayR, (vL0, vR0) = equalize_blur(grayL, grayR)
        print(f"[blur-match] LapVar L/R ~ {vL0:.1f}/{vR0:.1f}")
    neg_window = -(min(64, num_disp // 4)) if ALLOW_NEGATIVE else 0
    sgbm = build_sgbm(num_disp, prof["mode"], prof["bs"], prof["uniq"],
                      prof["p2_mult"], prof["disp12MaxDiff"], min_disp=neg_window)
    dispL16 = sgbm.compute(grayL, grayR)
    if use_wls and hasattr(cv2, "ximgproc"):
        right_matcher = cv2.ximgproc.createRightMatcher(sgbm)
        dispR16 = right_matcher.compute(grayR, grayL)
        wls = cv2.ximgproc.createDisparityWLSFilter(matcher_left=sgbm)
        wls.setLambda(float(prof["wls_lambda"]))
        wls.setSigmaColor(float(prof["wls_sigma"]))
        disp = wls.filter(dispL16, rectL_bgr, disparity_map_right=dispR16).astype(np.float32) / 16.0
    else:
        disp = dispL16.astype(np.float32) / 16.0
    if AUTO_FLIP_SIGN and ALLOW_NEGATIVE:
        sgn = _dominant_sign(disp)
        if sgn < 0:
            disp = -disp
            print("[sign] disparity flipped to match x_left - x_right > 0 convention")
    d16 = (disp * 16.0).astype(np.int16)
    cv2.filterSpeckles(d16, 0, 0, 2 * 16)
    disp = d16.astype(np.float32) / 16.0
    disp[disp < 0] = 0.0
    disp = cv2.medianBlur(disp, 3)
    return disp

def disparity_to_depth_Q(disp: np.ndarray, Q: np.ndarray) -> np.ndarray:
    pts3d = cv2.reprojectImageTo3D(disp.astype(np.float32), Q)
    Z = pts3d[..., 2].astype(np.float32)
    Z[~np.isfinite(Z)] = 0.0
    Z[disp <= 0] = 0.0
    return Z

def disparity_to_depth_fxB(disp: np.ndarray, P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    fx = float(P1[0, 0])
    baseline = -float(P2[0, 3]) / fx
    if CALIB_T_UNIT.lower() == "mm":
        baseline *= 0.001
    Z = np.zeros_like(disp, dtype=np.float32)
    m = disp > 0
    if baseline != 0:
        Z[m] = (fx * baseline) / disp[m]
    return Z

def disp_to_vis(disp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vis = np.zeros_like(disp, dtype=np.uint8)
    m = disp > 0
    if np.any(m):
        v = disp[m].astype(np.float32)
        lo, hi = np.percentile(v, 1.0), np.percentile(v, 99.0)
        hi = max(hi, lo + 1e-6)
        vis[m] = np.clip(255.0 * (disp[m] - lo) / (hi - lo), 0, 255).astype(np.uint8)
        if SMOOTH_PREVIEW:
            vis = cv2.GaussianBlur(vis, (3, 3), 0)
    try:
        col = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
    except Exception:
        col = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    return vis, col

def _orientation_matrix(frame: str) -> np.ndarray:
    f = (frame or "camera").lower()
    if f == "camera": return np.eye(3, dtype=float)
    if f == "z_up":   return np.array([[1,0,0],[0,0,1],[0,-1,0]], dtype=float)
    if f == "y_up":   return np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=float)
    raise ValueError("EXPORT_FRAME must be 'camera', 'z_up', or 'y_up'")

# ---- Masked PLY export (uses foreground mask) ----
def save_point_cloud_from_disparity_masked(rect_left_bgr: np.ndarray,
                                           disparity: np.ndarray,
                                           Q: np.ndarray,
                                           out_ply_path: str,
                                           seg_mask_u8: np.ndarray,
                                           calib_t_unit: str = "mm",
                                           z_min: float = 50.0,
                                           z_max: float = 5000.0) -> None:
    pts3d = cv2.reprojectImageTo3D(disparity.astype(np.float32), Q)
    Z = pts3d[..., 2]
    finite3d = np.isfinite(pts3d).all(axis=2)
    disp_ok = disparity > 0
    seg_ok  = seg_mask_u8 > 0
    mask = disp_ok & finite3d & seg_ok & (Z > float(z_min)) & (Z < float(z_max))
    if not np.any(mask):
        mask = disp_ok & finite3d & seg_ok
        if not np.any(mask):
            raise RuntimeError("Segmentation removed all points; relax thresholds or check mask.")

    # pixel stride decimation
    if POINT_STRIDE > 1:
        H, W = mask.shape
        stride_mask = np.zeros_like(mask, dtype=bool)
        stride_mask[0:H:POINT_STRIDE, 0:W:POINT_STRIDE] = True
        mask = mask & stride_mask
        if not np.any(mask):
            raise RuntimeError("All points removed by POINT_STRIDE; lower the stride.")

    pts = pts3d[mask].astype(np.float32)
    cols_rgb = (rect_left_bgr[mask].astype(np.float32) / 255.0)[:, ::-1]

    if calib_t_unit.lower() == "mm":
        pts *= 0.001  # mm -> m

    # optional random thinning
    if 0.0 < RANDOM_KEEP_RATIO < 1.0:
        N = pts.shape[0]
        keep = max(1, int(N * float(RANDOM_KEEP_RATIO)))
        idx = np.random.choice(N, size=keep, replace=False)
        pts = pts[idx]; cols_rgb = cols_rgb[idx]

    Rw = _orientation_matrix(EXPORT_FRAME)
    pts = (Rw @ pts.T).T

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(cols_rgb.astype(np.float64))

    if VOXEL_DOWNSAMPLE_M and VOXEL_DOWNSAMPLE_M > 0:
        pcd = pcd.voxel_down_sample(float(VOXEL_DOWNSAMPLE_M))

    os.makedirs(os.path.dirname(out_ply_path) or ".", exist_ok=True)
    ok = o3d.io.write_point_cloud(out_ply_path, pcd, write_ascii=True, print_progress=True)
    if not ok:
        raise IOError(f"Failed to save PLY: {out_ply_path}")
    print(f"[OK] Masked point cloud saved -> {out_ply_path}  (points: {len(pcd.points)})")


def load_precomputed_keypoints(base: str) -> Optional[np.ndarray]:
    kp_path = os.path.join(FEATURE_EXTRACT_DIR, f"{base}_keypoints.npy")
    if not os.path.exists(kp_path):
        return None
    try:
        arr = np.load(kp_path)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return arr
    except Exception:
        return None
    return None


def extract_anchors_from_keypoints(rectL, rectR, keypoints_arr, min_disp, num_disp, template_thresh=0.6):
    """Template-match each keypoint patch from left to the right image along the epipolar line.
    Returns list of (x,y,disp)
    """
    anchors = []
    if keypoints_arr is None or keypoints_arr.size == 0:
        return anchors
    gL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    gR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)
    h, w = gL.shape
    for row in keypoints_arr:
        x = int(round(float(row[0]))); y = int(round(float(row[1])))
        if x < 8 or x >= w-8 or y < 8 or y >= h-8:
            continue
        half = 11 if min(h, w) > 300 else 7
        tpl = gL[y-half:y+half+1, x-half:x+half+1]
        if tpl.size == 0 or tpl.shape[0] < 3 or tpl.shape[1] < 3:
            continue
        sx_min = max(0, x - int(num_disp) - 8)
        sx_max = max(0, x - max(1, int(min_disp)))
        if sx_max <= sx_min:
            continue
        search = gR[y-half:y+half+1, sx_min:sx_max]
        if search.size == 0 or search.shape[1] < tpl.shape[1]:
            continue
        try:
            res = cv2.matchTemplate(search, tpl, cv2.TM_CCOEFF_NORMED)
            _, maxv, _, maxloc = cv2.minMaxLoc(res)
            if maxv < template_thresh:
                continue
            best_x = sx_min + maxloc[0] + tpl.shape[1]//2
            disp = x - best_x
            if disp <= min_disp or disp > min_disp + num_disp:
                continue
            anchors.append((x, y, float(disp)))
        except Exception:
            continue
    return anchors


def fit_plane_ls(xs, ys, ds, anchors=None):
    A = np.column_stack([xs, ys, np.ones_like(xs)])
    b = ds
    if anchors is not None and len(anchors) > 0:
        akp = np.array([[a[0], a[1], 1.0] for a in anchors], dtype=np.float32)
        ad = np.array([a[2] for a in anchors], dtype=np.float32)
        A = np.vstack([A, akp]); b = np.concatenate([b, ad])
    try:
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        return float(x[0]), float(x[1]), float(x[2])
    except Exception:
        return None


def refine_disparity_grid(disp, rectL, fg_mask, anchors, debug_dir=None):
    h, w = disp.shape
    out = disp.copy()
    region = max(32, int(np.sqrt(1024)))
    residual_map = np.full_like(disp, np.nan, dtype=np.float32)
    for y0 in range(0, h, region):
        for x0 in range(0, w, region):
            y1 = min(h, y0 + region); x1 = min(w, x0 + region)
            sub_disp = disp[y0:y1, x0:x1]
            coords = np.argwhere(sub_disp > 0)
            if coords.shape[0] < PLANE_MIN_SAMPLES:
                continue
            ys = coords[:, 0] + y0
            xs = coords[:, 1] + x0
            ds = sub_disp[coords[:, 0], coords[:, 1]]
            # region-local anchors
            reg_anchors = [(ax, ay, ad) for (ax, ay, ad) in anchors if x0 <= ax < x1 and y0 <= ay < y1]
            res = fit_plane_ls(xs.astype(np.float32), ys.astype(np.float32), ds.astype(np.float32), anchors=reg_anchors if reg_anchors else None)
            if res is None:
                continue
            a, b, c = res
            pred = a * xs + b * ys + c
            med_res = float(np.median(np.abs(pred - ds)))
            residual_map[y0:y1, x0:x1][coords[:,0], coords[:,1]] = med_res
            if med_res > PLANE_RESID_THRESH:
                continue
            # apply blended plane values
            for (yy, xx), pval in zip(zip(ys, xs), pred):
                out[yy, xx] = (1.0 - PLANE_BLEND) * out[yy, xx] + PLANE_BLEND * pval
    if debug_dir:
        try:
            vis = residual_map.copy()
            vis[np.isnan(vis)] = 0
            mm = np.percentile(vis[vis>0], 98) if np.any(vis>0) else 1.0
            im = (np.clip(vis / max(mm, 1e-6), 0, 1) * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(debug_dir, "plane_residual_map.png"), cv2.applyColorMap(im, cv2.COLORMAP_INFERNO))
            # anchors visualization
            visL = rectL.copy()
            for (ax, ay, ad) in anchors:
                cv2.circle(visL, (ax, ay), 3, (0,255,0), -1)
            cv2.imwrite(os.path.join(debug_dir, "anchors.png"), visL)
        except Exception:
            pass
    return out

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(IMAGE_PATH))[0]

    # 1) Load & split SBS
    left_bgr, right_bgr = split_sbs(IMAGE_PATH)
    if MIRROR_RIGHT:
        right_bgr = cv2.flip(right_bgr, 1)
        print("[ingest] mirrored right half (MIRROR_RIGHT=True)")

    # 2) Load calibration, rectify with NO resize
    calib = load_npz_calibration(CALIB_NPZ)
    rectL, rectR, P1, P2, Q = rectify_no_resize(left_bgr, right_bgr, calib)

    # 2.5) Foreground mask on rectified LEFT (saved by your function)
    fg_mask = segment_foreground(rectL, OUTPUT_DIR, base)   # 0/255 u8

    # 2.6) ORB features (segmented rectified-left)
    if DO_FEATURES:
        print("[feat] ORB on segmented rectified-left …")
        kps, desc, vis = extract_orb_features_from_bgr_and_mask(rectL, fg_mask, n_features=N_FEATURES)
        save_feature_results(OUTPUT_DIR, base, vis, desc, keypoints=kps)
        print(f"[feat] keypoints: {len(kps) if kps else 0}  desc: {None if desc is None else desc.shape}")

    # Baseline sanity print
    fx = float(P1[0, 0])
    Tx = -float(P2[0, 3]) / fx
    unit = "mm" if CALIB_T_UNIT.lower() == "mm" else "m"
    print(f"[calib] Tx (baseline) ~ {Tx:.6f} {unit}")

    # 3) Profile + disparity
    if USE_WLS and not (hasattr(cv2, "ximgproc") and
                        hasattr(cv2.ximgproc, "createRightMatcher") and
                        hasattr(cv2.ximgproc, "createDisparityWLSFilter")):
        raise RuntimeError("This cv2 build lacks ximgproc WLS. Install opencv-contrib-python(-headless).")

    prof = get_profile(PROFILE)
    disp = compute_disparity(rectL, rectR, NUM_DISP, prof, USE_WLS)

    # Optional refinement using precomputed keypoints (anchors)
    debug_dir = os.path.join(OUTPUT_DIR, f"debug_{base}")
    os.makedirs(debug_dir, exist_ok=True)
    if ENABLE_PLANE_REFINEMENT:
        kp_arr = load_precomputed_keypoints(base)
        anchors = extract_anchors_from_keypoints(rectL, rectR, kp_arr, min_disp=0, num_disp=NUM_DISP, template_thresh=0.6)
        print(f"[anchors] found {len(anchors)} anchors from precomputed keypoints")
        disp_ref = refine_disparity_grid(disp, rectL, fg_mask, anchors, debug_dir=debug_dir)
        # save refined vis
        vref, cref = disp_to_vis(disp_ref)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_disparity_refined_gray.png"), vref)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_disparity_refined_color.png"), cref)
        use_disp = disp_ref
    else:
        use_disp = disp

    # 3.5) Edge-preserving refinement (joint bilateral on disparity)
    if hasattr(cv2, "ximgproc"):
        guide32 = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY).astype(np.float32)
        disp_mm = (disp * 1000.0).astype(np.float32)
        d = 9
        sigmaColor = 8.0
        sigmaSpace = 9.0
        disp_mm_f = cv2.ximgproc.jointBilateralFilter(guide32, disp_mm, d, sigmaColor, sigmaSpace)
        disp = disp_mm_f / 1000.0
    else:
        print("[warn] cv2.ximgproc not available — jointBilateralFilter skipped")

    # 4) Depth maps
    depth_mm_from_Q  = disparity_to_depth_Q(disp, Q)
    depth_m_from_fxB = disparity_to_depth_fxB(disp, P1, P2)

    # 5) Visualization
    disp_u8, disp_col = disp_to_vis(disp)

    # 6) Save images & arrays (plus masked previews)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_rect_left.png"), rectL)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_rect_right.png"), rectR)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_disparity_gray.png"), disp_u8)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_disparity_color.png"), disp_col)

    disp_u8_masked  = cv2.bitwise_and(disp_u8,  disp_u8,  mask=fg_mask)
    disp_col_masked = cv2.bitwise_and(disp_col, disp_col, mask=fg_mask)
    rectL_masked    = cv2.bitwise_and(rectL,    rectL,    mask=fg_mask)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_disparity_gray_MASKED.png"),  disp_u8_masked)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_disparity_color_MASKED.png"), disp_col_masked)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_rect_left_MASKED.png"),       rectL_masked)

    # np.save(os.path.join(OUTPUT_DIR, f"{base}_disp_float32.npy"), disp.astype(np.float32))
    # np.save(os.path.join(OUTPUT_DIR, f"{base}_depth_Q_mm.npy"),   depth_mm_from_Q.astype(np.float32))
    # np.save(os.path.join(OUTPUT_DIR, f"{base}_depth_fxB_m.npy"),  depth_m_from_fxB.astype(np.float32))
    depth_mm16 = np.clip(depth_mm_from_Q, 0, 65535).astype(np.uint16)
    # cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_depth_Q_mm16.png"), depth_mm16)
    # np.save(os.path.join(OUTPUT_DIR, f"{base}_Q.npy"),  Q.astype(np.float64))
    # np.save(os.path.join(OUTPUT_DIR, f"{base}_P1.npy"), P1.astype(np.float64))
    # np.save(os.path.join(OUTPUT_DIR, f"{base}_P2.npy"), P2.astype(np.float64))

    # 7) Colored point cloud (PLY) — **masked foreground**, meters
    ply_out = os.path.join(OUTPUT_DIR, f"{base}_pointcloud_MASKED_from_Q.ply")
    save_point_cloud_from_disparity_masked(
        rectL, use_disp, Q, ply_out,
        seg_mask_u8=fg_mask,
        calib_t_unit=CALIB_T_UNIT,
        z_min=PLY_Z_MIN, z_max=PLY_Z_MAX
    )

    # 8) Quick stats
    valid = disp > 0
    print(f"Valid disparity: {valid.sum()}/{disp.size} = {100*valid.mean():.2f}%")
    if valid.any():
        v = disp[valid]
        print(f"d min/max p1/p99: {v.min():.2f}/{v.max():.2f}  {np.percentile(v,1):.2f}/{np.percentile(v,99):.2f}")
        print(f"Depth median (Q, mm): {np.median(depth_mm_from_Q[valid]):.1f}")
    print("[DONE] Depth maps and (masked) point cloud exported.")

if __name__ == "__main__":
    main()

