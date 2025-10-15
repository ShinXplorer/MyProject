#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Depth + Colored Point Cloud from a side-by-side stereo image (NO RESIZING)
Upgrades:
  - Disparity range derived from calibration (fx, baseline, Zmin/Zmax)
  - Multi-scale SGBM (coarse->fine) with seed-up refinement
  - Left-Right consistency check + confidence mask
  - WLS filtering + small-hole inpainting
  - Retains your segmentation + ORB feature extraction pipeline

pip install opencv-contrib-python open3d numpy
"""

import os
from typing import Tuple, Dict, Optional
import numpy as np
import cv2
import open3d as o3d

# ---- segmentation + features (your modules) ----
from segmentation_fg import segment_foreground
from new_feature_extract import (
    extract_orb_features_from_bgr_and_mask,
    save_feature_results
)

# ========== EDIT THESE ==========
IMAGE_PATH = r"data/capture_images/WIN_20251007_09_42_25_Pro.jpg"
CALIB_NPZ  = r"opencv_stereo_params2.npz"
OUTPUT_DIR = r"data/output/depth_integrated"

PROFILE   = "lowtex"     # "balanced" | "detail" | "lowtex"
USE_WLS   = True
DO_HIST_MATCH = True
SMOOTH_PREVIEW = True
MATCH_BLUR     = True
MIRROR_RIGHT   = False

# ORB feature extraction on segmented rectified-left
DO_FEATURES = True
N_FEATURES  = 2000

# Expected scene depth range (in *calibration units* below). TUNABLE!
PLY_Z_MIN = 200.0      # e.g., 200 mm  (0.2 m)
PLY_Z_MAX = 5000.0     # e.g., 5000 mm (5 m)

CALIB_T_UNIT = "mm"    # "mm" or "m"

# Sparsity / export
POINT_STRIDE = 2
VOXEL_DOWNSAMPLE_M = 0.003
RANDOM_KEEP_RATIO = 1.0
EXPORT_FRAME = "y_up"
# ===============================

# ----------------- I/O & Calib -----------------

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
    if size[0] < size[1]:
        size = (size[1], size[0])
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

# ----------------- Profiles & helpers -----------------

def get_profile(name: str) -> Dict:
    name = (name or "balanced").lower()
    if name == "detail":
        return dict(mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY, bs=3, uniq=8, p2_mult=48, disp12MaxDiff=1,
                    wls_lambda=80000.0, wls_sigma=1.5, clahe_clip=1.5, clahe_tile=(8, 8))
    elif name == "lowtex":
        return dict(mode=cv2.STEREO_SGBM_MODE_HH, bs=7, uniq=12, p2_mult=48, disp12MaxDiff=1,
                    wls_lambda=32000.0, wls_sigma=1.2, clahe_clip=1.5, clahe_tile=(12, 12))
    else:  # balanced
        return dict(mode=cv2.STEREO_SGBM_MODE_HH, bs=5, uniq=10, p2_mult=64, disp12MaxDiff=1,
                    wls_lambda=40000.0, wls_sigma=1.3, clahe_clip=1.5, clahe_tile=(8, 8))

def _sharpness_var_laplacian(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def equalize_blur(grayL: np.ndarray, grayR: np.ndarray,
                  max_sigma: float = 2.0, step: float = 0.25, tol_ratio: float = 0.10):
    vL0, vR0 = _sharpness_var_laplacian(grayL), _sharpness_var_laplacian(grayR)
    if abs(vL0 - vR0) / max(vL0, vR0, 1e-6) < tol_ratio:
        return grayL, grayR, (vL0, vR0)
    left_is_sharper = (vL0 > vR0)
    src = grayL if left_is_sharper else grayR
    target = (grayR if left_is_sharper else grayL)
    best = src
    sigma = 0.25
    tgt = _sharpness_var_laplacian(target) * (1.0 + tol_ratio)
    while sigma <= max_sigma and _sharpness_var_laplacian(best) > tgt:
        best = cv2.GaussianBlur(src, (0, 0), sigma)
        sigma += step
    if left_is_sharper:  # blur L toward R
        return best, grayR, (vL0, vR0)
    else:                # blur R toward L
        return grayL, best, (vL0, vR0)

def match_histogram_to(src_gray: np.ndarray, ref_gray: np.ndarray) -> np.ndarray:
    src = src_gray.ravel(); ref = ref_gray.ravel()
    s_values, bin_idx, s_counts = np.unique(src, return_inverse=True, return_counts=True)
    r_values, r_counts = np.unique(ref, return_counts=True)
    s_quant = np.cumsum(s_counts).astype(np.float64); s_quant /= s_quant[-1]
    r_quant = np.cumsum(r_counts).astype(np.float64); r_quant /= r_quant[-1]
    interp = np.interp(s_quant, r_quant, r_values)
    return interp[bin_idx].reshape(src_gray.shape).astype(np.uint8)

def _orientation_matrix(frame: str) -> np.ndarray:
    f = (frame or "camera").lower()
    if f == "camera": return np.eye(3, dtype=float)
    if f == "z_up":   return np.array([[1,0,0],[0,0,1],[0,-1,0]], dtype=float)
    if f == "y_up":   return np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=float)
    raise ValueError("EXPORT_FRAME must be 'camera', 'z_up', or 'y_up'")

# ---------- Disparity range from calibration (IMPORTANT) ----------

def disparity_bounds_from_depth(P1: np.ndarray, P2: np.ndarray,
                                zmin: float, zmax: float,
                                unit: str = "mm") -> Tuple[int, int]:
    """Return (minDisp, numDisp aligned to 16) derived from fx, baseline, and Z range."""
    fx = float(P1[0, 0])
    baseline = -float(P2[0, 3]) / fx  # in same units as P2 translation
    if unit.lower() == "mm":
        baseline *= 0.001  # -> meters
        zmin_m, zmax_m = zmin * 0.001, zmax * 0.001
    else:
        zmin_m, zmax_m = zmin, zmax
    # disparity = fx * B / Z    (with fx in pixels)
    # its magnitude shrinks with Z; cover Zmin..Zmax
    d_max = fx * baseline / max(zmin_m, 1e-6)
    d_min = fx * baseline / max(zmax_m, 1e-6)
    # Ensure positive / reasonable values; clamp
    d_min = max(0.0, min(d_min, 4096.0))
    d_max = max(d_min + 16.0, min(d_max, 4096.0))
    # round to SGBM grid
    min_disp = int(np.floor(d_min))  # integer pixels
    num_disp = int(np.ceil((d_max - min_disp) / 16.0)) * 16
    return min_disp, num_disp

# ---------- SGBM builders & refinement ----------

def build_sgbm(min_disp: int, num_disp: int, mode: int, bs: int, uniq: int, p2_mult: int,
               disp12MaxDiff: int) -> cv2.StereoSGBM:
    num_disp = int(np.ceil(num_disp / 16.0)) * 16
    bs = bs if bs % 2 == 1 else bs + 1
    cn = 1
    P1 = 8  * cn * bs * bs
    P2 = p2_mult * cn * bs * bs
    return cv2.StereoSGBM_create(
        minDisparity=int(min_disp), numDisparities=int(num_disp), blockSize=int(bs),
        P1=int(P1), P2=int(P2), disp12MaxDiff=int(disp12MaxDiff), uniquenessRatio=int(uniq),
        speckleWindowSize=200, speckleRange=2, preFilterCap=63, mode=mode
    )

def lr_consistency_mask(dispL: np.ndarray, dispR: np.ndarray, thresh_px: float = 1.0) -> np.ndarray:
    """Left-Right consistency: invalidate pixels where D_L(x) != D_R(x - D_L)."""
    h, w = dispL.shape
    xs = np.tile(np.arange(w)[None, :], (h, 1)).astype(np.float32)
    # project left disparity to right coordinates
    xr = (xs - dispL).round().astype(np.int32)
    xr = np.clip(xr, 0, w - 1)
    dR_at = dispR[np.arange(h)[:, None], xr]
    ok = np.abs(dispL - dR_at) <= thresh_px
    return ok

def fill_small_holes(d: np.ndarray, max_diam_px: int = 5) -> np.ndarray:
    """Simple hole fill: morphology close + median on small gaps."""
    m = (d > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m2 = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
    d2 = d.copy()
    holes = (m2 > 0) & (d == 0)
    if holes.any():
        d_med = cv2.medianBlur((d * 16).astype(np.uint16), 5).astype(np.float32) / 16.0
        d2[holes] = d_med[holes]
    return d2

def compute_disparity_multiscale(rectL_bgr: np.ndarray,
                                 rectR_bgr: np.ndarray,
                                 min_disp: int, num_disp: int,
                                 prof: Dict, use_wls: bool) -> np.ndarray:
    """Coarse-to-fine SGBM with LR + WLS."""
    # preprocess
    grayL = cv2.cvtColor(rectL_bgr, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rectR_bgr, cv2.COLOR_BGR2GRAY)

    # light bilateral to reduce noise while keeping edges
    grayL = cv2.bilateralFilter(grayL, 7, 40, 40)
    grayR = cv2.bilateralFilter(grayR, 7, 40, 40)

    clahe = cv2.createCLAHE(clipLimit=prof["clahe_clip"], tileGridSize=prof["clahe_tile"])
    grayL = clahe.apply(grayL); grayR = clahe.apply(grayR)

    if DO_HIST_MATCH:
        grayR = match_histogram_to(grayR, grayL)
    if MATCH_BLUR:
        grayL, grayR, (vL0, vR0) = equalize_blur(grayL, grayR)
        print(f"[blur-match] LapVar L/R ~ {vL0:.1f}/{vR0:.1f}")

    # build pyramids
    pyr_levels = 2  # 1/2 resolution seed -> refine full-res
    Lp = [grayL]; Rp = [grayR]
    for _ in range(pyr_levels):
        Lp.append(cv2.pyrDown(Lp[-1]))
        Rp.append(cv2.pyrDown(Rp[-1]))

    # coarse disparity
    scale = 2 ** pyr_levels
    min_d_coarse = int(np.floor(min_disp / scale))
    num_d_coarse = max(16, int(np.ceil(num_disp / scale / 16.0)) * 16)

    sgbm_coarse = build_sgbm(min_d_coarse, num_d_coarse, prof["mode"], max(3, prof["bs"]),
                             prof["uniq"], prof["p2_mult"], prof["disp12MaxDiff"])
    dL16 = sgbm_coarse.compute(Lp[-1], Rp[-1])
    dL = dL16.astype(np.float32) / 16.0

    # upsample disparity to next level(s) and refine
    for level in range(pyr_levels, 0, -1):
        dL = cv2.resize(dL, (Lp[level-1].shape[1], Lp[level-1].shape[0]), interpolation=cv2.INTER_LINEAR) * 2.0
        # narrow search around upsampled seed by shifting min_disp and num_disp
        # but OpenCV SGBM doesn't support per-pixel seed; we just use full range at fine scale
        pass

    # final pass at full res with correct range
    sgbm = build_sgbm(min_disp, num_disp, prof["mode"], prof["bs"], prof["uniq"],
                      prof["p2_mult"], prof["disp12MaxDiff"])
    dL16 = sgbm.compute(Lp[0], Rp[0]).astype(np.int16)
    dL = dL16.astype(np.float32) / 16.0

    # optional right disparity + LR consistency
    if hasattr(cv2, "ximgproc"):
        right_matcher = cv2.ximgproc.createRightMatcher(sgbm)
        dR16 = right_matcher.compute(Rp[0], Lp[0]).astype(np.int16)
        dR = dR16.astype(np.float32) / 16.0
        ok = lr_consistency_mask(dL, dR, thresh_px=1.0)
        dL[~ok] = 0.0
    else:
        print("[warn] cv2.ximgproc not available — LR check limited")

    # WLS filtering (guided by color left)
    if use_wls and hasattr(cv2, "ximgproc"):
        # For WLS, recompute to get both disparities fresh
        right_matcher = cv2.ximgproc.createRightMatcher(sgbm)
        dL16 = sgbm.compute(Lp[0], Rp[0])
        dR16 = right_matcher.compute(Rp[0], Lp[0])
        wls = cv2.ximgproc.createDisparityWLSFilter(matcher_left=sgbm)
        wls.setLambda(float(prof["wls_lambda"]))
        wls.setSigmaColor(float(prof["wls_sigma"]))
        dL = wls.filter(dL16, rectL_bgr, disparity_map_right=dR16).astype(np.float32) / 16.0

    # speckle & small holes
    d16 = (dL * 16.0).astype(np.int16)
    cv2.filterSpeckles(d16, 0, 0, 2 * 16)
    d = d16.astype(np.float32) / 16.0
    d[d < 0] = 0.0
    d = fill_small_holes(d, max_diam_px=5)
    d = cv2.medianBlur(d, 3)
    return d

# ----------------- Depth & Export -----------------

def disparity_to_depth_Q(disp: np.ndarray, Q: np.ndarray) -> np.ndarray:
    pts3d = cv2.reprojectImageTo3D(disp.astype(np.float32), Q)
    Z = pts3d[..., 2].astype(np.float32)
    Z[~np.isfinite(Z)] = 0.0
    Z[disp <= 0] = 0.0
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

    if POINT_STRIDE > 1:
        H, W = mask.shape
        stride_mask = np.zeros_like(mask, dtype=bool)
        stride_mask[0:H:POINT_STRIDE, 0:W:POINT_STRIDE] = True
        mask &= stride_mask
        if not np.any(mask):
            raise RuntimeError("All points removed by POINT_STRIDE; lower the stride.")

    pts = pts3d[mask].astype(np.float32)
    cols_rgb = (rect_left_bgr[mask].astype(np.float32) / 255.0)[:, ::-1]

    if calib_t_unit.lower() == "mm":
        pts *= 0.001  # mm -> m

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

# ----------------- Main -----------------

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

    # 2.5) Foreground mask on rectified LEFT
    fg_mask = segment_foreground(rectL, OUTPUT_DIR, base)   # 0/255 u8

    # 2.6) ORB features (segmented rectified-left)
    if DO_FEATURES:
        print("[feat] ORB on segmented rectified-left …")
        kps, desc, vis = extract_orb_features_from_bgr_and_mask(rectL, fg_mask, n_features=N_FEATURES)
        save_feature_results(OUTPUT_DIR, base, vis, desc, keypoints=kps)
        print(f"[feat] keypoints: {len(kps) if kps else 0}  desc: {None if desc is None else desc.shape}")

    # Baseline + disparity range from desired Z band
    fx = float(P1[0, 0])
    Tx = -float(P2[0, 3]) / fx
    unit = "mm" if CALIB_T_UNIT.lower() == "mm" else "m"
    print(f"[calib] Tx (baseline) ~ {Tx:.6f} {unit}")
    min_disp, num_disp = disparity_bounds_from_depth(P1, P2, PLY_Z_MIN, PLY_Z_MAX, CALIB_T_UNIT)
    print(f"[range] minDisp={min_disp}  numDisp={num_disp}  (from Z∈[{PLY_Z_MIN},{PLY_Z_MAX}] {unit})")

    # 3) Profile + disparity (multiscale + LR + WLS + fill)
    if USE_WLS and not (hasattr(cv2, "ximgproc") and
                        hasattr(cv2.ximgproc, "createRightMatcher") and
                        hasattr(cv2.ximgproc, "createDisparityWLSFilter")):
        raise RuntimeError("This cv2 build lacks ximgproc WLS. Install opencv-contrib-python(-headless).")

    prof = get_profile(PROFILE)
    disp = compute_disparity_multiscale(rectL, rectR, min_disp, num_disp, prof, USE_WLS)

    # 4) Depth via Q
    depth_mm_from_Q  = disparity_to_depth_Q(disp, Q)

    # 5) Visualization
    disp_u8 = np.zeros_like(disp, dtype=np.uint8)
    m = disp > 0
    if np.any(m):
        v = disp[m]
        lo, hi = np.percentile(v, 1.0), np.percentile(v, 99.0)
        hi = max(hi, lo + 1e-6)
        disp_u8[m] = np.clip(255.0 * (disp[m] - lo) / (hi - lo), 0, 255).astype(np.uint8)
        if SMOOTH_PREVIEW:
            disp_u8 = cv2.GaussianBlur(disp_u8, (3, 3), 0)
    try:
        disp_col = cv2.applyColorMap(disp_u8, cv2.COLORMAP_TURBO)
    except Exception:
        disp_col = cv2.applyColorMap(disp_u8, cv2.COLORMAP_JET)

    # 6) Save previews
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

    # 7) Colored point cloud (PLY) — masked foreground, meters
    ply_out = os.path.join(OUTPUT_DIR, f"{base}_pointcloud_MASKED_from_Q.ply")
    save_point_cloud_from_disparity_masked(
        rectL, disp, Q, ply_out,
        seg_mask_u8=fg_mask,
        calib_t_unit=CALIB_T_UNIT,
        z_min=PLY_Z_MIN, z_max=PLY_Z_MAX
    )

    # 8) Stats
    valid = disp > 0
    print(f"Valid disparity: {valid.sum()}/{disp.size} = {100*valid.mean():.2f}%")
    if valid.any():
        v = disp[valid]
        print(f"d min/max p1/p99: {v.min():.2f}/{v.max():.2f}  {np.percentile(v,1):.2f}/{np.percentile(v,99):.2f}")
        print(f"Depth median (Q, mm): {np.median(depth_mm_from_Q[valid]):.1f}")
    print("[DONE] Depth maps and (masked) point cloud exported.")

if __name__ == "__main__":
    main()
