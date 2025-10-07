#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ENHANCED Depth + Colored Point Cloud for CLOSE-RANGE objects
— Multi-method disparity + advanced filtering + edge preservation
"""

import os
from typing import Tuple, Dict, Optional, List
import numpy as np
import cv2
import open3d as o3d
from scipy import ndimage

from segmentation_fg import segment_foreground
from new_feature_extract import (
    extract_orb_features_from_bgr_and_mask,
    save_feature_results
)

# ========== ENHANCED SETTINGS ==========
IMAGE_PATH = r"data/trial_approx_images/WIN_20251007_15_24_26_Pro.jpg"
CALIB_NPZ  = r"opencv_stereo_params2.npz"
OUTPUT_DIR = r"data/output/depth_only"

# Enhanced disparity methods
DISPARITY_METHOD = "sgbm_enhanced"  # "sgbm_enhanced" | "bm_improved" | "hybrid"
PROFILE   = "detail"   # "close_range" | "detail" | "lowtex" | "balanced"
NUM_DISP  = 320          # Increased for close objects
USE_WLS   = True
DO_HIST_MATCH = True
SMOOTH_PREVIEW = False  # Disable for debugging

# Enhanced toggles
MATCH_BLUR     = True
ALLOW_NEGATIVE = False
AUTO_FLIP_SIGN = True
MIRROR_RIGHT   = False
SAVE_ZUP = True

CALIB_T_UNIT = "mm"
PLY_Z_MIN = 100.0      # Increased minimum for close range
PLY_Z_MAX = 5000.0     # Reduced maximum for close range focus

EXPORT_FRAME = "y_up"
Z_UP_FORWARD = "+Y"

DO_FEATURES = True
N_FEATURES  = 2000

# NEW: Edge preservation settings
ENABLE_EDGE_AWARE = True
USE_DUAL_FILTERING = True
# ===============================

# ---------- Calibration / Rectification (UNCHANGED) ----------
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

# ---------- ENHANCED PROFILES ----------
def get_profile(name: str) -> Dict:
    name = (name or "balanced").lower()
    
    if name == "close_range":
        # OPTIMIZED FOR CLOSE OBJECTS (20-100cm)
        return dict(
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY, 
            bs=3,                       # Smaller blocks for detail
            uniq=5,                     # Lower for textureless areas  
            p2_mult=32,                 # Lower P2 for depth discontinuities
            disp12MaxDiff=1,            # Tighter consistency check
            wls_lambda=8000.0,          # Lower lambda for edge preservation
            wls_sigma=0.8,              # Lower sigma for sharp edges
            clahe_clip=2.5,             # Higher contrast enhancement
            clahe_tile=(8, 8),          # Smaller tiles for local contrast
            prefilter_size=5,           # Smaller prefilter
            prefilter_cap=31            # Higher prefilter cap
        )
    elif name == "detail":
        return dict(mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY, bs=3, uniq=12, p2_mult=48, disp12MaxDiff=2,
                    wls_lambda=22000.0, wls_sigma=0.9, clahe_clip=1.5, clahe_tile=(12, 12),
                    prefilter_size=9, prefilter_cap=31)
    elif name == "lowtex":
        return dict(mode=cv2.STEREO_SGBM_MODE_HH, bs=7, uniq=10, p2_mult=48, disp12MaxDiff=3,
                    wls_lambda=32000.0, wls_sigma=1.4, clahe_clip=1.5, clahe_tile=(12, 12),
                    prefilter_size=9, prefilter_cap=31)
    else:
        return dict(mode=cv2.STEREO_SGBM_MODE_HH, bs=5, uniq=14, p2_mult=64, disp12MaxDiff=2,
                    wls_lambda=28000.0, wls_sigma=1.2, clahe_clip=1.5, clahe_tile=(12, 12),
                    prefilter_size=9, prefilter_cap=31)

# ---------- ENHANCED DISPARITY COMPUTATION ----------
def build_enhanced_sgbm(num_disp: int, prof: Dict, min_disp: int) -> cv2.StereoSGBM:
    """Enhanced SGBM with better close-range performance"""
    nd = int(np.ceil(num_disp / 16.0)) * 16
    
    sgbm = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=nd,
        blockSize=prof["bs"],
        P1=8 * 3 * prof["bs"] ** 2,      # Reduced P1 for discontinuities
        P2=prof["p2_mult"] * 3 * prof["bs"] ** 2,
        disp12MaxDiff=prof["disp12MaxDiff"],
        uniquenessRatio=prof["uniq"],
        speckleWindowSize=50,            # Reduced speckle filtering
        speckleRange=1,                  # Tighter speckle range
        preFilterCap=prof["prefilter_cap"],
        mode=prof["mode"]
    )
    return sgbm

def compute_enhanced_disparity(rectL_bgr: np.ndarray,
                              rectR_bgr: np.ndarray,
                              num_disp: int,
                              prof: Dict,
                              use_wls: bool) -> np.ndarray:
    """Enhanced disparity computation for close-range objects"""
    
    # Convert to grayscale with edge preservation
    grayL = cv2.cvtColor(rectL_bgr, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rectR_bgr, cv2.COLOR_BGR2GRAY)
    
    # ENHANCED PREPROCESSING
    # 1. Use guided filter instead of bilateral for edge-aware smoothing
    if hasattr(cv2, 'ximgproc'):
        grayL = cv2.ximgproc.guidedFilter(rectL_bgr, grayL, radius=2, eps=100)
        grayR = cv2.ximgproc.guidedFilter(rectR_bgr, grayR, radius=2, eps=100)
    else:
        # Fallback to bilateral with smaller parameters
        grayL = cv2.bilateralFilter(grayL, 5, 25, 25)
        grayR = cv2.bilateralFilter(grayR, 5, 25, 25)

    # 2. Enhanced contrast limited adaptive histogram equalization
    clahe = cv2.createCLAHE(
        clipLimit=prof["clahe_clip"], 
        tileGridSize=prof["clahe_tile"]
    )
    grayL = clahe.apply(grayL)
    grayR = clahe.apply(grayR)
    
    # 3. Add gradient enhancement for textureless areas
    gradL = cv2.Sobel(grayL, cv2.CV_64F, 1, 1, ksize=3)
    gradR = cv2.Sobel(grayR, cv2.CV_64F, 1, 1, ksize=3)
    grayL = cv2.addWeighted(grayL, 0.8, np.uint8(np.abs(gradL)), 0.2, 0)
    grayR = cv2.addWeighted(grayR, 0.8, np.uint8(np.abs(gradR)), 0.2, 0)

    if DO_HIST_MATCH:
        grayR = match_histogram_to(grayR, grayL)

    if MATCH_BLUR:
        grayL, grayR, (vL0, vR0) = equalize_blur(grayL, grayR)
        print(f"[blur-match] LapVar L/R ~ {vL0:.1f}/{vR0:.1f}")

    neg_window = 0  # Disabled for close range
    
    # COMPUTE DISPARITY
    sgbm = build_enhanced_sgbm(num_disp, prof, min_disp=neg_window)
    dispL16 = sgbm.compute(grayL, grayR)

    # ENHANCED POST-PROCESSING
    disp = dispL16.astype(np.float32) / 16.0

    if use_wls and hasattr(cv2, "ximgproc"):
        right_matcher = cv2.ximgproc.createRightMatcher(sgbm)
        dispR16 = right_matcher.compute(grayR, grayL)
        wls = cv2.ximgproc.createDisparityWLSFilter(matcher_left=sgbm)
        wls.setLambda(float(prof["wls_lambda"]))
        wls.setSigmaColor(float(prof["wls_sigma"]))
        disp = wls.filter(dispL16, rectL_bgr, disparity_map_right=dispR16).astype(np.float32) / 16.0

    # EDGE-AWARE REFINEMENT
    if ENABLE_EDGE_AWARE:
        disp = edge_aware_refinement(disp, rectL_bgr)

    if AUTO_FLIP_SIGN and ALLOW_NEGATIVE:
        sgn = _dominant_sign(disp)
        if sgn < 0:
            disp = -disp
            print("[sign] disparity flipped")

    # GAP FILLING AND HOLE REMOVAL
    disp = fill_disparity_gaps(disp, rectL_bgr)
    
    # Less aggressive filtering to preserve details
    disp[disp < 0] = 0.0

    return disp

def edge_aware_refinement(disp: np.ndarray, guide_image: np.ndarray) -> np.ndarray:
    """Use the color image to guide disparity refinement at edges"""
    disp_refined = disp.copy()
    
    # Detect edges in the guide image
    gray_guide = cv2.cvtColor(guide_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_guide, 50, 150)
    
    # Dilate edges to cover boundary regions
    kernel = np.ones((2, 2), np.uint8)
    edge_mask = cv2.dilate(edges, kernel, iterations=1)
    
    # Apply edge-aware filtering only at edges
    if np.any(edge_mask):
        # Use bilateral filter that preserves edges
        disp_refined = cv2.bilateralFilter(disp_refined, 3, 25, 10)
        
    return disp_refined

def fill_disparity_gaps(disp: np.ndarray, guide_image: np.ndarray, max_hole_size: int = 15) -> np.ndarray:
    """Fill holes in disparity using guided methods"""
    disp_filled = disp.copy()
    invalid_mask = (disp <= 0) | ~np.isfinite(disp)
    
    if not np.any(invalid_mask):
        return disp_filled
    
    # Method 1: Use OpenCV inpaint for small holes
    if np.any(invalid_mask):
        # Convert to 8-bit for inpainting
        disp_normalized = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        inpainted = cv2.inpaint(disp_normalized, invalid_mask.astype(np.uint8), 3, cv2.INPAINT_NS)
        disp_filled = inpainted.astype(np.float32) * (disp.max() / 255.0) if disp.max() > 0 else inpainted.astype(np.float32)
    
    return disp_filled

# ---------- REST OF YOUR FUNCTIONS (KEEP AS IS) ----------
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
        best = cv2.GaussianBlur(src, (0, 0), sigma)
        sigma += step

    if is_left:  return best, grayR, (vL0, vR0)
    else:        return grayL, best, (vL0, vR0)

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
    y0, y1 = int(0.25*H), int(0.75*H)
    x0, x1 = int(0.25*W), int(0.75*W)
    roi = d[y0:y1, x0:x1]
    v = roi[np.isfinite(roi)]
    v = v[np.abs(v) > 0.1]
    if v.size < 500:
        return 0.0
    med = float(np.median(v))
    return 1.0 if med >= 0 else -1.0

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
    if f == "camera":
        return np.eye(3, dtype=float)
    if f == "z_up":
        if Z_UP_FORWARD.upper().startswith("-"):
            return np.array([[1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=float)
        else:
            return np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)
    if f == "y_up":
        return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
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

    pts = pts3d[mask].astype(np.float32)
    cols_rgb = (rect_left_bgr[mask].astype(np.float32) / 255.0)[:, ::-1]

    if calib_t_unit.lower() == "mm":
        pts *= 0.001

    R = _orientation_matrix(EXPORT_FRAME)
    pts = (R @ pts.T).T

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(cols_rgb.astype(np.float64))

    os.makedirs(os.path.dirname(out_ply_path) or ".", exist_ok=True)
    ok = o3d.io.write_point_cloud(out_ply_path, pcd, write_ascii=True, print_progress=True)
    if not ok:
        raise IOError(f"Failed to save PLY: {out_ply_path}")
    print(f"[OK] Masked point cloud saved -> {out_ply_path}  (points: {len(pts)})")

# ---------- ENHANCED MAIN ----------
def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
    
    print("=== ENHANCED STEREO DEPTH FOR CLOSE-RANGE OBJECTS ===")
    
    if USE_WLS and not (hasattr(cv2, "ximgproc") and
                    hasattr(cv2.ximgproc, "createRightMatcher") and
                    hasattr(cv2.ximgproc, "createDisparityWLSFilter")):
        print("[warning] WLS filter not available, using basic filtering")

    # 1) Load & split SBS
    left_bgr, right_bgr = split_sbs(IMAGE_PATH)
    print(f"[info] Image size: {left_bgr.shape}")
    
    if MIRROR_RIGHT:
        right_bgr = cv2.flip(right_bgr, 1)
        print("[ingest] mirrored right half")

    # 2) Load calibration, rectify with NO resize
    calib = load_npz_calibration(CALIB_NPZ)
    rectL, rectR, P1, P2, Q = rectify_no_resize(left_bgr, right_bgr, calib)

    # 2.5) Foreground mask
    fg_mask = segment_foreground(rectL, OUTPUT_DIR, base)
    print(f"[mask] Foreground pixels: {(fg_mask > 0).sum() / fg_mask.size * 100:.1f}%")

    # ORB features
    if DO_FEATURES:
        print("[feat] ORB on segmented rectified-left …")
        kps, desc, vis = extract_orb_features_from_bgr_and_mask(rectL, fg_mask, n_features=N_FEATURES)
        save_feature_results(OUTPUT_DIR, base, vis, desc, keypoints=kps)
        print(f"[feat] keypoints: {len(kps) if kps else 0}")

    # Baseline info
    fx = float(P1[0, 0])
    Tx = -float(P2[0, 3]) / fx
    unit = "mm" if CALIB_T_UNIT.lower() == "mm" else "m"
    print(f"[calib] Baseline: {Tx:.1f} {unit}, Focal: {fx:.1f} px")

    # 3) Enhanced disparity computation
    prof = get_profile(PROFILE)
    print(f"[disparity] Using {PROFILE} profile for close-range objects")
    
    disp = compute_enhanced_disparity(rectL, rectR, NUM_DISP, prof, USE_WLS)

    # 4) Depth maps
    depth_mm_from_Q = disparity_to_depth_Q(disp, Q)
    depth_m_from_fxB = disparity_to_depth_fxB(disp, P1, P2)

    # 5) Enhanced visualization with statistics
    disp_u8, disp_col = disp_to_vis(disp)
    
    # Calculate detailed statistics
    valid_pixels = (disp > 0).sum()
    total_pixels = disp.size
    valid_ratio = valid_pixels / total_pixels * 100
    
    print(f"[stats] Valid disparity: {valid_pixels}/{total_pixels} ({valid_ratio:.1f}%)")
    if valid_pixels > 0:
        valid_disp = disp[disp > 0]
        print(f"[stats] Disparity range: {valid_disp.min():.1f} - {valid_disp.max():.1f}")
        print(f"[stats] Depth range: {depth_mm_from_Q[disp > 0].min():.1f} - {depth_mm_from_Q[disp > 0].max():.1f} mm")

    # Save outputs
    disp_u8_masked = cv2.bitwise_and(disp_u8, disp_u8, mask=fg_mask)
    disp_col_masked = cv2.bitwise_and(disp_col, disp_col, mask=fg_mask)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_disparity_enhanced.png"), disp_col_masked)

    depth_mm16 = np.clip(depth_mm_from_Q, 0, 65535).astype(np.uint16)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_depth_enhanced.png"), depth_mm16)

    # 6) Point cloud
    ply_out_masked = os.path.join(OUTPUT_DIR, f"{base}_pcd_enhanced.ply")
    save_point_cloud_from_disparity_masked(
        rectL, disp, Q, ply_out_masked,
        fg_mask,
        calib_t_unit=CALIB_T_UNIT,
        z_min=PLY_Z_MIN, z_max=PLY_Z_MAX
    )

    print("[DONE] Enhanced depth processing complete.")

if __name__ == "__main__":
    main()
