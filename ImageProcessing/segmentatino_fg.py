# segmentation_fg.py
# Minimal-deps foreground mask for a rectified LEFT image (BGR).
# rembg is optional; we fall back to HSV+GrabCut if not available.

from typing import Optional, Tuple
import os
import numpy as np
import cv2

# --- knobs you can tweak here (kept local so depth code stays unchanged) ---
USE_REMBG = True
TRY_TWO_MODELS = True
REMBG_MODEL_LOCAL_PATH = r""

SAFETY_CORE_RADIUS  = 6
SAFETY_BAND_WIDTH   = 8

TRIMAP_FG_ERODE     = 2
TRIMAP_BG_DILATE    = 6
GRABCUT_ITERS       = 3

SMALL_HOLE_AREA     = 3000
GUIDED_RADIUS       = 8
GUIDED_EPS          = 1e-6
ERODE_SHRINK_PX     = 0

COLOR_CLEANUP_ENABLE = True
COLOR_UNKNOWN_BAND   = 8
COLOR_REG_EPS        = 5.0
COLOR_LLR_BIAS       = +0.35
# ---------------------------------------------------------------------------

def _keep_largest(mask_u8: np.ndarray) -> np.ndarray:
    m = (mask_u8 > 0).astype(np.uint8)
    n,labels,stats,_ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1: return (m*255).astype(np.uint8)
    i = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return ((labels == i).astype(np.uint8)*255)

def _fill_small_holes(mask_u8: np.ndarray, max_hole_area: int) -> np.ndarray:
    inv = cv2.bitwise_not(mask_u8)
    n,labels,stats,_ = cv2.connectedComponentsWithStats(inv, 4)
    out = mask_u8.copy()
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] <= max_hole_area:
            out[labels == i] = 255
    return out

def _refine_mask_guided(mask_u8: np.ndarray, guide_bgr: np.ndarray,
                        radius: int, eps: float, shrink_px: int) -> np.ndarray:
    m = mask_u8.copy()
    if shrink_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        m = cv2.erode(m, k, shrink_px)
    src = (m.astype(np.float32)/255.0)
    # Use guided filter if available; otherwise bilateral as a safe fallback
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
        guide = cv2.cvtColor(guide_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
        gf = cv2.ximgproc.guidedFilter(guide, src, radius, eps)
        ref = (gf*255.0).astype(np.uint8)
    else:
        ref = cv2.bilateralFilter(m, 7, 40, 40)
    _, hard = cv2.threshold(ref, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return hard

def _make_core_and_band(alpha_u8: np.ndarray, core_radius: int, band_width: int):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    core_fg = cv2.erode(alpha_u8, k, iterations=max(1, core_radius))
    dil = cv2.dilate(alpha_u8, k, iterations=max(1, band_width))
    ero = cv2.erode(alpha_u8, k, iterations=max(1, band_width))
    band = ((dil > 0) & (ero == 0)).astype(np.uint8) * 255
    return core_fg, band

def _merge_with_constraints(core_fg: np.ndarray, band: np.ndarray, refined_candidate: np.ndarray) -> np.ndarray:
    final = np.zeros_like(core_fg, np.uint8)
    final[core_fg > 0] = 255
    inside_band = (band > 0)
    final[inside_band] = np.where(refined_candidate[inside_band] > 0, 255, final[inside_band])
    return final

def _make_trimap(mask_u8: np.ndarray, fg_erode: int, bg_dilate: int) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    sure_fg = cv2.erode(mask_u8, k, fg_erode)
    sure_bg = cv2.dilate(cv2.bitwise_not(mask_u8), k, bg_dilate)
    trimap = np.full_like(mask_u8, 128, np.uint8)
    trimap[sure_bg > 0] = 0
    trimap[sure_fg > 0] = 255
    return trimap

def _grabcut_refine(img_bgr: np.ndarray, trimap: np.ndarray, iters: int=3) -> np.ndarray:
    h,w = trimap.shape
    gmask = np.zeros((h,w), np.uint8)
    GC_BGD,GC_FGD,GC_PR_BGD,GC_PR_FGD = 0,1,2,3
    gmask[trimap==0]   = GC_BGD
    gmask[trimap==255] = GC_FGD
    gmask[trimap==128] = GC_PR_BGD
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    cv2.grabCut(img_bgr, gmask, None, bgdModel, fgdModel, iters, cv2.GC_INIT_WITH_MASK)
    out = ((gmask==GC_FGD) | (gmask==GC_PR_FGD)).astype(np.uint8)*255
    return out

def _fit_gaussian(X: np.ndarray, eps: float):
    mu = X.mean(axis=0)
    S  = np.cov(X.T) + np.eye(3)*(eps**2)
    iS = np.linalg.inv(S)
    c  = -0.5*np.log(np.linalg.det(S) + 1e-12)
    return mu, iS, c

def _loglike(x: np.ndarray, mu: np.ndarray, iS: np.ndarray, c: float):
    d = x - mu
    return c - 0.5*np.sum(d @ iS * d, axis=1)

def _color_model_cleanup(mask_u8: np.ndarray, img_bgr: np.ndarray,
                         inner_erode: int, outer_dilate: int, unknown_band: int,
                         reg_eps: float, llr_bias: float) -> np.ndarray:
    h,w = mask_u8.shape
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    sure_fg = cv2.erode(mask_u8, k, inner_erode)
    sure_bg = cv2.dilate(cv2.bitwise_not(mask_u8), k, outer_dilate)
    dil = cv2.dilate(mask_u8, k, unknown_band)
    ero = cv2.erode(mask_u8, k, unknown_band)
    band = ((dil > 0) & (ero == 0)).astype(np.uint8) * 255

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).reshape(-1,3).astype(np.float32)
    idx_fg = (sure_fg.reshape(-1) > 0)
    idx_bg = (sure_bg.reshape(-1) > 0)
    idx_bd = (band.reshape(-1) > 0)

    if idx_fg.sum() < 500 or idx_bg.sum() < 500 or idx_bd.sum() == 0:
        return mask_u8

    mu_f, iS_f, c_f = _fit_gaussian(lab[idx_fg], reg_eps)
    mu_b, iS_b, c_b = _fit_gaussian(lab[idx_bg], reg_eps)
    ll_f = _loglike(lab[idx_bd], mu_f, iS_f, c_f)
    ll_b = _loglike(lab[idx_bd], mu_b, iS_b, c_b)
    llr  = ll_f - ll_b + llr_bias

    out = mask_u8.reshape(-1).copy()
    out[idx_bd] = np.where(llr >= 0.0, 255, 0).astype(np.uint8)
    out = out.reshape(h,w)
    return cv2.medianBlur(out, 3)

def _rembg_alpha(img_bgr: np.ndarray, model_name: str, model_local_override: str="") -> Optional[np.ndarray]:
    if not USE_REMBG:
        return None
    try:
        from rembg import new_session, remove
    except Exception:
        return None
    # Optional: copy local model into cache
    try:
        import shutil
        cache_dir = os.path.join(os.path.expanduser("~"), ".u2net")
        os.makedirs(cache_dir, exist_ok=True)
        if model_local_override and os.path.exists(model_local_override):
            shutil.copyfile(model_local_override, os.path.join(cache_dir, f"{model_name}.onnx"))
    except Exception:
        pass
    try:
        session = new_session(model_name)
    except Exception:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    try:
        out = remove(
            img_rgb, session=session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=235,
            alpha_matting_background_threshold=20,
            alpha_matting_erode_size=5,
            post_process_mask=True
        )
    except Exception:
        return None
    if out is None or out.size == 0: return None
    if out.ndim == 3 and out.shape[2] == 4: return out[:,:,3]
    if out.ndim == 2: return out
    return None

def segment_foreground(rectL_bgr: np.ndarray, out_dir: str, base: str) -> np.ndarray:
    """
    Return uint8 mask 0/255 aligned with rectL/disp. Saves a few debug images.
    """
    alpha_candidates = []
    a_u2 = _rembg_alpha(rectL_bgr, "u2net_human_seg", REMBG_MODEL_LOCAL_PATH)
    if a_u2 is not None:
        # cv2.imwrite(os.path.join(out_dir, f"{base}_rembg_alpha_u2.png"), a_u2)
        alpha_candidates.append(("u2", a_u2))
    if TRY_TWO_MODELS and USE_REMBG:
        a_is = _rembg_alpha(rectL_bgr, "isnet-general-use", REMBG_MODEL_LOCAL_PATH)
        if a_is is not None:
            # cv2.imwrite(os.path.join(out_dir, f"{base}_rembg_alpha_is.png"), a_is)
            alpha_candidates.append(("is", a_is))

    if not alpha_candidates:
        hsv = cv2.cvtColor(rectL_bgr, cv2.COLOR_BGR2HSV)
        s = hsv[:,:,1]
        _, a_fallback = cv2.threshold(cv2.GaussianBlur(s,(0,0),3), 0, 255,
                                      cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        alpha = _keep_largest(a_fallback)
    else:
        def boundary_score(a):
            m = (a>0).astype(np.uint8)
            return float(cv2.Canny(m, 50, 150).mean())
        alpha = alpha_candidates[0][1] if len(alpha_candidates)==1 else (
            alpha_candidates[0][1] if boundary_score(alpha_candidates[0][1]) <= boundary_score(alpha_candidates[1][1])
            else alpha_candidates[1][1]
        )

    alpha = (alpha>0).astype(np.uint8)*255
    alpha = _keep_largest(alpha)

    core_fg, band = _make_core_and_band(alpha, SAFETY_CORE_RADIUS, SAFETY_BAND_WIDTH)
    # cv2.imwrite(os.path.join(out_dir, f"{base}_safety_core.png"), core_fg)
    # cv2.imwrite(os.path.join(out_dir, f"{base}_safety_band.png"), band)

    trimap = _make_trimap(core_fg, TRIMAP_FG_ERODE, TRIMAP_BG_DILATE)
    # cv2.imwrite(os.path.join(out_dir, f"{base}_trimap.png"), trimap)
    gc_mask = _grabcut_refine(rectL_bgr, trimap, GRABCUT_ITERS)
    # cv2.imwrite(os.path.join(out_dir, f"{base}_grabcut.png"), gc_mask)

    if COLOR_CLEANUP_ENABLE:
        gc_mask = _color_model_cleanup(gc_mask, rectL_bgr,
                                       inner_erode=max(1, TRIMAP_FG_ERODE),
                                       outer_dilate=max(1, TRIMAP_BG_DILATE),
                                       unknown_band=COLOR_UNKNOWN_BAND,
                                       reg_eps=COLOR_REG_EPS,
                                       llr_bias=COLOR_LLR_BIAS)
        # cv2.imwrite(os.path.join(out_dir, f"{base}_color_cleaned.png"), gc_mask)

    candidate = _refine_mask_guided(gc_mask, rectL_bgr, GUIDED_RADIUS, GUIDED_EPS, ERODE_SHRINK_PX)
    candidate = _fill_small_holes(candidate, SMALL_HOLE_AREA)

    final_mask = _merge_with_constraints(core_fg, band, candidate)
    final_mask = _keep_largest(final_mask)
    # cv2.imwrite(os.path.join(out_dir, f"{base}_final_mask_refined.png"), final_mask)

    seg = cv2.bitwise_and(rectL_bgr, rectL_bgr, mask=final_mask)
    # cv2.imwrite(os.path.join(out_dir, f"{base}_segmented.png"), seg)
    return final_mask
