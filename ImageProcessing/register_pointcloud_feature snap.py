#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pairwise merge with fixed 90° yaw and feature-guided corner snap (optional) for y_up point clouds.

New in this version
-------------------
- SNAP_MODE="feature_corner": pick A's top-right 3D feature and B's top-left 3D feature (after rotation),
  then translate B so those two points meet (plus small X-gap). Translation is locked to the main axis (X) by default.
- If 3D features are not provided, falls back to geometric corner selection (robust).
- Keeps existing "face" and "corner" modes and "auto" placement if you need it.

Assumptions
-----------
- PLYs are exported in y_up: X=right, Y=up, Z=forward (same as your depth exporter).
- A is the FRONT/world, B is the next view.
- Positive ROT_DEG about +Y is CCW (right-hand rule when viewed from +Y).

Outputs
-------
- <A>_world.ply
- <B>_turned_aligned.ply
- <A>__plus__<B>_merged_turned.ply
- Optional *_centered.ply
- T_<B>_to_<A>_turned.{npy,txt}
"""

import os, glob
from typing import Optional, Tuple, Literal
import numpy as np
import open3d as o3d

# =================== USER SETTINGS ===================
INPUT_DIR   = r"data/output/ply_inputs"
OUTPUT_DIR  = r"data/output/registration_feature_snap"

PLY_A = r"data/output/ply_inputs/WIN_20250929_09_22_09_Pro_pcd_MASKED.ply"
PLY_B = r"data/output/ply_inputs/WIN_20250929_09_22_25_Pro_pcd_MASKED.ply"

INPUT_FRAME = "y_up"   # X=right, Y=up, Z=forward

# Rotation about +Y (y_up). +90=CCW, -90=CW.
ROT_AXIS = "y"
ROT_DEG  = -90.0

# Pivot for rotation
PIVOT_MODE    = "centroid_A"  # "centroid_A" | "centroid_B" | "midcentroid" | "origin"

# Placement strategy:
# - "auto"   -> search axis/side + rotation sign and pick best
# - "corner" -> manual axis using corner snap (geometric or feature-guided)
# - "face"   -> manual axis using face-touch translation
PLACEMENT_MODE: Literal["auto","corner","face"] = "corner"

# Manual placement axis and gap
TOUCH_AXIS    = "x"     # 'x'|'y'|'z'
TOUCH_GAP_M   = 0.001   # ~1 mm gap

# Snap mode:
# - "face"          -> face-touch
# - "corner"        -> geometric corner (robust, from points)
# - "feature_corner"-> use 3D features if present, else fallback to "corner"
SNAP_MODE: Literal["face","corner","feature_corner"] = "feature_corner"

# When using "corner" or "feature_corner", choose which corner on the touching faces:
# For main axis X: A uses far face (+X), B uses near face (-X) after rotation.
# Along Y and Z, choose 'min'/'max' (e.g., top-right => Y=max, Z=max).
CORNER_Y = "max"  # 'min' or 'max'
CORNER_Z = "max"  # 'min' or 'max'

# Feature (optional): paths to 3D keypoints for A and B (Nx3, meters, same frame as their PLYs)
# If left None or file missing, code falls back to geometric corners.
FEAT_A_3D: Optional[str] = r"data/output/depth_only/WIN_20250929_09_22_09_Pro_keypoints.npy"  # e.g., r"data/output/depth_only/WIN_20250929_09_22_09_Pro_features_3d.npy"
FEAT_B_3D: Optional[str] = r"data/output/depth_only/WIN_20250929_09_22_25_Pro_keypoints.npy"  # e.g., r"data/output/depth_only/WIN_20250929_09_22_25_Pro_features_3d.npy"

# Use Z when picking feature corner? (If False, choose extreme on X and Y only.)
FEATURE_USE_Z = False

# Robustness / outliers for geometric corners
PLANE_THICK_RATIO = 0.02
PERCENT_LOW  = 5.0
PERCENT_HIGH = 95.0

# Lock translation to main axis only? (recommended to avoid Y/Z drift)
LOCK_TO_AXIS = True

# ICP polish
VOXEL_FOR_SAVE: Optional[float] = None
RUN_TINY_ICP   = True
ICP_MAX_ITERS  = 30
ICP_MAX_DIST   = 0.01

SAVE_RECENTERED = True
# =====================================================


# -------------------- helpers --------------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def scan_two_plys(folder: str) -> Tuple[str, str]:
    fs = sorted(glob.glob(os.path.join(folder, "*.ply")))
    if len(fs) < 2:
        raise FileNotFoundError(f"Need at least two .ply files in: {folder}")
    if len(fs) > 2:
        print(f"[warn] Found {len(fs)} PLYs; using first two after sort.")
    return fs[0], fs[1]

def load_pcd(path: str) -> o3d.geometry.PointCloud:
    p = o3d.io.read_point_cloud(path)
    if p.is_empty():
        raise IOError(f"Empty/invalid PLY: {path}")
    return p

def centroid(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    return np.asarray(pcd.get_center(), dtype=np.float64)

def recenter_to_origin(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    out = o3d.geometry.PointCloud(pcd)  # copy
    out.translate(-centroid(out))
    return out

def rot_T(axis: str, deg: float, pivot: np.ndarray) -> np.ndarray:
    axis = axis.lower()
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    if axis == "x":
        R = np.array([[1, 0, 0],[0, c,-s],[0, s, c]], dtype=np.float64)
    elif axis == "y":
        R = np.array([[ c, 0, s],[ 0, 1, 0],[-s, 0, c]], dtype=np.float64)
    elif axis == "z":
        R = np.array([[ c,-s, 0],[ s, c, 0],[ 0, 0, 1]], dtype=np.float64)
    else:
        raise ValueError("ROT_AXIS must be 'x', 'y', or 'z'")
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = R
    # rotate about pivot: x' = R(x - p) + p = R x + (p - R p)
    T[:3,3] = pivot - R @ pivot
    return T

def robust_min_max(arr: np.ndarray) -> Tuple[float, float]:
    return float(np.percentile(arr, PERCENT_LOW)), float(np.percentile(arr, PERCENT_HIGH))

def aabb(pcd: o3d.geometry.PointCloud):
    pts = np.asarray(pcd.points)
    mins = np.percentile(pts, PERCENT_LOW, axis=0)
    maxs = np.percentile(pts, PERCENT_HIGH, axis=0)
    return mins, maxs

def bbox_diag(pcd: o3d.geometry.PointCloud) -> float:
    mn, mx = aabb(pcd)
    return float(np.linalg.norm(mx - mn))

def choose_pivot(mode: str, A: o3d.geometry.PointCloud, B: o3d.geometry.PointCloud) -> np.ndarray:
    cA = centroid(A); cB = centroid(B)
    m = (mode or "").lower()
    if m == "centroid_a": return cA
    if m == "centroid_b": return cB
    if m == "origin":     return np.zeros(3, dtype=np.float64)
    return 0.5 * (cA + cB)

def face_touch_translation(A: o3d.geometry.PointCloud,
                           B_rot: o3d.geometry.PointCloud,
                           axis: str = 'x',
                           gap: float = 0.0) -> np.ndarray:
    (Amin, Amax) = aabb(A)
    (Bmin, Bmax) = aabb(B_rot)
    t = np.zeros(3, dtype=np.float64)
    ax = axis.lower()
    if ax == 'x':
        t[0] = (Amax[0] + gap) - Bmin[0]
    elif ax == 'y':
        t[1] = (Amax[1] + gap) - Bmin[1]
    elif ax == 'z':
        t[2] = (Amax[2] + gap) - Bmin[2]
    else:
        raise ValueError("axis must be 'x','y', or 'z'")
    return t

def select_face_indices(pts: np.ndarray, axis: str, side: str, plane_thick: float) -> np.ndarray:
    ax_i = {'x':0,'y':1,'z':2}[axis.lower()]
    vals = pts[:, ax_i]
    vmin, vmax = robust_min_max(vals)
    v_plane = vmin if side == 'min' else vmax
    return np.abs(vals - v_plane) <= plane_thick

def pick_corner_point_on_face(pcd: o3d.geometry.PointCloud,
                              main_axis: str,
                              side_main: str,
                              corner_y: str,
                              corner_z: str,
                              plane_thick_ratio: float) -> np.ndarray:
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        raise ValueError("Empty point cloud")

    diag = bbox_diag(pcd)
    plane_thick = max(1e-8, plane_thick_ratio * diag)
    face_mask = select_face_indices(pts, main_axis, side_main, plane_thick)
    face_pts = pts[face_mask] if np.any(face_mask) else pts

    y = face_pts[:,1]
    y_val = np.percentile(y, PERCENT_HIGH if corner_y == 'max' else PERCENT_LOW)
    y_mask = np.abs(y - y_val) <= plane_thick
    yz_pts = face_pts[y_mask] if np.any(y_mask) else face_pts

    z = yz_pts[:,2]
    z_val = np.percentile(z, PERCENT_HIGH if corner_z == 'max' else PERCENT_LOW)
    z_mask = np.abs(z - z_val) <= plane_thick
    cand = yz_pts[z_mask] if np.any(z_mask) else yz_pts

    c = np.mean(cand, axis=0)
    idx = np.argmax(np.sum((cand - c)**2, axis=1))
    return cand[idx].astype(np.float64)

def _lock_translation_to_axis(t: np.ndarray, axis: str) -> np.ndarray:
    ax_i = {'x':0, 'y':1, 'z':2}[axis.lower()]
    out = np.zeros_like(t)
    out[ax_i] = t[ax_i]
    return out

def corner_snap_translation(A: o3d.geometry.PointCloud,
                            B_rot: o3d.geometry.PointCloud,
                            main_axis: str,
                            gap: float,
                            corner_y: str,
                            corner_z: str,
                            plane_thick_ratio: float,
                            lock_to_axis: bool = True) -> np.ndarray:
    axis = main_axis.lower()
    if axis not in ('x','y','z'):
        raise ValueError("main_axis must be 'x','y','z'")

    side_A = 'max'  # far face on A along +axis
    side_B = 'min'  # near face on B along +axis after rotation

    cA = pick_corner_point_on_face(A, axis, side_A, corner_y, corner_z, plane_thick_ratio)
    cB = pick_corner_point_on_face(B_rot, axis, side_B, corner_y, corner_z, plane_thick_ratio)

    t = cA - cB
    ax_i = {'x':0,'y':1,'z':2}[axis]
    t[ax_i] += gap
    if lock_to_axis:
        t = _lock_translation_to_axis(t, axis)
    return t

# ---------- Feature-guided corner helpers ----------
def _load_3d_features(path: Optional[str]) -> Optional[np.ndarray]:
    if not path:
        return None
    if not os.path.exists(path):
        print(f"[feat] 3D feature file not found: {path}")
        return None
    try:
        arr = np.load(path)
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] < 1:
            print(f"[feat] invalid shape in {path}: {arr.shape}")
            return None
        return arr
    except Exception as e:
        print(f"[feat] failed to load features {path}: {e}")
        return None

def _pick_feature_extreme(feat3d: np.ndarray,
                          x_side: str = 'max',
                          y_side: str = 'max',
                          use_z: bool = False,
                          z_side: str = 'max') -> np.ndarray:
    """
    Pick extreme feature: X -> x_side ('min'/'max'), Y -> y_side, and optionally Z -> z_side.
    Returns a single (3,) point.
    """
    f = feat3d
    if f.size == 0:
        raise ValueError("Empty feature set")
    def _ext(vals, side): return np.argmin(vals) if side=='min' else np.argmax(vals)
    # Filter progressively: first X, then Y, then optional Z
    x_idx = _ext(f[:,0], x_side)
    x_val = f[x_idx,0]
    # Keep near that extreme to avoid outliers that are too isolated
    keep = np.isclose(f[:,0], x_val, rtol=0, atol=1e-6) | ( (f[:,0] - x_val) * (1 if x_side=='max' else -1) >= -1e-6 )
    cand = f[keep]
    # Y extreme from candidates
    y_idx = _ext(cand[:,1], y_side)
    cand = cand[[y_idx]]
    # Optionally Z tie-break
    if use_z and cand.shape[0] > 1:
        z_idx = _ext(cand[:,2], z_side)
        cand = cand[[z_idx]]
    return cand[0].astype(np.float64)

def feature_corner_translation(A: o3d.geometry.PointCloud,
                               B_rot: o3d.geometry.PointCloud,
                               featA: Optional[np.ndarray],
                               featB: Optional[np.ndarray],
                               main_axis: str,
                               gap: float,
                               corner_y: str,
                               corner_z: str,
                               use_z: bool,
                               lock_to_axis: bool) -> np.ndarray:
    """
    Use 3D features if available:
      - pick A: top-right (X=max, Y=max)   -> far face along +X
      - pick B: top-left  (X=min, Y=max)   -> near face along +X after rotation
    Fallback to geometric corner if features missing.
    """
    axis = main_axis.lower()
    if (featA is not None) and (featB is not None):
        # interpret "top-right" as Y=max, X=max; "top-left" as Y=max, X=min (y_up frame)
        cA = _pick_feature_extreme(featA, x_side='max', y_side='max', use_z=use_z, z_side=corner_z)
        cB = _pick_feature_extreme(featB, x_side='min', y_side='max', use_z=use_z, z_side=corner_z)
        t = cA - cB
        ax_i = {'x':0,'y':1,'z':2}[axis]
        t[ax_i] += gap
        if lock_to_axis:
            t = _lock_translation_to_axis(t, axis)
        print("[feat] using 3D feature anchors (A: top-right, B: top-left)")
        return t
    else:
        print("[feat] features missing -> fallback to geometric corner")
        return corner_snap_translation(A, B_rot, main_axis, gap, corner_y, corner_z,
                                       PLANE_THICK_RATIO, lock_to_axis=lock_to_axis)

# ---------- AUTO SOLVER (unchanged core) ----------
_SIDES = [
    ("x","+"), ("x","-"),
    ("y","+"), ("y","-"),
    ("z","+"), ("z","-"),
]

def _apply_side_translation(A, B_rot, axis, sign, gap, mode_corner):
    axis = axis.lower()
    if mode_corner == "feature_corner":
        # For AUTO we still try both sides. If sign '+': A.max with B.min; if '-': A.min with B.max.
        # Here we do not know which feature sides to use cleanly for all axes; prefer geometric fallback.
        # We thus reuse geometric "corner" here for AUTO (features mainly for manual X case).
        mode_corner = "corner"

    if mode_corner == "corner":
        if sign == "+":
            t = corner_snap_translation(A, B_rot, axis, gap, CORNER_Y, CORNER_Z, PLANE_THICK_RATIO, lock_to_axis=LOCK_TO_AXIS)
        else:
            # Place on negative side: mirror roles (A.min with B.max), negative gap
            ax_i = {'x':0,'y':1,'z':2}[axis]
            # Pick corners explicitly
            def pick(pcd, side_main):
                return pick_corner_point_on_face(pcd, axis, side_main, CORNER_Y, CORNER_Z, PLANE_THICK_RATIO)
            cA = pick(A, 'min'); cB = pick(B_rot, 'max')
            t = cA - cB
            t[ax_i] -= gap
            if LOCK_TO_AXIS:
                t = _lock_translation_to_axis(t, axis)
    else:
        # face-touch
        if sign == "+":
            t = face_touch_translation(A, B_rot, axis=axis, gap=gap)
        else:
            t = np.zeros(3, float)
            ax_i = {'x':0,'y':1,'z':2}[axis]
            Amin, _ = aabb(A)
            _, Bmax = aabb(B_rot)
            t[ax_i] = (Amin[ax_i] - gap) - Bmax[ax_i]
    return t

def _overlap_penalty(A, B_after, axis, sign):
    axis = axis.lower()
    ax_i = {'x':0,'y':1,'z':2}[axis]
    Amin, Amax = aabb(A)
    Bmin, Bmax = aabb(B_after)
    if sign == "+":
        clearance = Bmin[ax_i] - Amax[ax_i]
    else:
        clearance = Amin[ax_i] - Bmax[ax_i]
    pen = 0.0
    if clearance < 0: pen += 1e6 * (-clearance)
    pen += 10.0 * (clearance)**2
    other = [0,1,2]; other.remove(ax_i)
    Acenter_other = 0.5*(Amin[other] + Amax[other])
    Bcenter_other = 0.5*(Bmin[other] + Bmax[other])
    pen += 100.0 * np.sum((Acenter_other - Bcenter_other)**2)
    return pen

def auto_place(A, B, piv_mode, base_rot_deg, gap, snap_mode_for_auto: str) -> Tuple[o3d.geometry.PointCloud, np.ndarray, dict]:
    """
    Try rot in {+|−}|base_rot_deg| and all six sides; pick min penalty.
    'snap_mode_for_auto' is "corner" or "face" (we don't use 'feature_corner' in AUTO).
    """
    best = None
    pivot = choose_pivot(piv_mode, A, B)
    for rot_deg in (+abs(base_rot_deg), -abs(base_rot_deg)):
        T_r = rot_T("y", rot_deg, pivot)
        B_rot = o3d.geometry.PointCloud(B).transform(T_r.copy())
        for axis, sign in _SIDES:
            t = _apply_side_translation(A, B_rot, axis, sign, gap, mode_corner=snap_mode_for_auto)
            T_t = np.eye(4); T_t[:3,3] = t
            B_try = o3d.geometry.PointCloud(B_rot).transform(T_t.copy())
            pen = _overlap_penalty(A, B_try, axis, sign)
            cand = (pen, rot_deg, axis, sign, T_t @ T_r, B_try)
            if (best is None) or (pen < best[0]):
                best = cand
    pen, rot_deg, axis, sign, T_final, B_best = best
    return B_best, T_final, {"penalty": pen, "rot_deg": rot_deg, "axis": axis, "sign": sign}

# -------------------- main --------------------
def run():
    ensure_dir(OUTPUT_DIR)

    if PLY_A and PLY_B:
        fA, fB = PLY_A, PLY_B
    else:
        fA, fB = scan_two_plys(INPUT_DIR)

    print(f"[in] A (world): {fA}")
    print(f"[in] B (turned then placed): {fB}")
    if INPUT_FRAME.lower() != "y_up":
        print(f"[note] INPUT_FRAME={INPUT_FRAME} (script assumes y_up for yaw about +Y).")

    baseA = os.path.splitext(os.path.basename(fA))[0]
    baseB = os.path.splitext(os.path.basename(fB))[0]
    out_A   = os.path.join(OUTPUT_DIR, f"{baseA}_world.ply")
    out_B   = os.path.join(OUTPUT_DIR, f"{baseB}_turned_aligned.ply")
    out_M   = os.path.join(OUTPUT_DIR, f"{baseA}__plus__{baseB}_merged_turned.ply")
    out_Tnp = os.path.join(OUTPUT_DIR, f"T_{baseB}_to_{baseA}_turned.npy")
    out_Ttx = os.path.join(OUTPUT_DIR, f"T_{baseB}_to_{baseA}_turned.txt")
    out_A_c = os.path.join(OUTPUT_DIR, f"{baseA}_world_centered.ply")
    out_B_c = os.path.join(OUTPUT_DIR, f"{baseB}_turned_aligned_centered.ply")
    out_M_c = os.path.join(OUTPUT_DIR, f"{baseA}__plus__{baseB}_merged_turned_centered.ply")

    A = load_pcd(fA)
    B = load_pcd(fB)

    # Rotation first
    pivot = choose_pivot(PIVOT_MODE, A, B)
    T_r = rot_T(ROT_AXIS, ROT_DEG, pivot)
    B_rot = o3d.geometry.PointCloud(B).transform(T_r.copy())

    if PLACEMENT_MODE == "auto":
        # For auto, we keep using geometric "corner" or "face" (features optionality is ambiguous across all sides)
        snap_mode_for_auto = "corner" if SNAP_MODE != "face" else "face"
        B_aligned, T_final, info = auto_place(A, B, PIVOT_MODE, ROT_DEG, TOUCH_GAP_M, snap_mode_for_auto)
        print(f"[auto] chose rot={info['rot_deg']:+.1f}°, axis={info['axis']}, side={'+' if info['sign']=='+' else '-'}, penalty={info['penalty']:.2f}")
    else:
        # Manual placement on TOUCH_AXIS using the chosen SNAP mode
        if SNAP_MODE == "face":
            t = face_touch_translation(A, B_rot, axis=TOUCH_AXIS, gap=TOUCH_GAP_M)
        elif SNAP_MODE == "corner":
            t = corner_snap_translation(A, B_rot, main_axis=TOUCH_AXIS, gap=TOUCH_GAP_M,
                                        corner_y=CORNER_Y, corner_z=CORNER_Z,
                                        plane_thick_ratio=PLANE_THICK_RATIO, lock_to_axis=LOCK_TO_AXIS)
        elif SNAP_MODE == "feature_corner":
            featA = _load_3d_features(FEAT_A_3D)
            featB = _load_3d_features(FEAT_B_3D)
            t = feature_corner_translation(A, B_rot, featA, featB,
                                           main_axis=TOUCH_AXIS, gap=TOUCH_GAP_M,
                                           corner_y=CORNER_Y, corner_z=CORNER_Z,
                                           use_z=FEATURE_USE_Z, lock_to_axis=LOCK_TO_AXIS)
        else:
            raise ValueError("SNAP_MODE must be 'face', 'corner', or 'feature_corner'")

        T_t = np.eye(4); T_t[:3,3] = t
        T_final = T_t @ T_r
        B_aligned = o3d.geometry.PointCloud(B_rot).transform(T_t.copy())

    # Optional ICP polish
    if RUN_TINY_ICP:
        print("[icp] tiny ICP refine (polish placement).")
        A.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=50))
        B_aligned.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=50))
        icp = o3d.pipelines.registration.registration_icp(
            B_aligned, A, ICP_MAX_DIST, np.eye(4),
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=ICP_MAX_ITERS)
        )
        T_final = icp.transformation @ T_final
        B_aligned.transform(icp.transformation)
        print(f"[icp] fitness={icp.fitness:.3f}, rmse={icp.inlier_rmse:.4f}")

    merged = A + B_aligned
    if VOXEL_FOR_SAVE:
        merged = merged.voxel_down_sample(VOXEL_FOR_SAVE)

    o3d.io.write_point_cloud(out_A, A, write_ascii=True)
    o3d.io.write_point_cloud(out_B, B_aligned, write_ascii=True)
    o3d.io.write_point_cloud(out_M, merged, write_ascii=True)
    np.save(out_Tnp, T_final)
    np.savetxt(out_Ttx, T_final, fmt="%.8f")

    if SAVE_RECENTERED:
        o3d.io.write_point_cloud(out_A_c, recenter_to_origin(A), write_ascii=True)
        o3d.io.write_point_cloud(out_B_c, recenter_to_origin(B_aligned), write_ascii=True)
        o3d.io.write_point_cloud(out_M_c, recenter_to_origin(merged), write_ascii=True)

    print("[ok] Saved:")
    print("  ", out_A)
    print("  ", out_B)
    print("  ", out_M)
    if SAVE_RECENTERED:
        print("  ", out_A_c)
        print("  ", out_B_c)
        print("  ", out_M_c)
    print("  ", out_Tnp)
    print("  ", out_Ttx)

if __name__ == "__main__":
    run()
