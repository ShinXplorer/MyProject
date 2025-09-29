#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Deterministic 2-view merge with a known fixed rotation about a chosen axis.

- Keeps first PLY as world.
- Rotates second PLY by ROT_DEG about axis AXIS in right-hand convention.
- Rotation is applied about a chosen pivot (centroids midpoint by default).
- Then translation:
    * X/Z by centroid match (as before)
    * Y by a selectable statistic (centroid / min / max / median / robust percentiles)
- Optional tiny ICP refine modes:
    * AXIS-LOCKED REFINE (recommended): small ICP then keep only the tweak about
      the chosen axis (e.g., yaw for AXIS='y'). Translation is kept.
    * UNCONSTRAINED ICP (original): disabled by default.

Axis is in the current PLY coordinate frame (as stored in your point clouds).
"""

import os, glob, math
from typing import Optional, Tuple
import numpy as np
import open3d as o3d

# =============== USER SETTINGS ===============
INPUT_DIR   = r"data/output/ply_inputs"      # or leave and set PLY_A / PLY_B
OUTPUT_DIR  = r"data/output/registration_fixed_axis"

PLY_A       = r"data/output/ply_inputs/WIN_20250929_09_22_09_Pro_pcd_MASKED.ply"  # world
PLY_B       = r"data/output/ply_inputs/WIN_20250929_09_22_25_Pro_pcd_MASKED.ply"  # to be aligned

AXIS        = "y"             # "x" | "y" | "z"
ROT_DEG     = 90.0            # + = CCW by right-hand rule about the chosen axis
PIVOT_MODE  = "midcentroid"   # "midcentroid" | "centroid_A" | "centroid_B" | "origin"

# Control vertical (Y) alignment after rotation
Y_ALIGN_MODE = "robust_min"   # "centroid" | "min" | "max" | "median" | "robust_min" | "robust_max"
Y_ALIGN_Q    = 5.0            # percentile for robust_* (use 5..10)

VOXEL_FOR_SAVE: Optional[float] = None  # e.g., 0.008 to lightly fuse duplicates

# ---- NEW: axis-locked micro-refine (recommended) ----
RUN_AXIS_LOCKED_REFINE = True   # keeps rotation around AXIS exact + small extra about AXIS only
REFINE_VOX             = 0.01   # m (downsample for ICP)
REFINE_MAX_DIST        = 0.03   # m (ICP correspondence distance)
REFINE_MAX_ITERS       = 30
REFINE_DEG_CLAMP       = 5.0    # allow ± this many degrees tweak around the chosen AXIS
# -----------------------------------------------------

# (Legacy) unconstrained ICP (off by default)
RUN_TINY_ICP  = False
ICP_MAX_ITERS = 30
ICP_MAX_DIST  = 0.01            # meters
# ============================================


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def scan_two_plys(folder: str) -> Tuple[str, str]:
    fs = sorted(glob.glob(os.path.join(folder, "*.ply")))
    if len(fs) < 2:
        raise FileNotFoundError(f"Need at least two .ply files in: {folder}")
    if len(fs) > 2:
        print(f"[warn] Found {len(fs)} PLYs, using first two after sort.")
    return fs[0], fs[1]


def load_pcd(path: str) -> o3d.geometry.PointCloud:
    p = o3d.io.read_point_cloud(path)
    if p.is_empty():
        raise IOError(f"Empty/invalid PLY: {path}")
    return p


def centroid(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    return np.asarray(pcd.get_center(), dtype=np.float64)


def _axis_basis(axis: str) -> Tuple[int, np.ndarray]:
    a = axis.lower()
    if a == "x": return 0, np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if a == "y": return 1, np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if a == "z": return 2, np.array([0.0, 0.0, 1.0], dtype=np.float64)
    raise ValueError("AXIS must be 'x', 'y', or 'z'")


def _rodrigues_from_axis_angle(u: np.ndarray, theta: float) -> np.ndarray:
    """Return 3x3 rotation for unit axis u and angle theta (radians)."""
    u = u / (np.linalg.norm(u) + 1e-12)
    ux, uy, uz = u
    K = np.array([[0, -uz, uy],
                  [uz, 0, -ux],
                  [-uy, ux, 0]], dtype=np.float64)
    I = np.eye(3, dtype=np.float64)
    s, c = math.sin(theta), math.cos(theta)
    return I + s * K + (1 - c) * (K @ K)


def _axis_angle_vector_from_R(R: np.ndarray) -> np.ndarray:
    """Return axis-angle vector r (rx, ry, rz) where |r|=theta, r_hat = axis."""
    tr = np.trace(R)
    cos_theta = max(-1.0, min(1.0, 0.5 * (tr - 1.0)))
    theta = math.acos(cos_theta)
    if theta < 1e-8:
        return np.zeros(3, dtype=np.float64)
    v = np.array([R[2,1] - R[1,2],
                  R[0,2] - R[2,0],
                  R[1,0] - R[0,1]], dtype=np.float64)
    axis = v / (2.0 * math.sin(theta))
    return axis * theta


def rot_T_axis(axis: str, deg: float, pivot: np.ndarray) -> np.ndarray:
    axis = axis.lower()
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    if axis == "x":
        R = np.array([[1, 0, 0],
                      [0, c,-s],
                      [0, s, c]], dtype=np.float64)
    elif axis == "y":
        R = np.array([[ c, 0, s],
                      [ 0, 1, 0],
                      [-s, 0, c]], dtype=np.float64)
    elif axis == "z":
        R = np.array([[ c,-s, 0],
                      [ s, c, 0],
                      [ 0, 0, 1]], dtype=np.float64)
    else:
        raise ValueError("AXIS must be 'x', 'y', or 'z'")
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = R
    # rotate about pivot: x' = R(x - p) + p = R x + (p - R p)
    T[:3,3] = pivot - R @ pivot
    return T


def choose_pivot(mode: str, cA: np.ndarray, cB: np.ndarray) -> np.ndarray:
    mode = mode.lower()
    if mode == "centroid_a": return cA
    if mode == "centroid_b": return cB
    if mode == "origin":     return np.zeros(3, dtype=np.float64)
    return 0.5 * (cA + cB)  # "midcentroid"


def _axis_stat(pcd: o3d.geometry.PointCloud, axis_idx: int, mode: str, q: float=5.0) -> float:
    """Return a statistic (mean/median/min/max/robust) along a coordinate axis."""
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        return 0.0
    col = pts[:, axis_idx]
    m = (mode or "centroid").lower()
    if m == "centroid":   return float(col.mean())
    if m == "median":     return float(np.median(col))
    if m == "min":        return float(col.min())
    if m == "max":        return float(col.max())
    if m == "robust_min": return float(np.percentile(col, q))
    if m == "robust_max": return float(np.percentile(col, 100.0 - q))
    raise ValueError("Y_ALIGN_MODE must be one of: centroid|min|max|median|robust_min|robust_max")


def _clone_and_transform(pcd: o3d.geometry.PointCloud, T: np.ndarray) -> o3d.geometry.PointCloud:
    """Open3D transform is in-place; this returns a transformed copy safely."""
    q = pcd.clone() if hasattr(pcd, "clone") else o3d.geometry.PointCloud(pcd)
    q.transform(T)
    return q


def run():
    ensure_dir(OUTPUT_DIR)

    if PLY_A and PLY_B:
        fA, fB = PLY_A, PLY_B
    else:
        fA, fB = scan_two_plys(INPUT_DIR)

    print(f"[in] A (world): {fA}")
    print(f"[in] B (to rotate+translate): {fB}")

    A = load_pcd(fA)   # world
    B = load_pcd(fB)   # to be aligned

    cA = centroid(A)
    cB = centroid(B)
    pivot = choose_pivot(PIVOT_MODE, cA, cB)
    print(f"[centroid] cA={cA}, cB={cB}, pivot({PIVOT_MODE})={pivot}")

    # 1) Fixed rotation about requested axis
    T_rot = rot_T_axis(AXIS, ROT_DEG, pivot)
    B_rot = _clone_and_transform(B, T_rot)

    # 2) Translation:
    #    - X/Z by centroids (keep previous behavior)
    #    - Y by selected statistic to avoid "B slightly under A"
    cB_rot = centroid(B_rot)
    t = cA - cB_rot  # start from centroid alignment for all axes

    try:
        yA     = _axis_stat(A,     axis_idx=1, mode=Y_ALIGN_MODE, q=Y_ALIGN_Q)
        yB_rot = _axis_stat(B_rot, axis_idx=1, mode=Y_ALIGN_MODE, q=Y_ALIGN_Q)
        t[1]   = yA - yB_rot
        print(f"[y-align] mode={Y_ALIGN_MODE} (q={Y_ALIGN_Q}) -> yA={yA:.5f}, yB'={yB_rot:.5f}, Δy={t[1]:.5f}")
    except Exception as e:
        print(f"[warn] Y alignment fallback to centroid due to: {e}")

    T_trans = np.eye(4, dtype=np.float64); T_trans[:3,3] = t
    B_aligned = _clone_and_transform(B_rot, T_trans)

    T_final = T_trans @ T_rot  # Transform B -> A
    print(f"[transform] Applied {ROT_DEG:.1f}° around {AXIS.upper()} and translation with Y_ALIGN_MODE='{Y_ALIGN_MODE}'.")
    print("[T_final]\n", np.array_str(T_final, precision=6, suppress_small=True))

    # 2.5) NEW: axis-locked micro-refine (keeps only the tweak about AXIS)
    if RUN_AXIS_LOCKED_REFINE:
        print("[refine] axis-locked micro-ICP around fixed rotation.")
        # Downsample for stability/speed
        A_icp = A.voxel_down_sample(REFINE_VOX)
        B_icp = B_aligned.voxel_down_sample(REFINE_VOX)

        # Normals (point-to-plane)
        A_icp.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3*REFINE_VOX, max_nn=50))
        B_icp.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3*REFINE_VOX, max_nn=50))

        icp = o3d.pipelines.registration.registration_icp(
            B_icp, A_icp, REFINE_MAX_DIST, np.eye(4),
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=REFINE_MAX_ITERS)
        )
        R_icp = icp.transformation[:3,:3]
        t_icp = icp.transformation[:3,3]

        # Project small rotation onto chosen axis
        rvec = _axis_angle_vector_from_R(R_icp)        # (rx, ry, rz)
        idx, e = _axis_basis(AXIS)
        theta_axis = float(rvec[idx])                  # radians about AXIS
        theta_deg  = math.degrees(theta_axis)
        theta_deg  = max(-REFINE_DEG_CLAMP, min(REFINE_DEG_CLAMP, theta_deg))
        theta_axis = math.radians(theta_deg)

        R_axis = _rodrigues_from_axis_angle(e, theta_axis)

        T_axis_locked = np.eye(4, dtype=np.float64)
        T_axis_locked[:3,:3] = R_axis
        T_axis_locked[:3, 3] = t_icp

        B_aligned = _clone_and_transform(B_aligned, T_axis_locked)
        T_final   = T_axis_locked @ T_final

        print(f"[refine] fitness={icp.fitness:.3f}, rmse={icp.inlier_rmse:.4f}, "
              f"Δ{AXIS}-deg={theta_deg:+.2f}° (clamped ±{REFINE_DEG_CLAMP}°)")

    # 3) Optional legacy tiny ICP (unconstrained)
    if RUN_TINY_ICP:
        print("[icp] tiny unconstrained ICP refine (may slightly change rotation).")
        A.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=50))
        B_aligned.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=50))
        icp = o3d.pipelines.registration.registration_icp(
            B_aligned, A, ICP_MAX_DIST, np.eye(4),
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=ICP_MAX_ITERS)
        )
        T_final = icp.transformation @ T_final
        B_aligned = _clone_and_transform(B_aligned, icp.transformation)
        print(f"[icp] fitness={icp.fitness:.3f}, rmse={icp.inlier_rmse:.4f}")

    # 4) Merge & save
    merged = A + B_aligned
    if VOXEL_FOR_SAVE:
        merged = merged.voxel_down_sample(VOXEL_FOR_SAVE)

    baseA = os.path.splitext(os.path.basename(fA))[0]
    baseB = os.path.splitext(os.path.basename(fB))[0]
    out_A   = os.path.join(OUTPUT_DIR, f"{baseA}_world.ply")
    out_B   = os.path.join(OUTPUT_DIR, f"{baseB}_aligned_axis{AXIS}_{int(ROT_DEG)}deg.ply")
    out_M   = os.path.join(OUTPUT_DIR, f"{baseA}__plus__{baseB}_merged_axis{AXIS}_{int(ROT_DEG)}deg.ply")
    out_Tnp = os.path.join(OUTPUT_DIR, f"T_{baseB}_to_{baseA}_axis{AXIS}_{int(ROT_DEG)}deg.npy")
    out_Ttx = os.path.join(OUTPUT_DIR, f"T_{baseB}_to_{baseA}_axis{AXIS}_{int(ROT_DEG)}deg.txt")

    o3d.io.write_point_cloud(out_A, A, write_ascii=True)
    o3d.io.write_point_cloud(out_B, B_aligned, write_ascii=True)
    o3d.io.write_point_cloud(out_M, merged, write_ascii=True)
    np.save(out_Tnp, T_final); np.savetxt(out_Ttx, T_final, fmt="%.8f")

    print(f"[ok] Saved:\n  {out_A}\n  {out_B}\n  {out_M}\n  {out_Tnp}\n  {out_Ttx}")


if __name__ == "__main__":
    run()
