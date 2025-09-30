#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pairwise merge with a fixed 90° turn and **face-touch** placement for y_up point clouds.

Assumptions
----------
- Both PLYs were exported in the same frame (EXPORT_FRAME="y_up" from your depth exporter):
    X = right, Y = up, Z = forward
- A is the FRONT view (world). B is the next view (e.g., RIGHT view after turning CCW).
- We rotate B by a fixed yaw about +Y, then translate it so its near face **touches** A’s far face
  along the chosen axis (default X), instead of aligning centroids (which can cause overlap).

Notes
-----
- If your rig turns CCW around the object, use ROT_DEG = +90.0 (default).
- If it turns CW, use ROT_DEG = -90.0.
- The tiny ICP step is optional and only polishes the placement slightly.

Outputs
-------
- <A>_world.ply
- <B>_turned_aligned.ply
- <A>__plus__<B>_merged_turned.ply
- (optional) *_centered.ply versions with centroid moved to origin (better initial view in some viewers)
- T_<B>_to_<A>_turned.npy / .txt  (final 4x4 transform that maps original B -> aligned B in A’s frame)
"""

import os, glob
from typing import Optional, Tuple
import numpy as np
import open3d as o3d

# =================== USER SETTINGS ===================
INPUT_DIR   = r"data/output/ply_inputs"   # used if PLY_A/PLY_B not set
OUTPUT_DIR  = r"data/output/registration_face_touch"

# Explicit files (recommended). If left empty, the first two .ply from INPUT_DIR are used.
PLY_A = r"data/output/ply_inputs/WIN_20250929_09_22_09_Pro_pcd_MASKED.ply"  # FRONT (world)
PLY_B = r"data/output/ply_inputs/WIN_20250929_09_22_25_Pro_pcd_MASKED.ply"  # NEXT view (right)

# Coordinate frame of the PLYs (must match your exporter)
INPUT_FRAME = "y_up"      # keep as "y_up" given your depth script

# Fixed yaw to apply to B (about +Y in y_up):
ROT_AXIS = "y"            # "x" | "y" | "z"
ROT_DEG  = -90.0          # CCW turn; use -90.0 for CW rigs

# Pivot for the rotation (world-space point)
PIVOT_MODE    = "centroid_A"  # "centroid_A" | "centroid_B" | "midcentroid" | "origin"

# After rotation, place B so it **touches** A along this axis (not centroid-aligned)
TOUCH_AXIS    = "x"       # 'x' | 'y' | 'z'  (default 'x' to place to the right of A)
TOUCH_GAP_M   = 0.001     # small gap (meters) to avoid z-fighting (1 mm)

# Optional finishing
VOXEL_FOR_SAVE: Optional[float] = None   # e.g., 0.008 to lightly fuse duplicates on merged output
RUN_TINY_ICP   = True
ICP_MAX_ITERS  = 30
ICP_MAX_DIST   = 0.01     # keep small so ICP only polishes, doesn't slide through A

# Viewer convenience: also save recentered copies (centroid at origin)
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

def aabb(pcd: o3d.geometry.PointCloud):
    a = np.asarray(pcd.points)
    mins = a.min(axis=0)
    maxs = a.max(axis=0)
    return mins, maxs

def choose_pivot(mode: str, A: o3d.geometry.PointCloud, B: o3d.geometry.PointCloud) -> np.ndarray:
    cA = centroid(A); cB = centroid(B)
    m = (mode or "").lower()
    if m == "centroid_a": return cA
    if m == "centroid_b": return cB
    if m == "origin":     return np.zeros(3, dtype=np.float64)
    # default: midcentroid
    return 0.5 * (cA + cB)

def face_touch_translation(A: o3d.geometry.PointCloud,
                           B_rot: o3d.geometry.PointCloud,
                           axis: str = 'x',
                           gap: float = 0.0) -> np.ndarray:
    """
    Translate B_rot so its near face just touches A's far face along `axis`.
    - For axis='x': set B.minX = A.maxX + gap.
    """
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

# -------------------- main pipeline --------------------

def run():
    ensure_dir(OUTPUT_DIR)

    # Resolve inputs
    if PLY_A and PLY_B:
        fA, fB = PLY_A, PLY_B
    else:
        fA, fB = scan_two_plys(INPUT_DIR)

    print(f"[in] A (world): {fA}")
    print(f"[in] B (turned then placed to touch): {fB}")
    if INPUT_FRAME.lower() != "y_up":
        print(f"[note] INPUT_FRAME={INPUT_FRAME} (script assumes y_up for yaw about +Y).")

    # Output paths
    baseA = os.path.splitext(os.path.basename(fA))[0]
    baseB = os.path.splitext(os.path.basename(fB))[0]
    out_A   = os.path.join(OUTPUT_DIR, f"{baseA}_world.ply")
    out_B   = os.path.join(OUTPUT_DIR, f"{baseB}_turned_aligned.ply")
    out_M   = os.path.join(OUTPUT_DIR, f"{baseA}__plus__{baseB}_merged_turned.ply")
    out_Tnp = os.path.join(OUTPUT_DIR, f"T_{baseB}_to_{baseA}_turned.npy")
    out_Ttx = os.path.join(OUTPUT_DIR, f"T_{baseB}_to_{baseA}_turned.txt")

    # Optional recentered versions
    out_A_c = os.path.join(OUTPUT_DIR, f"{baseA}_world_centered.ply")
    out_B_c = os.path.join(OUTPUT_DIR, f"{baseB}_turned_aligned_centered.ply")
    out_M_c = os.path.join(OUTPUT_DIR, f"{baseA}__plus__{baseB}_merged_turned_centered.ply")

    # Load point clouds
    A = load_pcd(fA)
    B = load_pcd(fB)

    # 1) Rotate B by fixed yaw about chosen pivot
    pivot = choose_pivot(PIVOT_MODE, A, B)
    T_r = rot_T(ROT_AXIS, ROT_DEG, pivot)
    B_rot = B.transform(T_r.copy())

    # 2) Face-touch translate along the intended axis (avoid centroid overlap)
    t = face_touch_translation(A, B_rot, axis=TOUCH_AXIS, gap=TOUCH_GAP_M)
    T_t = np.eye(4, dtype=np.float64); T_t[:3,3] = t
    B_aligned = B_rot.transform(T_t.copy())
    T_final = T_t @ T_r

    print(f"[turn] axis={ROT_AXIS.upper()}, deg={ROT_DEG:.1f}, pivot={pivot}")
    print(f"[touch] axis={TOUCH_AXIS.lower()}, t={t}, gap={TOUCH_GAP_M} m")
    print("[T_final]\n", np.array_str(T_final, precision=6, suppress_small=True))

    # 3) Optional tiny ICP (polish only; keep dist small)
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
        B_aligned = B_aligned.transform(icp.transformation)
        print(f"[icp] fitness={icp.fitness:.3f}, rmse={icp.inlier_rmse:.4f}")

    # 4) Merge
    merged = A + B_aligned
    if VOXEL_FOR_SAVE:
        merged = merged.voxel_down_sample(VOXEL_FOR_SAVE)

    # 5) Save (raw)
    o3d.io.write_point_cloud(out_A, A, write_ascii=True)
    o3d.io.write_point_cloud(out_B, B_aligned, write_ascii=True)
    o3d.io.write_point_cloud(out_M, merged, write_ascii=True)
    np.save(out_Tnp, T_final)
    np.savetxt(out_Ttx, T_final, fmt="%.8f")

    # 6) Also save recentered copies (for nicer default framing in viewers)
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
