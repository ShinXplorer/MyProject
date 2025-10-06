#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Four-view registration by -90° yaw steps using feature-anchored near↔far snapping (y_up frame).

Fixes vs previous:
- Snapping and gap are applied along the current view's local +X direction in world:
  d = R_world[:,0] from this step's rotation. No more always-+X gaps.
- Extremes (near/far) are chosen by projection onto d (dot product), not world X only.
- Geometric fallback selects faces by the same projection direction d.

Sequence
--------
A is world.
ref0 = A's "far along d" (for step 1, d is world +X so this is rightmost/top).
For each V in [B, C, D]:
  1) Rotate V by ROT_DEG_PER_STEP about +Y around merged centroid.
  2) Let d = current R_world[:,0] (local +X in world).
  3) Pick V's "near along d" (min dot(p,d)) as anchor; translate so it hits ref; add small gap along +d.
  4) Merge V; set ref = V's "far along d" (max dot(p,d)).

Assumptions
-----------
- y_up PLYs: X=right, Y=up, Z=forward.
- ROT_DEG_PER_STEP = -90 (CW around +Y). Adjust if you want CCW (+90).

Outputs
-------
A_world.ply
B_turned_aligned.ply, C_turned_aligned.ply, D_turned_aligned.ply
A__plus__B__plus__C__plus__D_merged_turned.ply
Transforms T_<name>_to_A_turned.{npy,txt}
"""

import os
from typing import Optional, List, Dict, Literal
import numpy as np
import open3d as o3d

# =================== USER SETTINGS ===================

OUTPUT_DIR  = r"data/output/registration_walk_four"

# A/B inputs (C uses A, D uses B to simulate)
PLY_A = r"data/output/ply_inputs/WIN_20250929_09_22_09_Pro_pcd_MASKED.ply"
PLY_B = r"data/output/ply_inputs/WIN_20250929_09_22_25_Pro_pcd_MASKED.ply"
FEAT_A = r"data/output/depth_only/WIN_20250929_09_22_09_Pro_keypoints.npy"
FEAT_B = r"data/output/depth_only/WIN_20250929_09_22_25_Pro_keypoints.npy"

VIEWS: List[Dict] = [
    dict(name="A", ply=PLY_A, feat=FEAT_A),
    dict(name="B", ply=PLY_B, feat=FEAT_B),
    dict(name="C", ply=PLY_A, feat=FEAT_A),  # simulate
    dict(name="D", ply=PLY_B, feat=FEAT_B),  # simulate
]

ROT_AXIS = "y"
ROT_DEG_PER_STEP = -90.0   # CW about +Y each step

TOUCH_GAP_M = 0.001        # along +d (meters)

# Geometric fallback params
CORNER_Y = "max"           # 'min' or 'max' (top)
CORNER_Z = "max"           # only used to break ties on fallback
PLANE_THICK_RATIO = 0.02
PERCENT_LOW  = 5.0
PERCENT_HIGH = 95.0

# ICP (small polish)
RUN_TINY_ICP   = True
ICP_MAX_ITERS  = 30
ICP_MAX_DIST   = 0.01

VOXEL_FOR_SAVE: Optional[float] = None
SAVE_RECENTERED = True

# =================== HELPERS ===================

def ensure_dir(p): os.makedirs(p, exist_ok=True)

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

def aabb(pcd: o3d.geometry.PointCloud):
    pts = np.asarray(pcd.points)
    mins = np.percentile(pts, PERCENT_LOW, axis=0)
    maxs = np.percentile(pts, PERCENT_HIGH, axis=0)
    return mins, maxs

def bbox_diag(pcd: o3d.geometry.PointCloud) -> float:
    mn, mx = aabb(pcd)
    return float(np.linalg.norm(mx - mn))

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
        raise ValueError("ROT_AXIS must be 'x','y', or 'z'")
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = R
    T[:3,3] = pivot - R @ pivot   # rotate about pivot
    return T

def _load_3d_features(path: Optional[str]) -> Optional[np.ndarray]:
    if not path or not os.path.exists(path):
        return None
    try:
        arr = np.load(path)
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] < 1:
            return None
        return arr
    except Exception:
        return None

def _transform_points(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    if pts is None:
        return None
    H = np.ones((pts.shape[0], 4), dtype=np.float64)
    H[:,:3] = pts
    WH = (T @ H.T).T
    return WH[:,:3]

def _R_from_T(T: np.ndarray) -> np.ndarray:
    return T[:3,:3]

def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + 1e-12)

# ---- Feature extremes by an arbitrary direction d (world 3-vector) ----

def pick_extreme_along_dir(feats_world: np.ndarray,
                           d_world: np.ndarray,
                           side: Literal['min','max']) -> np.ndarray:
    """Pick point with min/max dot(p, d_world)."""
    if feats_world is None or feats_world.size == 0:
        raise ValueError("No features")
    s = feats_world @ d_world
    idx = np.argmin(s) if side == 'min' else np.argmax(s)
    return feats_world[idx].astype(np.float64)

# ---- Geometric fallback: face defined by dot(p, d) ----

def pick_geometric_face_corner_along_dir(pcd_world: o3d.geometry.PointCloud,
                                         d_world: np.ndarray,
                                         side: Literal['min','max'],
                                         plane_thick_ratio: float,
                                         corner_y: str,
                                         corner_z: str) -> np.ndarray:
    """
    Select points on the near/far plane wrt direction d (dot-based),
    then break ties by Y and Z extremes.
    """
    pts = np.asarray(pcd_world.points)
    if pts.size == 0:
        raise ValueError("Empty point cloud")

    d = _unit(d_world)
    proj = pts @ d  # scalar projection
    diag = bbox_diag(pcd_world)
    plane_thick = max(1e-8, plane_thick_ratio * diag)

    lo = np.percentile(proj, PERCENT_LOW)
    hi = np.percentile(proj, PERCENT_HIGH)
    plane = lo if side == 'min' else hi
    mask_face = np.abs(proj - plane) <= plane_thick
    face_pts = pts[mask_face] if np.any(mask_face) else pts

    # tie-break by Y extreme (top/bottom)
    y = face_pts[:,1]
    y_val = np.percentile(y, 95.0 if corner_y == 'max' else 5.0)
    mask_y = np.abs(y - y_val) <= plane_thick
    yz_pts = face_pts[mask_y] if np.any(mask_y) else face_pts

    # tie-break by Z extreme (front/back)
    z = yz_pts[:,2]
    z_val = np.percentile(z, 95.0 if corner_z == 'max' else 5.0)
    mask_z = np.abs(z - z_val) <= plane_thick
    cand = yz_pts[mask_z] if np.any(mask_z) else yz_pts

    # pick stable: farthest from cluster center
    c = np.mean(cand, axis=0)
    idx = np.argmax(np.sum((cand - c)**2, axis=1))
    return cand[idx].astype(np.float64)

# ---- Unified pickers (features preferred, else geometry) ----

def pick_view_anchor(feat_local: Optional[np.ndarray],
                     view_rotated: o3d.geometry.PointCloud,
                     T_to_world: np.ndarray,
                     d_world: np.ndarray,
                     side: Literal['min','max']) -> np.ndarray:
    """
    Anchor on the near/far plane along d_world.
    If feature array exists -> transform to world then pick extreme along d.
    Else -> geometric fallback on the rotated cloud (already in world with T_to_world).
    """
    if feat_local is not None and feat_local.size:
        feats_w = _transform_points(feat_local, T_to_world)
        return pick_extreme_along_dir(feats_w, d_world, side)

    tmp = o3d.geometry.PointCloud(view_rotated)   # already R-applied; apply T_to_world if needed
    tmp.transform(T_to_world)
    return pick_geometric_face_corner_along_dir(tmp, d_world, side,
                                                plane_thick_ratio=PLANE_THICK_RATIO,
                                                corner_y=CORNER_Y, corner_z=CORNER_Z)

# =================== MAIN ===================

def run():
    ensure_dir(OUTPUT_DIR)
    assert len(VIEWS) == 4, "This script expects exactly A, B, C, D in VIEWS."

    # A as world
    A = load_pcd(VIEWS[0]['ply'])
    feats_A = _load_3d_features(VIEWS[0].get('feat'))
    merged = A

    out_A = os.path.join(OUTPUT_DIR, f"{VIEWS[0]['name']}_world.ply")
    o3d.io.write_point_cloud(out_A, merged, write_ascii=True)
    if SAVE_RECENTERED:
        o3d.io.write_point_cloud(os.path.join(OUTPUT_DIR, f"{VIEWS[0]['name']}_world_centered.ply"),
                                 recenter_to_origin(merged), write_ascii=True)
    print(f"[in] A: {VIEWS[0]['ply']}  -> saved as world")

    # Step 0 seam direction d0 = world +X (same as local +X of A in world)
    d0 = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    # Initial reference = far along d0 (rightmost/top on A)
    if feats_A is not None and feats_A.size:
        ref_point_world = pick_extreme_along_dir(feats_A, d0, side='max')
    else:
        ref_point_world = pick_geometric_face_corner_along_dir(merged, d0, side='max',
                                                               plane_thick_ratio=PLANE_THICK_RATIO,
                                                               corner_y=CORNER_Y, corner_z=CORNER_Z)
    print(f"[ref0] A far(+d0) anchor @ {ref_point_world}")

    merged_name_parts = [VIEWS[0]['name']]

    # Walk B, C, D
    for i in range(1, 4):
        name = VIEWS[i]['name']
        ply  = VIEWS[i]['ply']
        feat = _load_3d_features(VIEWS[i].get('feat'))

        print(f"\n[step {i}] Place {name}: rotate {ROT_DEG_PER_STEP:+.1f}° about +Y, snap NEAR(min) along current d")

        next_raw = load_pcd(ply)

        # Rotate about merged centroid
        # --- rotate this view by the cumulative yaw (i * step) about merged centroid ---
        angle_deg = i * ROT_DEG_PER_STEP          # <- cumulative, not just one step
        pivot = centroid(merged)
        T_r = rot_T(ROT_AXIS, angle_deg, pivot)   # rotate by the cumulative angle
        R = _R_from_T(T_r)

        # current seam direction = this view's local +X mapped into world
        d_world = _unit(R[:, 0])

        # apply rotation
        next_rot = o3d.geometry.PointCloud(next_raw).transform(T_r.copy())


        # Pick NEAR (min along d) of this view in world
        next_near_world = pick_view_anchor(
            feat_local=feat,
            view_rotated=next_rot,
            T_to_world=np.eye(4),      # next_rot is already in world (T_r applied),
            d_world=d_world,
            side='min'
        )

        # Translate to snap near->ref, then add gap along +d
        t = ref_point_world - next_near_world
        t = t + (TOUCH_GAP_M * d_world)
        T_t = np.eye(4); T_t[:3,3] = t

        # Final transform
        T_final = T_t @ T_r
        next_aligned = o3d.geometry.PointCloud(next_rot).transform(T_t.copy())

        # ICP polish
        if RUN_TINY_ICP:
            merged.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=50))
            next_aligned.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=50))
            icp = o3d.pipelines.registration.registration_icp(
                next_aligned, merged, ICP_MAX_DIST, np.eye(4),
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=ICP_MAX_ITERS)
            )
            next_aligned.transform(icp.transformation)
            T_final = icp.transformation @ T_final
            print(f"[icp] fitness={icp.fitness:.3f}, rmse={icp.inlier_rmse:.4f}")

        # Save aligned view & T
        out_vi = os.path.join(OUTPUT_DIR, f"{name}_turned_aligned.ply")
        o3d.io.write_point_cloud(out_vi, next_aligned, write_ascii=True)
        if SAVE_RECENTERED:
            o3d.io.write_point_cloud(os.path.join(OUTPUT_DIR, f"{name}_turned_aligned_centered.ply"),
                                     recenter_to_origin(next_aligned), write_ascii=True)
        np.save(os.path.join(OUTPUT_DIR, f"T_{name}_to_{VIEWS[0]['name']}_turned.npy"), T_final)
        np.savetxt(os.path.join(OUTPUT_DIR, f"T_{name}_to_{VIEWS[0]['name']}_turned.txt"), T_final, fmt="%.8f")

        # Merge
        merged = merged + next_aligned
        if VOXEL_FOR_SAVE:
            merged = merged.voxel_down_sample(VOXEL_FOR_SAVE)

        # Update reference to FAR (max along same d) of the just-placed view in world
        # (we must use the *final* transform on features if present)
        if feat is not None and feat.size:
            feats_world_final = _transform_points(feat, T_final)
            ref_point_world = pick_extreme_along_dir(feats_world_final, d_world, side='max')
        else:
            # Use the aligned geometry as fallback
            ref_point_world = pick_geometric_face_corner_along_dir(next_aligned, d_world, side='max',
                                                                   plane_thick_ratio=PLANE_THICK_RATIO,
                                                                   corner_y=CORNER_Y, corner_z=CORNER_Z)
        print(f"[ref{i}] {name} far(+d) anchor @ {ref_point_world}")

        merged_name_parts.append(name)

    # Save final merged
    final_name = "__plus__".join(merged_name_parts) + "_merged_turned.ply"
    out_M = os.path.join(OUTPUT_DIR, final_name)
    o3d.io.write_point_cloud(out_M, merged, write_ascii=True)
    if SAVE_RECENTERED:
        o3d.io.write_point_cloud(os.path.join(OUTPUT_DIR, final_name.replace('.ply','_centered.ply')),
                                 recenter_to_origin(merged), write_ascii=True)

    print("\n[done] Saved:")
    print("  ", out_A)
    for i in range(1,4):
        print("  ", os.path.join(OUTPUT_DIR, f"{VIEWS[i]['name']}_turned_aligned.ply"))
    print("  ", out_M)

if __name__ == "__main__":
    run()
