#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import cv2
from pathlib import Path

# =========================
#  Utility: calib loader
# =========================
def load_kitti_calib(calib_txt_path: str):
    """
    calib.txt を読み取り、Velodyne -> Camera の 4x4 変換行列 Tr のみを返す。
    ファイルは "Tr: ..." の1行のみを含むことを想定。
    """
    assert os.path.exists(calib_txt_path), f"calib file not found: {calib_txt_path}"
    Tr = None

    with open(calib_txt_path, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    for ln in lines:
        key = ln[:2]   # expect 'Tr'
        if key == 'Tr':
            arr = np.fromstring(ln[4:], sep=' ').reshape(3, 4).astype(np.float32)
            T = np.eye(4, dtype=np.float32)
            T[:3, :] = arr
            Tr = T

    assert Tr is not None, "calib.txt must contain Tr"
    return Tr


def load_frame_paths(root: str, seq: str, frame_idx: int):
    """
    corri2p_data 配下の想定パスから、該当フレームのファイルパスを返す。
    単一カメラ版: img, K ディレクトリを想定。
    """
    seq_dir = Path(root) / "sequences" / f"{int(seq):02d}"
    img_dir = seq_dir / "img"
    K_dir = seq_dir / "K"
    pc_dir = seq_dir / "pc_with_normal"
    calib_path = Path(root) / "calib" / f"{int(seq):02d}" / "calib.txt"

    img_path = img_dir / f"{frame_idx:06d}.npy"
    K_path = K_dir / f"{frame_idx:06d}.npy"
    pc_path = pc_dir / f"{frame_idx:06d}.npy"

    for p in [img_path, K_path, pc_path, calib_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    return str(img_path), str(K_path), str(pc_path), str(calib_path)


# =========================
#  Projection utilities
# =========================
def project_points(K: np.ndarray, T_cam_velo: np.ndarray, xyz_velo: np.ndarray):
    """
    点群(velo座標, shape=(3,N))をカメラ画像へ投影して、画素座標(u,v)と深度zを返す。
    T_cam_velo: Velodyne -> Camera の 4x4 変換（例えば P{2,3}*Tr）
    K: 3x3 カメラ内部行列
    """
    assert xyz_velo.shape[0] == 3
    N = xyz_velo.shape[1]
    xyz1 = np.vstack([xyz_velo, np.ones((1, N), dtype=xyz_velo.dtype)])  # (4,N)

    # カメラ座標へ
    xyz_cam = (T_cam_velo @ xyz1)[:3, :]  # (3,N)

    # 透視投影
    uvw = K @ xyz_cam  # (3,N)
    u = uvw[0, :] / (uvw[2, :] + 1e-8)
    v = uvw[1, :] / (uvw[2, :] + 1e-8)
    z = xyz_cam[2, :]

    return u, v, z


# =========================
#  Visualization
# =========================
def visualize_point_cloud(xyz: np.ndarray, intensity: np.ndarray, normals: np.ndarray,
                          color_mode: str = "intensity", voxel_size: float = None,
                          intensity_cmap: str = "plasma"):
    """
    Open3Dで点群を描画。color_mode は 'intensity' または 'normal'。
    voxel_size を与えるとOpen3Dで体素ダウンサンプルしてから表示。
    """
    assert xyz.shape[0] == 3
    N = xyz.shape[1]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.T.copy())  # (N,3)

    if color_mode == "intensity" and intensity is not None:
        intens = intensity.reshape(-1)
        if intens.size != N:
            raise ValueError("intensity length mismatch")
        # robust normalization: clip to [pmin, pmax] percentiles to avoid outliers
        pmin, pmax = np.percentile(intens, [2.0, 98.0])
        if pmax - pmin < 1e-8:
            norm = np.clip(intens - pmin, 0.0, 1.0)
        else:
            norm = (intens - pmin) / (pmax - pmin)
            norm = np.clip(norm, 0.0, 1.0)
        cmap = cm.get_cmap(intensity_cmap)
        colors = cmap(norm)[:, :3].astype(np.float32)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        # print summary for debugging
        print(f"Intensity range raw [{float(intens.min()):.3f}, {float(intens.max()):.3f}], clipped [{pmin:.3f}, {pmax:.3f}], cmap={intensity_cmap}")

    elif color_mode == "normal" and normals is not None and normals.shape[0] >= 3:
        # 法線可視化（[-1,1] -> [0,1] に）
        nrm = normals[:3, :].T.copy()
        pcd.normals = o3d.utility.Vector3dVector(nrm.astype(np.float32))
        colors = (nrm * 0.5 + 0.5).clip(0, 1)
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float32))

    else:
        # デフォルト色（グレー）
        colors = np.full((N, 3), 0.7, dtype=np.float32)
        pcd.colors = o3d.utility.Vector3dVector(colors)

    if voxel_size is not None and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=float(voxel_size))

    o3d.visualization.draw_geometries([pcd], window_name="Open3D Point Cloud")


def visualize_image_with_projection(img: np.ndarray, u: np.ndarray, v: np.ndarray, z: np.ndarray,
                                    point_stride: int = 1, dot_size: int = 2,
                                    depth_cmap: str = "jet"):
    """
    Matplotlibで画像を表示し、(u,v) に点群投影を重ねる。
    z>0 かつ 画像内の点のみを描画。
    """
    assert img.ndim == 3 and img.shape[2] == 3
    H, W = img.shape[:2]

    # 画像内かつ手前側のみ
    mask = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u_plot = u[mask][::point_stride]
    v_plot = v[mask][::point_stride]
    z_plot = z[mask][::point_stride]

    plt.figure(figsize=(10, 5))
    plt.imshow(img)
    sc = plt.scatter(u_plot, v_plot, c=z_plot, s=dot_size, cmap=depth_cmap, alpha=0.8)
    cbar = plt.colorbar(sc, fraction=0.046, pad=0.04)
    cbar.set_label("Depth (Z in camera frame)")
    plt.title("Image with Projected Point Cloud")
    plt.axis('off')
    plt.show()


# =========================
#  Main
# =========================
def main():
    parser = argparse.ArgumentParser(description="Visualize single-camera KITTI-like dataset (image + point cloud).")
    parser.add_argument("--root", required=True, help="Path to corri2p_data root")
    parser.add_argument("--seq", required=True, help="Sequence id, e.g., 09 or 10")
    parser.add_argument("--frame", type=int, required=True, help="Frame index, e.g., 0")
    parser.add_argument("--color_mode", choices=["intensity", "normal", "gray"], default="intensity",
                        help="Coloring for 3D point cloud")
    parser.add_argument("--voxel_size", type=float, default=0.0, help="Voxel size for O3D downsampling (meters)")
    parser.add_argument("--stride", type=int, default=1, help="Plot every N-th projected point on image")
    parser.add_argument("--dot_size", type=int, default=1, help="Dot size for projected points on image")
    parser.add_argument("--intensity_cmap", type=str, default="plasma",
                        help="Matplotlib colormap name for intensity coloring (e.g. plasma, viridis, inferno)")
    args = parser.parse_args()

    # 1) ファイルパスを取得
    img_path, K_path, pc_path, calib_path = load_frame_paths(args.root, args.seq, args.frame)

    # 2) ロード
    img = np.load(img_path)  # (H,W,3) uint8
    assert img.ndim == 3 and img.shape[2] == 3, f"Unexpected image shape: {img.shape}"
    K = np.load(K_path).astype(np.float32)  # (3,3)

    # --- Added: compute and print image resolution and FoV (horizontal, vertical, diagonal) ---
    H, W = img.shape[:2]
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    # print resolution first
    if fx <= 0 or fy <= 0:
        print(f"Image resolution: {W} x {H} (W x H) -- Invalid focal length in K: fx={fx}, fy={fy}")
    else:
        fov_h_rad = 2.0 * np.arctan((W / 2.0) / fx)
        fov_v_rad = 2.0 * np.arctan((H / 2.0) / fy)
        diag_half = np.sqrt((W / 2.0) ** 2 + (H / 2.0) ** 2)
        f_mean = max(1e-8, np.sqrt(fx * fy))
        fov_d_rad = 2.0 * np.arctan(diag_half / f_mean)
        fov_h_deg = np.degrees(fov_h_rad)
        fov_v_deg = np.degrees(fov_v_rad)
        fov_d_deg = np.degrees(fov_d_rad)
        print(f"Image resolution: {W} x {H} (W x H) | FoV: horizontal={fov_h_deg:.3f}°, vertical={fov_v_deg:.3f}°, diagonal={fov_d_deg:.3f}°")
    # --- end added ---

    pc_pack = np.load(pc_path).astype(np.float32)  # (C,N)

    # 3) pc を分解: XYZ / intensity / normals
    assert pc_pack.ndim == 2 and pc_pack.shape[0] >= 7, \
        f"pc shape must be (C>=7, N). got {pc_pack.shape}"
    xyz = pc_pack[0:3, :]
    intensity = pc_pack[3:4, :].reshape(-1)
    normals = pc_pack[4:7, :]

    # 4) calib 読み込み → Velodyne -> Camera の変換 (calib.txt に Tr のみが入っている想定)
    Tr = load_kitti_calib(calib_path)
    T_cam_velo = Tr.astype(np.float32)

    # 5) 投影 (画像側のKはフレームごとに保存されたものを使用)
    u, v, z = project_points(K, T_cam_velo, xyz)

    # 6) 可視化
    # 6-1. 3D (Open3D)
    if args.color_mode == "gray":
        color_mode = "none"
    else:
        color_mode = args.color_mode
    visualize_point_cloud(xyz, intensity, normals,
                          color_mode=color_mode,
                          voxel_size=(args.voxel_size if args.voxel_size > 0 else None),
                          intensity_cmap=args.intensity_cmap)

    # 6-2. 2D (Matplotlib) with projection overlay
    visualize_image_with_projection(img, u, v, z,
                                    point_stride=max(1, args.stride),
                                    dot_size=max(1, args.dot_size),
                                    depth_cmap="jet")


if __name__ == "__main__":
    main()
