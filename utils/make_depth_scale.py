import numpy as np
import argparse
import os
from joblib import delayed, Parallel
import json
from read_write_model import *

def get_scales(key, cameras, images, points3d_ordered, args):
    image_meta = images[key]
    cam_intrinsic = cameras[image_meta.camera_id]

    # 获取该图像可见的 3D 点索引
    pts_idx = images[key].point3D_ids

    # 过滤无效点
    mask = pts_idx >= 0
    mask *= pts_idx < len(points3d_ordered)

    pts_idx = pts_idx[mask]
    valid_xys = image_meta.xys[mask]

    if len(pts_idx) > 0:
        pts = points3d_ordered[pts_idx]
    else:
        # 如果没有可见点，返回 scale=0 (跳过此图)
        return {"image_name": image_meta.name[:-4], "scale": 0.0, "offset": 0.0}

    # 将 3D 点投影到相机坐标系
    R = qvec2rotmat(image_meta.qvec)
    pts = np.dot(pts, R.T) + image_meta.tvec

    # 获取 COLMAP 的深度 (Z坐标) 并转为逆深度
    # 增加物理保护，防止除以0或负数
    valid_depth_mask = pts[..., 2] > 1e-6
    
    if valid_depth_mask.sum() == 0:
        return {"image_name": image_meta.name[:-4], "scale": 0.0, "offset": 0.0}
        
    pts = pts[valid_depth_mask]
    valid_xys = valid_xys[valid_depth_mask]
    
    invcolmapdepth = 1. / (pts[..., 2] + 1.0)

    # 构建 .npy 文件路径
    # 假设后缀长度为4 (.png, .jpg)
    n_remove = len(image_meta.name.split('.')[-1]) + 1
    base_name = image_meta.name[:-n_remove]
    img_path = os.path.join(args.depths_dir, f"{base_name}.npy")
    
    if not os.path.exists(img_path):
        return None
        
    try:
        invmonodepthmap = np.load(img_path)
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        return None
    
    if invmonodepthmap is None:
        return None
    
    # 维度检查
    if invmonodepthmap.ndim == 3:
        invmonodepthmap = invmonodepthmap.squeeze()

    invmonodepthmap = invmonodepthmap.astype(np.float32)
    
    # 计算缩放比例
    h, w = invmonodepthmap.shape
    if h != cam_intrinsic.height:
        s = h / cam_intrinsic.height
    else:
        s = 1.0

    # 计算采样坐标 (浮点数)
    sample_x = valid_xys[..., 0] * s
    sample_y = valid_xys[..., 1] * s
    
    # 边界检查
    valid_mask = (
        (sample_x >= 0) & (sample_x < w - 1) & 
        (sample_y >= 0) & (sample_y < h - 1)
    )
    
    if valid_mask.sum() < 10:
        return {"image_name": base_name, "scale": 0.0, "offset": 0.0}

    # 应用掩码
    sample_x = sample_x[valid_mask]
    sample_y = sample_y[valid_mask]
    invcolmapdepth = invcolmapdepth[valid_mask]

    # =========================================================
    # [修正] 使用 Numpy 双线性插值替代 cv2.remap
    # 彻底解决 cv2.error: SHRT_MAX 问题
    # =========================================================
    x0 = np.floor(sample_x).astype(np.int32)
    x1 = x0 + 1
    y0 = np.floor(sample_y).astype(np.int32)
    y1 = y0 + 1

    # 再次钳制边界 (防止索引越界)
    x0 = np.clip(x0, 0, w-1); x1 = np.clip(x1, 0, w-1)
    y0 = np.clip(y0, 0, h-1); y1 = np.clip(y1, 0, h-1)

    # 获取四个角的深度值
    # 注意 numpy 索引是 [y, x]
    Ia = invmonodepthmap[y0, x0]
    Ib = invmonodepthmap[y1, x0]
    Ic = invmonodepthmap[y0, x1]
    Id = invmonodepthmap[y1, x1]

    # 计算权重
    wa = (x1 - sample_x) * (y1 - sample_y)
    wb = (x1 - sample_x) * (sample_y - y0)
    wc = (sample_x - x0) * (y1 - sample_y)
    wd = (sample_x - x0) * (sample_y - y0)

    # 加权求和
    invmonodepth = wa*Ia + wb*Ib + wc*Ic + wd*Id
    # =========================================================

    # 鲁棒统计计算 Scale 和 Offset
    if len(invmonodepth) > 0:
        t_colmap = np.median(invcolmapdepth)
        s_colmap = np.mean(np.abs(invcolmapdepth - t_colmap))

        t_mono = np.median(invmonodepth)
        s_mono = np.mean(np.abs(invmonodepth - t_mono))
        
        if s_mono < 1e-6:
            scale = 0.0
            offset = 0.0
        else:
            scale = s_colmap / s_mono
            offset = t_colmap - t_mono * scale
    else:
        scale = 0.0
        offset = 0.0
        
    return {"image_name": base_name, "scale": float(scale), "offset": float(offset)}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', required=True, help="Path to the scene directory (containing sparse/0)")
    parser.add_argument('--depths_dir', required=True, help="Path to the directory containing .npy depth maps")
    parser.add_argument('--model_type', default="bin", choices=["bin", "txt"], help="COLMAP model type")
    args = parser.parse_args()

    sparse_path = os.path.join(args.base_dir, "sparse", "0")
    print(f"Reading model from: {sparse_path}")
    cam_intrinsics, images_metas, points3d = read_model(sparse_path, ext=f".{args.model_type}")

    pts_indices = np.array([points3d[key].id for key in points3d])
    pts_xyzs = np.array([points3d[key].xyz for key in points3d])
    points3d_ordered = np.zeros([pts_indices.max()+1, 3])
    points3d_ordered[pts_indices] = pts_xyzs

    print("Computing scales from .npy files...")
    # 使用多线程并行处理
    depth_param_list = Parallel(n_jobs=-1, backend="threading")(
        delayed(get_scales)(key, cam_intrinsics, images_metas, points3d_ordered, args) for key in images_metas
    )

    depth_params = {
        depth_param["image_name"]: {"scale": depth_param["scale"], "offset": depth_param["offset"]}
        for depth_param in depth_param_list if depth_param != None
    }

    output_path = os.path.join(sparse_path, "depth_params.json")
    with open(output_path, "w") as f:
        json.dump(depth_params, f, indent=2)

    print(f"Done! Parameters saved to: {output_path}")