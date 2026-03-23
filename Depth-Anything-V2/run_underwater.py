import cv2
import numpy as np
import os
import torch
import glob
import matplotlib.pyplot as plt
from depth_anything_v2.dpt import DepthAnythingV2

def process_batch(input_dir, output_dir):
    # --- 1. 初始化模型 ---
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_configs = {
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    }
    encoder = 'vitb'
    checkpoint_path = 'checkpoints/depth_anything_v2_vitb.pth'

    print(f"正在加载模型到 {DEVICE}...")
    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model = model.to(DEVICE).eval()
    
    # --- 2. 准备路径 ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出文件夹: {output_dir}")

    valid_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif']
    image_paths = []
    for ext in valid_extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
    
    image_paths.sort()
    
    total_imgs = len(image_paths)
    if total_imgs == 0:
        print(f"错误: 在 {input_dir} 中没有找到图片。")
        return

    print(f"找到 {total_imgs} 张图片，开始处理...")

    # --- 3. 循环处理 ---
    for i, img_path in enumerate(image_paths):
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(filename)[0]
        
        raw_img = cv2.imread(img_path)
        if raw_img is None:
            print(f"跳过损坏文件: {filename}")
            continue

        # 获取原始深度推理结果
        depth = model.infer_image(raw_img)

        # ==========================================
        # 修改重点 1: 保存原始深度图 (.npy) 用于训练
        # ==========================================
        # 强制转换为 float32，这是大多数 3DGS/2DGS dataloader 期望的格式
        depth_npy = depth.astype(np.float32)
        npy_save_path = os.path.join(output_dir, base_name + '.npy')
        np.save(npy_save_path, depth_npy)

        # ==========================================
        # 修改重点 2: 保存可视化图 (.png) 用于检查
        # ==========================================
        # 1. 归一化
        min_val, max_val = depth.min(), depth.max()
        if max_val - min_val > 1e-5:
            depth_norm = (depth - min_val) / (max_val - min_val)
        else:
            depth_norm = np.zeros_like(depth)

        # 2. 应用颜色映射 (viridis_r)
        depth_colored = plt.cm.viridis_r(depth_norm)[:, :, :3]
        
        # 3. 转换格式并保存
        depth_colored = (depth_colored * 255).astype(np.uint8)
        depth_colored_bgr = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)

        # 给可视化图片加个后缀 _vis，以免混淆，或者你想保持原名也可以
        vis_save_path = os.path.join(output_dir, base_name + '_vis.png')
        cv2.imwrite(vis_save_path, depth_colored_bgr)

        if (i + 1) % 10 == 0 or (i + 1) == total_imgs:
            print(f"进度: {i + 1}/{total_imgs} - 已保存 .npy 和 .png")

    print("全部处理完成！")

if __name__ == '__main__':
    # === 路径 ===
    input_folder = '/root/autodl-tmp/Sea-thru/Panama/images'
    output_folder = '/root/autodl-tmp/Sea-thru/Panama/depth_anything' 
    
    process_batch(input_folder, output_folder)