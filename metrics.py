#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from utils.image_utils import psnr
from argparse import ArgumentParser
import json
from tqdm import tqdm

# 尝试导入 LPIPS，如果失败打印警告
try:
    from lpipsPyTorch import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    print("[WARNING] 'lpipsPyTorch' module not found. LPIPS metric will be 0.")
    LPIPS_AVAILABLE = False

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    
    if not os.path.exists(renders_dir) or not os.path.exists(gt_dir):
        return [], [], []

    # 获取所有图片文件
    files = [f for f in os.listdir(renders_dir) if f.endswith('.png') or f.endswith('.jpg')]
    files.sort()

    for fname in files:
        if os.path.exists(os.path.join(gt_dir, fname)):
            render = Image.open(os.path.join(renders_dir, fname))
            gt = Image.open(os.path.join(gt_dir, fname))
            
            renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
            gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
            image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths):
    full_dict = {}
    per_view_dict = {}
    print("")

    for scene_dir in model_paths:
        print(f"Scene: {scene_dir}")
        full_dict[scene_dir] = {}
        per_view_dict[scene_dir] = {}

        # 检查 test 文件夹 (如果不存在则检查 train，兼容不同配置)
        test_dir = Path(scene_dir) / "test"
        if not test_dir.exists():
            test_dir = Path(scene_dir) / "train"
            if not test_dir.exists():
                print(f"[Skipping] No 'test' or 'train' directory found in {scene_dir}")
                continue

        # 遍历所有方法 (例如 ours_30000)
        for method in os.listdir(test_dir):
            print(f"Method: {method}")

            full_dict[scene_dir][method] = {}
            per_view_dict[scene_dir][method] = {}

            method_dir = test_dir / method
            gt_dir = method_dir / "gt"
            
            # [修改关键点] 优先读取 SeaSplat 的 synthesized_wet，没有则读取 renders
            renders_dir = method_dir / "synthesized_wet"
            if not renders_dir.exists():
                print(f"  Note: 'synthesized_wet' not found, falling back to 'renders'")
                renders_dir = method_dir / "renders"

            if not renders_dir.exists():
                print(f"  [Skipping] No renders found in {method_dir}")
                continue

            renders, gts, image_names = readImages(renders_dir, gt_dir)
            
            if len(renders) == 0:
                print(f"  [Skipping] No matching images found in {renders_dir}")
                continue

            ssims = []
            psnrs = []
            lpipss = []

            for idx in tqdm(range(len(renders)), desc="Metric evaluation"):
                ssims.append(ssim(renders[idx], gts[idx]))
                psnrs.append(psnr(renders[idx], gts[idx]))
                
                if LPIPS_AVAILABLE:
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))
                else:
                    lpipss.append(torch.tensor(0.0).cuda())

            mean_ssim = torch.tensor(ssims).mean().item()
            mean_psnr = torch.tensor(psnrs).mean().item()
            mean_lpips = torch.tensor(lpipss).mean().item()

            print("  SSIM : {:>12.7f}".format(mean_ssim))
            print("  PSNR : {:>12.7f}".format(mean_psnr))
            print("  LPIPS: {:>12.7f}".format(mean_lpips))
            print("")

            full_dict[scene_dir][method].update({
                "SSIM": mean_ssim,
                "PSNR": mean_psnr,
                "LPIPS": mean_lpips
            })
            per_view_dict[scene_dir][method].update({
                "SSIM": {name: s.item() for s, name in zip(ssims, image_names)},
                "PSNR": {name: p.item() for p, name in zip(psnrs, image_names)},
                "LPIPS": {name: l.item() for l, name in zip(lpipss, image_names)}
            })

        # 保存结果
        results_path = os.path.join(scene_dir, "results.json")
        per_view_path = os.path.join(scene_dir, "per_view.json")
        
        with open(results_path, 'w') as fp:
            json.dump(full_dict[scene_dir], fp, indent=4)
        with open(per_view_path, 'w') as fp:
            json.dump(per_view_dict[scene_dir], fp, indent=4)
            
        print(f"Metrics saved to {results_path}")

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)