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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh
from utils.render_utils import generate_path, create_videos
import open3d as o3d

# === [新增] 引入 SeaSplat 模块 ===
# 确保你的目录下有 seasplat_utils 文件夹
from seasplat_utils.models import BackscatterNetV2, AttenuateNetV3
# =================================

def render_seasplat_set(model_path, name, iteration, views, gaussians, pipe, background, bs_model, at_model):
    """
    SeaSplat 专用渲染函数：输出干图、湿图、物理参数图
    """
    # 基础路径
    base_dir = os.path.join(model_path, name, "ours_{}".format(iteration))
    
    # 定义输出目录
    dry_dir = os.path.join(base_dir, "restored_dry")       # 修复后的清晰图像 (J)
    wet_dir = os.path.join(base_dir, "synthesized_wet")    # 合成的水下图像 (I_wet)
    gt_dir = os.path.join(base_dir, "gt")                  # 真实图像 (GT)
    att_dir = os.path.join(base_dir, "attenuation_map")    # 衰减系数图 (A)
    bs_dir = os.path.join(base_dir, "backscatter_map")     # 后向散射图 (B)

    makedirs(dry_dir, exist_ok=True)
    makedirs(wet_dir, exist_ok=True)
    makedirs(gt_dir, exist_ok=True)
    makedirs(att_dir, exist_ok=True)
    makedirs(bs_dir, exist_ok=True)

    print(f"[SeaSplat] Rendering {name} set with physical models...")
    
    for idx, view in enumerate(tqdm(views, desc="Rendering SeaSplat")):
        # 1. 2DGS 渲染 (获取几何与干颜色)
        render_pkg = render(view, gaussians, pipe, background)
        
        image_dry = render_pkg["render"]        # [3, H, W]
        depth_map = render_pkg["surf_depth"]    # [1, H, W] (2DGS 特有高质量深度)
        gt_image = view.original_image[0:3, :, :]

        # 2. 应用物理模型 (Wet Stage)
        # 增加 Batch 维度 [1, C, H, W]
        depth_batch = depth_map.unsqueeze(0)
        rgb_batch = image_dry.unsqueeze(0)

        # 推理物理参数
        attenuation = at_model(depth_batch)
        backscatter = bs_model(depth_batch)

        # 合成水下图像 I = J * A + B
        direct_signal = rgb_batch * attenuation
        image_wet = direct_signal + backscatter
        image_wet = torch.clamp(image_wet, 0.0, 1.0)

        # 3. 保存所有结果
        # 保存 Restored (Dry)
        torchvision.utils.save_image(image_dry, os.path.join(dry_dir, '{0:05d}.png'.format(idx)))
        # 保存 Synthesized (Wet)
        torchvision.utils.save_image(image_wet.squeeze(0), os.path.join(wet_dir, '{0:05d}.png'.format(idx)))
        # 保存 GT
        torchvision.utils.save_image(gt_image, os.path.join(gt_dir, '{0:05d}.png'.format(idx)))
        
        # 保存物理图 (归一化以便可视化)
        att_vis = attenuation.squeeze(0)
        att_vis = att_vis / (att_vis.max() + 1e-6)
        torchvision.utils.save_image(att_vis, os.path.join(att_dir, '{0:05d}.png'.format(idx)))
        
        bs_vis = backscatter.squeeze(0)
        bs_vis = bs_vis / (bs_vis.max() + 1e-6)
        torchvision.utils.save_image(bs_vis, os.path.join(bs_dir, '{0:05d}.png'.format(idx)))

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_mesh", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_path", action="store_true")
    
    # Mesh 参数
    parser.add_argument("--voxel_size", default=-1.0, type=float, help='Mesh: voxel size for TSDF')
    parser.add_argument("--depth_trunc", default=-1.0, type=float, help='Mesh: Max depth range for TSDF')
    parser.add_argument("--sdf_trunc", default=-1.0, type=float, help='Mesh: truncation value for TSDF')
    parser.add_argument("--num_cluster", default=50, type=int, help='Mesh: number of connected clusters to export')
    parser.add_argument("--unbounded", action="store_true", help='Mesh: using unbounded mode for meshing')
    parser.add_argument("--mesh_res", default=1024, type=int, help='Mesh: resolution for unbounded mesh extraction')
    
    # === [新增] SeaSplat 开关 ===
    parser.add_argument("--use_seasplat", action="store_true", help="Enable SeaSplat physical rendering (Wet/Dry/Att/BS)")
    # ============================
    
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # === [新增] 加载物理模型 (如果启用) ===
    bs_model = None
    at_model = None
    if args.use_seasplat:
        print("[SeaSplat] Loading physical models...")
        bs_model = BackscatterNetV2().cuda()
        at_model = AttenuateNetV3().cuda()
        
        # 尝试加载对应 iteration 的物理参数
        bs_path = os.path.join(scene.model_path, f"bs_model_{scene.loaded_iter}.pth")
        at_path = os.path.join(scene.model_path, f"at_model_{scene.loaded_iter}.pth")
        
        if os.path.exists(bs_path) and os.path.exists(at_path):
            bs_model.load_state_dict(torch.load(bs_path))
            at_model.load_state_dict(torch.load(at_path))
            bs_model.eval()
            at_model.eval()
            print(f"[SeaSplat] Successfully loaded physics models for iteration {scene.loaded_iter}")
        else:
            print(f"[SeaSplat] Warning: Checkpoints not found at {bs_path}. Rendering dry only.")
            bs_model = None
            at_model = None
            args.use_seasplat = False # 没找到模型就回退到普通模式
    # ====================================

    # 初始化提取器 (用于普通渲染和网格提取)
    gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color)    
    
    # --------------------------------------------------------------------------------
    # 1. 渲染训练集 (Train Set)
    # --------------------------------------------------------------------------------
    if not args.skip_train:
        train_dir = os.path.join(args.model_path, 'train', "ours_{}".format(scene.loaded_iter))
        os.makedirs(train_dir, exist_ok=True)
        
        if args.use_seasplat:
            # SeaSplat 模式：自定义渲染循环，输出全套物理图
            render_seasplat_set(args.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), 
                                gaussians, pipe, background, bs_model, at_model)
        else:
            # 原始 2DGS 模式：只输出 Dry Image
            print("export training images ...")
            gaussExtractor.reconstruction(scene.getTrainCameras())
            gaussExtractor.export_image(train_dir)
        
    # --------------------------------------------------------------------------------
    # 2. 渲染测试集 (Test Set)
    # --------------------------------------------------------------------------------
    if (not args.skip_test) and (len(scene.getTestCameras()) > 0):
        test_dir = os.path.join(args.model_path, 'test', "ours_{}".format(scene.loaded_iter))
        os.makedirs(test_dir, exist_ok=True)
        
        if args.use_seasplat:
            render_seasplat_set(args.model_path, "test", scene.loaded_iter, scene.getTestCameras(), 
                                gaussians, pipe, background, bs_model, at_model)
        else:
            print("export rendered testing images ...")
            gaussExtractor.reconstruction(scene.getTestCameras())
            gaussExtractor.export_image(test_dir)
    
    # --------------------------------------------------------------------------------
    # 3. 渲染视频轨迹 (Trajectory Video)
    # --------------------------------------------------------------------------------
    if args.render_path:
        print("render videos ...")
        traj_dir = os.path.join(args.model_path, 'traj', "ours_{}".format(scene.loaded_iter))
        os.makedirs(traj_dir, exist_ok=True)
        n_fames = 240
        cam_traj = generate_path(scene.getTrainCameras(), n_frames=n_fames)
        
        # 目前轨迹渲染保持原始逻辑 (Dry)，如果需要湿视频，可以仿照 render_seasplat_set 修改
        gaussExtractor.reconstruction(cam_traj)
        gaussExtractor.export_image(traj_dir)
        create_videos(base_dir=traj_dir,
                    input_dir=traj_dir, 
                    out_name='render_traj', 
                    num_frames=n_fames)

    # --------------------------------------------------------------------------------
    # 4. 导出网格 (Mesh Extraction) - 永远基于 Dry Gaussians
    # --------------------------------------------------------------------------------
    if not args.skip_mesh:
        print("export mesh ...")
        # Mesh 输出目录通常在 train 文件夹下
        mesh_out_dir = os.path.join(args.model_path, 'train', "ours_{}".format(scene.loaded_iter))
        os.makedirs(mesh_out_dir, exist_ok=True)
        
        # set the active_sh to 0 to export only diffuse texture
        gaussExtractor.gaussians.active_sh_degree = 0
        gaussExtractor.reconstruction(scene.getTrainCameras())
        
        # extract the mesh and save
        if args.unbounded:
            name = 'fuse_unbounded.ply'
            mesh = gaussExtractor.extract_mesh_unbounded(resolution=args.mesh_res)
        else:
            name = 'fuse.ply'
            depth_trunc = (gaussExtractor.radius * 2.0) if args.depth_trunc < 0  else args.depth_trunc
            voxel_size = (depth_trunc / args.mesh_res) if args.voxel_size < 0 else args.voxel_size
            sdf_trunc = 5.0 * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc
            mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
        
        o3d.io.write_triangle_mesh(os.path.join(mesh_out_dir, name), mesh)
        print("mesh saved at {}".format(os.path.join(mesh_out_dir, name)))
        
        # post-process the mesh and save, saving the largest N clusters
        mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
        o3d.io.write_triangle_mesh(os.path.join(mesh_out_dir, name.replace('.ply', '_post.ply')), mesh_post)
        print("mesh post processed saved at {}".format(os.path.join(mesh_out_dir, name.replace('.ply', '_post.ply'))))