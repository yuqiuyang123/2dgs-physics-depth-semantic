# 皮尔逊深度约束
# 使用了1/Z的标准定义，并且完全移除了仿射变换，确保了与伪深度真值的严格线性关系，从而使得皮尔逊相关系数能够正确反映预测深度与真值之间的相关性。
# 加入了冻结机制，在物理模型开始训练的前1000步内冻结高斯几何参数，允许物理网络在相对稳定的几何基础上进行预热，减少初期训练的不稳定性。
# 加入了语义特征掩码机制


import os
import torch
import torch.nn.functional as F # <--- [新增] 移到顶部的导包
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui 
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import subprocess 

import torchvision
from utils.mesh_utils import GaussianExtractor, post_process_mesh
import numpy as np
import open3d as o3d
from PIL import Image
import json

import matplotlib.pyplot as plt 

from seasplat_utils.models import BackscatterNetV2, AttenuateNetV3
# [修改] 导入所有新增的损失函数
from seasplat_utils.losses import (
    BackscatterLoss, GrayWorldLoss, SaturationLoss, AlphaBackgroundLoss,
    AttenuateLoss, DarkChannelPriorLossV3, RgbSpatialVariationLoss, SmoothDepthLoss
)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# ==============================================================================
# 皮尔逊深度损失 (Pearson Correlation Depth Loss)
# ==============================================================================
def pearson_depth_loss(pred, gt, epsilon=1e-6):
    pred_flat = pred.view(-1)
    gt_flat = gt.view(-1)
    
    mask = (gt_flat > 0) & (pred_flat > 0) & (~torch.isnan(pred_flat)) & (~torch.isinf(pred_flat))
    if mask.sum() < 100:
        print("\n[警告] 深度掩码有效像素不足 100，深度损失被跳过！请检查深度图预处理。")
        return torch.tensor(0.0).cuda()
        
    pred_val = pred_flat[mask]
    gt_val = gt_flat[mask]
    
    pred_mean = pred_val.mean()
    gt_mean = gt_val.mean()
    pred_centered = pred_val - pred_mean
    gt_centered = gt_val - gt_mean
    
    covariance = (pred_centered * gt_centered).sum()
    pred_std = torch.sqrt((pred_centered ** 2).sum()) + epsilon
    gt_std = torch.sqrt((gt_centered ** 2).sum()) + epsilon
    
    corr = covariance / (pred_std * gt_std)
    return 1.0 - corr

# ==============================================================================
# 渲染函数定义
# ==============================================================================
def render_seasplat_set(model_path, name, iteration, views, gaussians, pipe, background, bs_model, at_model):
    base_dir = os.path.join(model_path, name, "ours_{}".format(iteration))
    dry_dir = os.path.join(base_dir, "restored_dry")        
    wet_dir = os.path.join(base_dir, "synthesized_wet")     
    gt_dir = os.path.join(base_dir, "gt")                   
    depth_dir = os.path.join(base_dir, "depth")             
    att_dir = os.path.join(base_dir, "attenuation_map")     
    bs_dir = os.path.join(base_dir, "backscatter_map")      

    os.makedirs(dry_dir, exist_ok=True)
    os.makedirs(wet_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(att_dir, exist_ok=True) 
    os.makedirs(bs_dir, exist_ok=True)  

    print(f"[自动渲染-物理模式] 正在渲染 {name} 集...")
    
    for idx, view in enumerate(tqdm(views, desc=f"渲染 {name}")):
        render_pkg = render(view, gaussians, pipe, background)
        image_dry = render_pkg["render"]
        depth_map = render_pkg["surf_depth"]
        gt_image = view.original_image[0:3, :, :]

        torchvision.utils.save_image(image_dry, os.path.join(dry_dir, '{0:05d}.png'.format(idx)))
        torchvision.utils.save_image(gt_image, os.path.join(gt_dir, '{0:05d}.png'.format(idx)))
        
        depth_vis = depth_map.detach().clone().squeeze()
        depth_max = depth_vis.max()
        if depth_max > 0: depth_vis = depth_vis / depth_max
        depth_np = depth_vis.cpu().numpy()
        depth_colored_np = plt.get_cmap('viridis')(depth_np)[:, :, :3]
        depth_colored = torch.from_numpy(depth_colored_np).permute(2, 0, 1)
        torchvision.utils.save_image(depth_colored, os.path.join(depth_dir, '{0:05d}.png'.format(idx)))

        if bs_model is not None and at_model is not None:
            depth_input = depth_map.unsqueeze(0) / (depth_map.max() + 1e-6)
            rgb_batch = image_dry.unsqueeze(0)
            attenuation = at_model(depth_input)
            backscatter = bs_model(depth_input)
            direct_signal = rgb_batch * attenuation
            image_wet = direct_signal + backscatter
            image_wet = torch.clamp(image_wet, 0.0, 1.0).squeeze(0)
            torchvision.utils.save_image(image_wet, os.path.join(wet_dir, '{0:05d}.png'.format(idx)))
            
            att_vis = attenuation.squeeze(0)
            att_vis = att_vis / (att_vis.max() + 1e-6)
            torchvision.utils.save_image(att_vis, os.path.join(att_dir, '{0:05d}.png'.format(idx)))
            bs_vis = backscatter.squeeze(0)
            bs_vis = bs_vis / (bs_vis.max() + 1e-6)
            torchvision.utils.save_image(bs_vis, os.path.join(bs_dir, '{0:05d}.png'.format(idx)))

def render_standard_set(model_path, name, iteration, views, gaussians, pipe, background):
    base_dir = os.path.join(model_path, name, "ours_{}".format(iteration))
    renders_dir = os.path.join(base_dir, "renders") 
    gt_dir = os.path.join(base_dir, "gt")
    depth_dir = os.path.join(base_dir, "depth")

    os.makedirs(renders_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    print(f"[自动渲染-标准模式] 正在渲染 {name} 集...")

    for idx, view in enumerate(tqdm(views, desc=f"渲染 {name}")):
        render_pkg = render(view, gaussians, pipe, background)
        image = render_pkg["render"]
        depth_map = render_pkg["surf_depth"]
        gt_image = view.original_image[0:3, :, :]

        torchvision.utils.save_image(image, os.path.join(renders_dir, '{0:05d}.png'.format(idx)))
        torchvision.utils.save_image(gt_image, os.path.join(gt_dir, '{0:05d}.png'.format(idx)))

        depth_vis = depth_map.detach().clone().squeeze()
        depth_max = depth_vis.max()
        if depth_max > 0: depth_vis = depth_vis / depth_max
        depth_np = depth_vis.cpu().numpy()
        depth_colored_np = plt.get_cmap('viridis')(depth_np)[:, :, :3]
        depth_colored = torch.from_numpy(depth_colored_np).permute(2, 0, 1)
        torchvision.utils.save_image(depth_colored, os.path.join(depth_dir, '{0:05d}.png'.format(idx)))

# ==============================================================================
# 训练主循环
# ==============================================================================
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    bs_model = None
    at_model = None
    physics_optimizer = None
    
    if opt.use_seasplat:
        bs_model = BackscatterNetV2().cuda()
        at_model = AttenuateNetV3().cuda()
        physics_optimizer = torch.optim.Adam([
            {'params': bs_model.parameters()},
            {'params': at_model.parameters()}
        ], lr=opt.seasplat_lr)
        
        loss_fn_bs = BackscatterLoss()
        loss_fn_gw = GrayWorldLoss()
        loss_fn_sat = SaturationLoss()
        loss_fn_alpha = AlphaBackgroundLoss()
        
        loss_fn_at = AttenuateLoss().cuda()
        loss_fn_dcp = DarkChannelPriorLossV3().cuda()
        loss_fn_depth_smooth = SmoothDepthLoss().cuda()
        loss_fn_rgb_sv = RgbSpatialVariationLoss().cuda()

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="训练进度")
    first_iter += 1
    
    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        if opt.use_seasplat:
            freeze_window = 1000 
            
            if iteration == opt.start_seasplat_iter + 1:
                tqdm.write(f"\n[{iteration}] 冻结高斯几何参数，进行物理网络预热 ({freeze_window}步)...")
                gaussians.freeze_parameters(xyz=True, colors=False, opacity=True, scaling=True, rotation=True)
            
            elif iteration == opt.start_seasplat_iter + freeze_window + 1:
                tqdm.write(f"\n[{iteration}] 物理网络预热完成，解冻高斯几何参数，恢复全面联合训练...")
                gaussians.freeze_parameters(xyz=False, colors=False, opacity=False, scaling=False, rotation=False)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image_dry = render_pkg["render"]
        depth_map = render_pkg["surf_depth"] 
        gt_image = viewpoint_cam.original_image.cuda()
        
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        loss_bs = torch.tensor(0.0).cuda()
        
        if opt.use_seasplat and iteration > opt.start_seasplat_iter:
            depth_max = depth_map.max().detach()
            depth_input = depth_map.unsqueeze(0) / (depth_max + 1e-6)
            rgb_batch = image_dry.unsqueeze(0)
            gt_rgb_batch = gt_image.unsqueeze(0) 
            
            attenuation = at_model(depth_input)
            backscatter = bs_model(depth_input)
            direct_signal = rgb_batch * attenuation
            image_wet = direct_signal + backscatter
            image_wet = torch.clamp(image_wet, 0.0, 1.0).squeeze(0)
            
            Ll1 = l1_loss(image_wet, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image_wet, gt_image))
            
            bs_detached = bs_model(depth_input.detach())
            direct_from_gt_detached = (gt_rgb_batch - bs_detached).detach()
            
            loss_bs = loss_fn_bs(direct_from_gt_detached)
            loss_dcp, _ = loss_fn_dcp(direct_from_gt_detached, depth_input.detach())
            
            at_detached = at_model(depth_input.detach())
            J_through_atmodel = direct_from_gt_detached / torch.clamp(at_detached, min=1e-6)
            loss_at = loss_fn_at(direct_from_gt_detached, J_through_atmodel)
            
            loss_depth_smooth = loss_fn_depth_smooth(gt_rgb_batch, depth_input)
            loss_rgb_sv = loss_fn_rgb_sv(rgb_batch.detach(), direct_signal)
            loss_binf = bs_model.forward_rgb(gt_rgb_batch)

            loss_gw = loss_fn_gw(rgb_batch)
            loss_sat = loss_fn_sat(rgb_batch)
            b_inf_val = torch.sigmoid(bs_model.B_inf).detach()
            render_alpha_batch = render_pkg["rend_alpha"].unsqueeze(0) if render_pkg["rend_alpha"].dim() == 2 else render_pkg["rend_alpha"]
            loss_alpha = loss_fn_alpha(image_wet, b_inf_val, render_alpha_batch)
            
            total_loss = loss \
                         + getattr(opt, 'lambda_bs', 1.0) * loss_bs \
                         + getattr(opt, 'lambda_dcp', 1.0) * loss_dcp \
                         + getattr(opt, 'lambda_at', 1.0) * loss_at \
                         + getattr(opt, 'lambda_depth_smooth', 2.0) * loss_depth_smooth \
                         + getattr(opt, 'lambda_binf', 1.0) * loss_binf \
                         + getattr(opt, 'lambda_gw', 0.1) * loss_gw \
                         + 0.1 * loss_sat \
                         + 0.01 * loss_rgb_sv \
                         + 0.01 * loss_alpha
        else:
            Ll1 = l1_loss(image_dry, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image_dry, gt_image))
            total_loss = loss

        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        rend_normal = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        total_loss += normal_loss

        use_depth_constraint = (opt.lambda_depth > 0) and (iteration > 1000)
        loss_depth_val = torch.tensor(0.0).cuda()
        
        if use_depth_constraint and viewpoint_cam.original_depth is not None:
            gt_inv_depth = viewpoint_cam.original_depth
            depth_map_clamped = torch.clamp(depth_map, min=1e-6)
            pred_inv_depth = 1.0 / depth_map_clamped
            loss_depth_val = pearson_depth_loss(pred_inv_depth, gt_inv_depth)
            total_loss += opt.lambda_depth * loss_depth_val 
        
        total_loss.backward()
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "Depth": f"{loss_depth_val.item():.{5}f}" 
                })
                progress_bar.update(10)
            
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/normal_loss', normal_loss.item(), iteration)
                if use_depth_constraint:
                    tb_writer.add_scalar('train_loss_patches/depth_loss', loss_depth_val.item(), iteration)
                
                if opt.use_seasplat and iteration > opt.start_seasplat_iter:
                    tb_writer.add_scalar('train_loss_patches/bs_loss', loss_bs.item(), iteration)
                    tb_writer.add_scalar('train_loss_patches/dcp_loss', loss_dcp.item(), iteration)
                    tb_writer.add_scalar('train_loss_patches/at_loss', loss_at.item(), iteration)
                    tb_writer.add_scalar('train_loss_patches/depth_smooth_loss', loss_depth_smooth.item(), iteration)
                    tb_writer.add_scalar('train_loss_patches/binf_loss', loss_binf.item(), iteration)
            
            if iteration % 100 == 0:
                if opt.use_seasplat and iteration > opt.start_seasplat_iter:
                    att_mean = torch.sigmoid(at_model.attenuation_conv_params).mean().item()
                    bs_mean = torch.sigmoid(bs_model.backscatter_conv_params).mean().item()
                    tqdm.write(f"\n[物理参数] 迭代 {iteration}: 衰减(Sig)={att_mean:.4f}, 散射(Sig)={bs_mean:.4f}")

            if iteration == opt.iterations:
                progress_bar.close()

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            
            if (iteration in saving_iterations):
                print("\n[迭代 {}] 保存高斯模型".format(iteration))
                scene.save(iteration)

            # =========================================================
            # [核心合并] 几何控制与基于语义的自适应密度控制
            # =========================================================
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                
                if opt.use_seasplat and viewspace_point_tensor.grad is not None:
                    cam_center = viewpoint_cam.camera_center.cuda()
                    dists = torch.norm(gaussians.get_xyz - cam_center, dim=1)
                    dist_factor = dists / 5.0 
                    grad_weight = torch.clamp(dist_factor, min=1.0, max=5.0)
                    viewspace_point_tensor.grad[visibility_filter] *= grad_weight[visibility_filter].unsqueeze(1)
                
                # [新增] 语义与局部图像梯度联合驱动的自适应密度控制 
                # (Semantic & Image-Gradient Driven Densification)
                if hasattr(viewpoint_cam, 'gt_semantic_mask') and viewpoint_cam.gt_semantic_mask is not None and viewspace_point_tensor.grad is not None:
                    
                    # 1. 备份真实梯度，隔离 Adam 优化
                    real_optim_grad = viewspace_point_tensor.grad.clone()
                    
                    xyz = gaussians.get_xyz[visibility_filter]
                    if xyz.shape[0] > 0:
                        # 2. 射影几何映射 (3D -> 2D Pixel)
                        homogenous_coords = torch.cat([xyz, torch.ones(xyz.shape[0], 1, device="cuda")], dim=1) @ viewpoint_cam.full_proj_transform
                        w = homogenous_coords[:, 3:] + 1e-7
                        ndc_coords = homogenous_coords[:, :2] / w
                        
                        H_mask, W_mask = viewpoint_cam.gt_semantic_mask.shape
                        pixel_x = ((ndc_coords[:, 0] + 1.0) * W_mask - 1.0) * 0.5
                        pixel_y = ((ndc_coords[:, 1] + 1.0) * H_mask - 1.0) * 0.5
                        
                        pixel_x = torch.nan_to_num(pixel_x, nan=0.0, posinf=0.0, neginf=0.0)
                        pixel_y = torch.nan_to_num(pixel_y, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        pixel_x = torch.clamp(pixel_x, 0, W_mask - 1).long()
                        pixel_y = torch.clamp(pixel_y, 0, H_mask - 1).long()
                        
                        # 3. 提取语义标签 (0:珊瑚, 1:海水, 2:岩石)
                        semantic_labels = viewpoint_cam.gt_semantic_mask[pixel_y, pixel_x]
                        
                        # 4. 计算 GT 图像的局部梯度幅值 (Gradient Magnitude)
                        gt_img_gray = viewpoint_cam.original_image.mean(dim=0, keepdim=True).unsqueeze(0) # [1, 1, H, W]
                        
                        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device="cuda").view(1, 1, 3, 3)
                        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device="cuda").view(1, 1, 3, 3)
                        
                        grad_x = F.conv2d(gt_img_gray, sobel_x, padding=1)
                        grad_y = F.conv2d(gt_img_gray, sobel_y, padding=1)
                        
                        grad_mag = torch.sqrt(grad_x**2 + grad_y**2).squeeze() # [H, W]
                        grad_mag_norm = grad_mag / (grad_mag.max() + 1e-6)
                        
                        local_img_grad = grad_mag_norm[pixel_y, pixel_x]
                        
                        # 5. 构建自适应权重方程
                        grad_modifier = torch.zeros_like(semantic_labels, dtype=torch.float32)
                        
                        mask_coral = (semantic_labels == 0)
                        grad_modifier[mask_coral] = 0.4 + 0.6 * local_img_grad[mask_coral]
                        
                        mask_water = (semantic_labels == 1)
                        grad_modifier[mask_water] = 0.01
                        
                        mask_rock = (semantic_labels == 2)
                        grad_modifier[mask_rock] = 0.2 + 0.3 * local_img_grad[mask_rock]
                        
                        grad_modifier = torch.clamp(grad_modifier, min=0.01, max=1.0)
                        
                        # 6. 施加自适应“伪装梯度”
                        viewspace_point_tensor.grad[visibility_filter] *= grad_modifier.unsqueeze(1)

                # 使用“伪装梯度”计算克隆和分裂指标
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                
                # 7. 统计完成后，还原真实梯度，保全 Adam 优化
                if hasattr(viewpoint_cam, 'gt_semantic_mask') and viewpoint_cam.gt_semantic_mask is not None and viewspace_point_tensor.grad is not None:
                    viewspace_point_tensor.grad = real_optim_grad
                # =========================================================

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                if opt.use_seasplat and iteration > opt.start_seasplat_iter:
                    physics_optimizer.step()
                    physics_optimizer.zero_grad()

            if (iteration in checkpoint_iterations):
                print("\n[迭代 {}] 保存检查点".format(iteration))
                try:
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                    if opt.use_seasplat:
                        torch.save(bs_model.state_dict(), scene.model_path + "/bs_model_" + str(iteration) + ".pth")
                        torch.save(at_model.state_dict(), scene.model_path + "/at_model_" + str(iteration) + ".pth")
                except Exception as e:
                    print(f"[错误] 保存检查点失败: {e}")

        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {"#": gaussians.get_opacity.shape[0], "loss": ema_loss_for_log}
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    network_gui.conn = None

    print("\n[训练完成] 开始自动渲染与指标计算...")
    
    if opt.use_seasplat:
        render_seasplat_set(dataset.model_path, "train", opt.iterations, scene.getTrainCameras(), gaussians, pipe, background, bs_model, at_model)
        if len(scene.getTestCameras()) > 0:
             render_seasplat_set(dataset.model_path, "test", opt.iterations, scene.getTestCameras(), gaussians, pipe, background, bs_model, at_model)
    else:
        render_standard_set(dataset.model_path, "train", opt.iterations, scene.getTrainCameras(), gaussians, pipe, background)
        if len(scene.getTestCameras()) > 0:
            render_standard_set(dataset.model_path, "test", opt.iterations, scene.getTestCameras(), gaussians, pipe, background)
    
    print("\n[自动渲染] 开始提取网格(Mesh)...")
    mesh_out_dir = os.path.join(dataset.model_path, 'train', "ours_{}".format(opt.iterations))
    os.makedirs(mesh_out_dir, exist_ok=True)
    gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color)
    gaussExtractor.reconstruction(scene.getTrainCameras()) 
    depth_trunc = gaussExtractor.radius * 2.0
    voxel_size = depth_trunc / 1024
    sdf_trunc = 5.0 * voxel_size
    print(f"网格参数: 体素大小={voxel_size:.4f}, 深度截断={depth_trunc:.2f}")
    mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
    if len(mesh.vertices) > 0:
        o3d.io.write_triangle_mesh(os.path.join(mesh_out_dir, 'fuse.ply'), mesh)
        try:
            mesh_post = post_process_mesh(mesh, cluster_to_keep=50)
            o3d.io.write_triangle_mesh(os.path.join(mesh_out_dir, 'fuse_post.ply'), mesh_post)
            print("后处理网格已保存。")
        except Exception as e:
            print(f"[警告] 网格后处理失败: {e}")
    else:
        print("[警告] 提取的网格为空（0顶点）。跳过后处理。")

    print(f"\n[系统调用] 启动 metrics.py 计算指标...")
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        metrics_script = os.path.join(current_dir, "metrics.py")
        if os.path.exists(metrics_script):
            subprocess.run(["python", metrics_script, "-m", dataset.model_path], check=True)
            print("[系统调用] 指标计算完成。")
        else:
            print(f"[警告] 找不到 metrics.py 文件: {metrics_script}")
    except Exception as e:
        print(f"[错误] 调用 metrics.py 失败: {e}")

# ==============================================================================
# 缺失的辅助函数
# ==============================================================================
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
    
    print("输出文件夹: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard 不可用: 不记录日志")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0).to("cuda")
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        depth = depth / depth.max()
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[迭代 {}] 评估 {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.save_iterations.extend(args.checkpoint_iterations) 
    
    print("正在优化 " + args.model_path)

    safe_state(args.quiet)
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    print("\n训练完成。")