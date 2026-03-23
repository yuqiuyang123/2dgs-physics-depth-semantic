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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

# ==============================================================================
# [新增] 3D-UIR 深度约束所需的 Loss 函数
# ==============================================================================

def anisotropic_total_variation_loss(img):
    """
    各向异性总变分损失 (TV Loss)，用于减少深度图噪点
    """
    d_w = torch.abs(img[:, :, :-1] - img[:, :, 1:])
    d_h = torch.abs(img[:, :-1, :] - img[:, 1:, :])
    w_variation = torch.mean(d_w)
    h_variance = torch.mean(d_h)
    return h_variance + w_variation

def depth_aware_smooth_loss(depth, img, lambda_edge=10.0):
    """
    边缘感知平滑损失 (Edge-Aware Smoothness Loss)
    原理: 在 RGB 图像梯度小的地方(平坦区域)强制深度平滑，
          在 RGB 梯度大的地方(边缘)允许深度突变。
    Args:
        depth: 预测的深度图 (B, 1, H, W) 或 (1, H, W)
        img: 对应的 RGB 图像 (B, 3, H, W) 或 (3, H, W)
    """
    # 计算图像的梯度 (用来衡量纹理/边缘)
    # 使用 L1 范数计算 RGB 梯度
    grad_img_x = torch.mean(torch.abs(img[..., :, :-1] - img[..., :, 1:]), -3, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[..., :-1, :] - img[..., 1:, :]), -3, keepdim=True)

    # 计算深度的梯度
    grad_depth_x = torch.abs(depth[..., :, :-1] - depth[..., :, 1:])
    grad_depth_y = torch.abs(depth[..., :-1, :] - depth[..., 1:, :])

    # 计算权重: e^(-|grad_I|)
    # 图像梯度越大，权重越小 -> 不惩罚深度突变
    weights_x = torch.exp(-lambda_edge * grad_img_x)
    weights_y = torch.exp(-lambda_edge * grad_img_y)

    # 加权平滑损失
    smoothness_x = grad_depth_x * weights_x
    smoothness_y = grad_depth_y * weights_y

    return smoothness_x.mean() + smoothness_y.mean()