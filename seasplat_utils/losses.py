import torch
import torch.nn as nn

class BackscatterLoss(nn.Module):
    def __init__(self, cost_ratio=1000.):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.smooth_l1 = nn.SmoothL1Loss(beta=0.2)
        self.relu = nn.ReLU()
        self.cost_ratio = cost_ratio

    def forward(self, direct_signal):
        neg = self.smooth_l1(self.relu(-direct_signal), torch.zeros_like(direct_signal))
        pos = self.l1(self.relu(direct_signal), torch.zeros_like(direct_signal))
        return self.cost_ratio * neg + pos

class GrayWorldLoss(nn.Module):
    def __init__(self, target=0.5):
        super().__init__()
        self.target = target

    def forward(self, restored_image):
        means = torch.mean(restored_image, dim=[2, 3])
        return ((means - self.target) ** 2).mean()

class SaturationLoss(nn.Module):
    def __init__(self, thresh=0.7):
        super().__init__()
        self.thresh = thresh
        self.relu = nn.ReLU()

    def forward(self, restored_image):
        return (self.relu(restored_image - 1.0) + self.relu(-restored_image)).mean()

class AlphaBackgroundLoss(nn.Module):
    def __init__(self, threshold=0.2):
        super().__init__()
        self.threshold = threshold
        self.l1 = nn.L1Loss()

    def forward(self, rgb_render, bg_color, alpha):
        # bg_color [3, 1, 1] vs rgb_render [3, H, W]
        diff = torch.norm(rgb_render - bg_color, dim=0, keepdim=True)
        mask = diff < self.threshold
        if mask.sum() > 0:
            return self.l1(alpha[mask], torch.zeros_like(alpha[mask]))
        return torch.tensor(0.0).cuda()


# ==========================================
# 补充的 SeaSplat 核心物理与正则化损失
# ==========================================

class AttenuateLoss(nn.Module):
    """
    衰减损失：约束恢复出的无水图像 J。
    要求其通道强度期望逼近 0.5，且空间方差与原始透射信号保持一致。
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.relu = nn.ReLU()
        self.target_intensity = 0.5

    def forward(self, direct, J):
        # 饱和度惩罚 (限制在 0-1 之间)
        saturation_loss = (self.relu(-J) + self.relu(J - 1)).square().mean()
        
        # 空间方差一致性
        init_spatial = torch.std(direct, dim=[2, 3])
        channel_spatial = torch.std(J, dim=[2, 3])
        spatial_variation_loss = self.mse(channel_spatial, init_spatial)
        
        # 通道强度期望
        channel_intensities = torch.mean(J, dim=[2, 3], keepdim=True)
        intensity_loss = (channel_intensities - self.target_intensity).square().mean()
        
        return intensity_loss + spatial_variation_loss + saturation_loss

class DarkChannelPriorLossV3(nn.Module):
    """
    暗通道先验损失 V3：在 SeaSplat 的实际源码中，DCP V3 
    被巧妙地实现为对直接透射信号负值的强惩罚，配合平滑 L1 损失。
    """
    def __init__(self, cost_ratio=1000.):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.smooth_l1 = nn.SmoothL1Loss(beta=0.2)
        self.relu = nn.ReLU()
        self.cost_ratio = cost_ratio

    def forward(self, direct, depth=None):
        pos = self.l1(self.relu(direct), torch.zeros_like(direct))
        neg = self.smooth_l1(self.relu(-direct), torch.zeros_like(direct))
        bs_loss = self.cost_ratio * neg + pos
        return bs_loss, torch.zeros_like(direct)

class RgbSpatialVariationLoss(nn.Module):
    """
    RGB 空间变化损失：独立约束恢复信号与直接信号之间的方差一致性。
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, J, direct):
        init_spatial = torch.std(direct, dim=[2, 3])
        channel_spatial = torch.std(J, dim=[2, 3])
        spatial_variation_loss = self.mse(channel_spatial, init_spatial)
        return spatial_variation_loss

class SmoothDepthLoss(nn.Module):
    """
    深度平滑损失：边缘感知平滑 (Edge-aware Smoothness)。
    利用图像 RGB 的梯度作为权重，约束深度图的梯度，这在处理 2D/3DGS 产生的伪影时极为有效。
    """
    def __init__(self):
        super().__init__()

    def forward(self, rgb, depth):
        # 计算深度的 X, Y 方向梯度
        depth_dx = depth.diff(dim=-1)
        depth_dy = depth.diff(dim=-2)

        # 计算 RGB 图像的 X, Y 方向梯度
        rgb_dx = torch.mean(rgb.diff(dim=-1), axis=-3, keepdim=True)
        rgb_dy = torch.mean(rgb.diff(dim=-2), axis=-3, keepdim=True)

        # 边缘感知：RGB 梯度越大的地方（物体边缘），对深度平滑的约束越小
        depth_dx *= torch.exp(-rgb_dx)
        depth_dy *= torch.exp(-rgb_dy)

        return torch.abs(depth_dx).mean() + torch.abs(depth_dy).mean()