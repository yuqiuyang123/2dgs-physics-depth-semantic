import math
import torch
import torch.nn as nn

# ==========================================
# [新增] 物理大气光估算 (用于约束 B_inf)
# 根据经典的暗通道先验理论，寻找图像中最暗的像素点
# 并在这些点对应的原始 RGB 中取均值，作为环境背景光的估算值
# ==========================================
def dark_channel_estimate(rgb):
    patch_size = 41
    padding = patch_size // 2
    rgb_max = nn.MaxPool3d(
        kernel_size=(3, patch_size, patch_size),
        stride=1,
        padding=(0, padding, padding)
    )
    if len(rgb.size()) == 3:
        dcp = torch.abs(rgb_max(-rgb.unsqueeze(0)))
    else:
        dcp = torch.abs(rgb_max(-rgb))
    return dcp.squeeze()

def estimate_atmospheric_light(rgb):
    dcp = dark_channel_estimate(rgb)
    flat_dcp = torch.flatten(dcp)
    flat_r = torch.flatten(rgb[0])
    flat_g = torch.flatten(rgb[1])
    flat_b = torch.flatten(rgb[2])

    # 取暗通道中最亮的前 0.1% 的像素
    k = math.ceil(0.001 * len(flat_dcp))
    vals, idxs = torch.topk(flat_dcp, k)

    median_val = torch.median(vals)
    median_idxs = torch.where(vals == median_val)
    color_idxs = idxs[median_idxs]

    # 计算这些位置对应的 RGB 均值，即为物理背景光
    atmospheric_light = torch.stack([
        torch.mean(flat_r[color_idxs]), 
        torch.mean(flat_g[color_idxs]), 
        torch.mean(flat_b[color_idxs])
    ])
    return atmospheric_light

# ==========================================

class BackscatterNetV2(nn.Module):
    '''
    估计后向散射 (Backscatter): B = B_inf * (1 - exp(-beta_b * z))
    '''
    def __init__(self, scale: float = 5.0):
        super().__init__()
        self.scale = scale
        # [已修复] 修正为 -5.0，使得 sigmoid(-5.0) 接近 0
        self.backscatter_conv_params = nn.Parameter(torch.ones(3, 1, 1, 1) * -5.0)
        self.B_inf = nn.Parameter(torch.rand(3, 1, 1))
        self.relu = nn.ReLU()
        self.l2 = nn.MSELoss()

    def forward(self, depth):
        beta_b_conv = self.relu(torch.nn.functional.conv2d(depth, self.scale * torch.sigmoid(self.backscatter_conv_params)))
        backscatter = torch.sigmoid(self.B_inf) * (1 - torch.exp(-beta_b_conv))
        return backscatter

    # [新增] B_inf 约束计算
    def forward_rgb(self, rgb):
        atmospheric_colors = []
        for rgb_image in rgb:
            atmospheric_colors.append(estimate_atmospheric_light(rgb_image.detach()))
        
        atmospheric_color = torch.mean(torch.stack(atmospheric_colors), dim=0)
        # 约束网络学习的 B_inf 趋近于物理估算的环境光
        return self.l2(atmospheric_color.squeeze(), torch.sigmoid(self.B_inf).squeeze())

class AttenuateNetV3(nn.Module):
    '''
    估计衰减 (Attenuation): A = exp(-beta_d * z)
    '''
    def __init__(self, scale: float = 5.0):
        super().__init__()
        # [已修复] 修正为 -5.0
        self.attenuation_conv_params = nn.Parameter(torch.ones(3, 1, 1, 1) * -5.0)
        self.scale = scale
        self.relu = nn.ReLU()

    def forward(self, depth):
        beta_d_conv = self.relu(torch.nn.functional.conv2d(depth, self.scale * torch.sigmoid(self.attenuation_conv_params)))
        attenuation_map = torch.exp(-beta_d_conv)
        return attenuation_map