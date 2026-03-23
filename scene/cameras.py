#加入了语义掩码读取


import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 gt_depth=None,
                 gt_semantic_mask=None): # <---- [新增] 接收语义特征参数
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
            self.gt_alpha_mask = gt_alpha_mask.to(self.data_device)
        else:
            self.gt_alpha_mask = None
        
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        # [关键] 存储深度图
        self.original_depth = None
        if gt_depth is not None: # 使用统一后的参数名 gt_depth
            if isinstance(gt_depth, np.ndarray):
                self.original_depth = torch.from_numpy(gt_depth).float().to(self.data_device).unsqueeze(0)
            elif isinstance(gt_depth, torch.Tensor):
                self.original_depth = gt_depth.to(self.data_device)
            
            # 确保维度是 (1, H, W)
            if self.original_depth.ndim == 2:
                self.original_depth = self.original_depth.unsqueeze(0)

        # [新增] 将语义图存入 GPU，放在 __init__ 函数的最末尾
        self.gt_semantic_mask = None
        if gt_semantic_mask is not None:
            # 确保传入的 mask 直接挂载到指定的设备上，避免后续前向传播时产生 cpu-gpu 同步开销
            self.gt_semantic_mask = gt_semantic_mask.to(self.data_device)

# =================================================================================
# [必须包含] MiniCam 类
# =================================================================================
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]