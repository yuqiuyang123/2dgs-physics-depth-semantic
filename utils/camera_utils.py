#读取深度图并且进行动态归一化，完全移除仿射变换，确保与伪深度真值的严格线性关系，从而使得皮尔逊相关系数能够正确反映预测深度与真值之间的相关性。
# 增加了语义特征读取

import numpy as np
import os
import cv2
import torch
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
from PIL import Image

WARNED = False
INFO_LOGGED = False # <---- [新增] 全局标志位，确保探针只打印一次，避免刷屏

def loadCam(args, id, cam_info, resolution_scale):
    # [关键] 延迟导入，防止循环引用
    from scene.cameras import Camera
    global WARNED, INFO_LOGGED # <---- [新增] 声明使用全局变量

    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  
        if args.resolution == -1:
            if orig_w > 1600:
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    # --- 加载 RGB ---
    if len(cam_info.image.split()) > 3:
        resized_image_rgb = torch.cat([PILtoTorch(im, resolution) for im in cam_info.image.split()[:3]], dim=0)
        loaded_mask = PILtoTorch(cam_info.image.split()[3], resolution)
        gt_image = resized_image_rgb
    else:
        resized_image_rgb = PILtoTorch(cam_info.image, resolution)
        loaded_mask = None
        gt_image = resized_image_rgb

    # --- 加载深度并进行归一化 ---
    aligned_inv_depth = None
    depth_loaded_success = False # <---- [新增] 状态记录
    
    try:
        # 自动推导 .npy 路径
        # 假设结构: .../images/xxx.png -> .../depth_anything/xxx.npy
        base_dir = os.path.dirname(os.path.dirname(cam_info.image_path)) 
        depth_folder = os.path.join(base_dir, "depth_anything")
        depth_path = os.path.join(depth_folder, cam_info.image_name + ".npy")
        
        # 兼容性检查：有些数据集可能叫 depths
        if not os.path.exists(depth_path):
            depth_path = os.path.join(base_dir, "depths", cam_info.image_name + ".npy")

        if os.path.exists(depth_path):
            # 1. 读取原始数据
            mono_depth = np.load(depth_path).astype(np.float32)
            
            # 维度修正
            if mono_depth.ndim == 3: mono_depth = mono_depth.squeeze()
            
            # 2. Resize 到当前分辨率
            if mono_depth.shape[0] != resolution[1] or mono_depth.shape[1] != resolution[0]:
                mono_depth = cv2.resize(mono_depth, resolution, interpolation=cv2.INTER_NEAREST)
            
            # 3. [核心修改] 动态 Min-Max 归一化 (完全移除仿射变换)
            depth_min = mono_depth.min()
            depth_max = mono_depth.max()
            
            # 防止除零 (当整张图深度完全一致时的极端情况)
            if depth_max - depth_min > 1e-6:
                normalized_depth_np = (mono_depth - depth_min) / (depth_max - depth_min)
            else:
                normalized_depth_np = np.ones_like(mono_depth) * 1e-6
                
            # 映射到 [1e-6, 1.0] 区间
            # 这里的极小值下界非常重要，它保证了数据送入 loss 后，gt_flat > 0 的条件必定满足
            normalized_depth_np = np.clip(normalized_depth_np, 1e-6, 1.0)
            
            # 4. 转 Tensor 并上 GPU
            aligned_inv_depth = torch.from_numpy(normalized_depth_np).unsqueeze(0).cuda()
            depth_loaded_success = True # <---- 标记成功
            
        else:
            pass 
            # print(f"Warning: Depth file missing: {depth_path}")

    except Exception as e:
        print(f"[Error] Failed to process depth for {cam_info.image_name}: {e}")

    # ==========================================
    # [新增] 读取 NumPy 语义掩码并处理缩放
    # ==========================================
    gt_semantic_mask = None
    mask_loaded_success = False # <---- [新增] 状态记录
    
    if getattr(cam_info, 'semantic_mask_path', None) is not None and os.path.exists(cam_info.semantic_mask_path):
        mask_np = np.load(cam_info.semantic_mask_path)
        
        # 维度检查与修正：确保 mask 是二维的 (H, W)
        if mask_np.ndim == 3: 
            mask_np = mask_np.squeeze()

        # 离散拓扑标签缩放，必须使用最近邻插值 (INTER_NEAREST) 以防止出现插值出的浮点“伪类别”
        if mask_np.shape[0] != resolution[1] or mask_np.shape[1] != resolution[0]:
            mask_np = cv2.resize(mask_np, resolution, interpolation=cv2.INTER_NEAREST)
        
        # 将语义标签转为 long 整型 Tensor (CrossEntropyLoss 或类似分类损失的要求)
        gt_semantic_mask = torch.from_numpy(mask_np).long() 
        mask_loaded_success = True # <---- 标记成功
    # ==========================================

    # ==========================================
    # [新增] 数据探针打印逻辑 (仅触发一次)
    # ==========================================
    if not INFO_LOGGED:
        print(f"\n[{'='*40}]")
        print(f"[ 数据流探针 ] 正在检查视点 (ID: {id}, Name: {cam_info.image_name})")
        print(f"[-] RGB 分辨率: {resolution[0]}x{resolution[1]}")
        
        if depth_loaded_success:
            print(f"[-] 深度图状态: [ V 成功加载 ] Tensor Shape: {aligned_inv_depth.shape}")
        else:
            print(f"[-] 深度图状态: [ X 未找到或加载失败 ]")
            
        if mask_loaded_success:
            print(f"[-] 语义掩码状态: [ V 成功加载 ] Tensor Shape: {gt_semantic_mask.shape}")
        else:
            print(f"[-] 语义掩码状态: [ X 未找到或加载失败 ]")
        print(f"[{'='*40}]\n")
        
        INFO_LOGGED = True # 拦截后续视点的打印
    # ==========================================

    # --- 实例化 Camera ---
    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  gt_depth=aligned_inv_depth, 
                  gt_semantic_mask=gt_semantic_mask) # <---- 传入语义掩码

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []
    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))
    return camera_list

def camera_to_JSON(id, camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    
    # 兼容性读取
    fovy_val = getattr(camera, 'FoVy', getattr(camera, 'FovY', None))
    fovx_val = getattr(camera, 'FoVx', getattr(camera, 'FovX', None))
    height = getattr(camera, 'image_height', getattr(camera, 'height', 0))
    width = getattr(camera, 'image_width', getattr(camera, 'width', 0))

    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : width,
        'height' : height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(fovy_val, height),
        'fx' : fov2focal(fovx_val, width)
    }
    return camera_entry