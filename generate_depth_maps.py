


import sys
import os
import cv2
import torch
import numpy as np

# ===================== 关键路径配置（必须替换为你的实际路径！） =====================
# 1. Depth-Anything源码目录（合并后的单层目录，如/root/Depth-Anything）
DEPTH_ANYTHING_SRC_PATH = "/root/Depth-Anything"
# 2. depth_anything_vits14权重目录（如/root/depth_anything_vits14）
MODEL_WEIGHTS_PATH = "/root/depth_anything_vits14"
# 3. 你的水下图像目录（如/root/autodl-tmp/Sea-thru/Curasao/images/jpg_images）
IMAGE_INPUT_DIR = "/root/2d-gaussian-splatting-old/output/sea-thru-1/train/ours_30000/gt"
# 4. 深度图保存目录（自动创建）
DEPTH_OUTPUT_DIR = "/root/2d-gaussian-splatting-old/output/sea-thru-1/train/ours_30000/deepanything"
# ====================================================================================

# 将Depth-Anything源码目录加入Python搜索路径（解决导入问题）
sys.path.append(DEPTH_ANYTHING_SRC_PATH)

# 导入Depth-Anything模块（路径添加后可正常导入）
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet


def init_depth_model(weights_path, device):
    """初始化Depth-Anything模型"""
    try:
        model = DepthAnything.from_pretrained(weights_path).to(device)
        model.eval()
        print(f"✅ 模型加载成功：{weights_path}")
    except Exception as e:
        print(f"❌ 模型加载失败：{e}")
        sys.exit(1)

    # 定义图像预处理流水线（匹配模型要求）
    transform = torch.nn.Sequential(
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='bilinear',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    )
    return model, transform


def process_single_image(model, transform, img_path, device):
    """处理单张图像生成深度图"""
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ 跳过无效图像：{img_path}")
        return None

    # 图像预处理（BGR转RGB + 模型输入格式）
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = transform({"image": img_rgb})["image"]
    img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).to(device)

    # 推理生成深度图
    with torch.no_grad():
        depth_pred = model(img_tensor)

    # 深度图后处理（归一化到0-255，便于保存）
    depth_map = depth_pred.squeeze().cpu().numpy()
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255.0
    depth_map = depth_map.astype(np.uint8)

    return depth_map


def batch_process_images(model, transform, input_dir, output_dir, device):
    """批量处理图像生成深度图"""
    # 创建保存目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 深度图将保存到：{output_dir}")

    # 遍历图像目录
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    img_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]

    if not img_files:
        print(f"❌ 图像目录下无有效图像：{input_dir}")
        sys.exit(1)

    # 批量处理
    total = len(img_files)
    for idx, img_name in enumerate(img_files):
        img_path = os.path.join(input_dir, img_name)
        depth_map = process_single_image(model, transform, img_path, device)

        if depth_map is not None:
            # 保存深度图（命名：原文件名_depth.png）
            depth_name = os.path.splitext(img_name)[0] + "_depth.png"
            depth_path = os.path.join(output_dir, depth_name)
            cv2.imwrite(depth_path, depth_map)
            print(f"[{idx+1}/{total}] ✅ 已生成：{depth_name}")

    print(f"\n🎉 批量处理完成！共生成{total}张深度图（保存至{output_dir}）")


if __name__ == "__main__":
    # 检查关键路径是否存在
    if not os.path.exists(DEPTH_ANYTHING_SRC_PATH):
        print(f"❌ Depth-Anything源码目录不存在：{DEPTH_ANYTHING_SRC_PATH}")
        sys.exit(1)
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"❌ 模型权重目录不存在：{MODEL_WEIGHTS_PATH}")
        sys.exit(1)
    if not os.path.exists(IMAGE_INPUT_DIR):
        print(f"❌ 图像目录不存在：{IMAGE_INPUT_DIR}")
        sys.exit(1)

    # 设置设备（优先GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ 使用设备：{device}")

    # 初始化模型
    model, transform = init_depth_model(MODEL_WEIGHTS_PATH, device)

    # 批量生成深度图
    batch_process_images(model, transform, IMAGE_INPUT_DIR, DEPTH_OUTPUT_DIR, device)