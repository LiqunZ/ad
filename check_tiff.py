import cv2
import numpy as np
import sys

# 换成你刚才上传的那个 tiff 文件路径
img_path = "39.tiff"

# 1. 尝试用默认方式读取
img = cv2.imread(img_path)

if img is None:
    print(f"❌ 默认读取失败！尝试使用无损模式读取...")
    # 2. 尝试用无损模式读取 (IMREAD_UNCHANGED)
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

if img is None:
    print("❌ 彻底读取失败，请检查路径或安装 libtiff 库")
    sys.exit()

print(f"✅ 读取成功！图片信息如下：")
print(f"-----------------------------")
print(f"📏 尺寸 (H, W): {img.shape[:2]}")
print(f"🎨 通道数: {img.shape[2] if len(img.shape)>2 else 1}")
print(f"🔢 数据类型 (Dtype): {img.dtype}")
print(f"📊 像素极值: Min={img.min()}, Max={img.max()}")

if img.dtype == 'uint16':
    print("\n⚠️ 警告：这是一张 16-bit 图片！")
    print("👉 必须先转成 8-bit 才能喂给 YOLO，否则模型看不懂。")
else:
    print("\n✅这是一张标准的 8-bit 图片，可以直接用。")