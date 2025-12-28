import cv2
import numpy as np
import torch
import os
from ultralytics import YOLO
from PIL import Image
from torchvision.transforms import ToTensor, Resize, Compose, Normalize
from anomalib.models import Patchcore

# ================= 配置区 =================
INPUT_DIR = 'ori_bad_pic'  # 输入文件夹
OUTPUT_DIR = 'detection_results'  # 输出文件夹

STAGE1_MODEL_PATH = 'runs/detect/train/weights/best.pt'
ANOMALY_CHECKPOINT_PATH = 'results/patchcore/Patchcore/rivet/v4/weights/lightning/model.ckpt'

DEFECT_COLOR = (0, 0, 255)  # 红色描边
MANUAL_THRESHOLD = 0.4  # 0.4 ~ 0.6
# =========================================

# 环境设置
os.environ["TRUST_REMOTE_CODE"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 预处理
transform = Compose([
    Resize((256, 256)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_models():
    print("🚀 正在初始化模型 (只加载一次)...")

    # 1. 加载 YOLO
    print("   - 加载 YOLO...")
    model_yolo = YOLO(STAGE1_MODEL_PATH)

    # 2. 加载 PatchCore
    print(f"   - 加载 PatchCore: {ANOMALY_CHECKPOINT_PATH}...")
    if not os.path.exists(ANOMALY_CHECKPOINT_PATH):
        print("❌ 找不到 .ckpt 文件！")
        exit()

    # 提取参数
    stats_min, stats_max, pixel_threshold = None, None, 0.5
    try:
        checkpoint = torch.load(ANOMALY_CHECKPOINT_PATH, map_location=device)
        state_dict = checkpoint['state_dict']

        # 查找归一化参数
        if 'normalization_metrics.min' in state_dict:
            stats_min = state_dict['normalization_metrics.min'].cpu()
            stats_max = state_dict['normalization_metrics.max'].cpu()
        elif 'min_max.min' in state_dict:
            stats_min = state_dict['min_max.min'].cpu()
            stats_max = state_dict['min_max.max'].cpu()

        # 查找阈值
        if 'pixel_threshold.value' in state_dict:
            pixel_threshold = state_dict['pixel_threshold.value'].item()
        elif 'image_threshold.value' in state_dict:
            pixel_threshold = state_dict['image_threshold.value'].item()

        p_min = f"{stats_min:.4f}" if stats_min is not None else "自动推断"
        p_max = f"{stats_max:.4f}" if stats_max is not None else "自动推断"
        print(f"✅ 参数加载状态 | Min: {p_min}, Max: {p_max}, Threshold: {pixel_threshold:.4f}")

        # 加载模型结构
        model_anomaly = Patchcore.load_from_checkpoint(ANOMALY_CHECKPOINT_PATH)
        model_anomaly.to(device)
        model_anomaly.eval()

    except Exception as e:
        print(f"⚠️ 模型参数读取微恙 (启用自适应模式): {e}")
        model_anomaly = Patchcore.load_from_checkpoint(ANOMALY_CHECKPOINT_PATH)
        model_anomaly.to(device)
        model_anomaly.eval()

    return model_yolo, model_anomaly, stats_min, stats_max, pixel_threshold


def robust_normalize_heatmap(heatmap, min_v, max_v):
    if min_v is not None and max_v is not None:
        denominator = max_v - min_v
        if denominator == 0: denominator = 1.0
        heatmap_norm = (heatmap - min_v) / denominator
    else:
        # 自适应兜底
        curr_min = heatmap.min()
        curr_max = heatmap.max()
        denominator = curr_max - curr_min
        if denominator == 0: denominator = 1.0
        heatmap_norm = (heatmap - curr_min) / denominator
    return torch.clamp(heatmap_norm, 0, 1)


def draw_mask_on_image(frame, crop_x, crop_y, heatmap, threshold, scale_factor):
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.squeeze().cpu().numpy()

    mask = ((heatmap > threshold) * 255).astype(np.uint8)
    real_size = int(256 * scale_factor)
    mask_resized = cv2.resize(mask, (real_size, real_size), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    has_defect = False
    for cnt in contours:
        if cv2.contourArea(cnt) < 20: continue
        has_defect = True
        cnt_shifted = cnt + np.array([crop_x, crop_y])
        cv2.drawContours(frame, [cnt_shifted], -1, DEFECT_COLOR, 2)

    return has_defect


def process_one_image(img_path, output_path, models, params):
    model_yolo, model_anomaly = models
    stats_min, stats_max, pixel_threshold = params

    frame = cv2.imread(img_path)
    if frame is None:
        print(f"❌ 无法读取: {img_path}")
        return

    h_img, w_img = frame.shape[:2]
    results = model_yolo(frame, verbose=False)

    rivet_count = 0
    defect_count = 0

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        for box in boxes:
            rivet_count += 1
            rx1, ry1, rx2, ry2 = map(int, box)

            if rx1 < 10 or ry1 < 10 or rx2 > w_img - 10 or ry2 > h_img - 10: continue

            w_box = rx2 - rx1
            h_box = ry2 - ry1
            max_side = max(w_box, h_box)
            pad = int(max_side * 0.2)
            final_size = max_side + pad

            cx, cy = (rx1 + rx2) // 2, (ry1 + ry2) // 2
            crop_x1 = max(0, cx - final_size // 2)
            crop_y1 = max(0, cy - final_size // 2)
            crop_x2 = min(w_img, cx + final_size // 2)
            crop_y2 = min(h_img, cy + final_size // 2)

            crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            if crop.size == 0: continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_pil = Image.fromarray(crop_rgb)
            input_tensor = transform(crop_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model_anomaly(input_tensor)
                if hasattr(output, "anomaly_map"):
                    heatmap = output.anomaly_map
                elif isinstance(output, tuple):
                    heatmap = output[1]
                else:
                    heatmap = output

            heatmap = torch.nn.functional.interpolate(
                heatmap.unsqueeze(0) if heatmap.dim() == 2 else heatmap,
                size=(256, 256), mode='bilinear'
            )

            heatmap_norm = robust_normalize_heatmap(heatmap, stats_min, stats_max)

            final_thresh = MANUAL_THRESHOLD if MANUAL_THRESHOLD else pixel_threshold
            scale = final_size / 256.0
            is_defect = draw_mask_on_image(frame, crop_x1, crop_y1, heatmap_norm, final_thresh, scale)

            if is_defect:
                defect_count += 1
                #cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
                #cv2.putText(frame, "NG", (rx1, ry1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                # 良品画绿框
                # cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
                pass

    cv2.imwrite(output_path, frame)
    print(f"📊 检测报告: 铆钉 {rivet_count} 个 | 缺陷 {defect_count} 个")
    print(f"💾 结果已保存: {output_path}")


def main():
    # 1. 准备目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 2. 扫描文件
    supported = ('.tiff', '.tif', '.jpg', '.png', '.jpeg')
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(supported)]
    files.sort()  # 按文件名排序

    if not files:
        print(f"❌ 目录 '{INPUT_DIR}' 为空！")
        return

    # 3. 加载模型
    model_yolo, model_anomaly, s_min, s_max, thresh = load_models()
    models = (model_yolo, model_anomaly)
    params = (s_min, s_max, thresh)

    # 4. 循环处理
    print(f"\n📂 准备处理 {len(files)} 张图片...")
    print("=" * 40)

    for i, filename in enumerate(files):
        print(f"\n📸 [{i + 1}/{len(files)}] 正在检测: {filename}")

        in_path = os.path.join(INPUT_DIR, filename)

        # 自动生成输出文件名 (原名_result.jpg)
        name_only = os.path.splitext(filename)[0]
        out_path = os.path.join(OUTPUT_DIR, f"{name_only}_result.jpg")

        process_one_image(in_path, out_path, models, params)

        # 暂停逻辑
        if i < len(files) - 1:
            input("\n👉 按 Enter 键继续检测下一张...")
        else:
            print("\n🎉 全部图片检测完毕！")


if __name__ == "__main__":
    main()