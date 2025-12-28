import cv2
import os
import numpy as np
from ultralytics import YOLO

# ================= 配置区 =================
# 1. 模型路径
yolo_path = 'runs/detect/train/weights/best.pt'

# 2. 原图文件夹 (大图)
source_images_dir = 'ori_bad_pic'

# 3. 结果保存路径
save_dir = 'anomalib_data/rivet/test/defect'


# =========================================

def crop_objects_v2():
    model = YOLO(yolo_path)
    os.makedirs(save_dir, exist_ok=True)

    # ✅ 修改点 1：在后缀列表中加入 .tiff 和 .tif
    img_files = [f for f in os.listdir(source_images_dir) if
                 f.lower().endswith(('.jpg', '.png', '.jpeg', '.tiff', '.tif'))]
    print(f"🔍 开始处理 {len(img_files)} 张大图...")

    count = 0
    for img_file in img_files:
        img_path = os.path.join(source_images_dir, img_file)

        # ✅ 修改点 2：增强读取的鲁棒性 
        # 使用 IMREAD_UNCHANGED 确保能读入各种格式，然后转为标准的 BGR
        frame = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if frame is None:
            print(f"❌ 无法读取文件: {img_file}")
            continue

        # 处理通道 (防止读入透明通道或者灰度图报错)
        if len(frame.shape) == 2:  # 灰度转彩色
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:  # 去掉透明通道
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        h_img, w_img = frame.shape[:2]

        # YOLO 推理
        results = model(frame, verbose=False)

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)

                # --- 1. 检查是否在边缘 (过滤掉只露出一半的) ---
                margin_check = 10
                if x1 < margin_check or y1 < margin_check or x2 > w_img - margin_check or y2 > h_img - margin_check:
                    continue

                # --- 2. 动态扩充 ---
                w_box = x2 - x1
                h_box = y2 - y1

                max_side = max(w_box, h_box)
                pad = int(max_side * 0.2)
                final_size = max_side + pad

                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                new_x1 = max(0, cx - final_size // 2)
                new_y1 = max(0, cy - final_size // 2)
                new_x2 = min(w_img, cx + final_size // 2)
                new_y2 = min(h_img, cy + final_size // 2)

                crop = frame[new_y1:new_y2, new_x1:new_x2]

                # --- 3. 统一缩放 (Resize) ---
                if crop.size == 0: continue

                try:
                    crop_resized = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA)

                    # 保存 (注意：保存为 jpg 比较通用，体积也小)
                    save_name = f"{os.path.splitext(img_file)[0]}_crop_{count}.jpg"
                    cv2.imwrite(os.path.join(save_dir, save_name), crop_resized)
                    count += 1
                except Exception as e:
                    print(f"⚠️ 跳过一张异常截图: {e}")

    print(f"\n✅ 修正完成！已生成 {count} 张图片。")
    print(f"👉 快去 {save_dir} 看看，这次应该是完整的铆钉了！")


if __name__ == '__main__':
    crop_objects_v2()