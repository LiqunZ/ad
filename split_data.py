import os
import shutil
import random

# ================= 配置区 =================
# 1. 这里填你刚才存放图片和txt的那个文件夹路径
#    (如果不确定，可以在终端输入 pwd 查看，或者直接把文件夹拖进终端获取路径)
source_folder = "./good_pic"  # 举例，请修改为你真实的文件夹名！

# 2. 这里是你想要生成的标准数据集文件夹名字
dataset_name = "yolo_dataset"

# 3. 划分比例 (0.8 表示 80% 训练，20% 验证)
train_ratio = 0.8


# =========================================

def split_dataset():
    # 1. 准备好目录结构
    dirs = [
        f"{dataset_name}/images/train",
        f"{dataset_name}/images/val",
        f"{dataset_name}/labels/train",
        f"{dataset_name}/labels/val"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    print(f"✅ 目录结构已创建: {dataset_name}/")

    # 2. 获取所有图片文件
    files = os.listdir(source_folder)
    images = [f for f in files if f.endswith(('.jpg', '.png', '.jpeg', '.bmp'))]

    # 打乱顺序，保证随机性
    random.shuffle(images)

    train_count = int(len(images) * train_ratio)

    print(f"🔍 发现 {len(images)} 张图片。准备划分：训练集 {train_count} 张，验证集 {len(images) - train_count} 张。")

    # 3. 开始搬运
    for i, img_name in enumerate(images):
        # 构建源文件路径
        src_img_path = os.path.join(source_folder, img_name)
        src_txt_path = os.path.join(source_folder, img_name.rsplit('.', 1)[0] + '.txt')

        # 检查对应的 txt 是否存在 (防止你有的图忘了标)
        if not os.path.exists(src_txt_path):
            print(f"⚠️ 警告：{img_name} 没有对应的 .txt 标签文件，已跳过！")
            continue

        # 决定是去 train 还是 val
        if i < train_count:
            type_dir = "train"
        else:
            type_dir = "val"

        # 复制图片
        shutil.copy(src_img_path, f"{dataset_name}/images/{type_dir}/{img_name}")
        # 复制标签
        shutil.copy(src_txt_path, f"{dataset_name}/labels/{type_dir}/{img_name.rsplit('.', 1)[0] + '.txt'}")

    print("\n🎉 大功告成！数据已整理完毕！")
    print(f"📁 新的数据集在文件夹: {dataset_name}")


if __name__ == "__main__":
    if not os.path.exists(source_folder):
        print(f"❌ 错误：找不到源文件夹 '{source_folder}'，请修改代码中的 source_folder 路径！")
    else:
        split_dataset()