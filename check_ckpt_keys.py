import torch
import os

# 你的 ckpt 文件路径 (请确保路径正确)
ckpt_path = 'results/patchcore/Patchcore/rivet/v4/weights/lightning/model.ckpt'

if not os.path.exists(ckpt_path):
    print(f"❌ 找不到文件: {ckpt_path}")
    exit()

print(f"🚀 正在加载: {ckpt_path} ...")

try:
    # 加载 checkpoint
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    # 检查是否有 state_dict
    if 'state_dict' not in checkpoint:
        print("❌ 这个 checkpoint 里没有 'state_dict'！它包含的 keys 是：", checkpoint.keys())
        exit()

    state_dict = checkpoint['state_dict']
    print(f"✅ 加载成功！包含 {len(state_dict)} 个参数。")
    print("-" * 40)
    print("🔍 正在搜索关键参数 (Threshold / Normalization)...")
    print("-" * 40)

    found_any = False

    # 遍历所有 key，寻找我们感兴趣的
    for key, value in state_dict.items():
        # 过滤关键词
        if any(x in key for x in ['threshold', 'normalization', 'min', 'max']):
            print(f"🔑 Key: {key}")
            # 打印一下值的类型和内容，看看是不是标量
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                print(f"   Value: {value.item()}")
            else:
                print(f"   Shape: {value.shape}")
            found_any = True

    if not found_any:
        print("❌ 没找到任何带 threshold/min/max 的 key。可能命名完全变了。")
        print("以下是前 20 个 key，供参考：")
        for i, key in enumerate(list(state_dict.keys())[:20]):
            print(f" - {key}")

except Exception as e:
    print(f"❌ 读取出错: {e}")