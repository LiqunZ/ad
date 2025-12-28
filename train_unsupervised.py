
import os
# 确保安装了 anomalib: pip install anomalib
from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.deploy import ExportType


def train():
    # 1. 告诉它数据在哪
    datamodule = Folder(
        name="rivet",
        root="anomalib_data/rivet",
        normal_dir="train/good",  # 指向 train 里的 good
        abnormal_dir="test/defect",  # 指向 test 里的 defect
        normal_test_dir="test/good",  # 指向 test 里的 good (如果有的话)
    )

    # 2. 定义模型 (PatchCore)
    # backbone 用 wide_resnet50_2 效果最好
    model = Patchcore(
        backbone="wide_resnet50_2",
        coreset_sampling_ratio=0.1,  # 采样 10% 特征，速度快
    )

    # 3. 训练引擎
    engine = Engine(
        accelerator="auto",
        devices=1,
        max_epochs=1,  # 只要 1 轮！它是特征库匹配，不是深度学习反向传播
        default_root_dir="results/patchcore"
    )

    print("🚀 开始提取良品特征...")
    engine.fit(datamodule=datamodule, model=model)

    print("👀 正在测试并生成热力图...")
    engine.test(datamodule=datamodule, model=model)

    # 4. 导出为 Torch 模型 (方便后续调用)
    print("💾 正在保存模型...")
    # 导出到 weights/model.pt
    engine.export(
        model=model,
        export_type=ExportType.TORCH,
        export_root="weights_anomaly",
    )


if __name__ == "__main__":
    train()