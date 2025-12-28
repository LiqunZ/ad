from ultralytics import YOLO

# 1. 加载模型
# 如果你刚才手动下载了 pt 文件，这里就会直接读取，不会卡下载
model = YOLO('yolov8n.pt')

# 2. 开始训练
# workers=0 是关键！在某些服务器上，多线程读取数据会卡死，先设为0测试
results = model.train(
    data='data.yaml',
    epochs=50,
    imgsz=640,
    workers=0
)