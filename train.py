import torch
from ultralytics import YOLO

# Model 로드
model = YOLO('yolo8n-p2.yaml')

# 모델 훈련
model.train(data='coco.yaml', epochs=100, imgsz=1280)



model.eval()
