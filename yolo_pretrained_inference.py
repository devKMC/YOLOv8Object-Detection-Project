from ultralytics import YOLO

# 사전 훈련된 YOLO 모델 로드 (예: YOLOv8n)
model = YOLO('yolov8n.pt')  # YOLOv8의 사전 훈련된 모델 로드

# 예측할 이미지 경로
image_path = 'imageTest/human15.jpg'

# 이미지에서 객체 탐지
results = model.predict(source=image_path, iou=0.001, conf=0.1)


# 탐지된 결과 시각화 리스트 가능(jpg 갯수에 따라) []
results[0].show(line_width=1, labels=False)
