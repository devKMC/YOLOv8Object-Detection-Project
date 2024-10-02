
from ultralytics import YOLO

# 학습된 모델 로드
model = YOLO('runs/detect/train2/weights/best.pt')

# 예측할 이미지 경로 (바깥단에 위치하면 됨)
image_path = 'human1.jpg'

# 이미지에서 객체 탐지
results = model.predict(source=image_path, iou=0.1, conf=0.15)


# 탐지된 결과 시각화 리스트 가능(jpg 갯수에 따라) []
results[0].show()

