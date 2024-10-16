
from ultralytics import YOLO

# 학습된 모델 로드
# model = YOLO('runs/detect/train10/weights/best.pt')

# 모델 로드
model = YOLO('C:/Users/user/Desktop/office/yolov8/runs/detect/train15/weights/best.pt')

# 예측할 이미지 경로 (바깥단에 위치하면 됨)
image_path = 'imageTest/human15.jpg'

# 이미지에서 객체 탐지
#human 1 ( iou= 0.45 , conf=0.1)
results = model.predict(source=image_path, iou=0, conf=0.02, classes=0)

#human 11 ( iou= )


# 탐지된 결과 시각화 리스트 가능(jpg 갯수에 따라) []
results[0].show(labels=False)


