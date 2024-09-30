from ultralytics import YOLO
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os


## 클래스가 여러개일때 쓰면 좋음  (사람만 감지할때는 혼동행렬 의미 없음)
# YOLOv8 모델 로드
model = YOLO('runs/detect/train/weights/best.pt')  # 모델 경로를 수정하세요.

# 테스트 이미지 경로와 실제 레이블 리스트
test_images = [
    'C:/Users/user/Desktop/office/yolov8/human1.jpg',
    'C:/Users/user/Desktop/office/yolov8/human2.jpg',
    
    # 추가 이미지 경로를 여기에 추가하세요.
]

# 실제 레이블 리스트 (모든 이미지가 클래스 0)
y_true = [0] * len(test_images)  # 테스트 이미지 수에 맞게 0으로 초기화

# 예측 결과 리스트
y_pred = []

# 각 테스트 이미지에 대해 예측 수행
for img_path in test_images:
    results = model.predict(source=img_path)
    if results[0].boxes is not None and len(results[0].boxes.cls) > 0:
        pred_label = int(results[0].boxes.cls[0].item())  # 첫 번째 예측 클래스 추출
    else:
        pred_label = 0  # 탐지된 객체가 없는 경우 기본값으로 0 설정
    y_pred.append(pred_label)

# 혼동 행렬 생성
cm = confusion_matrix(y_true, y_pred)

# 혼동 행렬 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0'], yticklabels=['Class 0'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
