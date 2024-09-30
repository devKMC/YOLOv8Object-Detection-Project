from ultralytics import YOLO

# YOLOv8 모델 로드
model = YOLO('yolov8n.pt')


# 다운로드 후 훈련 시작
# 데이터셋의 YAML 파일 경로를 지정합니다.
model.train(data='coco.yaml')  # 필요한 경우 경로 수정
