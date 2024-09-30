from ultralytics import YOLO

if __name__ == '__main__':
    # Model 로드 및 이전 훈련 상태에서 재개 
    # .loac (경로) 사용해서 학습 도중에 멈춘 부분에서 다시 학습 가능
    model = YOLO('yolo8n-p2.yaml')

    # 모델 훈련
    model.train(data='coco.yaml', epochs=100, imgsz=640, batch=16, device='0', workers=8, resume=True)
