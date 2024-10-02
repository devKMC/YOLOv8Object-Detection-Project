from ultralytics import YOLO

if __name__ == '__main__':
    # Model 로드 및 이전 훈련 상태에서 재개 
    # .load (경로) 사용해서 학습 도중에 멈춘 부분에서 다시 학습 가능
    model = YOLO('yolo8n-p2.yaml').load('runs/detect/train2/weights/last.pt')

    # 모델 훈련
    model.train(data='coco.yaml', epochs=100, imgsz=1024, batch=16, device='0', workers=8, resume=True)

    # train 폴더가 생기지 않으려면   project='runs/detect' 폴더 명시 ,  exist_ok=True 옵션으로 기존 폴더에 결과 덮기
    # model.train(data='coco.yaml', epochs=100, imgsz=640, batch=16, device='0', workers=8, resume=True, project='runs
    # /detect', name='train', exist_ok=True)
