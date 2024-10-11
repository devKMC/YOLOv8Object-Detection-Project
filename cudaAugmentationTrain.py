from ultralytics import YOLO

if __name__ == '__main__':
    # 이전 훈련 상태에서 가중치 로드
    # model = YOLO('yolo8n-p2.yaml').load('C:/Users/user/Desktop/office/yolov8/runs/detect/train4/weights/last.pt')
    model = YOLO('yolov8n-p2.yaml').load('C:/Users/user/Desktop/office/yolov8/yolov8n.pt')
    
    # 이어서 모델 훈련
    model.train(data='coco.yaml', 
                epochs=10,  # 추가로 훈련할 에포크 수
                imgsz=1024, 
                cls=0.0,
                mosaic=0.0,
                hsv_v=0.6, # 명도 
                hsv_s=0.5  # 채도
                # batch=16, 
                # device='0', 
                # workers=8, 
                # lr0=0.01,                # 초기 학습률
                # lrf=0.01,                # 최종 학습률
                # momentum=0.937,          # SGD 모멘텀 / Adam beta1
                # weight_decay=0.0005,     # 가중치 감쇠
                # warmup_epochs=3.0,       # 웜업 단계 학습 에포크 수
                # warmup_momentum=0.8,     # 웜업 단계 초기 모멘텀
                # warmup_bias_lr=0.1,      # 웜업 단계 초기 편향 학습률
                # box=7.5,                 # 바운딩 박스 손실 가중치
                # kobj=1.0,                # 객체 손실 가중치
                # hsv_h=0.015,             # 이미지 HSV 색조 증강 비율
                # hsv_s=0.7,               # 이미지 HSV 채도 증강 비율
                # hsv_v=0.4,               # 이미지 HSV 명도 증강 비율
                # translate=0.0,           # 이미지 평행 이동 증강 비율
                # scale=0.5,               # 이미지 스케일 증강 비율
                # mosaic=0.0,              # 여러 이미지를 하나로 합치는 비율
                # cls=0.0,                 # 클래스 비율
                # mixup=0.0,               # 두 이미지를 혼합하는 비율
                # shear=0.0,               # 기울기 비율
                # fliplr=0.5               # 좌우 이미지 반전 확률)
                )               
    