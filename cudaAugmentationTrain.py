from ultralytics import YOLO

# 증강을 사용한 학습
# YOLO v8 에서는 파일 전달 대신 딕셔너리 형태로 전달해야 함

if __name__ == '__main__':
    # Model 로드 및 이전 훈련 상태에서 재개 
    model = YOLO('yolo8n-p2.yaml').load('runs/detect/train2/weights/last.pt')
    
    # 모델 훈련
    model.train(data='coco.yaml', 
                epochs=100, 
                imgsz=1024, 
                batch=16, 
                device='0', 
                workers=8, 
                lr0=0.01,                # 초기 학습률
                lrf=0.01,                # 최종 학습률
                momentum=0.937,          # SGD 모멘텀 / Adam beta1
                weight_decay=0.0005,     # 가중치 감쇠
                warmup_epochs=3.0,       # 웜업 단계 학습 에포크 수
                warmup_momentum=0.8,     # 웜업 단계 초기 모멘텀
                warmup_bias_lr=0.1,      # 웜업 단계 초기 편향 학습률
                box=7.5,                 # 바운딩 박스 손실 가중치
                kobj=1.0,                # 객체 손실 가중치
                iou=0.2,                 # IoU 학습 임계값
                hsv_h=0.015,             # 이미지 HSV 색조 증강 비율
                hsv_s=0.7,               # 이미지 HSV 채도 증강 비율
                hsv_v=0.4,               # 이미지 HSV 명도 증강 비율
                translate=0.1,           # 이미지 평행 이동 증강 비율
                scale=0.5,               # 이미지 스케일 증강 비율
                fliplr=0.5)              # 좌우 이미지 반전 확률)