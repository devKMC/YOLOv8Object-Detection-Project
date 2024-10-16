train 13 =      epochs=10, 
                imgsz=1024, 
                cls=0.0,
                hsv_v=0.6,
                hsv_s=0.5 

train 14 =      data='coco.yaml', 
                epochs=10,  # 추가로 훈련할 에포크 수
                imgsz=1024, 
                cls=0.0,
                mosaic=1.0,  # 모자이크 증강 적용
                hsv_h=0.02,  # 색조 조정
                hsv_s=0.6,   # 채도 조정
                hsv_v=0.5,   # 밝기 조정
                degrees=10,  # 랜덤 회전 각도
                translate=0.1,  # 이미지 평행 이동
                scale=0.5,   # 이미지 스케일 조정
                mixup=0.3,   # 믹스업 증강
                copy_paste=0.2,  # 복사-붙여넣기 증강
                erasing=0.3,  # 무작위 지우기
                crop_fraction=0.9  # 중심 부분 강조 크롭

train 15 =      model.train(data='coco.yaml', 
                epochs=150,  # 추가로 훈련할 에포크 수
                imgsz=1024, 
                cls=0.0,
                hsv_v=0.6, # 명도 
                hsv_s=0.5,  # 채도
                save_period=10,
                patience=50,
                device='0'
                이미지 적게해서 돌림

train 16 = 15의 best pt로 재 학습