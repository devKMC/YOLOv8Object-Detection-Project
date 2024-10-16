COCO dataset Seource -> dataset create -> train
seource에서 변환하여 dataset을 만들어 학습이나 검증 및 훈련

├─dataset
│  ├─train
│  │  ├─images        # 학습용 이미지
│  │  └─labels        # 학습용 라벨
│  └─val
│      ├─images       # 검증용 이미지
│      └─labels       # 검증용 라벨
├─imageTest           # 테스트용 이미지
│  └─output           # 테스트용 이미지 기반 테스트
├─newDataset
│  ├─train
│  │  ├─images
│  │  └─labels
│  └─val
│      ├─images
│      └─labels
├─ runs
│  └─ detect
│      └─ train
│          ├─ weights # 학습된 모델 가중치
│          │  ├─ best.pt         # 최고의 성능을 가진 모델 가중치
│          │  ├─ epoch_10.pt      # 10 에포크 후의 모델 가중치
│          │  └─ epoch_20.pt      # 20 에포크 후의 모델 가중치
└─ source
    ├─ annotations    # 주석 파일 (예: COCO 형식)
    └─ images
        ├─ train2017  # 학습 이미지 (2017)
        └─ val2017    # 검증 이미지 (2017)

데이터셋 설명
dataset/train/images: 모델 학습에 사용되는 이미지 파일이 저장되는 디렉터리
dataset/train/labels: 학습 이미지에 대한 주석(라벨) 파일이 저장되는 디렉터리
dataset/val/images: 모델 검증에 사용되는 이미지 파일이 저장되는 디렉터리
dataset/val/labels: 검증 이미지에 대한 주석(라벨) 파일이 저장되는 디렉터리
runs/detect/train/weights: 모델 학습이 완료된 후 저장되는 가중치 파일
best.pt: 최고의 성능을 가진 모델 가중치
epoch_10.pt: 10 에포크 후의 모델 가중치
epoch_20.pt: 20 에포크 후의 모델 가중치
source/annotations: 데이터셋의 주석 파일이 저장되는 디렉터리입니다. (예: COCO 형식)
source/images/train2017: 2017년 학습 데이터에 해당하는 이미지 파일이 저장되는 디렉터리
source/images/val2017: 2017년 검증 데이터에 해당하는 이미지 파일이 저장되는 디렉터리

학습 및 검증 과정
데이터셋 준비: 위 구조에 맞게 학습 및 검증용 이미지와 라벨을 준비
모델 학습: 준비된 데이터셋을 사용하여 모델을 학습
모델 검증: 검증용 데이터셋을 사용하여 학습된 모델의 성능을 평가
가중치 저장: 학습이 완료되면, 모델의 가중치를 지정된 디렉터리에 저장
best.pt: 최고의 성능을 가진 모델
epoch_10.pt 및 epoch_20.pt: 각 에포크에서의 모델 상태를 저장
이 과정을 통해 효과적인 객체 탐지 모델을 개발

파일 내용 정리
bestptListTest.py: 최상의 모델 가중치를 테스트하는 리스트 스크립트
bestptTest.py: 최상의 모델 가중치를 사용하여 테스트하는 스크립트
coco.yaml: COCO 데이터셋 구성 파일
confusionMatrixTest.py: 혼동 행렬을 생성하여 모델 성능을 평가하는 스크립트
cudaAugmentationTrain.py: CUDA를 이용한 데이터 트
train.py: YOLO 모델 훈련 스크립트
yolo11n.pt: YOLOv8 모델 가중치 파일
yolo8n-p2.yaml: YOLOv8 모델 구성 파일
yolo_pretrained_inference.py: 사전 훈련된 YOLO 모델로 추론하는 스크립트
yolov8-p2.yaml: YOLOv8 p2 모델 구성 파일
yolov8n.pt: YOLOv8n 모델 가중치 파일
yolov8s.pt: YOLOv8s 모델 가중치 파일
