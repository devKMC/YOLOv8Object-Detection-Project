

├─ dataset
│  ├─ train
│  │  ├─ images       # 학습용 이미지
│  │  └─ labels       # 학습용 라벨
│  └─ val
│      ├─ images      # 검증용 이미지
│      └─ labels      # 검증용 라벨
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
