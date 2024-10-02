import json
import os
import shutil

def convert_to_yolo_format(json_file, images_dir, labels_dir, outputImageDir):
    # JSON 파일 열기
    with open(json_file) as f:
        data = json.load(f)

    # 각 이미지에 대해 반복
    for item in data['images']:
        image_id = item['id']  # 이미지 ID
        filename = item['file_name']  # 이미지 파일 이름
        width = item['width']  # 이미지 너비
        height = item['height']  # 이미지 높이

        # 현재 이미지에 대한 주석 찾기
        annotations = [a for a in data['annotations'] if a['image_id'] == image_id]

        yolo_labels = []
        for ann in annotations:
            class_id = ann['category_id'] - 1  # 클래스 ID가 1부터 시작한다고 가정
            if class_id != 0:continue            
            x_center = (ann['bbox'][0] + ann['bbox'][2] / 2) / width  # x 중심 좌표
            y_center = (ann['bbox'][1] + ann['bbox'][3] / 2) / height  # y 중심 좌표
            bbox_width = ann['bbox'][2] / width  # 바운딩 박스 너비
            bbox_height = ann['bbox'][3] / height  # 바운딩 박스 높이

            # YOLO 형식으로 레이블 추가
            yolo_labels.append(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}")
        if yolo_labels == []:continue

        # 해당 .txt 파일에 레이블 저장
        label_file_path = os.path.join(labels_dir, filename.replace('.jpg', '.txt'))  # 필요 시 확장자 변경
        with open(label_file_path, 'w') as label_file:
            label_file.write('\n'.join(yolo_labels))  # 레이블을 줄바꿈으로 연결하여 저장

        # 원본 이미지 경로 및 복사할 경로 생성
        source_image_path = os.path.join(images_dir, filename) 
        image_file_path = os.path.join(outputImageDir, filename)

        # 이미지 복사
        shutil.copy(source_image_path, image_file_path)


# 사용 예시
convert_to_yolo_format('source/annotations/instances_train2017.json', 'source/images/train2017/', 'dataset/train/labels/', 'dataset/train/images/')
convert_to_yolo_format('source/annotations/instances_val2017.json', 'source/images/val2017/', 'dataset/val/labels/', 'dataset/val/images/')
