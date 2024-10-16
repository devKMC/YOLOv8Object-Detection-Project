import os
from ultralytics import YOLO

model = YOLO('runs/detect/train7/weights/best.pt')

root = os.getcwd()
image_folder = 'imageTest'
image_dir = os.path.join(root, image_folder)
output = os.path.join(image_dir, 'output')
if not os.path.exists(output):
    os.mkdir(output)
imageFileList = os.listdir(image_dir)

for imageFile in imageFileList:
    image_path = os.path.join(image_dir, imageFile)
    print(image_path)
    results = model.predict(source=image_path, iou=0, conf=0.02, device='0', half=True)
    results[0].save(os.path.join(output, imageFile), labels=False, line_width= 2 )




