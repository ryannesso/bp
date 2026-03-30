import cv2
import numpy as np
import glob
import os

# Путь к скачанному датасету с Roboflow
DATASET_PATH = "/home/yehor/work_disk/downloads/test_dataset_2/raw_dataset_4" # Укажи свой путь

def yolo_to_mask(txt_path, img_shape):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    with open(txt_path, 'r') as f:
        for line in f:
            parts = list(map(float, line.split()))
            # Первая цифра - класс (обычно 0 для sidewalk)
            # Остальные - координаты x,y,x,y...
            coords = np.array(parts[1:]).reshape(-1, 2)
            pts = (coords * [img_shape[1], img_shape[0]]).astype(np.int32)
            cv2.fillPoly(mask, [pts], 255)
    return mask

# Основной цикл конвертации
images = glob.glob(f"{DATASET_PATH}/train/images/*.png")
for img_path in images:
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    # Путь к текстовому файлу с разметкой
    txt_path = img_path.replace("images", "labels").replace(".png", ".txt")
    
    if os.path.exists(txt_path):
        mask = yolo_to_mask(txt_path, img.shape)
        # Сохраняем в твою папку с масками
        cv2.imwrite(f"/home/yehor/work_disk/downloads/test_dataset_2/dataset_4/masks/{os.path.basename(img_path).replace('.png', '.png')}", mask)