import cv2
import numpy as np
import os
import glob
from sklearn.ensemble import RandomForestClassifier
import joblib

def extract_features(img):
    h, w = img.shape[:2]
    # Добавляем защиту от слишком маленьких картинок
    if h == 0 or w == 0: return None
    
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    texture = cv2.magnitude(sobel_x, sobel_y)
    texture = cv2.GaussianBlur(texture, (5, 5), 0)
    
    y_coord = np.tile(np.linspace(0, 1, h), (w, 1)).T
    feature_map = np.dstack([lab, hsv, texture, y_coord])
    return feature_map.reshape(-1, feature_map.shape[2])

def train():
    dataset_dir = "/home/yehor/work_disk/dataset_final"
    
    # Ищем файлы с ЛЮБЫМ расширением (jpg, png, JPG, PNG)
    # Используем glob с флагами или просто собираем все файлы и фильтруем
    def get_files(path, extensions):
        all_files = []
        for ext in extensions:
            all_files.extend(glob.glob(os.path.join(path, f"*.{ext}")))
            all_files.extend(glob.glob(os.path.join(path, f"*.{ext.upper()}")))
        return sorted(list(set(all_files)))

    img_paths = get_files(f"{dataset_dir}/images", ['jpg', 'jpeg', 'png'])
    mask_paths = get_files(f"{dataset_dir}/masks", ['png', 'jpg', 'jpeg'])
    
    print(f"Найдено картинок: {len(img_paths)}, масок: {len(mask_paths)}")
    
    if len(img_paths) == 0:
        print("!!! ОШИБКА: Картинки не найдены!")
        return
        
    X_train = []
    y_train = []
    
    # Поскольку мы переименовали их в 00001.png, 00002.png..., 
    # zip(img_paths, mask_paths) гарантированно даст пару (00001.jpg, 00001.png)
    for img_p, mask_p in zip(img_paths, mask_paths):
        img = cv2.imread(img_p)
        mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        
        if img is None or mask is None:
            print(f"Не удалось прочитать: {img_p}")
            continue
            
        # Ресайз до 160х120 для скорости
        img = cv2.resize(img, (160, 120))
        mask = cv2.resize(mask, (160, 120))
        
        features = extract_features(img)
        if features is None: continue
            
        labels = mask.reshape(-1)
        
        # Берем 10% пикселей для обучения (чтобы не переполнить RAM)
        indices = np.random.choice(len(labels), size=int(len(labels)*0.1), replace=False)
        
        X_train.append(features[indices])
        y_train.append(labels[indices])
    
    if not X_train:
        print("Данных для обучения не найдено!")
        return

    X_train = np.vstack(X_train)
    y_train = np.hstack(y_train)
    # 127 - это порог: все что выше (белое) - класс 1, ниже (черное) - класс 0
    y_train = np.where(y_train > 127, 1, 0)
    
    print(f"Обучение на {X_train.shape[0]} пикселях...")
    rf = RandomForestClassifier(n_estimators=50, max_depth=15, n_jobs=-1, class_weight='balanced')
    rf.fit(X_train, y_train)
    
    save_path = "/home/yehor/work_disk/catkin_ws/src/TP_MRVK/path_detector/src/road_detector_rf.pkl"
    joblib.dump(rf, save_path)
    print(f"Успех! Модель сохранена в: {save_path}")

if __name__ == "__main__":
    train()