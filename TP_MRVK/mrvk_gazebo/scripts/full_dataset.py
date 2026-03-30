import os
import shutil

# Твои пути (включая тот, что ты добавил третьим)
sources = [
    ("/home/yehor/work_disk/downloads/test_dataset_2/raw_dataset_1/train", "/home/yehor/work_disk/downloads/test_dataset_2/dataset_1"),
    ("/home/yehor/work_disk/downloads/test_dataset_2/raw_dataset_2/train", "/home/yehor/work_disk/downloads/test_dataset_2/dataset_2"),
    ("/home/yehor/work_disk/downloads/test_dataset_2/raw_dataset_3/train", "/home/yehor/work_disk/downloads/test_dataset_2/dataset_3"),
    ("/home/yehor/work_disk/downloads/test_dataset_2/raw_dataset_4/train", "/home/yehor/work_disk/downloads/test_dataset_2/dataset_4"),
    ("/home/yehor/work_disk/catkin_ws/src/TP_MRVK/dataset/2", "/home/yehor/work_disk/catkin_ws/src/TP_MRVK/dataset/2"),
]

target_dir = "/home/yehor/work_disk/dataset_final"
os.makedirs(f"{target_dir}/images", exist_ok=True)
os.makedirs(f"{target_dir}/masks", exist_ok=True)

counter = 1

for img_dir, mask_dir_path in sources:
    # Проверяем, есть ли папка 'images' внутри или надо искать прямо в папке
    img_folder = os.path.join(img_dir, "images") if os.path.exists(os.path.join(img_dir, "images")) else img_dir
    mask_folder = os.path.join(mask_dir_path, "masks") if os.path.exists(os.path.join(mask_dir_path, "masks")) else mask_dir_path
    
    # Ищем фото (jpg, jpeg, png)
    all_files = [f for f in os.listdir(img_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for f in sorted(all_files):
        name, ext = os.path.splitext(f)
        
        # Ищем маску (png, jpg, jpeg) с таким же именем
        found_mask = None
        for mask_ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG']:
            if os.path.exists(os.path.join(mask_folder, name + mask_ext)):
                found_mask = os.path.join(mask_folder, name + mask_ext)
                break
        
        if found_mask:
            new_name = f"{counter:05d}"
            # Копируем фото и маску
            shutil.copy(os.path.join(img_folder, f), f"{target_dir}/images/{new_name}{ext.lower()}")
            shutil.copy(found_mask, f"{target_dir}/masks/{new_name}.png")
            
            print(f"[{counter:04d}] Обработано: {f} -> {new_name}.png")
            counter += 1
        else:
            print(f"!!! Маска для {f} НЕ НАЙДЕНА в папке {mask_folder}")

print(f"Готово! Всего собрано пар: {counter - 1}")