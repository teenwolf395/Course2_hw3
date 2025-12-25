# Проверка загруженных данных
import os
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

print("Проверка загруженных данных...")
print("=" * 50)

# Считаем файлы
train_images = list(Path("datasets/images/train").glob("*.*"))
val_images = list(Path("datasets/images/val").glob("*.*"))
train_labels = list(Path("datasets/labels/train").glob("*.txt"))
val_labels = list(Path("datasets/labels/val").glob("*.txt"))

print(f"Обучающие изображения: {len(train_images)}")
print(f"Валидационные изображения: {len(val_images)}")
print(f"Обучающие разметки: {len(train_labels)}")
print(f"Валидационные разметки: {len(val_labels)}")

# Проверим несколько изображений
if train_images:
    print("\nПримеры изображений:")
    fig, axes = plt.subplots(1, min(3, len(train_images)), figsize=(15, 5))
    
    # Исправлено: если только одно изображение, axes не будет списком
    if min(3, len(train_images)) == 1:
        axes = [axes]
    
    for i in range(min(3, len(train_images))):
        img_path = train_images[i]
        img = cv2.imread(str(img_path))
        
        # Исправлено: проверка на None
        if img is None:
            print(f"Не удалось загрузить: {img_path}")
            continue
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Показываем метки если есть
        label_path = Path("datasets/labels/train") / f"{img_path.stem}.txt"
        if label_path.exists():
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            h, w = img.shape[:2]
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    _, x_center, y_center, box_w, box_h = map(float, parts)
                    
                    # Конвертируем в пиксели
                    x_center *= w
                    y_center *= h
                    box_w *= w
                    box_h *= h
                    
                    # Рисуем прямоугольник
                    x1 = int(x_center - box_w/2)
                    y1 = int(y_center - box_h/2)
                    x2 = int(x_center + box_w/2)
                    y2 = int(y_center + box_h/2)
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        axes[i].imshow(img)
        axes[i].set_title(f"{img_path.name}")
        axes[i].axis('off')
    
    plt.show()
else:
    print("Изображений не найдено!")


# Создание/проверка конфигурационного файла
import yaml

print("Проверка конфигурационного файла...")
print("=" * 50)

# Проверяем, есть ли data.yaml от Roboflow
roboflow_yaml = Path("datasets/data.yaml")
if roboflow_yaml.exists():
    print("data.yaml от Roboflow найден")
    
    with open(roboflow_yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    print("Содержимое data.yaml:")
    print(f"  Классы: {config.get('nc', 'не указано')}")
    print(f"  Имена классов: {config.get('names', 'не указаны')}")
    print(f"  Путь к train: {config.get('train', 'не указан')}")
    print(f"  Путь к val: {config.get('val', 'не указан')}")
    
    # Проверяем пути
    train_path = config.get('train', '')
    if not Path(train_path).exists() and train_path:
        # Пробуем исправить путь
        config['train'] = 'images/train'
        config['val'] = 'images/val'
        config['path'] = str(Path.cwd() / 'datasets')
        
        with open(roboflow_yaml, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print("Пути исправлены")
    
else:
    print("Создаю data.yaml...")
    
    # Создаём простой конфиг
    config = {
        'path': str(Path.cwd() / 'datasets'),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,  # Один класс - ваш человек
        'names': ['your_person']  # Название класса
    }
    
    with open(roboflow_yaml, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(" data.yaml создан")
    print(f" Классы: {config['nc']}")
    print(f" Имя класса: {config['names'][0]}")