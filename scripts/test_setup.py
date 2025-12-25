"""
Проверка установки
"""

import sys
import torch
import cv2
import numpy as np

print("=" * 60)
print("Проверка установки")
print("=" * 60)
print()

# 1. Python версия
print("1. Python версия:")
print(f"   {sys.version}")
print()

# 2. Проверяем PyTorch
print("2. PyTorch:")
try:
    print(f"   Версия: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"   GPU доступна: {torch.cuda.get_device_name(0)}")
    else:
        print("   GPU не найдена, будет использоваться CPU")
except Exception as e:
    print(f"   Ошибка: {e}")
print()

# 3. Проверяем OpenCV
print("3. OpenCV:")
try:
    print(f"   Версия: {cv2.__version__}")
except Exception as e:
    print(f"   Ошибка: {e}")
print()

# 4. Создаем тестовое изображение
print("4. Создание тестового изображения:")
try:
    # Создаем простое изображение
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Рисуем прямоугольник
    cv2.rectangle(img, (100, 100), (300, 300), (0, 0, 255), 2)
    cv2.putText(img, "Test Image", (120, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Сохраняем
    cv2.imwrite("test_image.jpg", img)
    print("   Изображение создано: test_image.jpg")
    print(f"   Размер: 640x480 пикселей")
    
except Exception as e:
    print(f"   Ошибка: {e}")

print()
print("=" * 60)
print("Проверка завершена!")
print("=" * 60)

if torch.cuda.is_available():
    print("Отлично! Все установлено, GPU доступна.")
else:
    print("Все установлено, но GPU не найдена.")
    print("Обучение будет медленнее на CPU.")