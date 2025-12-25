"""
Скачивание YOLOv5
"""

import os
import subprocess
import sys
from pathlib import Path

print("=" * 60)
print("Скачивание YOLOv5")
print("=" * 60)
print()

# Проверяем виртуальное окружение
print("Проверяю виртуальное окружение...")
if sys.prefix == sys.base_prefix:
    print("ВНИМАНИЕ: Виртуальное окружение не активировано!")
    print("Активируйте его командой:")
    print("  Windows: venv\\Scripts\\activate")
    print("  Linux/Mac: source venv/bin/activate")
    print()
else:
    print("Вирутальное окружение активировано")
    print()

# Скачиваем YOLOv5
print("Скачиваю YOLOv5...")

# Проверяем, есть ли уже YOLOv5
yolo_dir = Path("yolov5")
if yolo_dir.exists():
    print("YOLOv5 уже скачан")
else:
    try:
        # Пробуем скачать через git
        print("Клонирую репозиторий...")
        subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5.git"], 
                      check=True, capture_output=True)
        print("YOLOv5 успешно скачан")
    except:
        print("Не удалось скачать через git")
        print("Скачайте вручную: https://github.com/ultralytics/yolov5")
        print("И разархивируйте в папку 'yolov5'")

# Устанавливаем зависимости YOLOv5
print()
print("Устанавливаю зависимости YOLOv5...")
requirements_file = yolo_dir / "requirements.txt"

if requirements_file.exists():
    try:
        print("Запускаю установку зависимостей (это может занять несколько минут)...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)], 
            check=True, 
            capture_output=True,
            text=True
        )
        print("Зависимости YOLOv5 установлены")
    except subprocess.CalledProcessError as e:
        print("Ошибка при установке зависимостей YOLOv5:")
        print(e.stderr[-500:] if e.stderr else "Неизвестная ошибка")
        print("\nПопробуйте установить вручную:")
        print(f"  {sys.executable} -m pip install -r {requirements_file}")
else:
    print("Файл requirements.txt не найден")

# Проверяем ключевые зависимости
print("\nПроверяю ключевые зависимости...")
key_packages = ['numpy', 'torch', 'torchvision', 'opencv-python', 'Pillow']
missing = []

for package in key_packages:
    try:
        __import__(package.replace('-', '_'))
        print(f"  {package}")
    except ImportError:
        print(f"  {package} - не установлен")
        missing.append(package)

if missing:
    print(f"\nОтсутствуют пакеты: {', '.join(missing)}")
    print("Установите их командой:")
    print(f"  {sys.executable} -m pip install {' '.join(missing)}")

print()
print("=" * 60)
print("Готово!")
print("=" * 60)