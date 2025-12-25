"""
Создание структуры проекта для YOLOv5
"""

import os
from pathlib import Path

print("=" * 60)
print("Создание структуры проекта для YOLOv5")
print("=" * 60)
print()

# Создаем папки
folders = [
    "datasets",
    "datasets/images/train",
    "datasets/images/val",
    "datasets/images/test",
    "datasets/labels/train", 
    "datasets/labels/val",
    "datasets/labels/test",
    "models",
    "test_videos",
    "results",
    "scripts"
]

print("Создаю папки...")
for folder in folders:
    folder_path = Path(folder)
    folder_path.mkdir(parents=True, exist_ok=True)
    print(f"  Создано: {folder}")

# Создаем README файл
readme_content = """# Проект YOLOv5 для распознавания человека

## Структура папок:
- datasets/ - данные для обучения
  - images/ - фотографии
    - train/ - для обучения (80%)
    - val/ - для проверки (20%)
    - test/ - для тестирования
  - labels/ - разметка (файлы .txt)
- models/ - сохраненные модели
- test_videos/ - видео для тестирования (как развития проекта)
- results/ - результаты работы
- scripts/ - наши скрипты
"""

with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme_content)
print("  Создано: README.md")

print()
print("=" * 60)
print("Структура проекта создана!")
print("=" * 60)
print()
print("Теперь соберите фотографии человека и разместите их:")
print("  Фотографии: datasets/images/train/ и datasets/images/val/")
print("  Разметка:   datasets/labels/train/ и datasets/labels/val/")