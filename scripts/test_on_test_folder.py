#!/usr/bin/env python3
"""
Тестирование модели на фото из папки datasets/images/test/
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    print("Тестирование модели YOLOv5")
    print("-" * 40)
    
    # Путь к модели
    model_paths = [
        "results/training_run/weights/best.pt",
        "models/best.pt",
        "results/train/exp/weights/best.pt"
    ]
    
    model = None
    for path in model_paths:
        if Path(path).exists():
            model = path
            break
    
    if not model:
        print("Ошибка: модель не найдена!")
        print("Сначала обучите модель или проверьте путь.")
        return
    
    print(f"Модель: {model}")
    
    # Папка с тестовыми фото
    test_folder = Path("datasets/images/test")
    
    if not test_folder.exists():
        print(f"Создаю папку: {test_folder}")
        test_folder.mkdir(parents=True, exist_ok=True)
        print(f"Добавьте фото в {test_folder} и запустите снова")
        return
    
    test_images = list(test_folder.glob("*.*"))
    
    if not test_images:
        print(f"Папка {test_folder} пуста")
        print(f"Добавьте фото в эту папку и запустите снова")
        return
    
    print(f"Найдено фото для теста: {len(test_images)}")
    
    # Команда для тестирования (исправлено для Windows)
    # Используем sys.executable для гарантии использования правильного Python
    cmd = [
        sys.executable, "detect.py",
        "--weights", f"../{model}",
        "--source", "../datasets/images/test",
        "--conf", "0.5",
        "--project", "../results",
        "--name", "final_test",
        "--exist-ok"
    ]
    
    print("\nЗапускаю тестирование...")
    print(f"Команда: {' '.join(cmd)}")
    
    # Переходим в папку yolov5 и запускаем
    original_dir = os.getcwd()
    yolov5_dir = Path("yolov5")
    
    if not yolov5_dir.exists():
        print("Папка yolov5 не найдена!")
        return
    
    try:
        os.chdir("yolov5")
        result = subprocess.run(cmd, shell=False)
    finally:
        os.chdir(original_dir)
    
    if result.returncode == 0:
        print("\nТестирование завершено!")
        print(f"Результаты сохранены в: results/final_test/")
        
        # Показываем список обработанных файлов
        results_dir = Path("results/final_test")
        if results_dir.exists():
            processed = list(results_dir.glob("*.*"))
            if processed:
                print(f"\nОбработанные файлы ({len(processed)}):")
                for file in processed[:10]:  # первые 10
                    print(f"  - {file.name}")
                if len(processed) > 10:
                    print(f"  ... и ещё {len(processed) - 10} файлов")
    else:
        print("\nОшибка при тестировании")

if __name__ == "__main__":
    main()