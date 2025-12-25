# Обучение модели YOLOv5 с оптимизированными параметрами
# БЕЗ mosaic - для точного распознавания конкретного человека
import subprocess
import time
import os
import sys
from pathlib import Path

print("Начинаю обучение модели...")
print("=" * 60)
print("Оптимизировано для CPU 16GB")
print("БЕЗ mosaic - для точного распознавания")
print("=" * 60)

# Проверяем наличие данных и yolov5
yolov5_dir = Path("yolov5")
if not yolov5_dir.exists():
    print("Папка yolov5 не найдена! Сначала запустите scripts/download_yolo.py")
    exit(1)

# Команда для обучения с улучшенными параметрами
# Используем sys.executable для гарантии использования правильного Python
# Параметры аугментации задаются через файл hyp.yaml
train_cmd = [
    sys.executable, "train.py",
    "--img", "640",
    "--batch", "8",              # УВЕЛИЧЕНО: было 4, теперь 8 (для 16GB RAM)
    "--epochs", "200",           # УВЕЛИЧЕНО: было 150, теперь 200
    "--data", "../datasets/data.yaml",
    "--weights", "yolov5m.pt",   # ИЗМЕНЕНО: yolov5s -> yolov5m (лучше точность)
    "--project", "../results",
    "--name", "training_run",
    "--exist-ok",
    "--workers", "0",
    "--device", "cpu",
    "--cache",                   # Кэширование для CPU
    "--patience", "50",          # УВЕЛИЧЕНО: было 10, теперь 50
    "--rect",                    # Прямоугольное обучение (быстрее)
    "--multi-scale",             # ВКЛЮЧЕНО: обучение на разных размерах (безопасно)
    "--hyp", "data/hyps/hyp.custom.yaml",  # Используем наш файл с параметрами (БЕЗ mosaic)
]

print("\nПараметры обучения:")
print(f"  Batch size: 8 (было 4)")
print(f"  Epochs: 200 (было 150)")
print(f"  Модель: yolov5m (было yolov5s) - лучше точность")
print(f"  Patience: 50 (было 10)")
print(f"  Multi-scale: ВКЛЮЧЕНО (безопасно)")
print(f"  Файл аугментаций: data/hyps/hyp.custom.yaml")
print(f"  Mosaic: ОТКЛЮЧЕНО (в hyp.custom.yaml)")
print(f"  Другие аугментации: умеренные (безопасные)")
print("\nНачинаю обучение...")

# Запускаем обучение
start_time = time.time()

# Исправлено для Windows: используем os.chdir вместо cd &&
original_dir = os.getcwd()
try:
    os.chdir("yolov5")
    result = subprocess.run(
        train_cmd,
        shell=False,  # Используем shell=False для лучшей кроссплатформенности
        capture_output=True,
        text=True
    )
finally:
    os.chdir(original_dir)

# Выводим последние строки лога
print("\nПоследние строки вывода:")
lines = result.stdout.split('\n')
for line in lines[-30:]:  # Последние 30 строк
    if line.strip():
        print(line)

if result.returncode == 0:
    elapsed = (time.time() - start_time) / 60
    print(f"\nОбучение завершено за {elapsed:.1f} минут")
    print(f"Модель сохранена: results/training_run/weights/best.pt")
else:
    print(f"\nОшибка при обучении:")
    print(result.stderr[-500:])  # Последние 500 символов ошибки