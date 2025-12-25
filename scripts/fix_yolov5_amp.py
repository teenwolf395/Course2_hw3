#!/usr/bin/env python3
"""
Исправление устаревшего AMP API в YOLOv5 для PyTorch 2.4+
"""

import os
import sys
from pathlib import Path

def fix_amp_in_yolov5():
    """Исправляет устаревший код AMP в YOLOv5"""
    
    print("Исправление AMP API в YOLOv5...")
    print("=" * 50)
    
    # Проверяем наличие YOLOv5
    yolov5_dir = Path("yolov5")
    if not yolov5_dir.exists():
        print("Папка yolov5 не найдена!")
        return False
    
    # 1. Исправляем train.py
    train_py = yolov5_dir / "train.py"
    if train_py.exists():
        print(f"Исправляю {train_py}...")
        
        with open(train_py, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Заменяем устаревший autocast
        old_line = "    with torch.cuda.amp.autocast(amp):"
        new_line = "    with torch.amp.autocast('cuda', enabled=amp):"
        
        if old_line in content:
            content = content.replace(old_line, new_line)
            print("Исправлен torch.cuda.amp.autocast в train.py")
        else:
            # Пробуем другой вариант
            old_line2 = "with torch.cuda.amp.autocast(amp):"
            new_line2 = "with torch.amp.autocast('cuda', enabled=amp):"
            if old_line2 in content:
                content = content.replace(old_line2, new_line2)
                print("Исправлен torch.cuda.amp.autocast в train.py (вариант 2)")
        
        # Сохраняем изменения
        with open(train_py, 'w', encoding='utf-8') as f:
            f.write(content)
    else:
        print(f"Файл не найден: {train_py}")
    
    # 2. Исправляем utils/amp.py (если есть)
    amp_py = yolov5_dir / "utils" / "amp.py"
    if amp_py.exists():
        print(f"\nИсправляю {amp_py}...")
        
        with open(amp_py, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Заменяем GradScaler
        if "torch.cuda.amp.GradScaler()" in content:
            content = content.replace("torch.cuda.amp.GradScaler()", "torch.amp.GradScaler('cuda')")
            print("Исправлен torch.cuda.amp.GradScaler в amp.py")
        
        # Сохраняем изменения
        with open(amp_py, 'w', encoding='utf-8') as f:
            f.write(content)
    
    # 3. Проверяем другие возможные файлы
    print("\nПроверяю другие файлы...")
    files_to_check = [
        yolov5_dir / "utils" / "torch_utils.py",
        yolov5_dir / "utils" / "general.py",
        yolov5_dir / "models" / "common.py",
    ]
    
    for file_path in files_to_check:
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Ищем и заменяем устаревший autocast
            if "torch.cuda.amp.autocast(" in content:
                content = content.replace("torch.cuda.amp.autocast(", "torch.amp.autocast('cuda', ")
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"Исправлен autocast в {file_path.name}")
    
    print("\n" + "=" * 50)
    print("Исправления применены!")
    return True

def create_backup():
    """Создает резервную копию файлов перед изменением"""
    
    print("Создание резервных копий...")
    backup_dir = Path("yolov5_backup")
    backup_dir.mkdir(exist_ok=True)
    
    import shutil
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"yolov5_backup_{timestamp}"
    
    # Копируем весь yolov5
    shutil.copytree("yolov5", backup_path)
    
    print(f"Резервная копия создана: {backup_path}")
    return backup_path

if __name__ == "__main__":
    print("=" * 60)
    print("ИСПРАВЛЕНИЕ YOLOv5 ДЛЯ НОВОЙ ВЕРСИИ PyTorch")
    print("=" * 60)
    
    # Создаем backup
    backup = create_backup()
    
    # Применяем исправления
    fix_amp_in_yolov5()