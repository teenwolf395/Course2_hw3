from pathlib import Path
import pandas as pd
import shutil
import numpy as np

# Пробуем импортировать plotext для графиков в консоли
try:
    import plotext as plt
    HAS_PLOTEXT = True
except ImportError:
    # Если plotext не установлен, используем matplotlib
    import matplotlib.pyplot as plt
    HAS_PLOTEXT = False

def check_training_progress(results_dir_name="training_run"):
    """Выводит итоговый отчет о завершенном обучении"""
    
    results_dir = Path("results") / results_dir_name
    
    if not results_dir.exists():
        print(f"Результаты обучения не найдены в: {results_dir}")
        print("   Убедитесь, что обучение было запущено")
        return
    
    print("=" * 70)
    print(" " * 20 + "ИТОГОВЫЙ ОТЧЕТ ОБ ОБУЧЕНИИ")
    print("=" * 70)
    print(f"Папка результатов: {results_dir_name}")
    
    # Ищем файлы с метриками
    results_files = list(results_dir.glob("results*.csv"))
    
    if not results_files:
        print("\nФайлы с результатами не найдены")
        print("   Обучение может не завершиться")
        return
    
    try:
        # Читаем последний файл результатов
        latest_file = sorted(results_files)[-1]
        df = pd.read_csv(latest_file)
        
        # Убираем пробелы из имен колонок (YOLOv5 добавляет пробелы для форматирования)
        df.columns = df.columns.str.strip()
        
        # Обрабатываем NaN значения
        df = df.fillna(0)
        
        print(f"\nФайл результатов: {latest_file.name}")
        print(f"Всего эпох: {len(df)}")
        
        if len(df) == 0:
            print("Файл пуст - обучение не завершилось")
            return
        
        # Функция для безопасного форматирования
        def format_metric(value, name):
            if pd.isna(value) or value == '':
                return f"  {name}: N/A"
            try:
                if isinstance(value, (int, float)) and not np.isnan(value):
                    return f"  {name}: {value:.3f}"
                else:
                    return f"  {name}: {value}"
            except:
                return f"  {name}: {value}"
        
        # ФИНАЛЬНЫЕ МЕТРИКИ (последняя эпоха)
        print("\n" + "=" * 70)
        print("ФИНАЛЬНЫЕ МЕТРИКИ (последняя эпоха):")
        print("=" * 70)
        
        last_row = df.iloc[-1]
        final_precision = last_row.get('metrics/precision', None)
        final_recall = last_row.get('metrics/recall', None)
        final_map = last_row.get('metrics/mAP_0.5', None)
        final_map_95 = last_row.get('metrics/mAP_0.5:0.95', None)
        
        print(format_metric(final_precision, "Точность (precision)"))
        print(format_metric(final_recall, "Полнота (recall)"))
        print(format_metric(final_map, "mAP@0.5"))
        if final_map_95 is not None:
            print(format_metric(final_map_95, "mAP@0.5:0.95"))
        
        # ЛУЧШИЕ МЕТРИКИ (за все обучение)
        print("\n" + "=" * 70)
        print("ЛУЧШИЕ МЕТРИКИ (за все обучение):")
        print("=" * 70)
        
        if 'metrics/precision' in df.columns:
            precision_data = df['metrics/precision'].dropna()
            if len(precision_data) > 0:
                best_precision = precision_data.max()
                best_precision_epoch = precision_data.idxmax() + 1
                print(f"  Точность (precision): {best_precision:.3f} (эпоха {best_precision_epoch})")
        
        if 'metrics/recall' in df.columns:
            recall_data = df['metrics/recall'].dropna()
            if len(recall_data) > 0:
                best_recall = recall_data.max()
                best_recall_epoch = recall_data.idxmax() + 1
                print(f"  Полнота (recall): {best_recall:.3f} (эпоха {best_recall_epoch})")
        
        if 'metrics/mAP_0.5' in df.columns:
            map_data = df['metrics/mAP_0.5'].dropna()
            if len(map_data) > 0:
                best_map = map_data.max()
                best_map_epoch = map_data.idxmax() + 1
                print(f"  mAP@0.5: {best_map:.3f} (эпоха {best_map_epoch})")
        
        if 'metrics/mAP_0.5:0.95' in df.columns:
            map_95_data = df['metrics/mAP_0.5:0.95'].dropna()
            if len(map_95_data) > 0:
                best_map_95 = map_95_data.max()
                best_map_95_epoch = map_95_data.idxmax() + 1
                print(f"  mAP@0.5:0.95: {best_map_95:.3f} (эпоха {best_map_95_epoch})")
        
        # СТАТИСТИКА ПО ОБУЧЕНИЮ
        if len(df) > 1:
            print("\n" + "=" * 70)
            print("СТАТИСТИКА ПО ОБУЧЕНИЮ:")
            print("=" * 70)
            
            if 'metrics/mAP_0.5' in df.columns:
                map_data = df['metrics/mAP_0.5'].dropna()
                if len(map_data) > 0:
                    print(f"  mAP@0.5:")
                    print(f"    Лучший: {map_data.max():.3f}")
                    print(f"    Худший: {map_data.min():.3f}")
                    print(f"    Средний: {map_data.mean():.3f}")
                    print(f"    Финальный: {map_data.iloc[-1]:.3f}")
            
            if 'train/box_loss' in df.columns:
                train_loss = df['train/box_loss'].dropna()
                if len(train_loss) > 0:
                    print(f"\n  Train Loss:")
                    print(f"    Начальный: {train_loss.iloc[0]:.3f}")
                    print(f"    Финальный: {train_loss.iloc[-1]:.3f}")
                    improvement = ((train_loss.iloc[0] - train_loss.iloc[-1]) / train_loss.iloc[0] * 100) if train_loss.iloc[0] > 0 else 0
                    print(f"    Улучшение: {improvement:.1f}%")
        
        # Графики (опционально)
        if len(df) > 1:
            try:
                if HAS_PLOTEXT:
                    # Используем plotext для графиков в консоли
                    print("\n" + "=" * 70)
                    print("ГРАФИКИ ОБУЧЕНИЯ:")
                    print("=" * 70)
                    
                    print("\nГрафик Precision/Recall:")
                    print("-" * 70)
                    
                    # График 1: Precision/Recall
                    plt.clear_data()
                    plt.clear_figure()
                    
                    epochs = list(range(len(df)))
                    has_data = False
                    
                    if 'metrics/precision' in df.columns:
                        data = df['metrics/precision'].dropna()
                        if len(data) > 0:
                            plt.plot(epochs[:len(data)], data.tolist(), label='Precision')
                            has_data = True
                    if 'metrics/recall' in df.columns:
                        data = df['metrics/recall'].dropna()
                        if len(data) > 0:
                            plt.plot(epochs[:len(data)], data.tolist(), label='Recall')
                            has_data = True
                    
                    if has_data:
                        plt.title("Precision/Recall по эпохам")
                        plt.xlabel("Epoch")
                        plt.ylabel("Score")
                        plt.plotsize(80, 20)
                        plt.show()
                    else:
                        print("  Нет данных для отображения")
                    
                    print("\nГрафик Loss:")
                    print("-" * 70)
                    
                    # График 2: Loss
                    plt.clear_data()
                    plt.clear_figure()
                    
                    has_data = False
                    
                    if 'train/box_loss' in df.columns:
                        data = df['train/box_loss'].dropna()
                        if len(data) > 0:
                            plt.plot(epochs[:len(data)], data.tolist(), label='Train Loss')
                            has_data = True
                    if 'val/box_loss' in df.columns:
                        data = df['val/box_loss'].dropna()
                        if len(data) > 0:
                            plt.plot(epochs[:len(data)], data.tolist(), label='Val Loss')
                            has_data = True
                    
                    if has_data:
                        plt.title("Loss по эпохам")
                        plt.xlabel("Epoch")
                        plt.ylabel("Loss")
                        plt.plotsize(80, 20)
                        plt.show()
                    else:
                        print("  Нет данных для отображения")
                else:
                    # Используем matplotlib (если plotext не установлен)
                    print("\n" + "=" * 70)
                    print("ГРАФИКИ ОБУЧЕНИЯ (matplotlib):")
                    print("=" * 70)
                    print("   (Для графиков в консоли установите: pip install plotext)")
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # График 1: Precision/Recall
                    has_data = False
                    if 'metrics/precision' in df.columns:
                        data = df['metrics/precision'].dropna()
                        if len(data) > 0:
                            ax1.plot(data, label='Precision', marker='o', markersize=3)
                            has_data = True
                    if 'metrics/recall' in df.columns:
                        data = df['metrics/recall'].dropna()
                        if len(data) > 0:
                            ax1.plot(data, label='Recall', marker='s', markersize=3)
                            has_data = True
                    ax1.set_title('Precision/Recall')
                    ax1.set_xlabel('Epoch')
                    ax1.set_ylabel('Score')
                    ax1.grid(True, alpha=0.3)
                    if has_data:
                        ax1.legend()
                    
                    # График 2: Loss
                    has_data = False
                    if 'train/box_loss' in df.columns:
                        data = df['train/box_loss'].dropna()
                        if len(data) > 0:
                            ax2.plot(data, label='Train Loss', marker='o', markersize=3)
                            has_data = True
                    if 'val/box_loss' in df.columns:
                        data = df['val/box_loss'].dropna()
                        if len(data) > 0:
                            ax2.plot(data, label='Val Loss', marker='s', markersize=3)
                            has_data = True
                    ax2.set_title('Loss')
                    ax2.set_xlabel('Epoch')
                    ax2.set_ylabel('Loss')
                    ax2.grid(True, alpha=0.3)
                    if has_data:
                        ax2.legend()
                    
                    plt.tight_layout()
                    plt.show()
            except Exception as e:
                print(f"Ошибка при построении графиков: {e}")
    
    except Exception as e:
        print(f"Ошибка при чтении файла результатов: {e}")
        return
    
    # Ищем лучшую модель
    best_model = results_dir / "weights" / "best.pt"
    if best_model.exists():
        try:
            size_mb = best_model.stat().st_size / (1024 * 1024)
            print("\n" + "=" * 70)
            print("СОХРАНЕННАЯ МОДЕЛЬ:")
            print("=" * 70)
            print(f"  Путь: {best_model}")
            print(f"  Размер: {size_mb:.1f} MB")
            
            # Копируем в папку models
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            target_model = models_dir / "best.pt"
            
            # Проверяем, нужно ли копировать
            should_copy = True
            if target_model.exists():
                if target_model.stat().st_size == best_model.stat().st_size:
                    print(f"  Уже скопирована в: {target_model}")
                    should_copy = False
            
            if should_copy:
                shutil.copy(best_model, target_model)
                print(f"  Скопирована в: {target_model}")
        except Exception as e:
            print(f"Ошибка при работе с моделью: {e}")
    else:
        print(f"\nЛучшая модель не найдена")
    
    print("\n" + "=" * 70)
    print(" " * 25 + "ОТЧЕТ ЗАВЕРШЕН")
    print("=" * 70)

if __name__ == "__main__":
    check_training_progress()