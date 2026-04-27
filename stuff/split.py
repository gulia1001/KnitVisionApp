import os
import shutil
import glob
import random

# --- Настройки ---
SOURCE_DIR = "knitting_dataset" # Исходная папка с данными
TEST_DIR = "test_dataset"      # Папка для тестов (30%)
CLASSES = ["stockinette", "garter", "ribbing", "lace"]
SPLIT_RATIO = 0.9              # 70% / 30%

print("Начинаю разделение датасета...")

# 1. Создаем структуру новых папок
if not os.path.exists(TEST_DIR):
    os.makedirs(TEST_DIR)
for label in CLASSES:
    class_path = os.path.join(TEST_DIR, label)
    if not os.path.exists(class_path):
        os.makedirs(class_path)

# 2. Распределяем и копируем файлы
for label in CLASSES:
    # Ищем все файлы в исходной папке
    files = glob.glob(os.path.join(SOURCE_DIR, label, "*.*"))
    
    if not files:
        print(f"ВНИМАНИЕ: В исходной папке '{SOURCE_DIR}/{label}' не найдено файлов!")
        continue
        
    # Перемешиваем файлы случайным образом, чтобы выборка была честной
    random.seed(42) # Фиксируем seed, чтобы при повторном запуске разделение было таким же
    random.shuffle(files)
    
    # Считаем, где резать список
    split_idx = int(len(files) * SPLIT_RATIO)
    
    test_files = files[split_idx:]
    
   
    # Копируем в Test
    for f in test_files:
        shutil.copy(f, os.path.join(TEST_DIR, label, os.path.basename(f)))
        
    print(f"[{label}] Всего фото: {len(files)} | В Train скопировано: | В Test скопировано: {len(test_files)}")

print(f"\nУспешно! Созданы папки '{TEST_DIR}'.")