import os
import random
import uuid
from PIL import Image, ImageEnhance, ImageOps

def fill_dataset_to_300(base_dir="cleaned_dataset"):
    """
    Пайплайн для добивки датасета до 300 штук путем аугментации.
    Изменяет: Размер, Цвет, Поворот.
    """
    if not os.path.exists(base_dir):
        print(f"[!] Ошибка: Папка {base_dir} не найдена.")
        return

    for class_name in os.listdir(base_dir):
        class_path = os.path.join(base_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        # Список уже существующих чистых файлов
        existing_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        current_count = len(existing_files)

        if current_count >= 300:
            print(f"[=] Класс {class_name} уже заполнен ({current_count} фото). Пропуск.")
            continue

        needed = 300 - current_count
        print(f"[*] Класс {class_name}: найдено {current_count}. Генерируем еще {needed} фото...")

        for i in range(needed):
            # Берем случайное фото из имеющихся чистых для основы
            source_img_name = random.choice(existing_files)
            source_path = os.path.join(class_path, source_img_name)

            try:
                with Image.open(source_path) as img:
                    if img.mode in ("RGBA", "P"):
                        img = img.convert("RGB")

                    # --- 1. СЛУЧАЙНЫЙ ПОВОРОТ (15-20 градусов) ---
                    angle = random.uniform(15, 20) * random.choice([-1, 1])
                    img = img.rotate(angle, expand=True, resample=Image.BICUBIC)

                    # --- 2. СЛУЧАЙНЫЙ ЦВЕТ И ЯРКОСТЬ ---
                    # Яркость (от 0.7 до 1.3)
                    enhancer = ImageEnhance.Brightness(img)
                    img = enhancer.enhance(random.uniform(0.7, 1.3))
                    
                    # Цвет/Насыщенность (от 0.5 до 1.5)
                    enhancer = ImageEnhance.Color(img)
                    img = enhancer.enhance(random.uniform(0.5, 1.5))
                    
                    # Контраст
                    enhancer = ImageEnhance.Contrast(img)
                    img = enhancer.enhance(random.uniform(0.8, 1.2))

                    # --- 3. СЛУЧАЙНЫЙ РАЗМЕР (Resize) ---
                    # Меняем размер в диапазоне от -20% до +20% от оригинала
                    w, h = img.size
                    scale = random.uniform(0.8, 1.2)
                    new_size = (int(w * scale), int(h * scale))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)

                    # --- 4. СОХРАНЕНИЕ ---
                    new_name = f"aug_{uuid.uuid4().hex[:8]}.jpg"
                    img.save(os.path.join(class_path, new_name), "JPEG", quality=85)

            except Exception as e:
                print(f"[!] Ошибка при аугментации {source_img_name}: {e}")

        print(f"[+] Класс {class_name} успешно дополнен до 300.")

if __name__ == "__main__":
    fill_dataset_to_300("cleaned_dataset")