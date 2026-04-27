import os
import requests
import time

# --- ТВОИ НАСТРОЙКИ ---
ACCESS_KEY = 'read-57e587e67e4637d3db18dc96d94b1c9f'
PERSONAL_KEY = 'IUQ/5EKO3dnrac8eTRfgZtAfDIBSvxoDFJ/WaH3C'

CLASSES = ['garter', 'stockinette', 'ribbing', 'seed']
MAX_IMAGES_PER_CLASS = 200 # Максимальное количество изображений для каждого класса
BASE_FOLDER = 'raw_dataset'

def get_best_photo_url(photo):
    """
    Возвращает URL фото лучшего доступного качества.
    У Ravelry 'medium2_url' обычно большего размера (500px), чем 'medium_url'.
    """
    if 'medium2_url' in photo and photo['medium2_url']:
        return photo['medium2_url']
    elif 'medium_url' in photo and photo['medium_url']:
        return photo['medium_url']
    return None

def download_ravelry_images():
    # Проходимся по каждому классу из списка
    for knit_class in CLASSES:
        query = f"{knit_class} stitch "
        save_folder = os.path.join(BASE_FOLDER, knit_class)
        
        # Создаем папку для конкретного класса
        os.makedirs(save_folder, exist_ok=True)
        
        print(f"\n{'='*40}")
        print(f"Начинаем сбор данных для класса: '{knit_class.upper()}'")
        print(f"Поисковый запрос: '{query}'")
        print(f"{'='*40}")
        
        saved_count = 0
        page = 1
        
        # Цикл работает, пока не скачаем ровно MAX_IMAGES_PER_CLASS
        while saved_count < MAX_IMAGES_PER_CLASS:
            search_url = "https://api.ravelry.com/patterns/search.json"
            
            params = {
                'query': query,
                'page': page,
                'page_size': 50 # Запрашиваем по 50 результатов поиска
            }
            
            search_response = requests.get(search_url, params=params, auth=(ACCESS_KEY, PERSONAL_KEY))
            
            if search_response.status_code != 200:
                print(f"Ошибка доступа при поиске! Код: {search_response.status_code}.")
                break 

            data = search_response.json()
            patterns = data.get('patterns', [])
            
            if not patterns:
                print(f"Больше нет результатов для '{query}'. Собрано {saved_count}/{MAX_IMAGES_PER_CLASS}.")
                break
                
            for pattern in patterns:
                # Глобальная проверка лимита перед каждым паттерном
                if saved_count >= MAX_IMAGES_PER_CLASS:
                    break
                    
                pattern_id = pattern.get('id')
                
                # Запрашиваем детали конкретного паттерна, чтобы вытащить ВСЕ его фото
                details_url = f"https://api.ravelry.com/patterns/{pattern_id}.json"
                details_response = requests.get(details_url, auth=(ACCESS_KEY, PERSONAL_KEY))
                
                if details_response.status_code != 200:
                    print(f"Пропуск паттерна {pattern_id}: не удалось получить детали.")
                    continue
                    
                pattern_details = details_response.json().get('pattern', {})
                photos = pattern_details.get('photos', [])
                
                # Проходимся по всем фото внутри одного паттерна
                for photo in photos:
                    # Строгая проверка: если уже набрали 200, прерываем скачивание фото
                    if saved_count >= MAX_IMAGES_PER_CLASS:
                        break
                        
                    img_url = get_best_photo_url(photo)
                    
                    if img_url:
                        try:
                            img_data = requests.get(img_url).content
                            
                            # Генерируем уникальное имя файла (id паттерна + id самого фото)
                            photo_id = photo.get('id', 'no_id')
                            filename = os.path.join(save_folder, f"{knit_class}_{pattern_id}_{photo_id}.jpg")
                            
                            # Сохраняем на диск
                            with open(filename, 'wb') as handler:
                                handler.write(img_data)
                                
                            saved_count += 1
                            print(f"[{saved_count}/{MAX_IMAGES_PER_CLASS}] Сохранено: {filename}")
                            
                        except Exception as e:
                            print(f"Не удалось скачать {img_url}: {e}")
                
                # Небольшая пауза, чтобы не перегружать API Ravelry запросами деталей
                time.sleep(0.3)
        
            # Переходим на следующую страницу поиска
            page += 1

    print("\n--- ВСЕ ГОТОВО! Загрузка по всем классам завершена. ---")

if __name__ == "__main__":
    download_ravelry_images()