import os
import cv2
import numpy as np
import tkinter as tk
import joblib
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps
from skimage.feature import hog, local_binary_pattern

# --- НАСТРОЙКИ ---
CLASSES = ['garter', 'stockinette', 'ribbing', 'seed']
IMG_SIZE = 128

# Настройки Multi-scale LBP
LBP_PARAMS = [(1, 8), (3, 24), (5, 40)]

SCHEMES = {
    "stockinette": "Лицевая гладь:\nНечетные ряды — лицевые, четные — изнаночные.",
    "garter": "Платочная вязка:\nВсе ряды вяжутся лицевыми петлями.",
    "ribbing": "Резинка:\nЧередуйте 1 лиц., 1 изн. в каждом ряду.",
    "seed": "Жемчужный узор (Рис):\nЧередуйте 1 лиц., 1 изн., в следующем ряду смещайте узор."
}

# --- ПАЛИТРА (Dark Theme) ---
BG_COLOR = "#1E1E2E"
PANEL_COLOR = "#313244"
TEXT_COLOR = "#CDD6F4"
ACCENT_COLOR = "#89B4FA"
SUCCESS_COLOR = "#A6E3A1"
ERROR_COLOR = "#F38BA8"

# --- ЗАГРУЗКА МОДЕЛЕЙ ---
try:
    # Загружаем оптимизированную модель!
    svm_model = joblib.load(os.path.join("output", "best_svm_model.pkl"))
    scaler = joblib.load(os.path.join("output", "scaler.pkl"))
    model_loaded = True
except Exception as e:
    print(f"Ошибка загрузки моделей: {e}")
    model_loaded = False

# --- ФИЛЬТРЫ ГАБОРА ---
def build_gabor_filters():
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 4): 
        for lamda in [np.pi/4, np.pi/2, np.pi]:   
            kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5 * kern.sum()
            filters.append(kern)
    return filters

GABOR_FILTERS = build_gabor_filters()

# --- ПРЕПРОЦЕССИНГ ---
def center_crop_square(img):
    h, w = img.shape
    min_dim = min(h, w)
    start_x = (w // 2) - (min_dim // 2)
    start_y = (h // 2) - (min_dim // 2)
    return img[start_y:start_y+min_dim, start_x:start_x+min_dim]

def extract_features(img):
    """Мега-пайплайн: выдает ровно 8202 признака (HOG 8100 + LBP 78 + Gabor 24)"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    
    # 1. HOG
    features_hog = hog(img, orientations=9, pixels_per_cell=(8, 8), 
                       cells_per_block=(2, 2), block_norm='L2-Hys')
    
    # 2. Multi-scale LBP
    features_lbp = []
    for radius, points in LBP_PARAMS:
        lbp = local_binary_pattern(img, points, radius, method='uniform')
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        features_lbp.extend(hist)
        
    # 3. Gabor Filters
    features_gabor = []
    for kern in GABOR_FILTERS:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        features_gabor.append(fimg.mean())
        features_gabor.append(fimg.var())
        
    combined_features = np.hstack([features_hog, features_lbp, features_gabor])
    return combined_features

# --- ОСНОВНАЯ ЛОГИКА ---
def upload_and_predict():
    if not model_loaded:
        log_text.config(state=tk.NORMAL)
        log_text.insert(tk.END, "\n[ОШИБКА] Модели не найдены. Убедитесь, что best_svm_model.pkl и scaler.pkl лежат в 'output/'.")
        log_text.config(state=tk.DISABLED)
        return

    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not file_path:
        return
    
    # Отрисовка
    img_pil = Image.open(file_path)
    img_pil_padded = ImageOps.pad(img_pil, (250, 250), color=PANEL_COLOR) 
    
    img_tk = ImageTk.PhotoImage(img_pil_padded)
    image_label.configure(image=img_tk)
    image_label.image = img_tk
    
    # Чтение
    img_array = np.fromfile(file_path, np.uint8)
    img_orig = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    if img_orig is None: return

    orig_shape = img_orig.shape
    img_cropped = center_crop_square(img_orig)
    crop_shape = img_cropped.shape
    img_resized = cv2.resize(img_cropped, (IMG_SIZE, IMG_SIZE))
    
    # Извлечение и масштабирование
    features = extract_features(img_resized)
    features_scaled = scaler.transform([features])
    
    # Предсказание
    pred_idx = svm_model.predict(features_scaled)[0]
    predicted_class = CLASSES[pred_idx]
    
    # Обновление UI
    result_label.configure(text=f"УЗОР: {predicted_class.upper()}", fg=SUCCESS_COLOR)
    scheme_label.configure(text=f"{SCHEMES[predicted_class]}")
    
    # Журнал
    scripted_text = (
        f"--- СИСТЕМНЫЙ ЖУРНАЛ ИИ ---\n"
        f"[+] Файл: {file_path.split('/')[-1]} (Оригинал: {orig_shape[1]}x{orig_shape[0]})\n"
        f"[+] Size Correction: Центральный кроп {crop_shape[1]}x{crop_shape[0]}\n"
        f"[+] Препроцессинг: Resize({IMG_SIZE}x{IMG_SIZE}), CLAHE\n"
        f"[+] Экстракция признаков:\n"
        f"    - HOG (8100) + Multi-LBP (78) + Gabor (24)\n"
        f"    - Итоговый вектор: {len(features)} параметров\n"
        f"[+] Вердикт Scikit-Learn SVM:\n"
        f"    - ID Класса: {pred_idx}\n"
        f">>> ИТОГ: Текстура классифицирована как '{predicted_class}'."
    )
    
    log_text.config(state=tk.NORMAL)
    log_text.delete(1.0, tk.END)
    log_text.insert(tk.END, scripted_text)
    log_text.config(state=tk.DISABLED)

# --- ИНТЕРФЕЙС TKINTER ---
root = tk.Tk()
root.title("Knitting AI Vision (Scikit-Learn Edition)")
root.geometry("480x720")
root.configure(bg=BG_COLOR)
root.resizable(False, False)

title_label = tk.Label(root, text="Распознавание Узоров", font=("Segoe UI", 18, "bold"), bg=BG_COLOR, fg=TEXT_COLOR)
title_label.pack(pady=(20, 10))

image_frame = tk.Frame(root, bg=PANEL_COLOR, bd=0, highlightbackground="#45475A", highlightthickness=1)
image_frame.pack(pady=10, padx=20, fill="x")

image_label = tk.Label(image_frame, bg=PANEL_COLOR, text="[ Место для фото ]", fg="#7F849C", font=("Segoe UI", 12), width=30, height=12)
image_label.pack(pady=10)

btn_upload = tk.Button(root, text="ЗАГРУЗИТЬ ФОТО", font=("Segoe UI", 12, "bold"), 
                       command=upload_and_predict, bg=ACCENT_COLOR, fg=BG_COLOR, 
                       activebackground="#74C7EC", activeforeground=BG_COLOR,
                       relief="flat", cursor="hand2", padx=20, pady=8)
btn_upload.pack(pady=10)

result_frame = tk.Frame(root, bg=BG_COLOR)
result_frame.pack(pady=10, fill="x")

result_label = tk.Label(result_frame, text="Узор: Ожидание...", font=("Segoe UI", 16, "bold"), bg=BG_COLOR, fg=TEXT_COLOR)
result_label.pack(pady=5)

scheme_label = tk.Label(result_frame, text="Загрузите фото для получения схемы.", font=("Segoe UI", 11), bg=BG_COLOR, fg="#BAC2DE", justify="center")
scheme_label.pack(pady=5)

log_frame = tk.Frame(root, bg=BG_COLOR)
log_frame.pack(side=tk.BOTTOM, fill="both", expand=True, padx=20, pady=(10, 20))

log_title = tk.Label(log_frame, text="ЖУРНАЛ ИИ-МОДЕЛИ", font=("Segoe UI", 9, "bold"), bg=BG_COLOR, fg="#7F849C", anchor="w")
log_title.pack(fill="x", pady=(0, 5))

log_text = tk.Text(log_frame, height=9, bg=PANEL_COLOR, fg=SUCCESS_COLOR, font=("Consolas", 9), 
                   relief="flat", padx=10, pady=10, state=tk.DISABLED)
log_text.pack(fill="both", expand=True)

log_text.config(state=tk.NORMAL)
if model_loaded:
    log_text.insert(tk.END, "--- СИСТЕМНЫЙ ЖУРНАЛ ИИ ---\n[+] Система инициализирована.\n[+] best_svm_model.pkl и Scaler загружены.\n[+] Ожидание ввода пользователя...")
else:
    log_text.config(fg=ERROR_COLOR)
    log_text.insert(tk.END, "--- СИСТЕМНЫЙ ЖУРНАЛ ИИ ---\n[ОШИБКА] Файлы best_svm_model.pkl или scaler.pkl\nне найдены в папке 'output/'.")
log_text.config(state=tk.DISABLED)

root.mainloop()