import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import joblib
from skimage.feature import hog, local_binary_pattern
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# --- НАСТРОЙКИ ---
DATASET_DIR = 'cleaned_dataset'
OUTPUT_DIR = 'output'
CLASSES = ['garter', 'stockinette', 'ribbing', 'seed']
IMG_SIZE = (128, 128)

# Настройки для Мульти-масштабного LBP (Радиус, Количество точек)
LBP_PARAMS = [(1, 8), (3, 24), (5, 40)]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def build_gabor_filters():
    """Создает банк фильтров Габора для поиска мягких текстур под разными углами."""
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 4):  # 4 угла наклона
        for lamda in [np.pi/4, np.pi/2, np.pi]:   # 3 частоты волны
            kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5 * kern.sum()
            filters.append(kern)
    return filters

GABOR_FILTERS = build_gabor_filters()

def center_crop_square(img):
    h, w = img.shape
    min_dim = min(h, w)
    start_x = (w // 2) - (min_dim // 2)
    start_y = (h // 2) - (min_dim // 2)
    return img[start_y:start_y+min_dim, start_x:start_x+min_dim]

def extract_features(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    
    # 1. HOG (Жесткие контуры и углы)
    features_hog = hog(img, orientations=9, pixels_per_cell=(8, 8), 
                       cells_per_block=(2, 2), block_norm='L2-Hys')
    
    # 2. Мульти-масштабный LBP (Микро- и макро-текстуры)
    features_lbp = []
    for radius, points in LBP_PARAMS:
        lbp = local_binary_pattern(img, points, radius, method='uniform')
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        features_lbp.extend(hist)
        
    # 3. Фильтры Габора (Ритм и переплетения)
    features_gabor = []
    for kern in GABOR_FILTERS:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        features_gabor.append(fimg.mean())
        features_gabor.append(fimg.var())
        
    # Склеиваем все 3 типа признаков в один гигантский вектор
    combined_features = np.hstack([features_hog, features_lbp, features_gabor])
    return combined_features

def load_data():
    X, y = [], []
    print("Извлекаем МЕГА-признаки (HOG + Multi-LBP + Gabor)... Это займет немного времени.")
    for label, knit_class in enumerate(CLASSES):
        folder_path = os.path.join(DATASET_DIR, knit_class)
        if not os.path.exists(folder_path): continue
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            
            img_cropped = center_crop_square(img)
            img_resized = cv2.resize(img_cropped, IMG_SIZE)
            feat = extract_features(img_resized)
            X.append(feat)
            y.append(label)
    return np.array(X), np.array(y)

def plot_confusion_matrices(y_test, svm_pred, knn_pred):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cm_svm = confusion_matrix(y_test, svm_pred)
    disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=CLASSES)
    disp_svm.plot(ax=axes[0], cmap='Blues', xticks_rotation=45)
    axes[0].set_title('Матрица ошибок: SVM (GridSearch)')
    
    cm_knn = confusion_matrix(y_test, knn_pred)
    disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=CLASSES)
    disp_knn.plot(ax=axes[1], cmap='Oranges', xticks_rotation=45)
    axes[1].set_title('Матрица ошибок: KNN (GridSearch)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '02_confusion_matrices_optimized.png'))
    plt.close()

def train_models():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=999)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # --- Grid Search для SVM ---
    print("\n" + "="*30)
    print("Запускаем Grid Search для SVM... Ждем...")
    svm_param_grid = {
        'C': [1, 10, 50, 100],
        'gamma': ['scale', 'auto', 0.01, 0.001]
    }
    svm_grid = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), svm_param_grid, cv=5, n_jobs=-1)
    svm_grid.fit(X_train, y_train)
    best_svm = svm_grid.best_estimator_
    
    svm_pred = best_svm.predict(X_test)
    print(f"Лучшие параметры SVM: {svm_grid.best_params_}")
    print(f"Точность оптимизированного SVM: {accuracy_score(y_test, svm_pred) * 100:.2f}%")
    
    # --- Grid Search для KNN ---
    print("\n" + "="*30)
    print("Запускаем Grid Search для KNN... Ждем...")
    knn_param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance']
    }
    knn_grid = GridSearchCV(KNeighborsClassifier(), knn_param_grid, cv=5, n_jobs=-1)
    knn_grid.fit(X_train, y_train)
    best_knn = knn_grid.best_estimator_
    
    knn_pred = best_knn.predict(X_test)
    print(f"Лучшие параметры KNN: {knn_grid.best_params_}")
    print(f"Точность оптимизированного KNN: {accuracy_score(y_test, knn_pred) * 100:.2f}%")
    
    # Сохраняем матрицы с новыми предсказаниями
    plot_confusion_matrices(y_test, svm_pred, knn_pred)
    
    # --- СОХРАНЕНИЕ ЛУЧШИХ МОДЕЛЕЙ ---
    print("\n" + "="*30)
    joblib.dump(best_svm, os.path.join(OUTPUT_DIR, 'best_svm_model.pkl'))
    joblib.dump(best_knn, os.path.join(OUTPUT_DIR, 'best_knn_model.pkl'))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler.pkl'))
    print(f"Лучшие модели сохранены в папку '{OUTPUT_DIR}'!")

if __name__ == "__main__":
    train_models()