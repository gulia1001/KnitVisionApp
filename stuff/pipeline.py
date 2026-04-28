import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog, local_binary_pattern
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


DATASET   = 'cleaned_dataset'
OUTPUT    = 'output'
CLASSES   = ['garter', 'stockinette', 'ribbing', 'seed']
IMG_SIZE  = (128, 128)
LBP_PARAMS = [(1, 8), (3, 24), (5, 40)]

os.makedirs(OUTPUT, exist_ok=True)


def build_gabor():
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 4):
        for lamda in [np.pi / 4, np.pi / 2, np.pi]:
            kern = cv2.getGaborKernel(
                (ksize, ksize), 4.0, theta, lamda, 0.5, 0, ktype=cv2.CV_32F
            )
            kern /= 1.5 * kern.sum()
            filters.append(kern)
    return filters

GABOR_FILTERS = build_gabor()



def crop_square(img):
    h, w = img.shape
    s = min(h, w)
    x = (w - s) // 2
    y = (h - s) // 2
    return img[y:y + s, x:x + s]



def extract_features(img):
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img   = clahe.apply(img)

    # HOG
    f_hog = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
    )

    # Multi-scale LBP
    f_lbp = []
    for radius, n_points in LBP_PARAMS:
        lbp   = local_binary_pattern(img, n_points, radius, method='uniform')
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins,
                               range=(0, n_bins), density=True)
        f_lbp.extend(hist)

    # Gabor
    f_gabor = []
    for kern in GABOR_FILTERS:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        f_gabor.append(fimg.mean())
        f_gabor.append(fimg.var())

    return np.hstack([f_hog, f_lbp, f_gabor]).astype(np.float32)


def load_data():
    X, y = [], []
    print("Loading data  (HOG + Multi-LBP + Gabor) …")
    for label, cls in enumerate(CLASSES):
        folder = os.path.join(DATASET, cls)
        if not os.path.exists(folder):
            print(f"  [WARN] folder not found: {folder}")
            continue
        for fname in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(crop_square(img), IMG_SIZE)
            X.append(extract_features(img))
            y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def normalize_features(X_train, X_test):
    """
    Feature-wise z-score: compute mean/std on train, apply to both.
    Returns normalised arrays (float32) and the scale parameters.
    """
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0) + 1e-8          # avoid div-by-zero
    X_train_n = ((X_train - mean) / std).astype(np.float32)
    X_test_n  = ((X_test  - mean) / std).astype(np.float32)
    return X_train_n, X_test_n, mean, std


def build_svm(C: float, gamma: float) -> cv2.ml.SVM:
    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setC(C)
    svm.setGamma(gamma)
    svm.setTermCriteria(
        (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-6)
    )
    return svm


def svm_predict(svm_model, X: np.ndarray) -> np.ndarray:
    _, preds = svm_model.predict(X)
    return preds.ravel().astype(int)



def build_knn(k: int, use_distance_weights: bool) -> cv2.ml.KNearest:
    knn = cv2.ml.KNearest_create()
    knn.setIsClassifier(True)
    knn.setAlgorithmType(cv2.ml.KNearest_BRUTE_FORCE)
    return knn        


def knn_predict(knn_model, X: np.ndarray, k: int) -> np.ndarray:
    _, results, neighbours, dists = knn_model.findNearest(X, k=k)

    preds = []
    for i in range(len(results)):
        d = dists[i]
        n = neighbours[i].astype(int)
        w = np.where(d == 0, 1e9, 1.0 / (d + 1e-8))
        votes = {}
        for cls_id, weight in zip(n, w):
            votes[cls_id] = votes.get(cls_id, 0) + weight
        preds.append(max(votes, key=votes.get))
    return np.array(preds, dtype=int)



def grid_search_svm(X_train, y_train, param_grid, cv=5):
    """Returns (best_params, best_mean_acc)."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    best_acc, best_params = -1, {}

    print(f"  SVM grid: {len(param_grid['C']) * len(param_grid['gamma'])} combos × {cv} folds")
    for C in param_grid['C']:
        for gamma in param_grid['gamma']:
            fold_accs = []
            for train_idx, val_idx in skf.split(X_train, y_train):
                Xtr, Xval = X_train[train_idx], X_train[val_idx]
                ytr, yval = y_train[train_idx], y_train[val_idx]
                svm = build_svm(C, gamma)
                svm.train(Xtr, cv2.ml.ROW_SAMPLE, ytr)
                fold_accs.append(accuracy_score(yval, svm_predict(svm, Xval)))
            mean_acc = np.mean(fold_accs)
            if mean_acc > best_acc:
                best_acc, best_params = mean_acc, {'C': C, 'gamma': gamma}
    return best_params, best_acc


def grid_search_knn(X_train, y_train, param_grid, cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    best_acc, best_params = -1, {}

    combos = [(k, w) for k in param_grid['n_neighbors']
                      for w in param_grid['weights']]
    print(f"  KNN grid: {len(combos)} combos × {cv} folds")

    fold_splits = list(skf.split(X_train, y_train))

    for k in param_grid['n_neighbors']:
        for use_dist in [False, True]:       
            weight_label = 'distance' if use_dist else 'uniform'
            fold_accs = []
            for train_idx, val_idx in fold_splits:
                Xtr, Xval = X_train[train_idx], X_train[val_idx]
                ytr, yval = y_train[train_idx], y_train[val_idx]
                knn = build_knn(k, use_dist)
                knn.train(Xtr, cv2.ml.ROW_SAMPLE, ytr)
                if use_dist:
                    preds = knn_predict(knn, Xval, k)
                else:
                    _, res, _, _ = knn.findNearest(Xval, k=k)
                    preds = res.ravel().astype(int)
                fold_accs.append(accuracy_score(yval, preds))
            mean_acc = np.mean(fold_accs)
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_params = {'n_neighbors': k, 'weights': weight_label}
    return best_params, best_acc



def plot_conf_matrices(y_test, svm_pred, knn_pred):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ConfusionMatrixDisplay(
        confusion_matrix(y_test, svm_pred), display_labels=CLASSES
    ).plot(ax=axes[0], cmap='Blues', xticks_rotation=45)
    axes[0].set_title('OpenCV SVM (RBF, Grid Search)')

    ConfusionMatrixDisplay(
        confusion_matrix(y_test, knn_pred), display_labels=CLASSES
    ).plot(ax=axes[1], cmap='Oranges', xticks_rotation=45)
    axes[1].set_title('OpenCV KNN (Grid Search)')

    plt.tight_layout()
    path = os.path.join(OUTPUT, 'confusion_matrices_opencv.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Confusion matrices saved → {path}")


def train():
    X, y = load_data()
    print(f"Dataset: {len(X)} samples, {X.shape[1]} features, {len(CLASSES)} classes\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=999
    )

    X_train, X_test, feat_mean, feat_std = normalize_features(X_train, X_test)

    # ── SVM ──────────────────────────────────
    print("── Grid Search  OpenCV SVM ──────────────────")
    svm_param_grid = {
        'C':     [1, 10, 50, 100],
        'gamma': [0.001, 0.01, 'scale', 'auto'],
    }
    n_feats = X_train.shape[1]
    n_total = len(X_train)
    resolved_gammas = []
    for g in svm_param_grid['gamma']:
        if g == 'scale':
            resolved_gammas.append(1.0 / (n_feats * X_train.var()))
        elif g == 'auto':
            resolved_gammas.append(1.0 / n_feats)
        else:
            resolved_gammas.append(float(g))
    svm_param_grid['gamma'] = resolved_gammas

    best_svm_params, best_svm_cv_acc = grid_search_svm(
        X_train, y_train, svm_param_grid
    )
    print(f"Best SVM params  : {best_svm_params}")
    print(f"Best SVM CV acc  : {best_svm_cv_acc * 100:.2f}%")

    best_svm = build_svm(**best_svm_params)
    best_svm.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
    svm_pred = svm_predict(best_svm, X_test)
    print(f"Test  SVM acc    : {accuracy_score(y_test, svm_pred) * 100:.2f}%\n")

    print("── Grid Search  OpenCV KNN ──────────────────")
    knn_param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights':     ['uniform', 'distance'],
    }

    best_knn_params, best_knn_cv_acc = grid_search_knn(
        X_train, y_train, knn_param_grid
    )
    print(f"Best KNN params  : {best_knn_params}")
    print(f"Best KNN CV acc  : {best_knn_cv_acc * 100:.2f}%")

    use_dist   = best_knn_params['weights'] == 'distance'
    best_k     = best_knn_params['n_neighbors']
    best_knn   = build_knn(best_k, use_dist)
    best_knn.train(X_train, cv2.ml.ROW_SAMPLE, y_train)

    if use_dist:
        knn_pred = knn_predict(best_knn, X_test, best_k)
    else:
        _, res, _, _ = best_knn.findNearest(X_test, k=best_k)
        knn_pred = res.ravel().astype(int)

    print(f"Test  KNN acc    : {accuracy_score(y_test, knn_pred) * 100:.2f}%\n")

    best_svm.save(os.path.join(OUTPUT, 'best_svm_opencv.xml'))
    best_knn.save(os.path.join(OUTPUT, 'best_knn_opencv.xml'))
    np.save(os.path.join(OUTPUT, 'scaler_mean.npy'), feat_mean)
    np.save(os.path.join(OUTPUT, 'scaler_std.npy'),  feat_std)
    print(f"Models & scaler saved in '{OUTPUT}/'")

    plot_conf_matrices(y_test, svm_pred, knn_pred)



def predict_image(img_path: str, model_type: str = 'svm') -> str:
    # Load scaler
    mean = np.load(os.path.join(OUTPUT, 'scaler_mean.npy'))
    std  = np.load(os.path.join(OUTPUT, 'scaler_std.npy'))

    # Load model
    if model_type == 'svm':
        model = cv2.ml.SVM_load(os.path.join(OUTPUT, 'best_svm_opencv.xml'))
    else:
        model = cv2.ml.KNearest_load(os.path.join(OUTPUT, 'best_knn_opencv.xml'))

    # Preprocess
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    img   = cv2.resize(crop_square(img), IMG_SIZE)
    feat  = extract_features(img).reshape(1, -1)
    feat  = ((feat - mean) / (std + 1e-8)).astype(np.float32)

    # Predict
    if model_type == 'svm':
        label_idx = int(svm_predict(model, feat)[0])
    else:
        _, res, _, _ = model.findNearest(feat, k=5)
        label_idx = int(res.ravel()[0])

    return CLASSES[label_idx]


train()