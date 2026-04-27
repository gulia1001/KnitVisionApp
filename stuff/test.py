import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# --- Настройки ---
CLASSES = ["stockinette", "garter", "ribbing", "lace"]
IMG_SIZE = 128
TEST_DIR = "test_dataset" 

hog = cv2.HOGDescriptor(
    _winSize=(IMG_SIZE, IMG_SIZE),
    _blockSize=(16, 16),
    _blockStride=(8, 8),
    _cellSize=(8, 8),
    _nbins=9
)

def preprocess_for_test(img_path):
    img_array = np.fromfile(img_path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    
    if img is None: 
        return None
    
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.GaussianBlur(img, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    
    return img

print("loading the model...")
svm = cv2.ml.SVM_load("knitting_classifier.yml")

y_true = [] 
y_pred = [] 

print(f"Тesting: {TEST_DIR}")

for class_idx, label in enumerate(CLASSES):
    files = glob.glob(os.path.join(TEST_DIR, label, "*.*"))
    print(f"[{label}] Найдено фото для теста: {len(files)}")
    
    for f in files:
        img = preprocess_for_test(f)
        if img is None: continue
            
        features = hog.compute(img).flatten().reshape(1, -1)
        _, result = svm.predict(features)
        
        y_true.append(class_idx)
        y_pred.append(int(result[0][0]))

acc = accuracy_score(y_true, y_pred) * 100
print(f"\nAcc: {acc:.2f}%")

cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
disp.plot(cmap=plt.cm.Blues, ax=ax)

plt.title(f"Conf matrix (Точность: {acc:.2f}%)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()