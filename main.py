import os
import cv2
import numpy as np
import joblib
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from skimage.feature import hog, local_binary_pattern

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASSES = ['garter', 'stockinette', 'ribbing', 'seed']
IMG_SIZE = 128
LBP_PARAMS = [(1, 8), (3, 24), (5, 40)]

try:
    svm_model = joblib.load(os.path.join("output", "best_svm_model.pkl"))
    scaler = joblib.load(os.path.join("output", "scaler.pkl"))
except Exception as e:
    print(f"Ошибка загрузки моделей: {e}")

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

def center_crop_square(img):
    h, w = img.shape
    min_dim = min(h, w)
    start_x = (w // 2) - (min_dim // 2)
    start_y = (h // 2) - (min_dim // 2)
    return img[start_y:start_y+min_dim, start_x:start_x+min_dim]

def extract_features(img):
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

@app.post("/predict")
async def predict_stitch(file: UploadFile = File(...)):
    if not svm_model or not scaler:
        return JSONResponse({"error": "Models are not loaded on server."}, status_code=500)
    
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_orig = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img_orig is None:
        return JSONResponse({"error": "Invalid image format."}, status_code=400)
    
    orig_shape = img_orig.shape
    img_cropped = center_crop_square(img_orig)
    crop_shape = img_cropped.shape
    img_resized = cv2.resize(img_cropped, (IMG_SIZE, IMG_SIZE))
    
    features = extract_features(img_resized)
    features_scaled = scaler.transform([features])
    
    pred_idx = svm_model.predict(features_scaled)[0]
    predicted_class = CLASSES[pred_idx]
    
    # ML Details for the explanation UI
    ml_details = {
        "filename": file.filename,
        "original_shape": f"{orig_shape[1]}x{orig_shape[0]}",
        "crop_shape": f"{crop_shape[1]}x{crop_shape[0]}",
        "resize": f"{IMG_SIZE}x{IMG_SIZE}",
        "preprocessing": "CLAHE (Contrast Limited Adaptive Histogram Equalization)",
        "features": {
            "hog": "8100 parameters",
            "multi_lbp": "78 parameters",
            "gabor": "24 parameters",
            "total_vector": f"{len(features)} parameters"
        },
        "verdict": f"Class ID: {pred_idx}"
    }
    
    return {"class": predicted_class, "details": ml_details}

app.mount("/", StaticFiles(directory="web_app", html=True), name="web_app")
