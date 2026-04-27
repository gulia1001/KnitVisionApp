# Knit Vision App

Knit Vision is an interactive web application that uses artificial intelligence to visually recognize different knitting stitch patterns (Garter, Ribbing, Seed, and Stockinette). 

When you upload an image, the application relies on a computer vision machine learning pipeline (built with HOG, Multi-LBP, and Gabor filters) combined with an SVM classifier to analyze the texture of the fabric. Once identified, it presents an "AI System Log" detailing the algorithmic breakdown, and provides an interactive side-panel containing a full tutorial on how to knit the extracted pattern!

## 📸 How It Works
1. **Frontend**: A sleek, lavender-styled Vanilla JS/CSS/HTML interface (`web_app/`).
2. **Backend**: A blazing-fast FastAPI server (`main.py`) that handles image processing via OpenCV/Scikit-Image and serves the web frontend.
3. **Data**: The tutorial data for the side-panel has been extracted and pre-compiled into `web_app/data.js`, so no active internet connection is needed to parse html pages anymore.

---

## 🚀 How to Run the App

### 1. Activate your Python Environment
Make sure you are running from your project environment (`cv_env`) so that packages are isolated:
```powershell
# On Windows PowerShell
..\cv_env\Scripts\Activate.ps1
```

### 2. Install Requirements
If you haven't already, install all the necessary dependencies used by the ML engine and the web server:
```powershell
pip install -r requirements.txt
```

### 3. Start the Server
Launch the FastAPI development server using Uvicorn. You can specify whichever port you prefer (e.g., 8000 or 9099):
```powershell
uvicorn main:app --port 8000
```

### 4. Open the App!
Finally, open your web browser and navigate to:
**http://127.0.0.1:8000**

Drop a photo into the upload area and let the AI recognize your stitch!
