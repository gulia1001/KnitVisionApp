import stitchData from './data.js';

document.addEventListener('DOMContentLoaded', () => {
  const uploadZone = document.getElementById('uploadZone');
  const fileInput = document.getElementById('fileInput');
  const previewImage = document.getElementById('previewImage');
  const loader = document.getElementById('loader');
  const resultArea = document.getElementById('resultArea');
  const patternName = document.getElementById('patternName');
  const viewTutorialBtn = document.getElementById('viewTutorialBtn');
  
  const mainStage = document.querySelector('.main-stage');
  const sidePanel = document.getElementById('sidePanel');
  const closePanelBtn = document.getElementById('closePanelBtn');
  const tutorialTitle = document.getElementById('tutorialTitle');
  const tutorialImage = document.getElementById('tutorialImage');
  const tutorialText = document.getElementById('tutorialText');

  let currentPattern = null;

  // Handle Drag & Drop
  uploadZone.addEventListener('click', () => fileInput.click());

  uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
  });

  uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('dragover');
  });

  uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFile(e.target.files[0]);
    }
  });

  function handleFile(file) {
    if (!file.type.startsWith('image/')) {
      alert('Please upload an image file');
      return;
    }

    // Show Preview
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImage.src = e.target.result;
      previewImage.classList.remove('hidden');
    };
    reader.readAsDataURL(file);

    // Reset UI
    resultArea.classList.add('hidden');
    closePanel();
    
    // Upload image to FastAPI backend
    uploadAndPredict(file);
  }

  async function uploadAndPredict(file) {
    loader.classList.remove('hidden');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error('Prediction failed');
      }

      const data = await response.json();
      const recognizedClass = data.class; // e.g. "garter"
      const mlDetails = data.details;
      
      loader.classList.add('hidden');
      
      // Update result area
      currentPattern = stitchData[recognizedClass];
      patternName.textContent = currentPattern ? currentPattern.title : recognizedClass.toUpperCase();
      
      const logContent = document.getElementById('logContent');
      if (logContent && mlDetails) {
        let logText = `--- SYSTEM AI LOG ---\n`;
        logText += `[+] File: ${mlDetails.filename}\n`;
        logText += `    - Original: ${mlDetails.original_shape}\n`;
        logText += `[+] Size Correction: Center crop ${mlDetails.crop_shape}\n`;
        logText += `[+] Preprocessing: Resize(${mlDetails.resize}), ${mlDetails.preprocessing}\n`;
        logText += `[+] Feature Extraction:\n`;
        logText += `    - HOG: ${mlDetails.features.hog}\n`;
        logText += `    - Multi-LBP: ${mlDetails.features.multi_lbp}\n`;
        logText += `    - Gabor: ${mlDetails.features.gabor}\n`;
        logText += `    - Total Vector: ${mlDetails.features.total_vector}\n`;
        logText += `[+] Scikit-Learn SVM Verdict:\n`;
        logText += `    - ${mlDetails.verdict}\n`;
        logText += `>>> RESULT: Texture classified as '${recognizedClass.toUpperCase()}'.`;
        logContent.textContent = logText;
      }

      resultArea.classList.remove('hidden');

    } catch (error) {
      loader.classList.add('hidden');
      alert('Error analyzing image. Make sure the backend is running properly!');
      console.error(error);
    }
  }

  // Side Panel Logic
  viewTutorialBtn.addEventListener('click', () => {
    if (currentPattern) {
      openPanel();
      
      tutorialTitle.textContent = currentPattern.title;
      tutorialImage.src = currentPattern.image;
      
      // Build paragraphs
      tutorialText.innerHTML = '';
      currentPattern.description.forEach(pText => {
        const p = document.createElement('p');
        p.textContent = pText;
        tutorialText.appendChild(p);
      });
    }
  });

  closePanelBtn.addEventListener('click', closePanel);

  function openPanel() {
    sidePanel.classList.add('open');
    mainStage.classList.add('panel-open');
  }

  function closePanel() {
    sidePanel.classList.remove('open');
    mainStage.classList.remove('panel-open');
  }
});
