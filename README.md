# 🔬 DermaVision
### AI-Based Skin Lesion Classification Using ResNet50 and Explainable AI (Grad-CAM)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-orange?logo=pytorch" />
  <img src="https://img.shields.io/badge/Streamlit-Deployed-red?logo=streamlit" />
  <img src="https://img.shields.io/badge/Model-ResNet50-green" />
  <img src="https://img.shields.io/badge/XAI-Grad--CAM-purple" />
</p>

<p align="center">
  <b>An AI-powered research application for skin lesion classification and explainable predictions.</b>
</p>

## 🌐 Live Application

<p align="center">
  <a href="https://dermavision-skin-lesion-classification-idzuntchs874u8zwbzdqbr.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀_Launch_DermaVision-Live_Application-176B72?style=for-the-badge" />
  </a>
</p>

**Live Demo:** https://dermavision-skin-lesion-classification-idzuntchs874u8zwbzdqbr.streamlit.app/

## 📌 Project Overview

DermaVision is an Artificial Intelligence and Deep Learning-based research project designed to classify dermoscopic skin lesion images into seven diagnostic categories using a ResNet50 convolutional neural network.

The project integrates Explainable Artificial Intelligence (XAI) through Grad-CAM (Gradient-weighted Class Activation Mapping), generating visual heatmaps that highlight image regions influencing the model's predictions.

The interactive Streamlit application allows users to upload skin lesion images, view classification results, explore class probability scores, and generate visual explanations.

## 🎯 Project Objectives

- Develop an AI-based skin lesion classification system.
- Implement ResNet50 using transfer learning.
- Classify dermoscopic images into seven diagnostic categories.
- Apply image preprocessing and data augmentation.
- Evaluate model performance using classification metrics.
- Integrate Grad-CAM for model interpretability.
- Develop an interactive web application using Streamlit.

## 🧠 Model Architecture – ResNet50

The project uses ResNet50, a 50-layer deep convolutional neural network.

The original classification layer is replaced with a custom fully connected layer containing seven output neurons corresponding to the HAM10000 dataset categories.

### Model Workflow

1. Input dermoscopic image.
2. Image resizing to 224 × 224 pixels.
3. Image normalization.
4. Feature extraction using ResNet50.
5. Seven-class classification.
6. Softmax-based class score calculation.
7. Grad-CAM visualization.

## 🗂️ Dataset – HAM10000

The project uses the HAM10000 (Human Against Machine with 10,000 training images) dataset, containing approximately 10,015 dermoscopic images.

### Classification Categories

| Label | Diagnostic Category |
|---|---|
| AKIEC | Actinic keratoses / intraepithelial carcinoma |
| BCC | Basal cell carcinoma |
| BKL | Benign keratosis-like lesions |
| DF | Dermatofibroma |
| MEL | Melanoma |
| NV | Melanocytic nevi |
| VASC | Vascular lesions |

**Dataset:** [HAM10000 – Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

## 🔍 Explainable AI – Grad-CAM

Grad-CAM visualizes image regions that influence a selected model prediction.

The application provides:

- Original uploaded image.
- Grad-CAM activation heatmap.
- Heatmap overlay on the original image.
- Classification probability distribution.

## 💻 Application Features

1. **Image Upload:** Upload JPG, JPEG or PNG dermoscopic images.
2. **AI Classification:** Predict one of seven HAM10000 categories.
3. **Probability Distribution:** Explore model scores across all seven classes.
4. **Explainable AI:** Generate Grad-CAM heatmaps.
5. **Interactive Dashboard:** Explore results through a Streamlit interface.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyTorch | Deep learning framework |
| Torchvision | ResNet50 and image transformations |
| ResNet50 | Image classification |
| Grad-CAM | Explainable AI |
| Streamlit | Web application |
| NumPy | Numerical computations |
| Pandas | Data processing |
| Pillow | Image processing |
| Plotly / Matplotlib | Data visualization |
| Scikit-learn | Model evaluation |

## 📁 Project Structure

```text
DermaVision/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .gitattributes
│
└── models/
    └── ham10000_resnet50_best.pth
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/tejashagawane/DermaVision-Skin-Lesion-Classification.git
```

### 2. Navigate to the project

```bash
cd DermaVision-Skin-Lesion-Classification
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment (Windows)

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Add the trained model

Place the compatible trained checkpoint inside:

```text
models/ham10000_resnet50_best.pth
```

### 7. Run locally

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## 📊 Model Evaluation

The research notebook uses the following evaluation metrics:

- Accuracy
- Balanced Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Training and Validation Loss

Actual test results should be reported after the final model evaluation.

## 🚀 Future Enhancements

- Improve classification performance through hyperparameter optimization.
- Compare ResNet50 with EfficientNet and other architectures.
- Add prediction history and downloadable reports.
- Explore SHAP and LIME for additional explainability.
- Improve deployment scalability and model monitoring.

## ⚠️ Medical Disclaimer

DermaVision is an academic research and educational prototype. It is not a clinically validated medical diagnostic system and must not be used for diagnosis or treatment decisions.

Model predictions and Grad-CAM visualizations are experimental outputs and should not replace assessment by qualified healthcare professionals.

## 👨‍💻 Project Information

**Project Name:** DermaVision – AI-Based Skin Lesion Classification

**Domain:** Artificial Intelligence | Deep Learning | Healthcare | Explainable AI

**Architecture:** ResNet50

**Dataset:** HAM10000

**Explainability:** Grad-CAM

**Frontend:** Streamlit

**GitHub Repository:** https://github.com/tejashagawane/DermaVision-Skin-Lesion-Classification

**Live Application:** https://dermavision-skin-lesion-classification-idzuntchs874u8zwbzdqbr.streamlit.app/

---

⭐ If you find this project interesting, consider giving the repository a star!

**Developed for academic and research purposes.**
