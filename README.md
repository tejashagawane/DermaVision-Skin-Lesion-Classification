# DermaVision-Skin-Lesion-Classification
AI-Based Skin Lesion Classification using ResNet50 and Grad-CAM
# 🔬 DermaVision – AI-Based Skin Lesion Classification

### Skin Lesion Classification Using ResNet50 and Explainable AI (Grad-CAM)

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red?logo=streamlit)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-CNN-green)
![Dataset](https://img.shields.io/badge/Dataset-HAM10000-purple)

## 📌 Project Overview

DermaVision is an AI-powered skin lesion classification system developed using Deep Learning and Explainable Artificial Intelligence (XAI).

The project uses a fine-tuned **ResNet50 Convolutional Neural Network (CNN)** to classify dermoscopic skin lesion images into seven diagnostic categories from the HAM10000 dataset.

To improve model interpretability, the project integrates **Grad-CAM (Gradient-weighted Class Activation Mapping)**, which generates visual heatmaps highlighting image regions that influenced the model's predictions.

The application is developed using Streamlit, providing an interactive interface where users can upload skin lesion images, view classification results, explore class probabilities, and generate visual explanations.

## 🎯 Project Objectives

- Develop a deep-learning-based skin lesion classification system.
- Implement ResNet50 using transfer learning.
- Classify dermoscopic images into seven diagnostic categories.
- Apply image preprocessing and data augmentation.
- Evaluate model performance using classification metrics.
- Integrate Grad-CAM for visual model interpretability.
- Develop an interactive web application using Streamlit.

## 🧠 Model Architecture

The project uses **ResNet50**, a 50-layer deep residual neural network.

ResNet50 uses residual connections to improve information and gradient flow through deeper network layers.

The original classification layer is replaced with a custom fully connected layer containing seven output neurons corresponding to the HAM10000 diagnostic categories.

### Model Workflow

1. Input dermoscopic image.
2. Image resizing to 224 × 224 pixels.
3. Image normalization.
4. Feature extraction using ResNet50.
5. Seven-class classification.
6. Softmax probability calculation.
7. Grad-CAM visualization.

## 🗂️ Dataset – HAM10000

The project uses the HAM10000 (Human Against Machine with 10,000 training images) dataset.

The dataset contains approximately 10,015 dermoscopic images representing seven categories of pigmented skin lesions.

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

**Dataset source:** [HAM10000 – Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

## 🔍 Explainable AI – Grad-CAM

Grad-CAM is used to visualize the regions of an input image that contribute to a selected model prediction.

It uses gradients flowing into the final convolutional layer to generate a class-specific activation heatmap.

The application provides:

- Original uploaded image.
- Grad-CAM activation heatmap.
- Heatmap overlay on the original image.
- Class probability distribution.

This helps researchers inspect model behavior and understand which image regions influenced its output.

## 🛠️ Technology Stack

| Technology | Application |
|---|---|
| Python | Core programming language |
| PyTorch | Deep learning framework |
| Torchvision | ResNet50 architecture and image transformations |
| ResNet50 | Skin lesion classification |
| Grad-CAM | Explainable AI visualization |
| Streamlit | Interactive web application |
| NumPy | Numerical computations |
| Pandas | Dataset processing |
| Matplotlib / Plotly | Data visualization |
| Pillow | Image processing |
| Scikit-learn | Model evaluation and classification metrics |

## 📁 Project Structure

```text
DermaVision/
│
├── app.py
│
├── requirements.txt
│
├── README.md
│
├── .gitignore
│
└── models/
    └── ham10000_resnet50_best.pth
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/DermaVision-Skin-Lesion-Classification.git
```

### 2. Navigate to the project directory

```bash
cd DermaVision-Skin-Lesion-Classification
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

**Windows:**

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Add the trained model

Place the trained checkpoint inside the models directory:

```text
models/ham10000_resnet50_best.pth
```

The checkpoint must match the seven-class ResNet50 architecture and the label mapping used by the application.

### 7. Run the application

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

## 📊 Application Features

### 1. Image Upload
Upload dermoscopic images in JPG, JPEG, or PNG format.

### 2. AI-Based Classification
The trained ResNet50 model predicts one of seven skin lesion categories.

### 3. Probability Distribution
View the model's scores across all seven diagnostic categories.

### 4. Explainable AI
Generate Grad-CAM heatmaps to visualize image regions influencing the selected prediction.

### 5. Interactive Dashboard
Explore prediction results, image explanations, and model information through a Streamlit interface.

## 🚀 Future Enhancements

- Improve classification performance through hyper
