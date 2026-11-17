import os
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Skin Lesion AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #777777;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .prediction-box {
        padding: 20px;
        border-radius: 15px;
        background-color: #f5f7fa;
        border: 1px solid #dddddd;
        text-align: center;
    }

    .prediction-label {
        font-size: 28px;
        font-weight: 700;
    }

    .confidence {
        font-size: 20px;
        margin-top: 10px;
    }

    .warning-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. APPLICATION TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🩺 AI Skin Lesion Classification</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'HAM10000 • ResNet50 • Explainable AI • Grad-CAM'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# 4. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 5. CLASS INFORMATION
# ============================================================

CLASS_NAMES = {
    "akiec": "Actinic keratoses / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions"
}


DEFAULT_LABEL_TO_IDX = {
    "akiec": 0,
    "bcc": 1,
    "bkl": 2,
    "df": 3,
    "mel": 4,
    "nv": 5,
    "vasc": 6
}


# ============================================================
# 6. MODEL PATH
# ============================================================

MODEL_PATH = os.path.join(
    "models",
    "ham10000_resnet50_best.pth"
)


# ============================================================
# 7. IMAGE PREPROCESSING
# ============================================================

IMAGE_SIZE = 224

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 8. LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):

        st.error(
            "Model file not found.\n\n"
            "Please place:\n"
            "`ham10000_resnet50_best.pth`\n\n"
            "inside the `models` folder."
        )

        return None, DEFAULT_LABEL_TO_IDX

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    # --------------------------------------------------------
    # Get label mapping from checkpoint
    # --------------------------------------------------------

    label_to_idx = checkpoint.get(
        "label_to_idx",
        DEFAULT_LABEL_TO_IDX
    )

    # --------------------------------------------------------
    # Create ResNet50
    # --------------------------------------------------------

    model = models.resnet50(
        weights=None
    )

    # --------------------------------------------------------
    # Replace final layer
    # --------------------------------------------------------

    num_classes = len(label_to_idx)

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    # --------------------------------------------------------
    # Load trained weights
    # --------------------------------------------------------

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)

    model.eval()

    return model, label_to_idx


# ============================================================
# 9. LOAD MODEL
# ============================================================

model, label_to_idx = load_model()


if model is None:

    st.stop()


# Reverse mapping

idx_to_label = {
    value: key
    for key, value in label_to_idx.items()
}


# ============================================================
# 10. PREDICTION FUNCTION
# ============================================================

def predict_image(image):

    """
    Predict the skin lesion class.
    """

    # Convert image to RGB

    image = image.convert("RGB")

    # Apply preprocessing

    image_tensor = transform(image)

    # Add batch dimension

    image_tensor = image_tensor.unsqueeze(0)

    # Move to CPU/GPU

    image_tensor = image_tensor.to(DEVICE)

    # Disable gradients

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

    # Get prediction

    predicted_index = torch.argmax(
        probabilities,
        dim=1
    ).item()

    predicted_label = idx_to_label[
        predicted_index
    ]

    confidence = probabilities[
        0,
        predicted_index
    ].item()

    return (
        predicted_label,
        confidence,
        probabilities[0].cpu().numpy(),
        image_tensor
    )


# ============================================================
# 11. GRAD-CAM FUNCTION
# ============================================================

def generate_gradcam(
    image,
    image_tensor,
    target_class
):

    """
    Generate Grad-CAM visualization
    for the predicted class.
    """

    # Convert image

    rgb_image = image.convert("RGB")

    # Resize

    rgb_image = rgb_image.resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    # Convert to numpy

    rgb_array = np.array(
        rgb_image
    ).astype(
        np.float32
    ) / 255.0

    # --------------------------------------------------------
    # Select target layer
    # --------------------------------------------------------

    target_layers = [
        model.layer4[-1]
    ]

    # --------------------------------------------------------
    # Create GradCAM
    # --------------------------------------------------------

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    # --------------------------------------------------------
    # Specify target class
    # --------------------------------------------------------

    targets = [
        ClassifierOutputTarget(
            target_class
        )
    ]

    # --------------------------------------------------------
    # Generate heatmap
    # --------------------------------------------------------

    grayscale_cam = cam(
        input_tensor=image_tensor,
        targets=targets
    )

    grayscale_cam = grayscale_cam[0]

    # --------------------------------------------------------
    # Overlay heatmap
    # --------------------------------------------------------

    visualization = show_cam_on_image(
        rgb_array,
        grayscale_cam,
        use_rgb=True
    )

    return visualization


# ============================================================
# 12. SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Model Information")

    st.write(
        "**Architecture:** ResNet50"
    )

    st.write(
        "**Dataset:** HAM10000"
    )

    st.write(
        "**Image Size:** 224 × 224"
    )

    st.write(
        "**Explainability:** Grad-CAM"
    )

    st.write(
        f"**Device:** {DEVICE}"
    )

    st.divider()

    st.subheader(
        "Supported Classes"
    )

    for code, name in CLASS_NAMES.items():

        st.write(
            f"**{code}** — {name}"
        )


# ============================================================
# 13. IMAGE UPLOAD
# ============================================================

st.header("📤 Upload Skin Lesion Image")

uploaded_file = st.file_uploader(
    "Upload a JPG, JPEG or PNG image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# 14. PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # Open image
    # --------------------------------------------------------

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    # --------------------------------------------------------
    # Create columns
    # --------------------------------------------------------

    col1, col2 = st.columns(
        2
    )

    # --------------------------------------------------------
    # Display uploaded image
    # --------------------------------------------------------

    with col1:

        st.subheader(
            "📷 Uploaded Image"
        )

        st.image(
            image,
            use_container_width=True
        )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predicted_label, confidence, probabilities, image_tensor = (
        predict_image(image)
    )

    predicted_class_index = label_to_idx[
        predicted_label
    ]

    # --------------------------------------------------------
    # Prediction result
    # --------------------------------------------------------

    with col2:

        st.subheader(
            "🤖 AI Prediction"
        )

        st.markdown(
            '<div class="prediction-box">',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="prediction-label">'
            f'{CLASS_NAMES[predicted_label]}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="confidence">'
            f'Confidence: {confidence:.2%}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

        st.write("")

        # ----------------------------------------------------
        # Risk-style indicator
        # ----------------------------------------------------

        if predicted_label == "mel":

            st.error(
                "⚠️ Model prediction: Melanoma class"
            )

        elif predicted_label == "bcc":

            st.warning(
                "⚠️ Model prediction: Basal cell carcinoma"
            )

        else:

            st.success(
                "Model prediction generated successfully."
            )


    # ========================================================
    # 15. PROBABILITY DISTRIBUTION
    # ========================================================

    st.divider()

    st.header(
        "📊 Class Probability Distribution"
    )

    probability_data = []

    for index, probability in enumerate(
        probabilities
    ):

        label = idx_to_label[index]

        probability_data.append({

            "Class": label,

            "Description":
                CLASS_NAMES[label],

            "Probability":
                probability

        })


    probability_df = pd.DataFrame(
        probability_data
    )

    probability_df[
        "Probability (%)"
    ] = (
        probability_df[
            "Probability"
        ] * 100
    )


    # Sort descending

    probability_df = probability_df.sort_values(
        "Probability",
        ascending=False
    )


    # --------------------------------------------------------
    # Table
    # --------------------------------------------------------

    st.dataframe(
        probability_df[
            [
                "Class",
                "Description",
                "Probability (%)"
            ]
        ].style.format({
            "Probability (%)":
                "{:.2f}%"
        }),
        use_container_width=True
    )


    # --------------------------------------------------------
    # Bar chart
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.bar(
        probability_df["Class"],
        probability_df["Probability (%)"]
    )

    ax.set_xlabel(
        "Skin Lesion Class"
    )

    ax.set_ylabel(
        "Probability (%)"
    )

    ax.set_title(
        "HAM10000 Classification Probabilities"
    )

    ax.set_ylim(
        0,
        100
    )

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    st.pyplot(
        fig
    )


    # ========================================================
    # 16. GRAD-CAM EXPLANATION
    # ========================================================

    st.divider()

    st.header(
        "🔍 Explainable AI — Grad-CAM"
    )

    st.write(
        """
        Grad-CAM highlights the image regions that contributed
        most strongly to the model's prediction.
        """
    )

    with st.spinner(
        "Generating Grad-CAM explanation..."
    ):

        try:

            gradcam_image = generate_gradcam(
                image,
                image_tensor,
                predicted_class_index
            )

            st.image(
                gradcam_image,
                caption=(
                    "Grad-CAM visualization — "
                    "highlighted regions indicate areas "
                    "that influenced the prediction."
                ),
                use_container_width=True
            )

        except Exception as e:

            st.error(
                f"Grad-CAM generation failed: {e}"
            )


    # ========================================================
    # 17. MODEL INTERPRETATION
    # ========================================================

    st.divider()

    st.header(
        "🧠 Model Interpretation"
    )

    st.write(
        f"""
        The trained ResNet50 model classified the uploaded
        image as **{CLASS_NAMES[predicted_label]}** with a
        confidence of **{confidence:.2%}**.
        """
    )

    st.write(
        """
        The Grad-CAM visualization provides an explanation of
        which image regions were most influential for the
        predicted class.
        """
    )


# ============================================================
# 18. INITIAL SCREEN
# ============================================================

else:

    st.info(
        "👆 Upload a skin-lesion image to begin prediction."
    )

    st.markdown(
        """
        ### 🔬 Research Workflow

        **1. Upload Image**

        ↓

        **2. Image Preprocessing**

        Resize to 224 × 224 and apply ImageNet normalization.

        ↓

        **3. ResNet50 Classification**

        The fine-tuned ResNet50 predicts one of the seven
        HAM10000 diagnostic categories.

        ↓

        **4. Probability Analysis**

        The application displays probabilities for all
        seven classes.

        ↓

        **5. Grad-CAM**

        Grad-CAM highlights image regions that influenced
        the model's prediction.
        """
    )


# ============================================================
# 19. DISCLAIMER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="warning-box">

    <strong>⚠️ Research / Educational Disclaimer</strong>

    <br><br>

    This application is an academic research prototype.
    It is not a medical diagnostic system and should not
    be used to diagnose, treat, or make clinical decisions
    about skin conditions.

    Any prediction should be reviewed by a qualified
    healthcare professional.

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 20. FOOTER
# ============================================================

st.markdown(
    """
    <br>
    <center>
    <small>
    AI Skin Lesion Classification | HAM10000 | ResNet50 |
    Explainable AI using Grad-CAM
    </small>
    </center>
    """,
    unsafe_allow_html=True
)