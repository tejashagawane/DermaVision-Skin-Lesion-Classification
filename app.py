import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# ------------------------- Page setup -------------------------
st.set_page_config(
    page_title="DermaVision | Skin Lesion Research",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
MODEL_PATH = os.path.join("models", "ham10000_resnet50_best.pth")

CLASS_NAMES = {
    "akiec": "Actinic keratoses / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}
DEFAULT_LABEL_TO_IDX = {
    "akiec": 0, "bcc": 1, "bkl": 2, "df": 3,
    "mel": 4, "nv": 5, "vasc": 6
}

# ------------------------- Styling -------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #f6f8fc; color: #172033; }
.block-container { max-width: 1240px; padding-top: 2rem; padding-bottom: 3rem; }
.hero {
  background: linear-gradient(115deg, #14213d 0%, #203d69 58%, #286b78 100%);
  padding: 2rem 2.2rem; border-radius: 24px; color: white;
  box-shadow: 0 14px 35px rgba(20,33,61,.14); margin-bottom: 1.25rem;
}
.eyebrow { text-transform: uppercase; letter-spacing: .16em; font-size: .72rem;
  font-weight: 700; color: #a9d8e2; margin-bottom: .55rem; }
.hero h1 { font-family: 'Manrope', sans-serif; font-size: clamp(2rem, 4vw, 3rem);
  line-height: 1.1; margin: 0 0 .65rem 0; color: white; font-weight: 800; }
.hero p { color: #d9e6f5; margin: 0; font-size: 1rem; max-width: 720px; }
.section-kicker { color: #287c83; text-transform: uppercase; letter-spacing: .12em;
  font-size: .72rem; font-weight: 700; margin-bottom: .25rem; }
.section-title { font-family: 'Manrope', sans-serif; font-size: 1.45rem;
  font-weight: 800; color: #172033; margin: 0 0 .75rem 0; }
.panel {
  background: white; border: 1px solid #e7ebf2; border-radius: 18px;
  padding: 1.15rem 1.25rem; box-shadow: 0 5px 18px rgba(24,39,75,.045);
}
.muted { color: #64748b; font-size: .9rem; }
.result-label { color: #64748b; font-size: .78rem; text-transform: uppercase;
  letter-spacing: .08em; font-weight: 700; }
.result-value { font-family: 'Manrope', sans-serif; font-size: 1.65rem;
  font-weight: 800; color: #172033; }
.pill {
  display: inline-block; border-radius: 999px; padding: .28rem .65rem;
  background: #e7f5f3; color: #17666b; font-size: .75rem; font-weight: 700;
}
.disclaimer {
  border: 1px solid #f3d59a; background: #fff8e8; color: #805313;
  border-radius: 12px; padding: .8rem 1rem; font-size: .86rem;
}
div[data-testid="stFileUploader"] {
  background: white; border: 1.5px dashed #9bb8c5; border-radius: 16px;
  padding: .7rem;
}
.stButton button[kind="primary"] {
  background: #176b72; border: 0; border-radius: 10px; font-weight: 700;
}
.stButton button[kind="primary"]:hover { background: #12565c; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ------------------------- Model loading -------------------------
@st.cache_resource(show_spinner="Loading the ResNet50 model...")
def load_model():
    if not os.path.isfile(MODEL_PATH):
        return None, None, None, None
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    label_to_idx = checkpoint.get("label_to_idx", DEFAULT_LABEL_TO_IDX)
    raw_idx = checkpoint.get("idx_to_label")
    idx_to_label = (
        {int(k): v for k, v in raw_idx.items()}
        if raw_idx is not None
        else {int(v): k for k, v in label_to_idx.items()}
    )
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(label_to_idx))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE).eval()
    return model, label_to_idx, idx_to_label, checkpoint

model, label_to_idx, idx_to_label, checkpoint = load_model()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

def predict(image):
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
    predicted_index = int(np.argmax(probabilities))
    return tensor, predicted_index, probabilities

def make_gradcam(image, tensor, target_index):
    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
    target = [ClassifierOutputTarget(int(target_index))]
    heatmap = cam(input_tensor=tensor, targets=target)[0]
    rgb = np.asarray(image.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
    overlay = show_cam_on_image(rgb, heatmap, use_rgb=True)
    return heatmap, overlay

# ------------------------- Header -------------------------
st.markdown("""
<div class="hero">
  <div class="eyebrow">AI-assisted dermatology research</div>
  <h1>DermaVision</h1>
  <p>Skin-lesion image classification with ResNet50 and explainable AI.
  Explore model predictions and visual explanations in one workspace.</p>
</div>
""", unsafe_allow_html=True)

# Compact status strip
status_text = "Model ready" if model is not None else "Model file missing"
status_color = "#dff4ed" if model is not None else "#fff0ed"
status_fg = "#17634e" if model is not None else "#a33b2d"
st.markdown(
    f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:1rem">'
    f'<span class="pill" style="background:{status_color};color:{status_fg}">● {status_text}</span>'
    f'<span class="pill">ResNet50</span><span class="pill">HAM10000 · 7 classes</span>'
    f'<span class="pill">Grad-CAM</span>'
    f'<span class="pill">Device: {str(DEVICE).upper()}</span></div>',
    unsafe_allow_html=True
)

if model is None:
    st.error("The trained model checkpoint could not be found.")
    st.write(f"Expected location: `{MODEL_PATH}`")
    st.info("Place `ham10000_resnet50_best.pth` inside the `models` folder, then restart Streamlit.")
    st.stop()

# ------------------------- Sidebar -------------------------
with st.sidebar:
    st.markdown("## Workspace")
    st.caption("Research model configuration")
    st.write("**Architecture:** ResNet50")
    st.write("**Dataset:** HAM10000")
    st.write("**Input size:** 224 × 224")
    st.write(f"**Compute:** {DEVICE}")
    st.divider()
    st.markdown("### Supported categories")
    for code, name in CLASS_NAMES.items():
        st.caption(f"**{code.upper()}** · {name}")
    st.divider()
    st.caption("For research and educational use only.")

# ------------------------- Upload section -------------------------
st.markdown('<div class="section-kicker">01 / Image input</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Upload a lesion image</div>', unsafe_allow_html=True)
left, right = st.columns([1.25, .75], gap="large")
with left:
    uploaded = st.file_uploader(
        "Choose a JPG, JPEG or PNG image",
        type=["jpg", "jpeg", "png"],
        help="Use a clear dermoscopic image. The image is processed locally by this app."
    )
with right:
    st.markdown(
        '<div class="panel"><div class="result-label">Analysis pipeline</div>'
        '<p style="margin:.55rem 0 .2rem"><b>1.</b> Image preprocessing</p>'
        '<p style="margin:.2rem 0"><b>2.</b> ResNet50 classification</p>'
        '<p style="margin:.2rem 0"><b>3.</b> Class probability scores</p>'
        '<p style="margin:.2rem 0"><b>4.</b> Grad-CAM explanation</p>'
        '<div class="muted" style="margin-top:.55rem">224 × 224 · RGB · ImageNet normalization</div></div>',
        unsafe_allow_html=True
    )

st.markdown('<div class="disclaimer">⚠️ <b>Research prototype:</b> This tool is not a medical device. Model outputs are not diagnoses and must not guide treatment or delay professional medical care.</div>', unsafe_allow_html=True)

if uploaded is None:
    st.markdown("")
    st.markdown('<div class="panel" style="text-align:center;padding:2rem">'
                '<div style="font-size:2rem">🖼️</div><h3 style="margin:.4rem 0">Your analysis will appear here</h3>'
                '<p class="muted">Upload an image above to view the prediction, class distribution and Grad-CAM explanation.</p>'
                '</div>', unsafe_allow_html=True)
    st.stop()

try:
    image = Image.open(uploaded).convert("RGB")
except Exception:
    st.error("Unable to read this image. Please upload a valid JPG or PNG file.")
    st.stop()

tensor, predicted_index, probabilities = predict(image)
predicted_code = idx_to_label[predicted_index]
predicted_name = CLASS_NAMES.get(predicted_code, predicted_code)
confidence = float(probabilities[predicted_index])

# ------------------------- Results -------------------------
st.markdown('<div class="section-kicker">02 / Model output</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Analysis overview</div>', unsafe_allow_html=True)

image_col, result_col = st.columns([.9, 1.1], gap="large")
with image_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.image(image, caption="Uploaded image", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with result_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="result-label">Top predicted category</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="result-value">{predicted_code.upper()}</div>', unsafe_allow_html=True)
    st.write(predicted_name)
    st.progress(confidence, text=f"Model score: {confidence:.1%}")
    st.caption("A softmax score is not a calibrated clinical probability.")
    st.divider()
    top_indices = np.argsort(probabilities)[::-1][:3]
    metric_cols = st.columns(3)
    for col, idx in zip(metric_cols, top_indices):
        with col:
            code = idx_to_label[int(idx)]
            st.metric(code.upper(), f"{probabilities[idx]:.1%}")
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------- Detail tabs -------------------------
st.markdown("")
tab_prediction, tab_explain, tab_about = st.tabs(
    ["📊 Class distribution", "🔥 Explainability", "ℹ️ About this model"]
)

with tab_prediction:
    st.markdown("#### Scores across all seven categories")
    rows = [{
        "Code": idx_to_label[i],
        "Category": CLASS_NAMES.get(idx_to_label[i], idx_to_label[i]),
        "Score": float(probabilities[i])
    } for i in range(len(probabilities))]
    scores = pd.DataFrame(rows).sort_values("Score", ascending=False)
    chart = px.bar(
        scores, x="Score", y="Code", orientation="h",
        hover_data={"Category": True, "Score": ":.1%"},
        labels={"Score": "Model score", "Code": "Category"},
        color="Score", color_continuous_scale=["#c8e6e8", "#176b72"]
    )
    chart.update_layout(
        height=360, margin=dict(l=10, r=10, t=20, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
        coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"},
        xaxis_tickformat=".0%"
    )
    st.plotly_chart(chart, use_container_width=True)
    display_scores = scores.copy()
    display_scores["Score"] = display_scores["Score"].map(lambda x: f"{x:.2%}")
    st.dataframe(display_scores, use_container_width=True, hide_index=True)

with tab_explain:
    st.markdown("#### Visual explanation with Grad-CAM")
    st.write("Select a category to inspect which image regions influenced that class score.")
    explain_options = [
        f"{idx_to_label[i].upper()} — {CLASS_NAMES.get(idx_to_label[i], idx_to_label[i])}"
        for i in range(len(idx_to_label))
    ]
    default_option = next(
        (i for i, item in enumerate(explain_options)
         if item.startswith(predicted_code.upper())),
        0
    )
    selected_option = st.selectbox(
        "Category to explain", explain_options, index=default_option
    )
    target_index = explain_options.index(selected_option)
    if st.button("Generate visual explanation", type="primary"):
        with st.spinner("Generating Grad-CAM heatmap..."):
            try:
                heatmap, overlay = make_gradcam(image, tensor, target_index)
                heat_col1, heat_col2 = st.columns(2)
                with heat_col1:
                    st.image(heatmap, caption="Activation heatmap", use_container_width=True)
                with heat_col2:
                    st.image(overlay, caption="Heatmap over original image", use_container_width=True)
                st.success("Grad-CAM explanation generated.")
            except Exception as exc:
                st.error(f"Could not generate Grad-CAM: {exc}")
    st.caption("Grad-CAM visualizes model sensitivity for a selected class; it does not identify a medically causal region.")

with tab_about:
    st.markdown("#### Model details")
    detail_col1, detail_col2 = st.columns(2)
    with detail_col1:
        st.markdown("**Architecture**")
        st.write("ResNet50 convolutional neural network")
        st.markdown("**Training dataset**")
        st.write("HAM10000")
        st.markdown("**Output categories**")
        st.write("Seven diagnostic labels")
    with detail_col2:
        st.markdown("**Input pipeline**")
        st.write("RGB · 224 × 224 · ImageNet normalization")
        st.markdown("**Explainability**")
        st.write("Gradient-weighted Class Activation Mapping (Grad-CAM)")
        st.markdown("**Runtime device**")
        st.write(str(DEVICE))
    st.info("Performance depends on the quality of the training, validation and test process. Validate the model independently before presenting any performance claims.")

st.markdown("---")
st.caption("DermaVision · Academic research prototype · HAM10000 / ResNet50 / Grad-CAM")
