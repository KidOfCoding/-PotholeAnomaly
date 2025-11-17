import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="Pothole Anomaly Detection",
    page_icon="🚧",
    layout="wide"
)

# -------------------------------------------------------
# CSS — Streamlit 1.51.0 Compatible
# -------------------------------------------------------
st.markdown("""
<style>

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0f0f, #1a1a1a);
    color: #e8e8e8 !important;
    font-family: 'Inter', sans-serif !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111 !important;
    border-right: 1px solid #333 !important;
}
[data-testid="stSidebar"] * {
    color: #ddd !important;
}

/* File uploader */
[data-testid="stFileUploaderDropzone"] {
    background-color: #111 !important;
    border: 2px dashed #444 !important;
    border-radius: 15px !important;
}

/* Tabs */
[data-testid="stTabs"] button {
    background-color: #1d1d1d !important;
    color: #aaa !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 10px 18px !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    background-color: #333 !important;
    color: #fff !important;
    border-bottom: 2px solid #888 !important;
}

/* Cards */
.card {
    padding: 22px;
    border-radius: 15px;
    background: #1d1d1d;
    border: 1px solid #333;
    box-shadow: 0 0 18px rgba(0,0,0,0.4);
}

/* Image styling */
.architecture-img {
    border-radius: 10px;
    border: 1px solid #444;
    margin: 10px 0;
}

.result-caption {
    text-align: center;
    font-weight: bold;
    margin-top: 10px;
    color: #fff;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# DEVICE & GLOBAL CONFIG
# -------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "attentive_autoencoder_clahe.pth"
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512

# -------------------------------------------------------
# MODEL ARCHITECTURE (EXACT TRAINING VERSION)
# -------------------------------------------------------
class CoordinateAttention(nn.Module):
    def __init__(self, inp, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, 1)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(mip, inp, 1)
        self.conv_w = nn.Conv2d(mip, inp, 1)

    def forward(self, x):
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0,1,3,2)

        y = torch.cat([x_h, x_w], 2)
        y = self.act(self.bn1(self.conv1(y)))

        x_h, x_w = torch.split(y, [h, w], dim=2)
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w.permute(0,1,3,2)))

        return x * a_h * a_w


class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class AttentiveUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.d1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.d2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.d3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.d4 = DoubleConv(256, 512)

        self.att1 = CoordinateAttention(64)
        self.att2 = CoordinateAttention(128)
        self.att3 = CoordinateAttention(256)

        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.u1 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.u2 = DoubleConv(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.u3 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, 3, 1)
        self.final_activation = nn.Sigmoid()

    def forward(self, x):
        c1 = self.d1(x); p1 = self.pool1(c1)
        c2 = self.d2(p1); p2 = self.pool2(c2)
        c3 = self.d3(p2); p3 = self.pool3(c3)
        c4 = self.d4(p3)

        x = self.up1(c4)
        x = torch.cat([self.att3(c3), x], 1)
        x = self.u1(x)

        x = self.up2(x)
        x = torch.cat([self.att2(c2), x], 1)
        x = self.u2(x)

        x = self.up3(x)
        x = torch.cat([self.att1(c1), x], 1)
        x = self.u3(x)

        x = self.final(x)
        return self.final_activation(x)

# -------------------------------------------------------
# HELPERS
# -------------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    model = AttentiveUNet().to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)
    model.eval()
    return model


def apply_clahe(img):
    clahe = cv2.createCLAHE(2.0, (8,8))
    arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    l2 = clahe.apply(l)
    lab = cv2.merge((l2,a,b))
    arr2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return Image.fromarray(cv2.cvtColor(arr2, cv2.COLOR_BGR2RGB))


def to_np(t):
    x = t.squeeze().cpu().permute(1,2,0).numpy()
    x = (x * 0.5) + 0.5
    return np.clip(x, 0, 1)


def run_inference(img, model, threshold):
    img_c = apply_clahe(img)
    t = TF.to_tensor(img_c)
    t = TF.resize(t, (512,512))
    t = TF.normalize(t, [0.5]*3, [0.5]*3).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        rec = model(t)

    orig = to_np(t)
    recon = to_np(rec)

    err = np.abs(orig - recon)
    gray = np.mean(err, 2)
    mask = (gray > threshold).astype(float)

    fig, ax = plt.subplots(1, 3, figsize=(19,6))

    ax[0].imshow(recon); ax[0].set_title("Reconstruction", color="white")
    ax[1].imshow(gray, cmap="jet"); ax[1].set_title("Error Heatmap", color="white")
    ax[2].imshow(mask, cmap="gray"); ax[2].set_title("Binary Mask", color="white")

    for a in ax: a.axis("off")

    fig.patch.set_facecolor("#1a1a1a")
    return fig

def run_inference_for_result(img, model, threshold):
    """Run inference and return the result figure for pre-loaded result images"""
    img_c = apply_clahe(img)
    t = TF.to_tensor(img_c)
    t = TF.resize(t, (512,512))
    t = TF.normalize(t, [0.5]*3, [0.5]*3).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        rec = model(t)

    orig = to_np(t)
    recon = to_np(rec)

    err = np.abs(orig - recon)
    gray = np.mean(err, 2)
    mask = (gray > threshold).astype(float)

    fig, ax = plt.subplots(1, 3, figsize=(15,5))

    ax[0].imshow(recon); ax[0].set_title("Reconstruction", color="white")
    ax[1].imshow(gray, cmap="jet"); ax[1].set_title("Error Heatmap", color="white")
    ax[2].imshow(mask, cmap="gray"); ax[2].set_title("Binary Mask", color="white")

    for a in ax: a.axis("off")

    fig.patch.set_facecolor("#1a1a1a")
    return fig

# -------------------------------------------------------
# HEADER
# -------------------------------------------------------
st.markdown("<h1 style='text-align:center;'>🚧 Attentive U-Net Pothole Anomaly Detection</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center;color:#ccc;'>Unsupervised Reconstruction Error Mapping</h3>", unsafe_allow_html=True)

# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------
st.sidebar.header("⚙ Settings")
threshold = st.sidebar.slider("Anomaly Threshold", 0.1, 0.5, 0.3, 0.01)

model = load_model()
if not model:
    st.error("❌ Model file not found! Place attentive_autoencoder_clahe.pth in this folder.")
    st.stop()

# -------------------------------------------------------
# TABS
# -------------------------------------------------------
tab_detect, tab_arch, tab_desc = st.tabs(["🔍 Detection", "🏗 Model Architecture", "ℹ Description & Results"])

# -------------------------------------------------------
# TAB 1 — Detection
# -------------------------------------------------------
with tab_detect:

    uploaded = st.file_uploader("Upload a road image", type=["jpg","jpeg","png"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.image(img, caption="Original Image", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            with st.spinner("🧠 Processing..."):
                fig = run_inference(img, model, threshold)
                st.pyplot(fig, use_container_width=True)
            st.success("Done!")
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("⬆ Upload an image to begin.")

# -------------------------------------------------------
# TAB 2 — Model Architecture
# -------------------------------------------------------
with tab_arch:
    st.header("Attentive U-Net Architecture")
    
    # Display the architecture images
    col1, col2 = st.columns(2)
    
    with col1:
        if os.path.exists("image.png"):
            st.image("image.png", 
                    caption="Attentive U-Net Architecture Diagram", 
                    use_container_width=True,
                    output_format="PNG")
        else:
            st.warning("Architecture diagram image (image.png) not found")
    
    with col2:
        if os.path.exists("Gemini_Generated_Image_nki7l0nki7l0nki7.png"):
            st.image("Gemini_Generated_Image_nki7l0nki7l0nki7.png", 
                    caption="Detailed Architecture Visualization", 
                    use_container_width=True,
                    output_format="PNG")
        else:
            st.warning("Gemini generated architecture image not found")

    st.subheader("Architecture Components")
    st.markdown("""
- **Encoder Path**: Progressive downsampling with DoubleConv blocks and MaxPool
- **Bottleneck**: High-level feature compression (512 channels)
- **Decoder Path**: Upsampling with ConvTranspose2D and skip connections
- **Coordinate Attention**: Spatial attention mechanisms on skip connections
- **Skip Connections**: Feature concatenation from encoder to decoder
- **Output**: Sigmoid activation for pixel-wise reconstruction
""")

# -------------------------------------------------------
# TAB 3 — Description + Results
# -------------------------------------------------------
with tab_desc:
    st.header("Project Description")
    st.markdown("""
This model is an **unsupervised anomaly detector** based on **Attentive U-Net Autoencoder**:

- **Training**: Trained on real-world road images without pothole annotations
- **Preprocessing**: CLAHE (Contrast Limited Adaptive Histogram Equalization) applied to emphasize texture
- **Learning Objective**: Learns to reconstruct normal road patterns
- **Anomaly Detection**: Potholes appear as high-error regions in reconstruction
- **Architecture**: U-Net with Coordinate Attention for better spatial feature learning
    """)

    st.header("Live Detection Results")
    st.markdown("Below are real-time inference results on sample images using the current model:")
    
    # Run inference on result images and display them
    result_images = [
        ("result_good.jpg", "High confidence detection (Threshold ~0.3)"),
        ("result_tune.jpg", "A smaller pothole, showing the importance of tuning the threshold (Try ~0.25)")
    ]
    
    # Create columns for results
    cols = st.columns(2)
    
    for idx, (img_path, caption) in enumerate(result_images):
        if os.path.exists(img_path):
            with cols[idx]:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                with st.spinner(f"Processing {img_path}..."):
                    # Load and process the image
                    img = Image.open(img_path).convert("RGB")
                    
                    # Display original image
                    st.image(img, caption="Original Image", use_container_width=True)
                    
                    # Run inference and display results
                    fig = run_inference_for_result(img, model, threshold)
                    st.pyplot(fig, use_container_width=True)
                    
                    # Display caption
                    st.markdown(f"<div class='result-caption'>{caption}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            with cols[idx]:
                st.warning(f"Missing: {img_path}")

    st.markdown("""
### Key Insights:
- **Threshold 0.3**: Works well for large, clear potholes with high reconstruction error
- **Threshold 0.25**: Better for detecting smaller or less prominent potholes
- **Adaptive Thresholding**: Consider using different thresholds based on road conditions and lighting
- **Coordinate Attention**: Improves spatial feature learning for better anomaly localization
- **Post-processing**: Morphological operations can clean up the binary mask for better visualization

### Performance Notes:
- The model processes images in real-time using the current threshold setting
- Results update automatically when you adjust the threshold in the sidebar
- Each result shows: Original → Reconstruction → Error Heatmap → Binary Mask
    """)
