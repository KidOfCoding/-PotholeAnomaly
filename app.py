import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance
import numpy as np
import os
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
# CSS
# -------------------------------------------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0f0f, #1a1a1a);
    color: #e8e8e8;
}
[data-testid="stSidebar"] {
    background-color: #111;
}
.card {
    padding: 20px;
    border-radius: 15px;
    background: #1d1d1d;
    border: 1px solid #333;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# DEVICE & CONFIG
# -------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "attentive_autoencoder_clahe.pth"

# -------------------------------------------------------
# SIMPLIFIED MODEL ARCHITECTURE
# -------------------------------------------------------
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

class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(256, 512)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = DoubleConv(128, 64)
        
        # Output
        self.final = nn.Conv2d(64, 3, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        
        # Bottleneck
        b = self.bottleneck(self.pool3(e3))
        
        # Decoder with skip connections
        d1 = self.up1(b)
        d1 = torch.cat([e3, d1], dim=1)
        d1 = self.dec1(d1)
        
        d2 = self.up2(d1)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)
        
        d3 = self.up3(d2)
        d3 = torch.cat([e1, d3], dim=1)
        d3 = self.dec3(d3)
        
        return self.sigmoid(self.final(d3))

# -------------------------------------------------------
# HELPERS (No OpenCV)
# -------------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model file not found! Upload 'attentive_autoencoder_clahe.pth'")
        return None
    try:
        model = SimpleUNet().to(DEVICE)
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def enhance_contrast_pil(img):
    """Simple contrast enhancement using PIL"""
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(1.2)  # 20% contrast boost

def to_np(tensor):
    """Convert tensor to numpy image"""
    img = tensor.squeeze().cpu().permute(1, 2, 0).numpy()
    img = (img * 0.5) + 0.5  # Denormalize
    return np.clip(img, 0, 1)

def run_inference(image, model, threshold):
    """Run inference on image"""
    # Preprocess
    enhanced_img = enhance_contrast_pil(image)
    tensor = TF.to_tensor(enhanced_img)
    tensor = TF.resize(tensor, (512, 512))
    tensor = TF.normalize(tensor, [0.5]*3, [0.5]*3).unsqueeze(0).to(DEVICE)
    
    # Inference
    with torch.no_grad():
        reconstructed = model(tensor)
    
    # Process results
    original_np = to_np(tensor)
    reconstructed_np = to_np(reconstructed)
    
    # Calculate error
    error = np.abs(original_np - reconstructed_np)
    error_gray = np.mean(error, axis=2)
    mask = (error_gray > threshold).astype(np.float32)
    
    # Create visualization
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original
    axes[0].imshow(original_np)
    axes[0].set_title("Original", color='white', fontsize=12)
    axes[0].axis('off')
    
    # Reconstruction
    axes[1].imshow(reconstructed_np)
    axes[1].set_title("Reconstruction", color='white', fontsize=12)
    axes[1].axis('off')
    
    # Error Heatmap
    im = axes[2].imshow(error_gray, cmap='jet')
    axes[2].set_title("Error Heatmap", color='white', fontsize=12)
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    
    # Binary Mask
    axes[3].imshow(mask, cmap='gray')
    axes[3].set_title(f"Mask (Threshold: {threshold})", color='white', fontsize=12)
    axes[3].axis('off')
    
    fig.patch.set_facecolor('#1a1a1a')
    fig.tight_layout()
    
    return fig

# -------------------------------------------------------
# APP INTERFACE
# -------------------------------------------------------
st.title("🚧 Pothole Anomaly Detection")
st.markdown("Unsupervised deep learning model for road defect identification")

# Sidebar
st.sidebar.header("⚙ Settings")
threshold = st.sidebar.slider("Detection Threshold", 0.1, 0.5, 0.3, 0.01)
st.sidebar.markdown("---")
st.sidebar.info("**Tip:** Lower threshold for subtle defects, higher for clear potholes")

# Load model
model = load_model()

# Main interface
tab1, tab2, tab3 = st.tabs(["🔍 Detection", "🏗 Architecture", "📊 Results"])

with tab1:
    st.header("Upload & Detect")
    
    uploaded_file = st.file_uploader(
        "Choose a road image", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image of a road surface"
    )
    
    if uploaded_file and model:
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            with st.spinner("🔄 Analyzing image..."):
                result_fig = run_inference(image, model, threshold)
                st.pyplot(result_fig, use_container_width=True)
            st.success("✅ Analysis complete!")
            st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.header("Model Architecture")
    st.image("image.png", caption="Attentive U-Net Architecture", use_container_width=True)
    
    st.markdown("""
    **Architecture Features:**
    - Encoder-decoder with skip connections
    - Progressive downsampling and upsampling
    - Coordinate attention mechanisms
    - Unsupervised anomaly detection
    """)

with tab3:
    st.header("Sample Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Clear Detection")
        st.image("result_good.jpg", caption="High confidence pothole detection (Threshold ~0.3)", use_container_width=True)
    
    with col2:
        st.subheader("Subtle Defects")
        st.image("result_tune.jpg", caption="Smaller defects require threshold tuning (~0.25)", use_container_width=True)

# Footer
st.markdown("---")
st.caption("PotholeAnomaly - Unsupervised Road Defect Detection System")
