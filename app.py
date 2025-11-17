import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance, ImageFilter
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
# EXACT ORIGINAL MODEL ARCHITECTURE
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
# HELPERS (No OpenCV)
# -------------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model file not found! Please upload 'attentive_autoencoder_clahe.pth'")
        return None
    try:
        model = AttentiveUNet().to(DEVICE)
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Load state dict with strict=False to handle minor mismatches
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("💡 Try using strict=False loading...")
        try:
            model = AttentiveUNet().to(DEVICE)
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            st.success("✅ Model loaded with strict=False!")
            return model
        except Exception as e2:
            st.error(f"❌ Failed to load model even with strict=False: {str(e2)}")
            return None

def enhance_contrast_pil(img):
    """Simple contrast enhancement using PIL to simulate CLAHE"""
    # Convert to grayscale for contrast enhancement
    gray = img.convert('L')
    enhancer = ImageEnhance.Contrast(gray)
    enhanced_gray = enhancer.enhance(2.0)  # Strong contrast enhancement
    
    # Merge back with original color
    enhanced_rgb = Image.merge('RGB', [enhanced_gray] * 3)
    return enhanced_rgb

def to_np(tensor):
    """Convert tensor to numpy image"""
    img = tensor.squeeze().cpu().permute(1, 2, 0).numpy()
    img = (img * 0.5) + 0.5  # Denormalize
    return np.clip(img, 0, 1)

def run_inference(image, model, threshold):
    """Run inference on image"""
    try:
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
        
        return fig, error_gray.max()
    
    except Exception as e:
        st.error(f"❌ Inference error: {str(e)}")
        # Return a simple error visualization
        fig, axes = plt.subplots(1, 1, figsize=(10, 5))
        axes.text(0.5, 0.5, f"Inference Error: {str(e)}", 
                 ha='center', va='center', transform=axes.transAxes, color='red')
        axes.axis('off')
        fig.patch.set_facecolor('#1a1a1a')
        return fig, 0.0

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
                result_fig, max_error = run_inference(image, model, threshold)
                st.pyplot(result_fig, use_container_width=True)
                st.info(f"📊 Maximum reconstruction error: {max_error:.4f}")
            st.success("✅ Analysis complete!")
            st.markdown("</div>", unsafe_allow_html=True)
    elif uploaded_file and not model:
        st.error("❌ Cannot process image - model failed to load")

with tab2:
    st.header("Model Architecture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if os.path.exists("image.png"):
            st.image("image.png", caption="Attentive U-Net Architecture", use_container_width=True)
        else:
            st.info("Architecture diagram image not found")
    
    with col2:
        st.markdown("""
        **Architecture Features:**
        - **Encoder-decoder** with skip connections
        - **Coordinate Attention** mechanisms
        - **DoubleConv blocks** for feature extraction
        - **Unsupervised** anomaly detection
        - **CLAHE-like** preprocessing
        """)

with tab3:
    st.header("Sample Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Clear Detection")
        if os.path.exists("result_good.jpg"):
            st.image("result_good.jpg", caption="High confidence pothole detection (Threshold ~0.3)", use_container_width=True)
        else:
            st.info("Sample result image not found")
    
    with col2:
        st.subheader("Subtle Defects")
        if os.path.exists("result_tune.jpg"):
            st.image("result_tune.jpg", caption="Smaller defects require threshold tuning (~0.25)", use_container_width=True)
        else:
            st.info("Sample result image not found")

# Footer
st.markdown("---")
st.caption("PotholeAnomaly - Unsupervised Road Defect Detection System")
