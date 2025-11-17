## 🚧 PotholeAnomaly: Attentive U-Net for Unsupervised Road Defect Detection

### 📖 Description

**PotholeAnomaly** is an advanced deep learning system for unsupervised pothole detection using an Attentive U-Net autoencoder with Coordinate Attention mechanisms. This project provides a complete pipeline for identifying road anomalies without requiring labeled pothole data during training.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-green)

### 🎯 Key Features

- **🚀 Unsupervised Learning**: Trained only on normal road images - no pothole annotations needed
- **🧠 Advanced Architecture**: Attentive U-Net with Coordinate Attention for enhanced feature learning
- **🎨 Interactive Web App**: Streamlit-based interface for real-time inference and visualization
- **⚡ CLAHE Preprocessing**: Contrast Limited Adaptive Histogram Equalization for better texture emphasis
- **📊 Multi-View Results**: Displays reconstruction, error heatmap, and binary mask simultaneously
- **🎛️ Adjustable Threshold**: Dynamic anomaly detection sensitivity control

### 🏗️ Model Architecture

```
Input (3×512×512) → Encoder (Downsampling) → Bottleneck → Decoder (Upsampling) → Output (3×512×512)
                      ↑         ↑              ↑              ↑
          Coordinate Attention ←┘              └─ Skip Connections
```

**Core Components:**
- **DoubleConv Blocks**: 2×(Conv → BN → ReLU) for feature extraction
- **Coordinate Attention**: Spatial attention mechanisms on skip connections
- **Encoder-Decoder**: Symmetric U-Net architecture with skip connections
- **Sigmoid Output**: Pixel-wise reconstruction for anomaly detection

### 🛠️ Installation & Usage

```bash
# Clone repository
git clone https://github.com/yourusername/PotholeAnomaly.git
cd PotholeAnomaly

# Install dependencies
pip install -r requirements.txt

# Run the web application
streamlit run app.py
```

### 📁 Project Structure

```
PotholeAnomaly/
├── app.py                 # Main Streamlit application
├── attentive_autoencoder_clahe.pth  # Pre-trained model weights
├── requirements.txt       # Python dependencies
├── result_good.jpg       # Example detection result 1
├── result_tune.jpg       # Example detection result 2
├── image.png            # Architecture diagram
└── README.md            # Project documentation
```

### 🎮 Quick Start

1. **Launch the app**: `streamlit run app.py`
2. **Upload a road image** through the web interface
3. **Adjust the threshold** (0.1-0.5) for detection sensitivity
4. **View results** in three panels: Reconstruction, Error Heatmap, Binary Mask

### 📈 Performance Highlights

- **High Confidence Detection**: Clear potholes detected at threshold ~0.3
- **Subtle Anomaly Detection**: Smaller defects visible at threshold ~0.25
- **Real-time Processing**: Fast inference on GPU/CPU environments
- **Robust Preprocessing**: CLAHE enhancement improves detection accuracy

### 🎯 Use Cases

- **Municipal Road Maintenance**: Automated pothole identification for city infrastructure
- **Autonomous Vehicles**: Road condition awareness for self-driving systems
- **Insurance Claims**: Objective evidence for road damage claims
- **Research & Education**: Benchmark for unsupervised anomaly detection methods

### 🤝 Contributing

We welcome contributions! Please feel free to submit issues, fork the repository, and create pull requests for any improvements.

### 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

### 🙏 Acknowledgments

- Inspired by U-Net architecture and attention mechanisms in computer vision
- Built with PyTorch and Streamlit frameworks
- CLAHE preprocessing for enhanced texture analysis

---

**⭐ Star this repo if you find it helpful for your road anomaly detection projects!**

---

This GitHub description provides:

1. **Eye-catching title** with emoji and clear purpose
2. **Comprehensive feature list** highlighting technical strengths
3. **Clear architecture explanation** with visual diagram
4. **Easy installation instructions**
5. **Practical use cases** for real-world applications
6. **Professional badges** for Python version, frameworks, and license
7. **Engaging call-to-action** for contributors and stargazers
8. **Well-structured sections** for easy navigation
