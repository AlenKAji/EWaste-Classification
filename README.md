# E-Waste Image Classification Using EfficientNetV2B0(Modified to MobileNetV3Large)

## 🌍 Project Overview

This project addresses the critical environmental challenge of electronic waste (e-waste) management through automated image classification. With e-waste becoming one of the fastest-growing waste streams globally, proper sorting and categorization are essential for efficient recycling and environmental protection.

Our solution leverages deep learning to automatically classify e-waste items into distinct categories, enabling automated sorting systems that can significantly improve recycling efficiency and reduce environmental impact.

## 🎯 Problem Statement

E-waste contains valuable materials that can be recovered and reused, but also hazardous substances that require careful handling. Manual sorting is:
- **Labor-intensive** and costly
- **Error-prone** leading to contamination
- **Slow** and inefficient for large-scale operations
- **Inconsistent** across different workers and facilities

## 🔧 Technical Solution

### Model Architecture
- **Base Model**: EfficientNetV2B0 (transfer learning)
- **Input Size**: 128x128 pixels updated to 224x224 pixels
- **Categories**: 10 distinct e-waste types
- **Batch Size**: 32
- **Framework**: TensorFlow/Keras

### Key Features
- **Transfer Learning**: Leverages pre-trained EfficientNetV2B0 for better performance with limited data
- **Data Augmentation**: Implements robust preprocessing pipeline
- **Visualization**: Comprehensive data analysis and model evaluation tools
- **Web Interface**: Gradio-based deployment for easy testing and demonstration

## 📊 Dataset Structure

```
data01/
└──data/
    ├── train/          # Training images
    ├── val/            # Validation images
    └── test/           # Test images
          ├── category_1/
          ├── category_2/
          ├── ...
          └── category_10/


```

## 🚀 Getting Started

### Prerequisites
```bash
pip install tensorflow
pip install gradio
pip install matplotlib
pip install seaborn
pip install scikit-learn
pip install pillow
pip install numpy
pip install gTTs
```

### Installation & Setup

1. **Use the .pyinb file in Google Colab**

2. **Prepare your dataset**
```python
# Extract the dataset
import zipfile
import os

zip_path = "path/to/your/data.zip"
extract_dir = "data"

os.makedirs(extract_dir, exist_ok=True)
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
```

3. **Run the training script** <br>
        -run all cells in colab


## 💻 Code Structure

### Data Loading & Preprocessing
```python
# Load datasets with appropriate preprocessing
datatrain = tf.keras.utils.image_dataset_from_directory(
    trainpath, 
    shuffle=True, 
    image_size=(128,128), 
    batch_size=32
)
similarly for datatest and data validation

#data preprocessing
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

datatrain = datatrain.map(lambda x, y: (preprocess_input(x), y))
datavalid = datavalid.map(lambda x, y: (preprocess_input(x), y))

AUTOTUNE = tf.data.AUTOTUNE
datatrain = datatrain.cache().prefetch(buffer_size=AUTOTUNE)
datavalid = datavalid.cache().prefetch(buffer_size=AUTOTUNE)
```

### Model Architecture
- **Input Layer**: 128x128x3 images
- **Base Model**: EfficientNetV2B0 (pre-trained on ImageNet) modified to MobileNetV3Large
- **Custom Head**: Dense layers for classification
- **Output**: 10 classes with softmax activation

### Training Pipeline
- **Optimizer**: Adam with adaptive learning rate
- **Loss Function**: Categorical crossentropy
- **Metrics**: Accuracy, precision, recall
- **Callbacks**: Early stopping, model checkpointing, Learning rate schedule

## 📈 Model Performance

### Evaluation Metrics
- **Accuracy**: Overall classification accuracy
- **Precision/Recall**: Per-class performance metrics
- **Confusion Matrix**: Detailed classification results
- **F1-Score**: Balanced performance measure

### Visualization Tools
- **Class Distribution**: Analyze dataset balance
- **Sample Images**: Visualize training data
- **Training History**: Monitor model performance
- **Confusion Matrix**: Evaluate classification errors

## 🌐 Deployment

### 🌐 Gradio Web Interface

The project features a clean and accessible web UI powered by **Gradio**:

- 📤 **Image Upload**: Supports drag-and-drop and webcam capture for fast input
- ⚡ **Real-time Classification**: Instant predictions using a MobileNetV3-based model
- 📊 **Confidence Scores**: Displays class probabilities with percentage confidence
- 🔄 **Quantity Input**: Calculates CO₂ savings based on the number of items
- ♻️ **Recycling Guidance**: Provides responsible disposal instructions per e-waste type
- 🔊 **Voice Output**: Uses TTS for audio feedback (via gTTS or pyttsx3)
- 🚀 **Deployment-Ready**: Designed for easy hosting on Hugging Face Spaces or locally

> This interface makes the AI system accessible to both end-users and field workers, promoting sustainability and awareness.

### Usage Example
```python
import gradio as gr
from tensorflow.keras.models import load_model

# Load trained model
model = load_model('ewaste_classifier.h5')

# Create Gradio interface
interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3)
)

interface.launch()
```
### 🔑 Key Functions

This project goes beyond basic classification by integrating meaningful features for real-world e-waste management:

- 🧠 **Image Classification**  
  Identifies e-waste items across 10 categories using a MobileNetV3-based deep learning model.

- 📋 **Recycling Instruction Generator**  
  Provides safe and environmentally friendly disposal methods for each identified item.

- 🌱 **CO₂ Savings Estimator**  
  Calculates estimated carbon savings based on the quantity of e-waste diverted from landfill.

- 🔊 **Voice Assistant (TTS)**  
  Speaks out the predicted category and recycling advice, improving accessibility and hands-free operation.

- 🧑‍💻 **Interactive Web Interface (Gradio)**  
  Enables real-time classification, guidance, and feedback through an intuitive, browser-based interface.



## 🎯 Industry Applications

### Manufacturing & Recycling
- **Automated Sorting**: Integration with robotic sorting systems
- **Quality Control**: Consistent classification standards
- **Inventory Management**: Automatic categorization of incoming e-waste

### Environmental Compliance
- **Regulatory Reporting**: Automated waste stream documentation
- **Hazardous Material Identification**: Proper handling protocols
- **Recycling Optimization**: Maximize material recovery rates

### Smart City Initiatives
- **Waste Collection**: Optimize pickup routes and scheduling
- **Public Awareness**: Educational tools for proper e-waste disposal
- **Data Analytics**: Trends and patterns in e-waste generation

## 📋 Future Enhancements

### Technical Improvements
- **Multi-modal Learning**: Combine image data with metadata
- **Edge Deployment**: Optimize for mobile and IoT devices
- **Real-time Processing**: Video stream classification
- **Active Learning**: Continuous model improvement

### Functionality Extensions
- **Material Composition**: Identify specific materials within devices
- **Condition Assessment**: Evaluate repair vs. recycling potential
- **Value Estimation**: Economic analysis of recoverable materials
- **Predictive Maintenance**: Forecast equipment lifecycle

## 🤝 Contributing

We welcome contributions to improve this project:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Contribution Areas
- **Dataset Expansion**: Add new e-waste categories
- **Model Optimization**: Improve accuracy and efficiency
- **Deployment Tools**: Create production-ready solutions
- **Documentation**: Enhance user guides and tutorials

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **EfficientNet Team**: For the pre-trained model architecture
- **TensorFlow Community**: For the robust ML framework
- **Environmental Organizations**: For highlighting the importance of e-waste management
- **Open Source Community**: For tools and libraries that made this project possible

## 📞 Contact & Support

For questions, suggestions, or collaboration opportunities:
- **Email**: alenkaji12@gmail.com
  

---

## 🌟 Project Impact

This project contributes to the UN Sustainable Development Goals, particularly:
- **Goal 12**: Responsible Consumption and Production
- **Goal 13**: Climate Action
- **Goal 14**: Life Below Water (reducing ocean pollution)

By automating e-waste classification, we're taking a crucial step toward a more sustainable future and circular economy.

---

*Built with ❤️ for a sustainable future*
