# E-Waste Image Classification Using EfficientNetV2B0

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
- **Input Size**: 128x128 pixels
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
```

### Model Architecture
- **Input Layer**: 128x128x3 images
- **Base Model**: EfficientNetV2B0 (pre-trained on ImageNet)
- **Custom Head**: Dense layers for classification
- **Output**: 10 classes with softmax activation

### Training Pipeline
- **Optimizer**: Adam with adaptive learning rate
- **Loss Function**: Categorical crossentropy
- **Metrics**: Accuracy, precision, recall
- **Callbacks**: Early stopping, model checkpointing

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

### Gradio Web Interface
The project includes a user-friendly web interface built with Gradio:
- **Image Upload**: Easy drag-and-drop functionality
- **Real-time Classification**: Instant prediction results
- **Confidence Scores**: Probability distribution across classes
- **Batch Processing**: Handle multiple images simultaneously

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

## 🔍 Key Functions

### `plot_class_distribution(dataset, title)`
Visualizes the distribution of samples across different e-waste categories to identify class imbalances.

**Parameters:**
- `dataset`: TensorFlow dataset object
- `title`: Plot title for identification

**Purpose:** Ensures balanced training and identifies potential bias in the dataset.

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
