# Image Tampering Detection using ELA and CNN

This project detects digitally tampered images using **Error Level Analysis (ELA)** combined with a **Convolutional Neural Network (CNN)**. The model is trained on the **CASIA 2.0** dataset, a well-known benchmark in digital image forensics.

---

## 📦 Dataset

- **Source**: [CASIA 2.0 Dataset](https://www.kaggle.com/datasets/divg07/casia-20-image-tampering-detection-dataset)
- **Contents**: Authentic and tampered images with corresponding labels.
- **Download**: Requires [Kaggle API key](https://www.kaggle.com/docs/api) (`kaggle.json`) placed at `/root/.kaggle/kaggle.json`.

---

## 🧠 Methodology

### 🔍 Error Level Analysis (ELA)
- Detects tampering by analyzing compression discrepancies.
- Steps:
  1. Save the image at a known JPEG quality.
  2. Subtract the resaved image from the original.
  3. Brighten the difference for visibility.

### 🧰 Model Architecture
- **Input Shape**: 128×128 grayscale (from ELA)
- **Layers**:
  - 2 × Conv2D + ReLU
  - MaxPooling + Dropout
  - Dense + Dropout
  - Output: Softmax (2 classes)

### 🔧 Training Details
- Optimizer: RMSprop
- Loss: Categorical Crossentropy
- Early Stopping & Learning Rate Reduction on Plateau
- Epochs: 30 | Batch Size: 100

---

## 📊 Evaluation

- **Metrics**: Accuracy, Loss, Confusion Matrix
- Visualization: Loss & Accuracy curves, Confusion Matrix heatmap

---

## 💾 Output

- Trained model saved as: `tamper_detection_model.h5`

---

## 🚀 How to Run

1. Install requirements:
    ```bash
    pip install numpy pandas matplotlib seaborn pillow scikit-learn keras kaggle
    ```

2. Place your `kaggle.json` in the correct path:
    ```bash
    mkdir -p ~/.kaggle
    cp /path/to/kaggle.json ~/.kaggle/
    chmod 600 ~/.kaggle/kaggle.json
    ```

3. Run the main script.

---

## 📈 Example Results

Plots of training vs. validation accuracy and loss, and a confusion matrix will be shown after training.

---

## 🛠️ Future Improvements

- Add model explainability (e.g., Grad-CAM, SHAP)
- Test with different image manipulation techniques
- Deploy as a web app using Flask or Streamlit

---

## 📌 Notes

- Ensure GPU usage if training time is a concern.
- Dataset path in CSV must be adjusted if directory changes.

---
