
# Traffic Sign Recognition System

This repository implements a traffic sign classification system using the German Traffic Sign Recognition Benchmark (GTSRB). It includes data loading, preprocessing, visualization, and training a deep learning model with TensorFlow/Keras. As the second phase of development the YOLO model is being used.

## 📌 Features

- Load and split custom GTSRB dataset (`labels.csv`, `Traffic_Data` folder)
- Preprocess images: grayscale conversion, histogram equalization, normalization
- Visualize sample images and label distribution
- Train CNN model to classify traffic signs
- Evaluate model performance

## 📂 Dataset Structure

Make sure your dataset is organized as follows:

```
Traffic_Data/
    ├── 0/
    │   ├── img1.png
    │   ├── img2.png
    ├── 1/
    │   ├── img3.png
    ...
labels.csv
```

The `labels.csv` should include image file paths and their corresponding class labels.

## 🧪 Preprocessing Pipeline

1. Convert to grayscale
2. Apply histogram equalization
3. Normalize pixel values to [0, 1]
4. Resize to 32x32x1

## 🛠️ Tech Stack

- Python 3.10+
- TensorFlow / Keras
- OpenCV
- NumPy, Pandas, Matplotlib

## 🚀 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the main program
python main.py
```
## Results Graphs

Accuracy for 32 x 32 size images wit 20 epochs
<img width="640" height="480" alt="Accuracy_20_epoch_32" src="https://github.com/user-attachments/assets/6d64b3c1-2915-4d0a-82bf-4cba769e16b0" />

Accuracy for 64 x 64 size images wit 20 epochs
<img width="640" height="480" alt="Accuracy_20_epoch_64" src="https://github.com/user-attachments/assets/5ad38811-0701-43a1-86a6-90fb4da303ba" />

Model Summary

<img width="677" height="439" alt="model_32" src="https://github.com/user-attachments/assets/dc815ddf-3335-40ea-81f6-561de119edbc" />

## 🧠 Future Improvements

- Add support for more signs using transfer learning
- Improve accuracy with data augmentation and better CNN architecture
- Deploy model using Flask / Streamlit

## 📄 License

This project is licensed under the MIT License.
