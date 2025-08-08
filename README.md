
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

## 🧠 Future Improvements

- Add support for more signs using transfer learning
- Improve accuracy with data augmentation and better CNN architecture
- Deploy model using Flask / Streamlit

## 📄 License

This project is licensed under the MIT License.
