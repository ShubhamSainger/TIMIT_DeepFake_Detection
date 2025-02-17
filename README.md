# Deepfake Detection using CNN-RNN Hybrid Model

## 📌 Project Overview
This project focuses on **deepfake video detection** using a **hybrid deep learning model** trained on the **TIMIT Deepfake Dataset**. The model effectively identifies manipulated face-swapped videos by leveraging **spatial feature extraction** and **temporal sequence learning**.

## 🚀 Key Features
- **Custom dataset creation** by combining real and fake videos
- **Face extraction and preprocessing** using **MTCNN**
- **Custom training data generator** for efficient batch processing
- **Hybrid model architecture** with **ResNet50 + RNN**
- **Frame sequence standardization** for uniform model input

## 🎥 Dataset
- **Fake Data:** Sourced from the **TIMIT Deepfake Dataset** ([Link](https://www.idiap.ch/en/scientific-research/data/deepfaketimit))
- **Real Data:** Downloaded separately from an VidTIMIT dataset ([Link](https://conradsanderson.id.au/vidtimit/)
- **Preprocessing:**
  - Detected face coordinates using **MTCNN**
  - Cropped and saved faces from each video frame in **.npz NumPy arrays**
  - Standardized frame sequence lengths for consistency

## 🛠️ Model Architecture
The model consists of two primary networks:
- **Feature Extractor:** Pretrained **ResNet50**, which extracts spatial features from video frames
- **Temporal Dependency Learner:** RNN-based architecture that learns sequential patterns in face movements
- **Final Classification:** A fully connected layer that classifies whether the video is real or fake


## 📈 Evaluation 
- **Validation Accuracy:** 0.9688
- **Training Accuracy:** 0.9941

## 🤝 Contributions
Feel free to contribute! Open an issue or create a pull request if you'd like to improve the project. 🚀

## 📧 Contact
For any queries, reach out via:
- 🔗 LinkedIn: [Shubham Singh Sainger](https://linkedin.com/in/shubham-singh-267214192)
- 📧 Email: shubhamsainger97@gmail.com
---
