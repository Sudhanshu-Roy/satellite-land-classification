# Satellite Land-Use Classification using Deep Learning

## Project Overview
This project builds a deep learning model to classify satellite images into different land-use categories using transfer learning. The model is trained on the EuroSAT dataset and uses a pretrained ResNet18 architecture to identify terrain types such as forests, rivers, residential areas, and agricultural land.

The project demonstrates a complete computer vision pipeline including dataset preprocessing, model training, evaluation, prediction visualization, and Grad-CAM interpretability.

---

## Dataset

This project uses the **EuroSAT satellite image dataset**, containing approximately **27,000 RGB satellite images across 10 land-use classes**.

Classes include:

- AnnualCrop  
- Forest  
- HerbaceousVegetation  
- Highway  
- Industrial  
- Pasture  
- PermanentCrop  
- Residential  
- River  
- SeaLake  

Dataset source:  
https://www.kaggle.com/datasets/apollo2506/eurosat-dataset

---

## Model Architecture

The model uses **transfer learning with ResNet18**.

Steps:
- Pretrained ResNet18 backbone
- Final fully connected layer modified for 10 classes
- Cross-entropy loss for classification
- Adam optimizer for training

---

## Training Pipeline

1. Dataset loading using `torchvision.datasets.ImageFolder`
2. Image preprocessing and resizing
3. Train / validation split
4. DataLoader for batch training
5. Model training using PyTorch
6. Best model checkpoint saved based on validation accuracy

---

## Model Performance

**Validation Accuracy:** `96.8%`

Evaluation metrics include:

- Accuracy
- Confusion Matrix
- Classification Report (Precision, Recall, F1-score)

---

## Confusion Matrix

<img width="932" height="845" alt="image" src="https://github.com/user-attachments/assets/ecc90b2c-083d-4f5b-b7e7-87eb320037bb" />

---

## Prediction Visualization

Example predictions on validation images.

```
Satellite Image → Predicted Land Class
```
<img width="950" height="672" alt="image" src="https://github.com/user-attachments/assets/de845b76-2122-4acf-b393-cb158f8f2fff" />


---

## Model Interpretability (Grad-CAM)

Grad-CAM visualization highlights the regions of the satellite image that influenced the model's prediction.

Example visualization:

```
Original Image | Heatmap | Overlay
```

<img width="950" height="315" alt="image" src="https://github.com/user-attachments/assets/89ffd154-648c-4c8e-a6cd-8ab6d21b4791" />


---

## Project Structure

```
satellite-land-classification
│
├── Notebooks/
│   ├── train.ipynb
│   ├── evaluate.ipynb
│   ├── predict.ipynb
│   └── gradcam.ipynb
│
├── models/
│   └── best_model.pth
│
├── .gitignore
├── LICENSE
├── download_dataset.py
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Sudhanshu-Roy/satellite-land-classification.git
cd satellite-land-classification
```

Install dependencies:

```bash
pip install -r requirements.txt
```
---

3. Open the notebooks

Run the notebooks in the following order:

```
Notebooks/training.ipynb
Notebooks/evaluate.ipynb
Notebooks/predict.ipynb
Notebooks/gradcam.ipynb
```

---

## Future Improvements

Possible improvements include:

- Training with deeper architectures (ResNet50 / EfficientNet)
- Hyperparameter tuning
- Real-time satellite image classification interface
- Satellite change detection using temporal image pairs

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Scikit-learn
- OpenCV
- Grad-CAM

---

## Author

**Sudhanshu Roy**  
Machine Learning & AI Enthusiast

GitHub: https://github.com/Sudhanshu-Roy 
LinkedIn: www.linkedin.com/in/gecraiandds240200143065
