# Satellite Land Cover Classifier

A deep learning model that classifies satellite images into land-use and land-cover types. A fine-tuned ResNet18, trained on the EuroSAT dataset, reaches **96.8% validation accuracy** across 10 classes. The repository includes the training notebooks and a web app for inference: a FastAPI backend and a React frontend.

Upload a satellite image, or try a sample tile, and the app returns the predicted class with a score for every class.

**Live demo:** https://satellite-land-cover-classifier.vercel.app

---

## Dataset

The model is trained on the **EuroSAT** dataset: about **27,000 RGB satellite images** across **10 land-use classes**.

| Group | Classes |
| --- | --- |
| Vegetation and agriculture | AnnualCrop, PermanentCrop, Pasture, HerbaceousVegetation, Forest |
| Built environment | Residential, Industrial, Highway |
| Water | River, SeaLake |

Dataset source:  
https://www.kaggle.com/datasets/apollo2506/eurosat-dataset

Download it with:

```bash
python download_dataset.py
```

---

## Model

Transfer learning with **ResNet18**:

- Pretrained ResNet18 backbone
- Final fully connected layer replaced with 10 outputs
- Cross-entropy loss
- Adam optimizer (`lr=0.001`)

Images are resized to **224×224** and scaled with `ToTensor()` only. There is no ImageNet normalization.

The best checkpoint, chosen by validation accuracy, is saved to `backend/model/best_model.pth`.

---

## Training and evaluation

Run the notebooks from the **project root**, in this order:

1. `Notebooks/training.ipynb`
2. `Notebooks/evaluate.ipynb`
3. `Notebooks/predict.ipynb`
4. `Notebooks/gradcam.ipynb`

Training loads EuroSAT with `torchvision.datasets.ImageFolder`, splits the data, and writes `backend/model/best_model.pth` whenever validation accuracy improves. Evaluation reports accuracy, a confusion matrix, and a classification report. Grad-CAM shows which regions influenced a prediction.

**Validation accuracy: 96.8%**

### Confusion matrix

<img width="932" height="845" alt="Confusion matrix" src="https://github.com/user-attachments/assets/ecc90b2c-083d-4f5b-b7e7-87eb320037bb" />

### Example predictions

<img width="950" height="672" alt="Prediction examples" src="https://github.com/user-attachments/assets/de845b76-2122-4acf-b393-cb158f8f2fff" />

### Grad-CAM

<img width="950" height="315" alt="Grad-CAM visualization" src="https://github.com/user-attachments/assets/89ffd154-648c-4c8e-a6cd-8ab6d21b4791" />

---

## Web app

`backend/` is a FastAPI service that loads `backend/model/best_model.pth` and classifies an uploaded image on CPU.

- `GET /health` checks that the service is up
- `POST /predict` accepts an image and returns the top class plus all 10 scores

`frontend/` is a Vite + React app. It waits on the loading screen until `/health` succeeds, then lets you upload an image or step through one sample tile from each EuroSAT class. Sample images live in `frontend/public/samples/`.

---

## Project structure

```
satellite-land-classification
│
├── Notebooks/
│   ├── training.ipynb
│   ├── evaluate.ipynb
│   ├── predict.ipynb
│   └── gradcam.ipynb
│
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   ├── Procfile
│   ├── .python-version
│   └── model/
│       └── best_model.pth
│
├── frontend/
│   ├── src/
│   ├── public/samples/
│   ├── package.json
│   └── vercel.json
│
├── download_dataset.py
├── requirements.txt
├── LICENSE
└── README.md
```

`requirements.txt` at the root is for the notebooks. `backend/requirements.txt` is for the API and installs the CPU build of PyTorch.

---

## Run locally

```bash
git clone https://github.com/Sudhanshu-Roy/satellite-land-classification.git
cd satellite-land-classification
```

Notebooks:

```bash
pip install -r requirements.txt
```

API:

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8000
```

Frontend, in another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. The page calls http://localhost:8000 unless `VITE_API_URL` is set.

---

## Deployment

The API is meant for **Render** and the frontend for **Vercel**.

**Render**

- Root directory: `backend`
- Runtime: Python. `.python-version` pins 3.12, because the current Render default does not have a matching PyTorch wheel.
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- Use an instance with at least 2 GB of RAM. The 512 MB plan runs out of memory while loading PyTorch.

**Vercel**

- Root directory: `frontend`
- Framework: Vite
- Environment variable: `VITE_API_URL` set to the Render URL, with no trailing path
- Redeploy after setting that variable. Vite reads it at build time.

The first request after Render has been idle can take about a minute. The page stays on the loading screen until the API responds.

---

## Technologies

- Python, PyTorch, Torchvision
- FastAPI, Uvicorn
- React, Vite
- NumPy, Matplotlib, Scikit-learn, OpenCV, Grad-CAM

---

## Possible next steps

- Train a deeper backbone, such as ResNet50 or EfficientNet
- Tune hyperparameters
- Detect change between satellite images taken at different times

---

## Author

**Sudhanshu Roy**  
Machine Learning & AI Enthusiast

GitHub: https://github.com/Sudhanshu-Roy  
LinkedIn: https://www.linkedin.com/in/gecraiandds240200143065
