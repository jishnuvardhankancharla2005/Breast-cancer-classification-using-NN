# 💡 Breast Cancer Prediction System

A full ML pipeline + Gradio web app built using Python, Scikit-Learn, and TensorFlow.

# 🚀 Project Overview

This project implements an end-to-end Breast Cancer Prediction System using feature selection, ML preprocessing, a deep learning model, and an interactive Gradio UI.

The workflow consists of:

▶️ mian.ipynb — Model training pipeline

Loads Breast Cancer dataset

Selects top-K most important features

Applies scaling

Trains a TensorFlow/Keras neural network

Saves model, scaler, and selected features

▶️ app.ipynb — Gradio prediction interface

Loads the trained model + scaler

Accepts user input

Predicts Benign or Malignant

Returns prediction + confidence score

This project demonstrates a complete ML lifecycle: data → feature selection → model → deployment UI.

# 🧠 Features
🔬 Model Training (mian.ipynb)

Selects top 10 features using feature importance (RandomForest)

Scales selected features using StandardScaler

Builds a Sequential neural network with Dense + Dropout layers

Splits data into train/test

Evaluates test accuracy

Saves:

reduced_model.h5

reduced_scaler.pkl

selected_features.json

# 🌐 Deployment UI (app.ipynb)

Fully interactive Gradio interface

Input fields auto-generated from selected features

Loads:

Trained model

Scaler

Feature list

Outputs:

Prediction label (Benign/Malignant)

Confidence score

# 📂 Project Structure
📦 Breast-Cancer-Prediction
├── mian.ipynb                 # Model training & feature selection
├── app.ipynb                  # Gradio-based prediction UI
├── reduced_model.h5           # Saved Keras model
├── reduced_scaler.pkl         # Saved Scikit-learn scaler
├── selected_features.json     # Selected feature names
└── README.md                  # Documentation

# ⚙️ Installation
1️⃣ Clone the repository
git clone https://github.com/yourusername/breast-cancer-prediction.git
cd breast-cancer-prediction

2️⃣ Install dependencies
pip install -r requirements.txt


(If you don’t have a requirements file, create one using:)

pip freeze > requirements.txt

3️⃣ Launch Jupyter
jupyter notebook

# ▶️ Usage Instructions
🔧 To Train the Model (mian.ipynb)

Open mian.ipynb

Run all cells

Model + scaler + feature list will be saved automatically

Accuracy will be printed at the end

🌐 To Run the Gradio App (app.ipynb)

Make sure the saved files exist in the same directory:

reduced_model.h5

reduced_scaler.pkl

selected_features.json

Run all cells in app.ipynb

A Gradio interface will launch

Enter feature values → get prediction + confidence

# 📈 Model Architecture

Input: Top 10 selected features

Layers:

Dense

Dropout

Dense Output (Sigmoid)

Optimizer: Adam

Loss: Binary Crossentropy

Metrics: Accuracy

# 🛠️ Technologies Used
Area	Tools
Model	TensorFlow / Keras
Feature Selection	RandomForestClassifier
Preprocessing	StandardScaler
Dataset	Scikit-learn Breast Cancer dataset
Deployment	Gradio
Language	Python
# 📌 Future Improvements

 Convert notebooks into .py scripts

 Add Docker support

 Create a beautiful web dashboard

 Add cross-validation

 Add interpretability (SHAP / LIME)

# 👤 Author

jishnu vardhan kancharla

GitHub: jishnuvardhankancharla2005

Email: jishnuvardhan558@gmail.com
