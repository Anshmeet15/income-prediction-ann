💰 Income Prediction using ANN (MLPClassifier)

This project predicts whether a person earns more than 50K or less based on demographic features from the Income Dataset.
It uses a scikit-learn Artificial Neural Network (MLPClassifier) and is deployed as an interactive Streamlit web app.

📌 Features

Upload your own CSV dataset

Automatic preprocessing (One-Hot Encoding + Standard Scaling)

Train-Test split configuration from sidebar

Train an ANN model (MLPClassifier) directly in the app

View Accuracy, Confusion Matrix, Classification Report

Interactive Training vs Validation performance curves

Easy deployment with Streamlit

⚙️ Tech Stack

Python 

Streamlit (Web App)

scikit-learn (MLPClassifier ANN, preprocessing, evaluation)

Pandas, Numpy (Data handling)

Matplotlib, Seaborn (Visualization)

📂 Project Structure
├── income.csv                 # Dataset
├── app.py                     # Streamlit App
├── mlp_model.pkl              # Saved ANN model (MLPClassifier)
├── scaler.pkl                 # Saved StandardScaler
├── features.pkl               # Saved feature list
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
