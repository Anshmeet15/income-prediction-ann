# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import (
#     accuracy_score,
#     confusion_matrix,
#     classification_report,
#     roc_curve,
#     auc
# )

# # ---------------------
# # Streamlit UI
# # ---------------------
# st.set_page_config(page_title="MLP ANN Classifier", layout="wide")
# st.title("🔮 ANN Classifier with Streamlit")
# st.write("Upload your dataset, explore it, and train an Artificial Neural Network (MLP).")

# # ---------------------
# # File Upload
# # ---------------------
# uploaded_file = st.file_uploader("📂 Upload your CSV dataset", type=["csv"])

# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)
#     st.subheader("📊 Dataset Preview")
#     st.dataframe(df.head())

#     # ---------------------
#     # EDA Section
#     # ---------------------
#     st.subheader("🔍 Exploratory Data Analysis")

#     st.write("**Dataset Shape:**", df.shape)
#     st.write("**Missing Values:**")
#     st.write(df.isnull().sum())
#     st.write("**Statistical Summary:**")
#     st.write(df.describe(include="all"))

#     st.write("**Correlation Heatmap (numerical only):**")
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
#     st.pyplot(plt)

#     # ---------------------
#     # Target Selection
#     # ---------------------
#     target_col = st.selectbox("🎯 Select Target Column", df.columns)

#     if target_col:
#         X = df.drop(columns=[target_col])
#         y = df[target_col]

#         # Encode target if categorical
#         if y.dtype == "object":
#             y = pd.factorize(y)[0]

#         # Identify categorical & numeric features
#         categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
#         numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

#         # Preprocessing pipeline
#         preprocessor = ColumnTransformer(
#             transformers=[
#                 ("num", StandardScaler(), numeric_features),
#                 ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
#             ]
#         )

#         # ---------------------
#         # Model Parameters
#         # ---------------------
#         st.subheader("⚙️ Model Configuration")
#         hidden_layer_sizes = st.slider("Hidden Layers (neurons per layer)", 10, 200, 100, step=10)
#         max_iter = st.slider("Max Iterations", 100, 1000, 300, step=100)
#         learning_rate_init = st.select_slider("Learning Rate", options=[0.001, 0.01, 0.1], value=0.001)
#         test_size = st.slider("Test size (%)", 10, 50, 20, step=5) / 100

#         # ---------------------
#         # Train/Test Split
#         # ---------------------
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=test_size, random_state=42
#         )

#         # ---------------------
#         # Training Pipeline
#         # ---------------------
#         @st.cache_resource
#         def train_model():
#             model = Pipeline(steps=[
#                 ("preprocessor", preprocessor),
#                 ("classifier", MLPClassifier(
#                     hidden_layer_sizes=(hidden_layer_sizes,),
#                     max_iter=max_iter,
#                     learning_rate_init=learning_rate_init,
#                     random_state=42
#                 ))
#             ])
#             model.fit(X_train, y_train)
#             return model

#         if st.button("🚀 Train Model"):
#             model = train_model()
#             y_pred = model.predict(X_test)

#             acc = accuracy_score(y_test, y_pred)
#             st.success(f"✅ Model Trained! Accuracy: **{acc:.2f}**")

#             # Classification Report
#             st.subheader("📑 Classification Report")
#             st.text(classification_report(y_test, y_pred))



#             # ROC Curve
#             if len(np.unique(y)) == 2:
#                 st.subheader("📈 ROC Curve")
#                 y_prob = model.predict_proba(X_test)[:, 1]
#                 fpr, tpr, _ = roc_curve(y_test, y_prob)
#                 roc_auc = auc(fpr, tpr)

#                 plt.figure(figsize=(6, 4))
#                 plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
#                 plt.plot([0, 1], [0, 1], linestyle="--")
#                 plt.xlabel("False Positive Rate")
#                 plt.ylabel("True Positive Rate")
#                 plt.title("ROC Curve")
#                 plt.legend()
#                 st.pyplot(plt)
# else:
#     st.info("👆 Please upload a CSV dataset to continue.")



# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle

# # -------------------------------
# # Load Saved Objects
# # -------------------------------
# @st.cache_resource
# def load_artifacts():
#     with open("mlp_model.pkl", "rb") as f:
#         model = pickle.load(f)
#     with open("scaler.pkl", "rb") as f:
#         scaler = pickle.load(f)
#     with open("features.pkl", "rb") as f:
#         features = pickle.load(f)
#     return model, scaler, features

# model, scaler, features = load_artifacts()

# # -------------------------------
# # Streamlit UI Setup
# # -------------------------------
# st.set_page_config(page_title="Income Prediction App", layout="wide")
# st.title("💰 ANN Model Deployment - Income Prediction")

# st.sidebar.header("Choose Input Method")
# option = st.sidebar.radio("Select Input Type:", ["Manual Input", "Upload CSV"])

# # -------------------------------
# # Manual Input
# # -------------------------------
# if option == "Manual Input":
#     st.subheader("🔹 Enter Citizen Details")

#     age = st.number_input("Age", min_value=18, max_value=100, value=30)
#     workclass = st.selectbox("Workclass", ["Private", "Self-emp", "Government", "Other"])
#     education = st.selectbox("Education", ["Bachelors", "Masters", "HS-grad", "Doctorate", "Other"])
#     hours_per_week = st.slider("Hours per Week", 1, 100, 40)

#     # Convert to dataframe
#     input_data = pd.DataFrame([[age, workclass, education, hours_per_week]], columns=["age", "workclass", "education", "hours_per_week"])

#     # Align with training features
#     input_data = pd.get_dummies(input_data)
#     input_data = input_data.reindex(columns=features, fill_value=0)

#     # Scale
#     input_scaled = scaler.transform(input_data)

#     # Predict
#     prediction = model.predict(input_scaled)[0]
#     income_class = ">50K" if prediction == 1 else "<=50K"

#     st.success(f"✅ Predicted Income Class: **{income_class}**")

# # -------------------------------
# # CSV Upload
# # -------------------------------
# elif option == "Upload CSV":
#     st.subheader("📂 Upload Citizen Dataset")
#     uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

#     if uploaded_file is not None:
#         df = pd.read_csv(uploaded_file)
#         st.write("Preview of Uploaded Data:")
#         st.dataframe(df.head())

#         # Preprocess
#         df_proc = pd.get_dummies(df)
#         df_proc = df_proc.reindex(columns=features, fill_value=0)

#         # Scale
#         df_scaled = scaler.transform(df_proc)

#         # Predict
#         preds = model.predict(df_scaled)
#         df["Predicted Income"] = np.where(preds == 1, ">50K", "<=50K")

#         st.subheader("📊 Prediction Results")
#         st.dataframe(df)

#         # Download predictions
#         csv = df.to_csv(index=False).encode("utf-8")
#         st.download_button("⬇️ Download Predictions", csv, "income_predictions.csv", "text/csv")




import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier

# -------------------------------
# Streamlit Setup
# -------------------------------
st.set_page_config(page_title="MLP ANN Deployment", layout="wide")
st.title("🤖 ANN (MLPClassifier) Deployment -  American citizen Income Prediction")

# -------------------------------
# Upload Dataset
# -------------------------------
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Select target column
    target_column = st.sidebar.selectbox("🎯 Select Target Column", df.columns)

    if target_column:
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Categorical + Numeric handling
        categorical_cols = X.select_dtypes(include=["object"]).columns
        numeric_cols = X.select_dtypes(exclude=["object"]).columns

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_cols),
                ('cat', OneHotEncoder(handle_unknown="ignore"), categorical_cols)
            ]
        )

        X = preprocessor.fit_transform(X)

        # Encode target if categorical
        if y.dtype == "object":
            y = pd.factorize(y)[0]

        # Train-Test Split
        test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20) / 100
        random_state = st.sidebar.number_input("Random State", 0, 100, 42)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # -------------------------------
        # MLP ANN Model
        # -------------------------------
        model = MLPClassifier(hidden_layer_sizes=(64, 32),
                              activation="relu",
                              solver="adam",
                              max_iter=300,
                              random_state=random_state)

        st.write("⏳ Training ANN (MLPClassifier)...")
        model.fit(X_train, y_train)

        # -------------------------------
        # Predictions & Evaluation
        # -------------------------------
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        st.metric("✅ Accuracy", f"{acc:.2f}")

        # # Confusion Matrix
        # cm = confusion_matrix(y_test, y_pred)
        # fig, ax = plt.subplots(figsize=(6, 4))
        # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
        #             xticklabels=np.unique(y), yticklabels=np.unique(y))
        # plt.xlabel("Predicted")
        # plt.ylabel("Actual")
        # st.pyplot(fig)

        # Classification Report
        report = classification_report(y_test, y_pred, output_dict=True)
        st.subheader("📑 Classification Report")
        st.dataframe(pd.DataFrame(report).transpose())

else:
    st.info("👆 Please upload a dataset to train an ANN model.")

