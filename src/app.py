import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open("./models/linear_regression_model.pkl", "rb") as f:
    model_artifacts = pickle.load(f)

weights = model_artifacts["weights"]
bias = model_artifacts["bias"]
mu = model_artifacts["mu"]
sigma = model_artifacts["sigma"]
features = model_artifacts["features"]

st.title("🏠 Boston Housing Price Predictor")

st.write("Enter housing details to predict the price:")

# Dynamically create input fields for each feature
inputs = []
for feature in features:
    value = st.number_input(f"{feature}", value=0.0)
    inputs.append(value)

# Convert inputs into numpy array
X = np.array(inputs)

# Apply normalization (same as training)
X_norm = (X - mu) / sigma

# Predict
if st.button("Predict"):
    y_pred = np.dot(X_norm, weights) + bias
    st.success(f"Predicted House Price: ${y_pred:.2f}k")
