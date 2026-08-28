# House Price Prediction

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A from-scratch implementation of **linear regression with batch gradient descent** to predict Boston housing prices, exposed through an interactive **Streamlit** web interface.

## Overview

This project builds a machine learning pipeline that predicts the median value of owner-occupied homes in the Boston area. Instead of relying on scikit-learn, the linear regression model — including z-score normalization, cost computation, gradient descent, and prediction — is implemented entirely from scratch using NumPy. A Streamlit web app lets you explore the model interactively by adjusting the 13 housing features and seeing the predicted price in real time.

## Features

- Custom linear regression model with batch gradient descent implemented from scratch
- Z-score (standardization) feature normalization
- Mean Squared Error (MSE) cost function and R-squared (R²) evaluation metrics
- 80/20 train-test split with a fixed seed for reproducible results
- Training loss curve visualization with Matplotlib
- Model serialization to a `.pkl` file (weights, bias, normalization statistics, feature names)
- Interactive Streamlit web application for price prediction

## Project Structure

```
house-price-prediction/
├── data/
│   ├── raw/                           # Original dataset
│   │   └── housing.csv                # Boston Housing dataset
│   └── processed/                     # (reserved for cleaned data)
├── models/
│   └── linear_regression_model.pkl    # Trained model artifacts
├── notebooks/
│   └── data_preprocessing.ipynb       # Exploratory data analysis
├── src/
│   ├── app.py                         # Streamlit web interface
│   ├── data_utils.py                  # Data loading, preparation, splitting
│   ├── linear_regression.py           # Custom LinearRegression implementation
│   ├── metrics.py                     # MSE and R² scoring functions
│   └── train.py                       # Training pipeline
├── requirements.txt                   # Python dependencies
└── README.md
```

## Dataset

The project uses the classic [Boston Housing dataset](https://www.kaggle.com/datasets/schirmerchad/bostonhoustingmlnd) (506 samples, 13 features + 1 target). Below is a description of each feature.

| Feature    | Description                                                       |
|------------|-------------------------------------------------------------------|
| `CRIM`     | Per-capita crime rate by town                                     |
| `ZN`       | Proportion of residential land zoned for lots over 25,000 sq.ft.  |
| `INDUS`    | Proportion of non-retail business acres per town                   |
| `CHAS`     | Charles River dummy variable (1 if bounds river, else 0)          |
| `NOX`      | Nitric oxides concentration (parts per 10 million)                 |
| `RM`       | Average number of rooms per dwelling                              |
| `AGE`      | Proportion of owner-occupied units built prior to 1940             |
| `DIS`      | Weighted distances to five Boston employment centres               |
| `RAD`      | Index of accessibility to radial highways                         |
| `TAX`      | Full-value property-tax rate per $10,000                           |
| `PTRATIO`  | Pupil–teacher ratio by town                                       |
| `B`        | `1000(Bk − 0.63)²`, where Bk is the proportion of Black residents |
| `LSTAT`    | Percentage of lower-status population                             |
| **`MEDV`** | **Target:** median value of owner-occupied homes (in $1000s)      |

## Tech Stack

- **Python 3.9+**
- **NumPy** — numerical computation and linear algebra
- **Pandas** — data loading and manipulation
- **Matplotlib** — training loss curve visualization
- **Streamlit** — interactive web interface

## Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/Priyanshu-CODERX/lr-house-price-estimation.git
cd lr-house-price-estimation
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### 1. Train the model

Run the training script from the **project root directory** (it uses relative paths):

```bash
python src/train.py
```

This will:
1. Load and clean the dataset (`data/raw/housing.csv`)
2. Z-score normalize the features
3. Split the data into 80% training / 20% test sets
4. Train the linear regression model using batch gradient descent (1,000 iterations, learning rate 0.01)
5. Print the MSE and R² scores
6. Display the training loss curve
7. Save the trained model to `models/linear_regression_model.pkl`

### 2. Run the web application

```bash
streamlit run src/app.py
```

Open your browser at `http://localhost:8501`. Adjust the values of the 13 housing features with the input sliders and click **Predict** to see the estimated home price (in thousands of dollars).

## Model Details

The linear regression model predicts the median home value (`MEDV`) as a linear combination of the standardized features:

```
ŷ = X_normalized · w + b
```

Training is done via **batch gradient descent**, minimizing the mean squared error cost:

```
J(w, b) = (1 / 2m) * Σ (ŷᵢ − yᵢ)²
```

### Hyperparameters

| Parameter     | Value  |
|---------------|--------|
| Learning rate | `0.01` |
| Iterations    | `1,000`|
| Test size     | `0.20` |
| Random seed   | `42`   |

### Evaluation

Model performance is measured on the held-out test set using:
- **Mean Squared Error (MSE)** — average squared difference between predictions and actual values
- **R-squared (R²)** — proportion of variance explained by the model

## Source Files

| File                        | Purpose                                                              |
|-----------------------------|----------------------------------------------------------------------|
| `src/linear_regression.py`  | `LinearRegression` class: normalization, cost, gradient, predict     |
| `src/train.py`              | End-to-end pipeline: load → normalize → train → evaluate → save      |
| `src/data_utils.py`         | CSV loading, feature preparation, and train/test splitting           |
| `src/metrics.py`            | `mean_squared_error` and `r2_score` implementations                  |
| `src/app.py`                | Streamlit UI that loads the saved model and predicts prices          |

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.