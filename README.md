# Credit Card Fraud Detection System

This project implements a machine learning-based credit card fraud detection system using Python. The system uses Random Forest Classifier to identify potentially fraudulent transactions.

## Dataset

This project uses the Credit Card Fraud Detection dataset from Kaggle:
- Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Contains real credit card transactions from European cardholders
- Features are anonymized for privacy
- Contains 284,807 transactions with 492 frauds (0.172% fraud rate)
- All features are numerical and have been preprocessed using PCA

## Features

- Data preprocessing and feature scaling
- Handling of class imbalance using SMOTE
- Random Forest Classifier for fraud detection
- Model evaluation with classification metrics
- Feature importance visualization
- Confusion matrix visualization

## Requirements

- Python 3.8 or higher
- Required packages are listed in `requirements.txt`

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```
3. Download the dataset:
   - Go to https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
   - Download the `creditcard.csv` file
   - Place it in the same directory as the script

## Usage

1. Run the main script:
```bash
python credit_card_fraud_detection.py
```

The script will:
- Load and preprocess the data
- Train the Random Forest model
- Evaluate the model's performance
- Display visualizations of the results

## Note

The current implementation uses synthetic data for demonstration purposes. In a real-world scenario, you would need to:

1. Replace the `load_data()` function with your actual credit card transaction data
2. Adjust the feature engineering process based on your specific data
3. Tune the model hyperparameters for optimal performance
4. Implement proper data security measures

## Model Details

- Algorithm: Random Forest Classifier
- Features: 30 numerical features (PCA transformed)
- Class imbalance handling: SMOTE (Synthetic Minority Over-sampling Technique)
- Evaluation metrics: Precision, Recall, F1-score, and Confusion Matrix 


what should we do follow this stepes 
*install visual c++ build tools 
# Deactivate current environment
deactivate

# Remove old environment
rm -r .venv

# Create new environment with Python 3.10
python3.10 -m venv .venv

# Activate new environment
.venv\Scripts\activate

# Upgrade pip and install wheel
python -m pip install --upgrade pip
pip install wheel

# Install packages one by one
pip install numpy==1.23.5
pip install pandas==1.5.3
pip install scikit-learn==1.2.2
pip install matplotlib==3.7.2
pip install seaborn==0.12.2
pip install imbalanced-learn==0.10.1
pip install tqdm==4.65.0
pip install joblib==1.2.0
pip install flask==2.2.3

to start python app.py
to train python credit_card_fraud_detection.py  