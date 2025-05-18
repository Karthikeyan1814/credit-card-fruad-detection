import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import warnings
from tqdm import tqdm
import streamlit as st
import joblib
import os

# Add caching decorators
@st.cache_data
def load_data():
    """
    Load the credit card fraud dataset from Kaggle.
    Dataset source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    
    Returns:
        pd.DataFrame: The credit card transaction dataset
    """
    try:
        print("Loading dataset...")
        df = pd.read_csv('creditcard.csv')
        
        # Take a smaller sample for faster training (50,000 transactions)
        df = df.sample(n=50000, random_state=42)
        
    except FileNotFoundError:
        print("Dataset not found in current directory.")
        print("\nTo use this code:")
        print("1. Download the dataset from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("2. Place the 'creditcard.csv' file in the same directory as this script")
        print("3. Run the script again")
        raise
    
    # Display basic information about the dataset
    print("\nDataset Information:")
    print(f"Total transactions: {len(df)}")
    print(f"Number of fraud cases: {df['Class'].sum()}")
    print(f"Fraud percentage: {(df['Class'].sum() / len(df) * 100):.3f}%")
    
    # Rename 'Class' column to 'fraud' for consistency
    df = df.rename(columns={'Class': 'fraud'})
    
    return df

@st.cache_data
def preprocess_data(df):
    """
    Preprocess the data for model training using only V1, V2, V3, Amount, and Time.
    """
    print("\nPreprocessing data...")
    # Use only selected features
    features = ['V1', 'V2', 'V3', 'Amount', 'Time']
    X = df[features]
    y = df['fraud']
    
    print("Splitting data...")
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Scaling features...")
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Balancing classes with SMOTE...")
    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    return X_train_balanced, X_test_scaled, y_train_balanced, y_test, scaler

def save_model_and_scaler(model, scaler):
    """
    Save the trained model and scaler to disk.
    """
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(model, 'models/fraud_detection_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')

def load_model_and_scaler():
    """
    Load the trained model and scaler from disk.
    """
    try:
        model = joblib.load('models/fraud_detection_model.joblib')
        scaler = joblib.load('models/scaler.joblib')
        return model, scaler
    except FileNotFoundError:
        return None, None

def train_model(X_train, y_train):
    """
    Train the Random Forest model with progress monitoring.
    """
    print("\nTraining model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        n_jobs=-1,        # Use all CPU cores
        random_state=42,
        verbose=1         # Show progress
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and print metrics.
    """
    y_pred = model.predict(X_test)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def plot_feature_importance(model, feature_names):
    """
    Plot feature importance.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance')
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()

def main():
    st.title("Credit Card Fraud Detection")
    
    # Try to load existing model and scaler
    model, scaler = load_model_and_scaler()
    
    if model is None or scaler is None:
        with st.spinner("Training model for the first time..."):
            # Load data
            df = load_data()
            
            # Preprocess data
            X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
            
            # Train model
            model = train_model(X_train, y_train)
            
            # Save model and scaler
            save_model_and_scaler(model, scaler)
            
            # Evaluate model
            evaluate_model(model, X_test, y_test)
            
            # Plot feature importance
            plot_feature_importance(model, ['V1', 'V2', 'V3', 'Amount', 'Time'])
    else:
        st.success("Model loaded successfully!")

    st.sidebar.header("Or Enter Transaction Details Manually")

    with st.sidebar.form("manual_entry_form"):
        manual_data = {}
        for col in ['V1', 'V2', 'V3', 'Amount', 'Time']:
            manual_data[col] = st.number_input(f"{col}", value=0.0)
        submitted = st.form_submit_button("Predict Manually")

    if submitted:
        # Prepare the data for prediction
        manual_df = pd.DataFrame([manual_data])
        manual_scaled = scaler.transform(manual_df)
        manual_pred = model.predict(manual_scaled)[0]
        result = "Fraud" if manual_pred == 1 else "Normal"
        st.write(f"### Manual Entry Prediction: **{result}**")

if __name__ == "__main__":
    main() 