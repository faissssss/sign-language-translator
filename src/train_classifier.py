import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

def train_model(data_path='data/landmarks_data.csv', model_save_path='models/isl_classifier.p'):
    # Load data
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found. Please run data_collector.py first.")
        return

    df = pd.read_csv(data_path)
    
    # Check if we have enough data
    if len(df) == 0:
        print("Error: Dataset is empty.")
        return
        
    print(f"Dataset loaded: {len(df)} samples")
    print(f"Classes found: {df['label'].unique()}")

    # Separate features and labels
    X = df.drop('label', axis=1).values
    y = df['label'].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Train model
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Save model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    with open(model_save_path, 'wb') as f:
        pickle.dump({'model': model}, f)
    
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    train_model()
