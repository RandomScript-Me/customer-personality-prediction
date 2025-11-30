#!/usr/bin/env python3
"""
Script to train the customer personality prediction model
This script extracts the key logic from the Jupyter notebook
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
import warnings
import joblib
import json
import os

# Try to import xgboost, but continue without it if it fails
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except (ImportError, Exception) as e:
    print(f"Warning: XGBoost not available ({str(e)[:50]}...), will skip XGBoost model")
    HAS_XGBOOST = False

# Try to import lightgbm, but continue without it if it fails
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except (ImportError, Exception) as e:
    print(f"Warning: LightGBM not available, will skip LightGBM model")
    HAS_LIGHTGBM = False

warnings.filterwarnings('ignore')

print("=" * 60)
print("Customer Personality Prediction - Model Training")
print("=" * 60)

# Load the dataset
print("\n[1/8] Loading dataset...")
df = pd.read_csv('dataset_file.rtfd/marketing_campaign.csv.xls', sep='\t')
print(f"Dataset shape: {df.shape}")

# Data Preprocessing
print("\n[2/8] Preprocessing data...")
df_processed = df.copy()

# Handle missing values in Income
if df_processed['Income'].isnull().sum() > 0:
    df_processed['Income'].fillna(df_processed['Income'].median(), inplace=True)

# Convert Dt_Customer to datetime and extract features
df_processed['Dt_Customer'] = pd.to_datetime(df_processed['Dt_Customer'], format='%d-%m-%Y')
df_processed['Customer_Age'] = 2024 - df_processed['Year_Birth']
df_processed['Days_Since_Customer'] = (pd.Timestamp('2024-01-01') - df_processed['Dt_Customer']).dt.days

# Drop unnecessary columns
df_processed = df_processed.drop(['ID', 'Year_Birth', 'Dt_Customer', 'Z_CostContact', 'Z_Revenue'], axis=1)

# Feature Engineering
print("\n[3/8] Engineering features...")
df_processed['Total_Spent'] = (df_processed['MntWines'] + df_processed['MntFruits'] + 
                                df_processed['MntMeatProducts'] + df_processed['MntFishProducts'] + 
                                df_processed['MntSweetProducts'] + df_processed['MntGoldProds'])

df_processed['Total_Purchases'] = (df_processed['NumDealsPurchases'] + df_processed['NumWebPurchases'] + 
                                    df_processed['NumCatalogPurchases'] + df_processed['NumStorePurchases'])

df_processed['Total_Accepted_Campaigns'] = (df_processed['AcceptedCmp1'] + df_processed['AcceptedCmp2'] + 
                                             df_processed['AcceptedCmp3'] + df_processed['AcceptedCmp4'] + 
                                             df_processed['AcceptedCmp5'])

df_processed['Avg_Purchase_Value'] = df_processed['Total_Spent'] / (df_processed['Total_Purchases'] + 1)
df_processed['Children'] = df_processed['Kidhome'] + df_processed['Teenhome']
df_processed['Family_Size'] = df_processed['Children'] + 1

# Spending patterns
df_processed['Wine_Ratio'] = df_processed['MntWines'] / (df_processed['Total_Spent'] + 1)
df_processed['Meat_Ratio'] = df_processed['MntMeatProducts'] / (df_processed['Total_Spent'] + 1)
df_processed['Gold_Ratio'] = df_processed['MntGoldProds'] / (df_processed['Total_Spent'] + 1)

# Purchase channel preferences
df_processed['Web_Purchase_Ratio'] = df_processed['NumWebPurchases'] / (df_processed['Total_Purchases'] + 1)
df_processed['Store_Purchase_Ratio'] = df_processed['NumStorePurchases'] / (df_processed['Total_Purchases'] + 1)
df_processed['Catalog_Purchase_Ratio'] = df_processed['NumCatalogPurchases'] / (df_processed['Total_Purchases'] + 1)

# Encode categorical variables
print("\n[4/8] Encoding categorical variables...")
le_education = LabelEncoder()
le_marital = LabelEncoder()

df_processed['Education_Encoded'] = le_education.fit_transform(df_processed['Education'])
df_processed['Marital_Status_Encoded'] = le_marital.fit_transform(df_processed['Marital_Status'])

# Drop original categorical columns
df_processed = df_processed.drop(['Education', 'Marital_Status'], axis=1)

# Separate features and target
X = df_processed.drop('Response', axis=1)
y = df_processed['Response']

print(f"Features shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# Split the data
print("\n[5/8] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Function to evaluate models
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Evaluate model performance"""
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    roc_auc = roc_auc_score(y_test, y_test_proba)
    
    cv_scores = cross_val_score(model, X_train, y_train, cv=StratifiedKFold(n_splits=5), scoring='roc_auc')
    
    results = {
        'Model': model_name,
        'Train_Accuracy': train_accuracy,
        'Test_Accuracy': test_accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1,
        'ROC_AUC': roc_auc,
        'CV_ROC_AUC_Mean': cv_scores.mean(),
        'CV_ROC_AUC_Std': cv_scores.std()
    }
    
    return model, results

# Train models
print("\n[6/8] Training models...")
results_list = []
models_dict = {}

# Random Forest
print("  - Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, min_samples_split=5)
rf_trained, rf_results = evaluate_model(rf_model, X_train, X_test, y_train, y_test, 'Random Forest')
results_list.append(rf_results)
models_dict['Random Forest'] = (rf_trained, False)

# XGBoost
if HAS_XGBOOST:
    print("  - Training XGBoost...")
    xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, max_depth=6, learning_rate=0.1)
    xgb_trained, xgb_results = evaluate_model(xgb_model, X_train, X_test, y_train, y_test, 'XGBoost')
    results_list.append(xgb_results)
    models_dict['XGBoost'] = (xgb_trained, False)

# LightGBM
if HAS_LIGHTGBM:
    print("  - Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42, max_depth=6, learning_rate=0.1, verbose=-1)
    lgb_trained, lgb_results = evaluate_model(lgb_model, X_train, X_test, y_train, y_test, 'LightGBM')
    results_list.append(lgb_results)
    models_dict['LightGBM'] = (lgb_trained, False)

# Neural Network
print("  - Training Neural Network...")
nn_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42, alpha=0.01)
nn_trained, nn_results = evaluate_model(nn_model, X_train_scaled, X_test_scaled, y_train, y_test, 'Neural Network')
results_list.append(nn_results)
models_dict['Neural Network'] = (nn_trained, True)

# Compare models
print("\n[7/8] Comparing models...")
results_df = pd.DataFrame(results_list)
results_df = results_df.set_index('Model')
print("\nModel Comparison:")
print(results_df.round(4))

# Select best model
best_model_name = results_df['ROC_AUC'].idxmax()
print(f"\nBest model: {best_model_name} with ROC-AUC: {results_df.loc[best_model_name, 'ROC_AUC']:.4f}")

best_model, use_scaled = models_dict[best_model_name]

# Save model
print("\n[8/8] Saving model...")
os.makedirs('models', exist_ok=True)

joblib.dump(best_model, 'models/best_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(le_education, 'models/le_education.pkl')
joblib.dump(le_marital, 'models/le_marital.pkl')

with open('models/feature_names.json', 'w') as f:
    json.dump(list(X.columns), f)

model_metadata = {
    'model_name': best_model_name,
    'use_scaled': use_scaled,
    'roc_auc': float(results_df.loc[best_model_name, 'ROC_AUC']),
    'accuracy': float(results_df.loc[best_model_name, 'Test_Accuracy']),
    'precision': float(results_df.loc[best_model_name, 'Precision']),
    'recall': float(results_df.loc[best_model_name, 'Recall']),
    'f1_score': float(results_df.loc[best_model_name, 'F1_Score'])
}

with open('models/model_metadata.json', 'w') as f:
    json.dump(model_metadata, f, indent=2)

print("\n" + "=" * 60)
print("✅ Model training completed successfully!")
print("=" * 60)
print(f"\nBest Model: {best_model_name}")
print(f"ROC-AUC: {model_metadata['roc_auc']:.4f}")
print(f"Accuracy: {model_metadata['accuracy']:.2%}")
print(f"\nModel saved to: models/best_model.pkl")
print("\nYou can now run the Streamlit app with: streamlit run app.py")

