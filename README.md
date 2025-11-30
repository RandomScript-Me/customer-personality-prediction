# Customer Personality Prediction Model

A machine learning project that predicts customer response to marketing campaigns based on demographic, behavioral, and purchase history data.

## 📋 Project Overview

This project aims to help businesses design effective marketing strategies by predicting customer personality traits and their likelihood to respond to marketing campaigns. The solution includes:

- **ML Model Development**: Jupyter notebook with comprehensive data preprocessing, feature engineering, and model training
- **Web Application**: Streamlit-based dashboard for real-time predictions
- **Multiple Models**: Random Forest, XGBoost, LightGBM, and Neural Networks

## 🎯 Objectives

- Develop a machine learning model to predict customer personality traits
- Improve marketing campaign effectiveness through targeted personalization
- Provide actionable insights for marketing teams to optimize customer engagement strategies

## 📁 Project Structure

```
P1/
├── customer_personality_prediction.ipynb  # Jupyter notebook with ML model
├── app.py                                 # Streamlit web application
├── requirements.txt                       # Python dependencies
├── README.md                              # This file
├── dataset_file.rtfd/
│   └── marketing_campaign.csv.xls         # Dataset
└── models/                                # Generated after running notebook
    ├── best_model.pkl
    ├── scaler.pkl
    ├── le_education.pkl
    ├── le_marital.pkl
    ├── feature_names.json
    └── model_metadata.json
```

## 🚀 Getting Started

> 📖 **For detailed step-by-step instructions, see [SETUP_GUIDE.md](SETUP_GUIDE.md)**
> 
> 🌐 **For deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md)**

### Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the model:**
   ```bash
   python3 train_model.py
   ```
   Or use the Jupyter notebook: `jupyter notebook customer_personality_prediction.ipynb`

3. **Run the app:**
   ```bash
   python3 -m streamlit run app.py
   ```

4. **Open in browser:** http://localhost:8501

## 📊 Features

### Jupyter Notebook (`customer_personality_prediction.ipynb`)

- **Data Preprocessing**: Handles missing values, encodes categorical variables, creates derived features
- **Feature Engineering**: 
  - Total spending and purchase counts
  - Spending patterns (wine ratio, meat ratio, etc.)
  - Purchase channel preferences
  - Family demographics
- **Model Training**: Trains and compares 4 different models
- **Model Evaluation**: Comprehensive metrics including accuracy, precision, recall, F1-score, and ROC-AUC
- **Model Selection**: Automatically selects the best model based on ROC-AUC score
- **Visualizations**: Feature importance, model comparison, confusion matrix

### Web Application (`app.py`)

- **Single Prediction**: Input customer data and get instant predictions
- **Model Information**: View model performance metrics in the sidebar
- **User-Friendly Interface**: Clean, intuitive design with real-time predictions
- **Recommendations**: Actionable marketing recommendations based on predictions

## 🔧 Model Details

The project trains and compares the following models:

1. **Random Forest**: Ensemble method using multiple decision trees
2. **XGBoost**: Gradient boosting framework optimized for performance
3. **LightGBM**: Fast, distributed gradient boosting framework
4. **Neural Network**: Multi-layer perceptron classifier

The best model (based on ROC-AUC score) is automatically selected and saved for deployment.

## 📈 Model Evaluation Metrics

- **Accuracy**: Overall correctness of predictions
- **Precision**: Proportion of positive predictions that are correct
- **Recall**: Proportion of actual positives that are correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve

## 🎨 Usage Example

### Using the Web App

1. Open the Streamlit app
2. Navigate to the "Single Prediction" tab
3. Enter customer information:
   - Demographic data (age, education, marital status, income)
   - Purchase history (spending amounts, purchase counts)
   - Campaign responses
4. Click "Predict Response"
5. View the prediction and marketing recommendations

### Using the Notebook

1. Open the Jupyter notebook
2. Run cells sequentially to:
   - Explore the data
   - Preprocess and engineer features
   - Train models
   - Evaluate performance
   - Save the best model

## 📝 Dataset

The dataset (`marketing_campaign.csv.xls`) contains:
- **Demographic data**: Year of birth, education, marital status, income
- **Family information**: Number of kids and teens at home
- **Purchase history**: Spending amounts across different product categories
- **Behavioral data**: Purchase channels, web visits, campaign responses
- **Target variable**: Response (1 = responded, 0 = did not respond)

## 🛠️ Tech Stack

- **Programming**: Python 3.8+
- **ML Libraries**: Scikit-learn, XGBoost, LightGBM
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Deployment**: Streamlit
- **Notebook**: Jupyter

## 📦 Dependencies

All required packages are listed in `requirements.txt`. Key dependencies include:

- `pandas`: Data manipulation
- `numpy`: Numerical computing
- `scikit-learn`: Machine learning algorithms
- `xgboost`: XGBoost implementation
- `lightgbm`: LightGBM implementation
- `streamlit`: Web application framework
- `joblib`: Model serialization

## 🔍 Model Performance

After training, the notebook displays:
- Comparison of all models
- Best model selection
- Feature importance analysis
- Confusion matrix
- Classification report

## 🚢 Deployment

The Streamlit app can be deployed to:
- **Streamlit Cloud**: Free hosting for Streamlit apps
- **Heroku**: Cloud platform
- **AWS/GCP/Azure**: Cloud providers
- **Docker**: Containerized deployment

## 📚 Future Enhancements

- Batch prediction functionality
- Model retraining pipeline
- Real-time data integration
- Advanced visualizations
- A/B testing framework
- Model monitoring and logging

## 🤝 Contributing

This is a project for learning and demonstration purposes. Feel free to:
- Experiment with different models
- Add new features
- Improve the UI/UX
- Optimize model performance

## 📄 License

This project is for educational purposes.

## 👤 Author

Created as part of a machine learning project for customer personality prediction.

---

**Note**: Make sure to run the Jupyter notebook first to generate the model files before using the web application.

