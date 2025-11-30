import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="Customer Personality Prediction",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model and preprocessing objects"""
    try:
        model = joblib.load('models/best_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        le_education = joblib.load('models/le_education.pkl')
        le_marital = joblib.load('models/le_marital.pkl')
        
        with open('models/feature_names.json', 'r') as f:
            feature_names = json.load(f)
        
        with open('models/model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        return model, scaler, le_education, le_marital, feature_names, metadata
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please run the Jupyter notebook first to train and save the model.")
        return None, None, None, None, None, None

def create_features(input_data):
    """Create all required features from user input"""
    features = input_data.copy()
    
    # Calculate derived features
    features['Total_Spent'] = (features['MntWines'] + features['MntFruits'] + 
                               features['MntMeatProducts'] + features['MntFishProducts'] + 
                               features['MntSweetProducts'] + features['MntGoldProds'])
    
    features['Total_Purchases'] = (features['NumDealsPurchases'] + features['NumWebPurchases'] + 
                                   features['NumCatalogPurchases'] + features['NumStorePurchases'])
    
    features['Total_Accepted_Campaigns'] = (features['AcceptedCmp1'] + features['AcceptedCmp2'] + 
                                             features['AcceptedCmp3'] + features['AcceptedCmp4'] + 
                                             features['AcceptedCmp5'])
    
    features['Avg_Purchase_Value'] = features['Total_Spent'] / (features['Total_Purchases'] + 1)
    features['Children'] = features['Kidhome'] + features['Teenhome']
    features['Family_Size'] = features['Children'] + 1
    
    # Spending patterns
    features['Wine_Ratio'] = features['MntWines'] / (features['Total_Spent'] + 1)
    features['Meat_Ratio'] = features['MntMeatProducts'] / (features['Total_Spent'] + 1)
    features['Gold_Ratio'] = features['MntGoldProds'] / (features['Total_Spent'] + 1)
    
    # Purchase channel preferences
    features['Web_Purchase_Ratio'] = features['NumWebPurchases'] / (features['Total_Purchases'] + 1)
    features['Store_Purchase_Ratio'] = features['NumStorePurchases'] / (features['Total_Purchases'] + 1)
    features['Catalog_Purchase_Ratio'] = features['NumCatalogPurchases'] / (features['Total_Purchases'] + 1)
    
    return features

def predict_customer_response(model, scaler, le_education, le_marital, feature_names, metadata, input_data):
    """Make prediction using the trained model"""
    try:
        # Create all features
        features = create_features(input_data)
        
        # Encode categorical variables
        features['Education_Encoded'] = le_education.transform([features['Education']])[0]
        features['Marital_Status_Encoded'] = le_marital.transform([features['Marital_Status']])[0]
        
        # Drop original categorical columns
        features = features.drop(['Education', 'Marital_Status'])
        
        # Create feature vector in correct order
        feature_vector = np.array([features[col] for col in feature_names]).reshape(1, -1)
        
        # Scale if needed
        if metadata['use_scaled']:
            feature_vector = scaler.transform(feature_vector)
        
        # Predict
        prediction = model.predict(feature_vector)[0]
        probability = model.predict_proba(feature_vector)[0]
        
        return prediction, probability
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None

def preprocess_batch_data(df):
    """Preprocess batch CSV data similar to training pipeline"""
    df_processed = df.copy()
    original_cols = list(df.columns)  # Store original columns for better error messages
    
    # Strip whitespace from column names
    df_processed.columns = df_processed.columns.str.strip()
    
    # Handle missing values in Income
    if 'Income' in df_processed.columns:
        # Convert to numeric, handling any non-numeric values
        df_processed['Income'] = pd.to_numeric(df_processed['Income'], errors='coerce')
        df_processed['Income'] = df_processed['Income'].fillna(df_processed['Income'].median())
    
    # Calculate Customer_Age if Year_Birth is provided
    if 'Year_Birth' in df_processed.columns and 'Customer_Age' not in df_processed.columns:
        df_processed['Year_Birth'] = pd.to_numeric(df_processed['Year_Birth'], errors='coerce')
        df_processed['Customer_Age'] = 2024 - df_processed['Year_Birth']
    elif 'Customer_Age' in df_processed.columns:
        df_processed['Customer_Age'] = pd.to_numeric(df_processed['Customer_Age'], errors='coerce')
    
    # Calculate Days_Since_Customer if Dt_Customer is provided
    if 'Dt_Customer' in df_processed.columns and 'Days_Since_Customer' not in df_processed.columns:
        # Try multiple date formats
        df_processed['Dt_Customer'] = pd.to_datetime(
            df_processed['Dt_Customer'], 
            format='%d-%m-%Y', 
            errors='coerce'
        )
        # If that fails, try other common formats
        if df_processed['Dt_Customer'].isna().any():
            df_processed['Dt_Customer'] = pd.to_datetime(
                df_processed['Dt_Customer'], 
                errors='coerce'
            )
        df_processed['Days_Since_Customer'] = (pd.Timestamp('2024-01-01') - df_processed['Dt_Customer']).dt.days
        # Fill any remaining NaN values with median
        if df_processed['Days_Since_Customer'].isna().any():
            median_days = df_processed['Days_Since_Customer'].median()
            if pd.isna(median_days):
                median_days = 365  # Default to 1 year if no valid dates
            df_processed['Days_Since_Customer'] = df_processed['Days_Since_Customer'].fillna(median_days)
    elif 'Days_Since_Customer' in df_processed.columns:
        df_processed['Days_Since_Customer'] = pd.to_numeric(df_processed['Days_Since_Customer'], errors='coerce')
    
    # Ensure numeric columns are numeric
    numeric_cols = ['Kidhome', 'Teenhome', 'Recency', 'MntWines', 'MntFruits', 
                    'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
                    'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 
                    'NumStorePurchases', 'NumWebVisitsMonth', 'AcceptedCmp1', 'AcceptedCmp2',
                    'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Complain']
    
    for col in numeric_cols:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0)
    
    # Now check for required columns AFTER all preprocessing (Customer_Age and Days_Since_Customer should exist now)
    required_base_cols = ['Income', 'Kidhome', 'Teenhome', 'Recency', 'MntWines', 'MntFruits', 
                          'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
                          'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 
                          'NumStorePurchases', 'NumWebVisitsMonth', 'AcceptedCmp1', 'AcceptedCmp2',
                          'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Complain', 
                          'Customer_Age', 'Days_Since_Customer', 'Education', 'Marital_Status']
    
    # Check what columns we actually have
    available_cols = list(df_processed.columns)
    missing_cols = [col for col in required_base_cols if col not in df_processed.columns]
    
    if missing_cols:
        # Provide helpful error message with available columns
        error_msg = f"Missing required columns: {', '.join(missing_cols)}\n\n"
        error_msg += f"Available columns in your file: {', '.join(available_cols[:20])}"
        if len(available_cols) > 20:
            error_msg += f" ... and {len(available_cols) - 20} more"
        
        # Provide helpful hints for missing columns
        hints = []
        if 'Customer_Age' in missing_cols:
            # Check if Year_Birth exists in original or processed columns
            if 'Year_Birth' in original_cols or 'Year_Birth' in df_processed.columns:
                hints.append("⚠️ Found 'Year_Birth' but couldn't calculate 'Customer_Age'. Please check that Year_Birth contains valid year values (e.g., 1950-2010).")
            else:
                hints.append("✗ Need either 'Customer_Age' or 'Year_Birth' column")
        
        if 'Days_Since_Customer' in missing_cols:
            # Check if Dt_Customer exists in original or processed columns
            if 'Dt_Customer' in original_cols or 'Dt_Customer' in df_processed.columns:
                hints.append("⚠️ Found 'Dt_Customer' but couldn't calculate 'Days_Since_Customer'. Please check that dates are in DD-MM-YYYY format (e.g., 04-09-2012).")
            else:
                hints.append("✗ Need either 'Days_Since_Customer' or 'Dt_Customer' column (format: DD-MM-YYYY)")
        
        if hints:
            error_msg += "\n\n" + "\n".join(hints)
        
        raise ValueError(error_msg)
    
    return df_processed

def predict_batch(model, scaler, le_education, le_marital, feature_names, metadata, df_processed):
    """Make batch predictions"""
    try:
        # Create all features for each row
        features_list = []
        
        for idx, row in df_processed.iterrows():
            # Create features for this row
            row_features = create_features(row)
            
            # Encode categorical variables
            try:
                row_features['Education_Encoded'] = le_education.transform([row_features['Education']])[0]
            except ValueError:
                # If education value not seen during training, use most common
                row_features['Education_Encoded'] = le_education.transform([le_education.classes_[0]])[0]
            
            try:
                row_features['Marital_Status_Encoded'] = le_marital.transform([row_features['Marital_Status']])[0]
            except ValueError:
                # If marital status not seen during training, use most common
                row_features['Marital_Status_Encoded'] = le_marital.transform([le_marital.classes_[0]])[0]
            
            # Drop original categorical columns
            row_features = row_features.drop(['Education', 'Marital_Status'])
            
            # Create feature vector in correct order
            feature_vector = np.array([row_features[col] for col in feature_names])
            features_list.append(feature_vector)
        
        # Convert to numpy array
        feature_matrix = np.array(features_list)
        
        # Scale if needed
        if metadata['use_scaled']:
            feature_matrix = scaler.transform(feature_matrix)
        
        # Predict
        predictions = model.predict(feature_matrix)
        probabilities = model.predict_proba(feature_matrix)
        
        return predictions, probabilities
    except Exception as e:
        raise Exception(f"Error making batch predictions: {str(e)}")

# Main app
def main():
    st.markdown('<div class="main-header">🎯 Customer Personality Prediction Dashboard</div>', unsafe_allow_html=True)
    
    # Load model
    model, scaler, le_education, le_marital, feature_names, metadata = load_model()
    
    if model is None:
        st.stop()
    
    # Sidebar with model info
    with st.sidebar:
        st.header("📊 Model Information")
        st.write(f"**Model:** {metadata['model_name']}")
        st.write(f"**Accuracy:** {metadata['accuracy']:.2%}")
        st.write(f"**ROC-AUC:** {metadata['roc_auc']:.4f}")
        st.write(f"**Precision:** {metadata['precision']:.2%}")
        st.write(f"**Recall:** {metadata['recall']:.2%}")
        st.write(f"**F1-Score:** {metadata['f1_score']:.4f}")
        
        st.markdown("---")
        st.header("ℹ️ About")
        st.write("This application predicts customer response to marketing campaigns based on demographic, behavioral, and purchase history data.")
        st.write("**Target:** Response (1 = Will respond, 0 = Will not respond)")
    
    # Main content
    tab1, tab2 = st.tabs(["🔮 Single Prediction", "📊 Batch Prediction"])
    
    with tab1:
        st.header("Predict Customer Response")
        st.write("Enter customer information to predict their response to marketing campaigns.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Demographic Information")
            year_birth = st.number_input("Year of Birth", min_value=1900, max_value=2010, value=1980)
            education = st.selectbox("Education", le_education.classes_)
            marital_status = st.selectbox("Marital Status", le_marital.classes_)
            income = st.number_input("Income", min_value=0, value=50000, step=1000)
            kidhome = st.number_input("Number of Kids at Home", min_value=0, max_value=5, value=0)
            teenhome = st.number_input("Number of Teens at Home", min_value=0, max_value=5, value=0)
            
            # Calculate age and days since customer
            customer_age = 2024 - year_birth
            dt_customer = st.date_input("Customer Since", value=datetime(2013, 1, 1))
            days_since_customer = (datetime.now().date() - dt_customer).days
            
            recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=365, value=30)
        
        with col2:
            st.subheader("Purchase History")
            st.write("**Spending Amounts:**")
            mnt_wines = st.number_input("Wines", min_value=0, value=100, step=10)
            mnt_fruits = st.number_input("Fruits", min_value=0, value=20, step=5)
            mnt_meat = st.number_input("Meat Products", min_value=0, value=50, step=5)
            mnt_fish = st.number_input("Fish Products", min_value=0, value=20, step=5)
            mnt_sweet = st.number_input("Sweet Products", min_value=0, value=10, step=5)
            mnt_gold = st.number_input("Gold Products", min_value=0, value=20, step=5)
            
            st.write("**Purchase Counts:**")
            num_deals = st.number_input("Deal Purchases", min_value=0, value=2, step=1)
            num_web = st.number_input("Web Purchases", min_value=0, value=4, step=1)
            num_catalog = st.number_input("Catalog Purchases", min_value=0, value=2, step=1)
            num_store = st.number_input("Store Purchases", min_value=0, value=5, step=1)
            num_web_visits = st.number_input("Web Visits per Month", min_value=0, value=5, step=1)
            
            st.write("**Campaign Responses:**")
            accepted_cmp1 = st.checkbox("Accepted Campaign 1", value=False)
            accepted_cmp2 = st.checkbox("Accepted Campaign 2", value=False)
            accepted_cmp3 = st.checkbox("Accepted Campaign 3", value=False)
            accepted_cmp4 = st.checkbox("Accepted Campaign 4", value=False)
            accepted_cmp5 = st.checkbox("Accepted Campaign 5", value=False)
            
            complain = st.checkbox("Complained", value=False)
        
        # Prepare input data
        input_data = pd.DataFrame({
            'Income': [income],
            'Kidhome': [kidhome],
            'Teenhome': [teenhome],
            'Recency': [recency],
            'MntWines': [mnt_wines],
            'MntFruits': [mnt_fruits],
            'MntMeatProducts': [mnt_meat],
            'MntFishProducts': [mnt_fish],
            'MntSweetProducts': [mnt_sweet],
            'MntGoldProds': [mnt_gold],
            'NumDealsPurchases': [num_deals],
            'NumWebPurchases': [num_web],
            'NumCatalogPurchases': [num_catalog],
            'NumStorePurchases': [num_store],
            'NumWebVisitsMonth': [num_web_visits],
            'AcceptedCmp1': [1 if accepted_cmp1 else 0],
            'AcceptedCmp2': [1 if accepted_cmp2 else 0],
            'AcceptedCmp3': [1 if accepted_cmp3 else 0],
            'AcceptedCmp4': [1 if accepted_cmp4 else 0],
            'AcceptedCmp5': [1 if accepted_cmp5 else 0],
            'Complain': [1 if complain else 0],
            'Customer_Age': [customer_age],
            'Days_Since_Customer': [days_since_customer],
            'Education': [education],
            'Marital_Status': [marital_status]
        })
        
        # Predict button
        if st.button("🔮 Predict Response", type="primary", use_container_width=True):
            prediction, probability = predict_customer_response(
                model, scaler, le_education, le_marital, feature_names, metadata, input_data.iloc[0]
            )
            
            if prediction is not None:
                st.markdown("---")
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 1:
                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.success("✅ **Prediction: Customer WILL RESPOND**")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                        st.warning("❌ **Prediction: Customer WILL NOT RESPOND**")
                        st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.metric("Probability of Response", f"{probability[1]:.2%}")
                    st.metric("Probability of No Response", f"{probability[0]:.2%}")
                
                # Progress bar
                st.progress(probability[1])
                st.caption(f"Confidence: {probability[1]:.2%}")
                
                # Recommendations
                st.markdown("### 💡 Marketing Recommendations")
                if prediction == 1:
                    st.success("""
                    **Recommended Actions:**
                    - This customer is likely to respond positively to marketing campaigns
                    - Consider offering personalized deals based on their purchase history
                    - Engage through their preferred channels (Web/Store/Catalog)
                    - Maintain regular communication to build loyalty
                    """)
                else:
                    st.info("""
                    **Recommended Actions:**
                    - This customer may need different engagement strategies
                    - Consider analyzing their preferences more deeply
                    - Try alternative marketing channels
                    - Focus on building trust and value proposition
                    """)
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.header("Batch Prediction")
        st.write("Upload a CSV or TSV file with customer data to predict responses for multiple customers.")
        st.info("💡 **Note:** Your dataset file (`marketing_campaign.csv.xls`) is tab-separated and can be uploaded directly. The app will automatically detect the file format.")
        
        # Show required columns
        with st.expander("📋 Required CSV Columns", expanded=False):
            st.write("""
            Your CSV file should include the following columns:
            
            **Required Columns:**
            - `Year_Birth` or `Customer_Age` (if Customer_Age is provided, Year_Birth is optional)
            - `Education` (e.g., 'Graduation', 'PhD', 'Master', 'Basic')
            - `Marital_Status` (e.g., 'Single', 'Married', 'Together', 'Divorced')
            - `Income` (numeric)
            - `Kidhome` (number of kids, 0-5)
            - `Teenhome` (number of teens, 0-5)
            - `Dt_Customer` (date format: DD-MM-YYYY) or `Days_Since_Customer` (numeric)
            - `Recency` (days since last purchase)
            - `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds` (spending amounts)
            - `NumDealsPurchases`, `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`, `NumWebVisitsMonth` (purchase counts)
            - `AcceptedCmp1`, `AcceptedCmp2`, `AcceptedCmp3`, `AcceptedCmp4`, `AcceptedCmp5` (0 or 1)
            - `Complain` (0 or 1)
            
            **Optional:** You can include an `ID` column to track customers (will be preserved in results).
            """)
        
        uploaded_file = st.file_uploader("Choose a CSV or TSV file", type=["csv", "txt", "xls"])
        
        if uploaded_file is not None:
            try:
                # Read file - try to detect separator (tab or comma)
                # Read first line to detect separator
                first_bytes = uploaded_file.read(1024)  # Read first 1KB
                uploaded_file.seek(0)  # Reset file pointer
                
                # Decode first line
                if isinstance(first_bytes, bytes):
                    first_line = first_bytes.decode('utf-8', errors='ignore').split('\n')[0]
                else:
                    first_line = str(first_bytes).split('\n')[0]
                
                # Count tabs and commas in first line
                tab_count = first_line.count('\t')
                comma_count = first_line.count(',')
                
                # Determine separator
                if tab_count > comma_count:
                    separator = '\t'
                    st.info("📄 Detected tab-separated file (TSV format)")
                elif comma_count > 0:
                    separator = ','
                    st.info("📄 Detected comma-separated file (CSV format)")
                else:
                    # Default to tab if no commas found (likely TSV)
                    separator = '\t'
                    st.info("📄 Using tab separator (TSV format)")
                
                # Read the file with detected separator
                df = pd.read_csv(uploaded_file, sep=separator, encoding='utf-8', low_memory=False)
                
                # Strip whitespace from column names
                df.columns = df.columns.str.strip()
                
                st.success(f"✅ File uploaded successfully! Found {len(df)} rows and {len(df.columns)} columns.")
                st.write("**Columns found in your file:**")
                st.write(", ".join(df.columns.tolist()))
                st.write("**Preview of uploaded data:**")
                st.dataframe(df.head(10))
                
                if st.button("🔮 Predict for All Customers", type="primary", use_container_width=True):
                    with st.spinner("Processing data and making predictions..."):
                        try:
                            # Preprocess the data
                            df_processed = preprocess_batch_data(df)
                            
                            # Make predictions
                            predictions, probabilities = predict_batch(
                                model, scaler, le_education, le_marital, 
                                feature_names, metadata, df_processed
                            )
                            
                            # Create results dataframe with original data
                            results_df = pd.DataFrame()
                            
                            # Add ID if present
                            if 'ID' in df.columns:
                                results_df['ID'] = df['ID']
                            else:
                                results_df['ID'] = range(1, len(df) + 1)
                            
                            # Add Education and Marital_Status if available (before encoding)
                            if 'Education' in df.columns:
                                results_df['Education'] = df['Education']
                            if 'Marital_Status' in df.columns:
                                results_df['Marital_Status'] = df['Marital_Status']
                            
                            # Add predictions
                            results_df['Predicted_Response'] = predictions
                            results_df['Probability_Response'] = probabilities[:, 1]
                            results_df['Probability_No_Response'] = probabilities[:, 0]
                            results_df['Prediction'] = results_df['Predicted_Response'].map({1: 'Will Respond', 0: 'Will Not Respond'})
                            
                            # Reorder columns
                            col_order = ['ID', 'Predicted_Response', 'Prediction', 'Probability_Response', 
                                       'Probability_No_Response', 'Education', 'Marital_Status']
                            if 'ID' in results_df.columns:
                                results_df = results_df[[col for col in col_order if col in results_df.columns] + 
                                                         [col for col in results_df.columns if col not in col_order]]
                            
                            st.success(f"✅ Predictions completed for {len(results_df)} customers!")
                            
                            # Display summary statistics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Customers", len(results_df))
                            with col2:
                                st.metric("Predicted to Respond", int(results_df['Predicted_Response'].sum()))
                            with col3:
                                st.metric("Predicted Not to Respond", int((results_df['Predicted_Response'] == 0).sum()))
                            with col4:
                                avg_prob = results_df['Probability_Response'].mean()
                                st.metric("Avg Response Probability", f"{avg_prob:.2%}")
                            
                            # Display results
                            st.markdown("### 📊 Prediction Results")
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Download button
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Predictions as CSV",
                                data=csv,
                                file_name=f"customer_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                            
                            # Visualization
                            st.markdown("### 📈 Prediction Distribution")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Response distribution
                                response_counts = results_df['Prediction'].value_counts()
                                st.bar_chart(response_counts)
                            
                            with col2:
                                # Probability distribution
                                st.line_chart(results_df['Probability_Response'])
                            
                            # Recommendations summary
                            st.markdown("### 💡 Marketing Recommendations Summary")
                            will_respond = results_df[results_df['Predicted_Response'] == 1]
                            will_not_respond = results_df[results_df['Predicted_Response'] == 0]
                            
                            if len(will_respond) > 0:
                                st.success(f"""
                                **{len(will_respond)} customers are predicted to respond:**
                                - These customers are likely to engage with marketing campaigns
                                - Consider offering personalized deals based on their purchase history
                                - Engage through their preferred channels
                                - Maintain regular communication to build loyalty
                                """)
                            
                            if len(will_not_respond) > 0:
                                st.info(f"""
                                **{len(will_not_respond)} customers are predicted not to respond:**
                                - These customers may need different engagement strategies
                                - Consider analyzing their preferences more deeply
                                - Try alternative marketing channels
                                - Focus on building trust and value proposition
                                """)
                            
                        except ValueError as e:
                            st.error(f"❌ Validation Error")
                            error_text = str(e)
                            # Split error message into lines for better display
                            error_lines = error_text.split('\n')
                            for line in error_lines:
                                if line.strip():
                                    st.write(line)
                            st.info("💡 **Tip:** Make sure your file has the exact column names as shown in the 'Required CSV Columns' section above. Column names are case-sensitive and should match exactly.")
                        except Exception as e:
                            st.error(f"❌ Error during prediction: {str(e)}")
                            st.exception(e)
                            
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
                st.info("Please ensure your file is a valid CSV format.")

if __name__ == "__main__":
    main()

