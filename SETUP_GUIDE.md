# Step-by-Step Setup and Deployment Guide

This guide will walk you through running the project manually and deploying the web application.

## 📋 Prerequisites

- Python 3.8 or higher installed
- pip (Python package installer)
- Terminal/Command Prompt access
- Web browser

---

## 🚀 Part 1: Manual Setup and Running

### Step 1: Open Terminal/Command Prompt

- **Mac/Linux**: Open Terminal
- **Windows**: Open Command Prompt or PowerShell
- Navigate to your project directory:
  ```bash
  cd /Users/ansh/P1
  ```
  (On Windows, use: `cd C:\path\to\P1`)

### Step 2: Check Python Installation

```bash
python3 --version
# or on Windows:
python --version
```

You should see Python 3.8 or higher.

### Step 3: Install Dependencies

```bash
# Install all required packages
python3 -m pip install -r requirements.txt

# If you get permission errors, use:
python3 -m pip install --user -r requirements.txt
```

**Expected output**: Packages will be installed. This may take a few minutes.

**Troubleshooting**:
- If `xgboost` or `lightgbm` fail to install, that's okay - the app will work with Random Forest and Neural Network models
- On Mac, if you see OpenMP errors for XGBoost, run: `brew install libomp`

### Step 4: Train the Machine Learning Model

You have two options:

#### Option A: Run the Training Script (Faster)

```bash
python3 train_model.py
```

This will:
- Load and preprocess the dataset
- Train multiple models (Random Forest, XGBoost, LightGBM, Neural Network)
- Compare models and select the best one
- Save the model to `models/` directory

**Expected output**: You'll see progress messages and finally:
```
✅ Model training completed successfully!
Best Model: Random Forest
ROC-AUC: 0.8841
Accuracy: 88.39%
```

#### Option B: Run the Jupyter Notebook (Interactive)

1. **Start Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```
   This will open Jupyter in your browser.

2. **Open the notebook**:
   - Click on `customer_personality_prediction.ipynb`
   - Run all cells: Click "Cell" → "Run All"
   - Or run cells one by one: Click each cell and press `Shift + Enter`

3. **Wait for completion**: The notebook will train models and save them.

**Note**: After training, you should see a `models/` directory with:
- `best_model.pkl` - The trained model
- `scaler.pkl` - Data scaler
- `le_education.pkl` - Education encoder
- `le_marital.pkl` - Marital status encoder
- `feature_names.json` - Feature names
- `model_metadata.json` - Model metadata

### Step 5: Run the Web Application

```bash
python3 -m streamlit run app.py
```

**Note:** If you get "command not found: streamlit", always use `python3 -m streamlit` instead.

**Expected output**:
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

The app will automatically open in your browser. If not, copy the Local URL and paste it in your browser.

### Step 6: Use the Application

1. **Single Prediction Tab**:
   - Enter customer information
   - Click "Predict Response"
   - View predictions and recommendations

2. **Batch Prediction Tab**:
   - Upload your CSV/TSV file (like `marketing_campaign.csv.xls`)
   - Click "Predict for All Customers"
   - View results and download predictions

### Step 7: Stop the Application

Press `Ctrl + C` in the terminal to stop the Streamlit app.

---

## 🌐 Part 2: Deploying the Website

You have several options to deploy your Streamlit app:

### Option 1: Streamlit Cloud (Easiest & Free) ⭐ Recommended

**Steps**:

1. **Create a GitHub account** (if you don't have one):
   - Go to https://github.com
   - Sign up for free

2. **Create a GitHub repository**:
   - Click "New repository"
   - Name it (e.g., `customer-personality-prediction`)
   - Make it public (required for free Streamlit Cloud)
   - Click "Create repository"

3. **Upload your files to GitHub**:
   ```bash
   # Initialize git (if not already done)
   git init
   
   # Add all files
   git add .
   
   # Commit
   git commit -m "Initial commit"
   
   # Add remote repository (replace YOUR_USERNAME and REPO_NAME)
   git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
   
   # Push to GitHub
   git branch -M main
   git push -u origin main
   ```
   
   **Or use GitHub Desktop** (easier for beginners):
   - Download GitHub Desktop
   - Add your repository
   - Commit and push files

4. **Deploy on Streamlit Cloud**:
   - Go to https://share.streamlit.io
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file path: `app.py`
   - Click "Deploy"

5. **Wait for deployment** (2-3 minutes)

6. **Your app is live!** You'll get a URL like: `https://your-app-name.streamlit.app`

**Important**: Make sure `models/` directory is in your repository (don't add it to `.gitignore`)

---

### Option 2: Heroku (Free tier available)

**Steps**:

1. **Install Heroku CLI**:
   - Download from https://devcenter.heroku.com/articles/heroku-cli

2. **Create required files**:

   Create `Procfile` (no extension):
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

   Create `setup.sh`:
   ```bash
   mkdir -p ~/.streamlit/
   echo "\
   [server]\n\
   headless = true\n\
   port = $PORT\n\
   enableCORS = false\n\
   \n\
   " > ~/.streamlit/config.toml
   ```

   Update `Procfile`:
   ```
   web: sh setup.sh && streamlit run app.py
   ```

3. **Deploy**:
   ```bash
   # Login to Heroku
   heroku login
   
   # Create app
   heroku create your-app-name
   
   # Deploy
   git push heroku main
   ```

---

### Option 3: AWS/Azure/GCP (Advanced)

These require cloud accounts and more setup. See Streamlit documentation for details.

---

### Option 4: Docker Deployment

1. **Create `Dockerfile`**:
   ```dockerfile
   FROM python:3.9-slim
   
   WORKDIR /app
   
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   
   COPY . .
   
   EXPOSE 8501
   
   HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
   
   ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. **Build and run**:
   ```bash
   docker build -t customer-prediction-app .
   docker run -p 8501:8501 customer-prediction-app
   ```

---

## 📝 Quick Reference Commands

### Daily Usage

```bash
# 1. Navigate to project
cd /Users/ansh/P1

# 2. Run the app
streamlit run app.py

# 3. Stop the app
# Press Ctrl + C
```

### Retrain Model

```bash
# Run training script
python3 train_model.py

# Or use Jupyter notebook
jupyter notebook customer_personality_prediction.ipynb
```

### Check if Everything Works

```bash
# Check Python version
python3 --version

# Check if packages are installed
python3 -m pip list | grep streamlit

# Check if model exists
ls -la models/

# Test the app
streamlit run app.py
```

---

## 🐛 Troubleshooting

### Issue: "Module not found" error

**Solution**:
```bash
python3 -m pip install -r requirements.txt
```

### Issue: "Model not found" error

**Solution**: Run the training script first:
```bash
python3 train_model.py
```

### Issue: Port 8501 already in use

**Solution**: 
```bash
# Kill the process using port 8501
lsof -ti:8501 | xargs kill -9

# Or use a different port
streamlit run app.py --server.port 8502
```

### Issue: Streamlit Cloud deployment fails

**Solutions**:
- Make sure `models/` directory is committed to GitHub
- Check that `app.py` is in the root directory
- Verify `requirements.txt` includes all dependencies
- Check deployment logs in Streamlit Cloud dashboard

### Issue: File upload not working in batch prediction

**Solutions**:
- Make sure file is CSV or TSV format
- Check that all required columns are present
- Verify column names match exactly (case-sensitive)
- Try the sample file: `sample_batch_input.csv`

---

## 📚 Additional Resources

- **Streamlit Documentation**: https://docs.streamlit.io
- **Streamlit Cloud**: https://streamlit.io/cloud
- **GitHub**: https://github.com
- **Python Documentation**: https://docs.python.org

---

## ✅ Checklist Before Deployment

- [ ] All dependencies installed (`requirements.txt`)
- [ ] Model trained and saved (`models/` directory exists)
- [ ] App runs locally without errors
- [ ] Tested single prediction
- [ ] Tested batch prediction with sample file
- [ ] Code pushed to GitHub (for Streamlit Cloud)
- [ ] `models/` directory included in repository
- [ ] `requirements.txt` is up to date

---

## 🎉 You're All Set!

Once deployed, you can:
- Share the URL with others
- Use it on any device
- Make predictions from anywhere
- No need to run locally anymore!

Good luck! 🚀

