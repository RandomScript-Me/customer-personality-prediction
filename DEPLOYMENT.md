# Deployment Guide - Quick Reference

## 🚀 Streamlit Cloud Deployment (Recommended)

### Prerequisites
1. GitHub account (free)
2. Your code pushed to GitHub

### Steps

1. **Prepare your repository**:
   ```bash
   # Make sure models/ directory is included
   git add models/
   git commit -m "Add trained models"
   git push
   ```

2. **Deploy**:
   - Go to https://share.streamlit.io
   - Sign in with GitHub
   - Click "New app"
   - Select repository
   - Main file: `app.py`
   - Click "Deploy"

3. **Your app URL**: `https://your-app-name.streamlit.app`

### Important Notes
- Repository must be **public** (for free tier)
- `models/` directory must be in the repository
- `requirements.txt` must be present
- App will automatically redeploy on git push

---

## 🐳 Docker Deployment

### Build and Run Locally
```bash
docker build -t customer-prediction-app .
docker run -p 8501:8501 customer-prediction-app
```

### Deploy to Cloud
- **AWS**: Use ECS or EC2
- **Google Cloud**: Use Cloud Run
- **Azure**: Use Container Instances

---

## ☁️ Heroku Deployment

1. **Install Heroku CLI**
2. **Login**: `heroku login`
3. **Create app**: `heroku create your-app-name`
4. **Deploy**: `git push heroku main`

---

## 📋 Files Needed for Deployment

- ✅ `app.py` - Main application
- ✅ `requirements.txt` - Dependencies
- ✅ `models/` - Trained models (IMPORTANT!)
- ✅ `Procfile` - For Heroku
- ✅ `setup.sh` - For Heroku
- ✅ `Dockerfile` - For Docker

---

## 🔒 Security Notes

- Don't commit sensitive data
- Use environment variables for API keys
- Keep models in repository (they're not sensitive)
- Consider authentication for production

---

## 📞 Need Help?

- Streamlit Community: https://discuss.streamlit.io
- Streamlit Docs: https://docs.streamlit.io
- GitHub Issues: Create an issue in your repo

