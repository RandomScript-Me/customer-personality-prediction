# ⚡ Quick Start Guide

## 🎯 Run Everything in 3 Steps

### Step 1️⃣: Install Dependencies
```bash
cd /Users/ansh/P1
python3 -m pip install -r requirements.txt
```

### Step 2️⃣: Train the Model
```bash
python3 train_model.py
```
⏱️ Takes 2-5 minutes

### Step 3️⃣: Run the App
```bash
python3 -m streamlit run app.py
```
🌐 Opens at http://localhost:8501

**Note:** If you get "command not found: streamlit", use `python3 -m streamlit` instead of just `streamlit`

---

## 📋 What Each File Does

| File | Purpose | When to Use |
|------|---------|-------------|
| `train_model.py` | Trains ML models | First time setup, or when you want to retrain |
| `app.py` | Web application | Every time you want to use the app |
| `customer_personality_prediction.ipynb` | Interactive notebook | When you want to explore/experiment |
| `requirements.txt` | Lists all packages needed | First time setup |

---

## 🌐 Deploy to Web (Streamlit Cloud)

### 5 Simple Steps:

1. **Create GitHub account** → https://github.com
2. **Create new repository** → Make it public
3. **Upload your files** → Push code to GitHub
4. **Go to Streamlit Cloud** → https://share.streamlit.io
5. **Deploy** → Select repo, click Deploy

**Done!** Your app is live on the internet! 🎉

---

## 🆘 Common Issues

**"Module not found"**
→ Run: `python3 -m pip install -r requirements.txt`

**"Model not found"**
→ Run: `python3 train_model.py` first

**"Port already in use"**
→ Use: `streamlit run app.py --server.port 8502`

---

## 📚 Need More Details?

- **Full Setup Guide**: See [SETUP_GUIDE.md](SETUP_GUIDE.md)
- **Deployment Guide**: See [DEPLOYMENT.md](DEPLOYMENT.md)
- **Main README**: See [README.md](README.md)

---

## ✅ Checklist

- [ ] Python 3.8+ installed
- [ ] Dependencies installed
- [ ] Model trained (`models/` folder exists)
- [ ] App runs locally
- [ ] Ready to deploy!

---

**That's it! You're ready to go! 🚀**

