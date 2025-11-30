# 🚀 How to Run the App - Step by Step

## Quick Start (3 Steps)

### Step 1: Make sure port 8501 is free
```bash
# Kill any process using port 8501 (if needed)
lsof -ti:8501 | xargs kill -9
```

### Step 2: Navigate to project directory
```bash
cd /Users/ansh/P1
```

### Step 3: Run the app
```bash
python3 -m streamlit run app.py
```

## ✅ That's it!

The app will:
- Start on port 8501
- Automatically open in your browser
- If it doesn't open automatically, go to: **http://localhost:8501**

---

## 🛑 How to Stop the App

Press `Ctrl + C` in the terminal where the app is running.

---

## 🔧 Troubleshooting

### Issue: "Port 8501 is already in use"

**Solution:**
```bash
# Kill the process using port 8501
lsof -ti:8501 | xargs kill -9

# Then run the app again
python3 -m streamlit run app.py
```

**Or use a different port:**
```bash
python3 -m streamlit run app.py --server.port 8502
```
Then open: http://localhost:8502

---

### Issue: "command not found: streamlit"

**Solution:** Always use:
```bash
python3 -m streamlit run app.py
```

---

### Issue: "Model not found"

**Solution:** Train the model first:
```bash
python3 train_model.py
```

---

## 📋 Complete Workflow

### First Time Setup:
```bash
# 1. Install dependencies
python3 -m pip install -r requirements.txt

# 2. Train the model
python3 train_model.py

# 3. Run the app
python3 -m streamlit run app.py
```

### Daily Use:
```bash
# Just run the app
cd /Users/ansh/P1
python3 -m streamlit run app.py
```

---

## 🎯 What You'll See

1. **Terminal output** showing the app is starting
2. **Browser opens automatically** with the app
3. **Two tabs** in the app:
   - **Single Prediction**: Enter customer data manually
   - **Batch Prediction**: Upload CSV file for multiple predictions

---

## 💡 Pro Tips

- Keep the terminal open while using the app
- The app auto-reloads when you make code changes
- Check the terminal for any error messages
- Use `Ctrl + C` to stop, then restart if needed

---

## ✅ Success Indicators

You'll know it's working when you see:
- ✅ Terminal shows: "You can now view your Streamlit app..."
- ✅ Browser opens automatically
- ✅ You see the "Customer Personality Prediction Dashboard"

Enjoy! 🎉

