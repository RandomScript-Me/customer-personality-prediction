#!/bin/bash

# Simple script to start the Streamlit app

echo "🚀 Starting Customer Personality Prediction App..."
echo ""

# Kill any process on port 8501
echo "Checking port 8501..."
if lsof -ti:8501 > /dev/null 2>&1; then
    echo "⚠️  Port 8501 is in use. Killing existing process..."
    lsof -ti:8501 | xargs kill -9 2>/dev/null
    sleep 1
fi

# Check if model exists
if [ ! -d "models" ] || [ ! -f "models/best_model.pkl" ]; then
    echo "⚠️  Model not found! Training model first..."
    python3 train_model.py
    echo ""
fi

# Start the app
echo "✅ Starting app on http://localhost:8501"
echo "📱 The app will open in your browser automatically"
echo "🛑 Press Ctrl+C to stop the app"
echo ""

python3 -m streamlit run app.py

