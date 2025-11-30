#!/bin/bash

# Quick start script for Customer Personality Prediction App

echo "🎯 Customer Personality Prediction App"
echo "======================================"
echo ""

# Check if models directory exists
if [ ! -d "models" ]; then
    echo "⚠️  Models directory not found!"
    echo "📝 Please run the Jupyter notebook first to train and save the model."
    echo "   Run: jupyter notebook customer_personality_prediction.ipynb"
    echo ""
    read -p "Do you want to continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
    echo ""
fi

echo "🚀 Starting Streamlit app..."
echo "📱 The app will open in your default browser"
echo "🔗 If not, navigate to: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

streamlit run app.py

