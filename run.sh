#!/bin/bash

# CryptoTrader Pro - Run Script

echo "🚀 CryptoTrader Pro - Advanced Cryptocurrency Trading Bot"
echo "=========================================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies if needed
if [ ! -f "venv/installed" ]; then
    echo "📥 Installing dependencies..."
    pip install -r requirements.txt
    touch venv/installed
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚙️ Creating .env file from template..."
    cp .env.example .env
    echo "🔑 Please edit .env file with your API keys before running in live mode"
fi

# Run the application
echo "🎯 Starting CryptoTrader Pro..."
echo "💻 Web interface will be available at: http://localhost:8080"
echo "🛑 Press Ctrl+C to stop"
echo ""

python main.py --mode web