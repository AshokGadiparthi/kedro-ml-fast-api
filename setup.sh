#!/bin/bash

echo "🚀 FastAPI Setup - 100% Working"
echo "=================================="
echo ""

# Check Python
echo "🔍 Checking Python..."
python3 --version || { echo "Python 3 not found!"; exit 1; }

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt

# Create .env if not exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please edit .env and set KEDRO_PROJECT_PATH"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env and update KEDRO_PROJECT_PATH"
echo "2. Terminal 1: redis-server"
echo "3. Terminal 2: celery -A worker worker --loglevel=info"
echo "4. Terminal 3: python main.py"
echo "5. Open: http://localhost:8000/docs"
echo ""
