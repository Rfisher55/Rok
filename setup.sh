#!/bin/bash
# ROK — Quick Setup Script

echo ""
echo "  ██████╗  ██████╗ ██╗  ██╗"
echo "  ██╔══██╗██╔═══██╗██║ ██╔╝"
echo "  ██████╔╝██║   ██║█████╔╝ "
echo "  ██╔══██╗██║   ██║██╔═██╗ "
echo "  ██║  ██║╚██████╔╝██║  ██╗"
echo "  ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝"
echo "  Stock Intelligence Briefing"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found. Install Python 3.11+ first."
    exit 1
fi
echo "✓ Python $(python3 --version)"

# Create venv if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate

# Install deps
echo "Installing dependencies..."
pip install -r requirements.txt -q

# Create .env if missing
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ""
    echo "⚠  .env file created from template."
    echo "   Add your ANTHROPIC_API_KEY to .env to enable AI analysis."
    echo "   Get your key at: https://console.anthropic.com"
    echo ""
fi

echo "✓ Setup complete."
echo ""
echo "To start ROK:"
echo "  source venv/bin/activate"
echo "  python app.py"
echo ""
echo "Then open: http://localhost:5000"
echo ""
