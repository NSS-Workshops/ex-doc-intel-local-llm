#!/bin/bash
# Setup script for Invoice Processor

set -e

echo "=========================================="
echo "Invoice Processor Setup"
echo "=========================================="

# Check Python version
echo ""
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create models directory
echo ""
echo "Creating models directory..."
mkdir -p models

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
echo "This may take a few minutes (llama-cpp-python needs to compile)..."
pip install -r requirements.txt

# Check if Tesseract is installed
echo ""
echo "Checking for Tesseract OCR..."
if command -v tesseract &> /dev/null; then
    tesseract_version=$(tesseract --version 2>&1 | head -n 1)
    echo "✓ Tesseract found: $tesseract_version"
else
    echo "✗ Tesseract not found!"
    echo ""
    echo "Please install Tesseract OCR:"
    echo "  macOS:   brew install tesseract"
    echo "  Ubuntu:  sudo apt-get install tesseract-ocr"
    echo "  Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki"
fi

# Check if model exists
echo ""
echo "Checking for LLM model..."
model_path="./models/Phi-3.5-mini-instruct-Q4_K_M.gguf"

if [ -f "$model_path" ]; then
    echo "✓ Model found: $model_path"
    model_size=$(du -h "$model_path" | cut -f1)
    echo "  Size: $model_size"
else
    echo "✗ Model not found: $model_path"
    echo ""
    echo "To download the model:"
    echo "  1. Visit: https://huggingface.co/microsoft/Phi-3.5-mini-instruct-GGUF"
    echo "  2. Download: Phi-3.5-mini-instruct-Q4_K_M.gguf"
    echo "  3. Place it in: ./models/"
    echo ""
    echo "Or use wget:"
    echo "  cd models"
    echo "  wget https://huggingface.co/microsoft/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Ensure Tesseract OCR is installed"
echo "  2. Download the model if not present"
echo "  3. Run: python main.py path/to/invoice.jpg"
echo ""
