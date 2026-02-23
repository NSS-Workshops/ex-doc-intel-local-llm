#!/bin/bash

# Invoice Processor Setup Script
# This script installs all dependencies for the invoice processing system

set -e  # Exit on error

echo "=========================================="
echo "Invoice Processor Setup"
echo "=========================================="
echo ""

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
else
    echo "Unsupported operating system: $OSTYPE"
    echo "Please install Tesseract OCR manually."
    exit 1
fi

echo "Detected OS: $OS"
echo ""

# Install Tesseract OCR
echo "Step 1: Installing Tesseract OCR..."
echo "----------------------------------------"

if [ "$OS" == "linux" ]; then
    # Check if running as root or with sudo
    if [ "$EUID" -ne 0 ]; then
        echo "Installing Tesseract OCR (requires sudo)..."
        sudo apt-get update
        sudo apt-get install -y tesseract-ocr
    else
        echo "Installing Tesseract OCR..."
        apt-get update
        apt-get install -y tesseract-ocr
    fi
elif [ "$OS" == "macos" ]; then
    if command -v brew &> /dev/null; then
        echo "Installing Tesseract OCR via Homebrew..."
        brew install tesseract
    else
        echo "Error: Homebrew not found. Please install Homebrew first:"
        echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi
fi

# Verify Tesseract installation
if command -v tesseract &> /dev/null; then
    echo "✓ Tesseract OCR installed successfully"
    tesseract --version | head -n 1
else
    echo "✗ Tesseract OCR installation failed"
    exit 1
fi

echo ""

# Install Python dependencies
echo "Step 2: Installing Python dependencies..."
echo "----------------------------------------"

if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "Python version: $(python3 --version)"

# Check if pip is available
if ! command -v pip3 &> /dev/null && ! command -v pip &> /dev/null; then
    echo "Error: pip is not installed. Please install pip first."
    exit 1
fi

# Use pip3 if available, otherwise pip
PIP_CMD="pip3"
if ! command -v pip3 &> /dev/null; then
    PIP_CMD="pip"
fi

echo "Installing Python packages from requirements.txt..."
$PIP_CMD install -r requirements.txt

echo "✓ Python dependencies installed successfully"
echo ""

# Create models directory
echo "Step 3: Setting up models directory..."
echo "----------------------------------------"

if [ ! -d "models" ]; then
    mkdir -p models
    echo "✓ Created models directory"
else
    echo "✓ Models directory already exists"
fi

echo ""

# Download model if it doesn't exist
MODEL_FILE="models/Phi-3.5-mini-instruct-Q4_K_M.gguf"
MODEL_URL="https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf"

if [ ! -f "$MODEL_FILE" ]; then
    echo "Downloading LLM model (Phi-3.5-mini-instruct-Q4_K_M.gguf)..."
    echo "This is a ~2.4GB file and may take several minutes..."
    echo ""
    
    # Check if wget or curl is available
    if command -v wget &> /dev/null; then
        echo "Using wget to download..."
        wget -O "$MODEL_FILE" "$MODEL_URL" --progress=bar:force 2>&1
    elif command -v curl &> /dev/null; then
        echo "Using curl to download..."
        curl -L -o "$MODEL_FILE" "$MODEL_URL" --progress-bar
    else
        echo "Error: Neither wget nor curl is available."
        echo "Please install wget or curl, then run:"
        echo "  wget -O $MODEL_FILE $MODEL_URL"
        echo "Or download manually from:"
        echo "  https://huggingface.co/microsoft/Phi-3.5-mini-instruct-GGUF"
        exit 1
    fi
    
    # Verify download
    if [ -f "$MODEL_FILE" ]; then
        MODEL_SIZE=$(du -h "$MODEL_FILE" | cut -f1)
        echo "✓ Model downloaded successfully ($MODEL_SIZE)"
    else
        echo "✗ Model download failed"
        exit 1
    fi
else
    echo "✓ LLM model already exists"
    MODEL_SIZE=$(du -h "$MODEL_FILE" | cut -f1)
    echo "  Size: $MODEL_SIZE"
fi

echo ""

# Create output directory
echo "Step 4: Setting up output directory..."
echo "----------------------------------------"

if [ ! -d "output" ]; then
    mkdir -p output
    echo "✓ Created output directory"
else
    echo "✓ Output directory already exists"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
if [ ! -f "$MODEL_FILE" ]; then
    echo "  1. Download the LLM model (see instructions above)"
    echo "  2. Run: python main.py path/to/invoice.jpg"
else
    echo "  1. Run: python main.py path/to/invoice.jpg"
fi
echo ""
echo "For more information, see README.md"
