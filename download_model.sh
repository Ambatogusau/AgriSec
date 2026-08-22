#!/bin/bash
# AgriSec Model Download Script
# Downloads the Qwen2.5 1.5B model in GGUF format for offline use
# No credentials required - uses publicly available model

set -e

echo "🌾 AgriSec Model Download"
echo "=========================="
echo ""

# Create model directory if it doesn't exist
if [ ! -d "model" ]; then
    echo "📁 Creating model/ directory..."
    mkdir -p model
fi

# Check if model already exists
if [ -f "model/model.gguf" ]; then
    echo "✅ Model already exists at model/model.gguf"
    echo ""
    echo "To use the model with Ollama:"
    echo "  ollama pull qwen2.5:1.5b"
    echo "  python -m src.web_app --model qwen2.5:1.5b"
    exit 0
fi

echo "📥 Downloading Qwen2.5 1.5B GGUF model..."
echo ""
echo "This model is optimized for CPU inference on budget laptops."
echo "Size: ~1.0 GB (compressed)"
echo ""

# Download from Hugging Face (public model, no credentials needed)
MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"

# Use wget or curl (whichever is available)
if command -v wget &> /dev/null; then
    echo "Using wget to download..."
    wget -O model/model.gguf "$MODEL_URL" --progress=bar:force 2>&1
elif command -v curl &> /dev/null; then
    echo "Using curl to download..."
    curl -L -o model/model.gguf "$MODEL_URL" --progress-bar
else
    echo "❌ Error: Neither wget nor curl is installed."
    echo "Please install one of them and try again:"
    echo "  Ubuntu/Debian: sudo apt-get install wget curl"
    echo "  macOS: brew install wget curl"
    echo "  Windows: Use WSL or download manually from:"
    echo "  $MODEL_URL"
    exit 1
fi

echo ""
echo "✅ Model downloaded successfully!"
echo ""
echo "📊 Model Details:"
echo "  - Name: Qwen2.5 1.5B Instruct"
echo "  - Quantization: Q4_K_M"
echo "  - Format: GGUF"
echo "  - Size: ~1.0 GB"
echo "  - Path: model/model.gguf"
echo ""
echo "🚀 Next steps:"
echo "  1. Install Ollama: https://ollama.ai"
echo "  2. Pull the model: ollama pull qwen2.5:1.5b"
echo "  3. Build the knowledge index: python -m src.rag --build"
echo "  4. Run the web app: python -m src.web_app --model qwen2.5:1.5b"
echo "  5. Open browser at: http://127.0.0.1:7860"
echo ""
echo "For ADTC profiler validation:"
echo "  adtc-profiler run --submission . --mode participant --output submission.json --skip-accuracy"
echo ""
