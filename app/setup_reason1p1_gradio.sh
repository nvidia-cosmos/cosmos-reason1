#!/bin/bash

# Setup script for Cosmos-Reason1.1 Gradio App
echo "🚀 Setting up Cosmos-Reason1.1 Gradio App..."

# Check if we're in the right directory
if [ ! -f "reason1p1_gradio.py" ]; then
    echo "❌ Error: reason1p1_gradio.py not found in current directory"
    echo "Please run this script from the directory containing reason1p1_gradio.py"
    exit 1
fi

# Function to install packages with error handling
install_packages() {
    local packages="$1"
    echo "📦 Installing packages: $packages"
    
    # Try normal pip install first
    python3 -m pip install $packages 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "✅ Packages installed successfully!"
        return 0
    fi
    
    # If that fails, try with --break-system-packages
    echo "⚠️  Normal pip install failed. Trying with --break-system-packages..."
    echo "   (This is safe in containerized environments like this one)"
    python3 -m pip install --break-system-packages $packages
    if [ $? -eq 0 ]; then
        echo "✅ Packages installed successfully with --break-system-packages!"
        return 0
    fi
    
    # If that also fails, suggest virtual environment
    echo "❌ Failed to install packages. Consider creating a virtual environment:"
    echo "   python3 -m venv vllm_env"
    echo "   source vllm_env/bin/activate"
    echo "   pip install $packages"
    return 1
}

# Install core dependencies for Cosmos-Reason1.1
echo "📦 Installing core dependencies..."
install_packages "torch torchvision gradio pillow numpy huggingface_hub"

# Install latest transformers with Qwen2_VL support
echo "📦 Installing latest transformers..."
python3 -m pip install --break-system-packages --upgrade transformers

# Install vLLM (for hybrid approach - tries vLLM first, falls back to transformers)
echo "📦 Installing vLLM for faster inference..."
install_packages "vllm"

# Check if installation was successful
if [ $? -ne 0 ]; then
    echo "❌ Failed to install required packages"
    exit 1
fi

# Check if qwen_vl_utils is available (might need special installation)
python3 -c "import qwen_vl_utils" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Warning: qwen_vl_utils not found. Installing..."
    install_packages "qwen-vl-utils"
    
    # If that fails, try alternative installation
    if [ $? -ne 0 ]; then
        echo "⚠️  Trying alternative qwen_vl_utils installation..."
        python3 -m pip install --break-system-packages git+https://github.com/QwenLM/Qwen-VL.git 2>/dev/null
        if [ $? -ne 0 ]; then
            echo "⚠️  Could not install qwen_vl_utils. You may need to install it manually:"
            echo "   python3 -m pip install --break-system-packages qwen-vl-utils"
            echo "   or"
            echo "   python3 -m pip install --break-system-packages git+https://github.com/QwenLM/Qwen-VL.git"
        fi
    fi
fi

# fix the error with binary conflicts
PYTHONPATH=$(pwd) python3 -m pip install --break-system-packages --upgrade --force-reinstall numpy pandas gradio
echo "✅ Fixed the error with binary conflicts"
echo "✅ Installed numpy pandas gradio"

echo ""
echo "🎉 Setup complete! You can now run the Gradio app with:"
echo "   python3 reason1p1_gradio.py"
echo ""
echo "Or run this script with the 'run' argument to start automatically:"
echo "   ./setup_reason1p1_gradio.sh run"

# If 'run' argument is provided, start the app
if [ "$1" = "run" ]; then
    echo ""
    echo "🚀 Starting Gradio app..."
    python3 reason1p1_gradio.py
fi
