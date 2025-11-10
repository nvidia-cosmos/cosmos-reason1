#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting setup..."

# Optionally ensure pip is up to date
echo "Upgrading pip..."
python3 -m pip install --upgrade pip

# Install gradio
echo "Installing gradio..."
python3 -m pip install gradio
echo "Finished Installing gradio..."

echo "Installing transformers..."
python3 -m pip install transformers==4.49.0
echo "Finished Installing transformers..."

echo "Upgrading typer..."
pip install --upgrade "typer>=0.7.0"
echo "Finished Upgrading typer..."

# echo "Uninstalling torch..."
# pip uninstall torch
# echo "Finished Uninstalling torch..."

# echo "Uninstalling torchvision..."
# pip uninstall torchvision
# echo "Finished Uninstalling torchvision..."

# echo "Installing torch..."
# pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
# echo "Finished Installing torch..."

# echo "Installing torchvision..."
# pip install torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
# echo "Finished Installing torchvision..."

echo "Setup complete!"