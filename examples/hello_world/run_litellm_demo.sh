#!/bin/bash

# Fenic Local LLM Demo Runner
# This script helps you run the LiteLLM demo with proper setup

set -e

echo "🏠 Fenic Local LLM Demo Setup"
echo "=================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

# Check if Ollama is running
echo "🔍 Checking Ollama connection..."
if curl -s http://localhost:11434/api/version > /dev/null; then
    echo "✅ Ollama is running on localhost:11434"
else
    echo "❌ Ollama is not running or not accessible"
    echo ""
    echo "📋 Please ensure:"
    echo "   1. Install Ollama: https://ollama.ai"
    echo "   2. Start Ollama: ollama serve"
    echo "   3. Pull a model: ollama pull qwen3:30b"
    echo ""
    echo "🔧 Quick setup commands:"
    echo "   curl -fsSL https://ollama.ai/install.sh | sh"
    echo "   ollama serve &"
    echo "   ollama pull qwen3:30b"
    exit 1
fi

# Check for required model
echo "🔍 Checking for required models..."
if ollama list | grep -q "qwen3:30b"; then
    echo "✅ Model qwen3:30b is available"
    MODEL="qwen3:30b"
elif ollama list | grep -q "gpt-oss"; then
    echo "✅ Model gpt-oss is available"
    MODEL="gpt-oss"
elif ollama list | grep -q "deepseek-r1"; then
    echo "✅ Model deepseek-r1 is available"
    MODEL="deepseek-r1"
else
    echo "❌ No compatible models found"
    echo ""
    echo "📦 Please pull a compatible model:"
    echo "   ollama pull qwen3:30b    # Recommended"
    echo "   ollama pull gpt-oss      # Alternative"
    echo "   ollama pull deepseek-r1  # Alternative"
    exit 1
fi

# Check Python dependencies
echo "🔍 Checking Python dependencies..."
if python3 -c "import fenic, litellm" 2>/dev/null; then
    echo "✅ Required packages are installed"
else
    echo "📦 Installing required packages..."
    pip install fenic litellm
fi

echo ""
echo "🚀 Starting Fenic Local LLM Demo..."
echo "📊 Using model: $MODEL"
echo "=================================="
echo ""

# Run the demo
cd "$(dirname "$0")"
python3 hello_world_litellm.py

echo ""
echo "🎉 Demo completed successfully!"
echo "💡 Try modifying hello_world_litellm.py to experiment with different models and prompts"