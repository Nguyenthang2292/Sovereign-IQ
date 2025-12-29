#!/bin/bash
# Bash Script to configure API Keys using environment variables (Linux/Mac)
# Run this script and append to ~/.bashrc

echo "=== Configure API Keys for Crypto Probability ==="
echo ""

# Enter API Keys
read -p "Binance API Key (or press Enter to skip): " BINANCE_API_KEY_INPUT
if [ ! -z "$BINANCE_API_KEY_INPUT" ]; then
    # Remove any existing BINANCE_API_KEY entry to avoid duplicates
    if [ -f ~/.bashrc ]; then
        grep -v "^export BINANCE_API_KEY=" ~/.bashrc > ~/.bashrc.tmp && mv ~/.bashrc.tmp ~/.bashrc
        if [ $? -ne 0 ]; then
            echo "Error removing existing BINANCE_API_KEY entry from ~/.bashrc"
            exit 1
        fi
    fi
    echo "export BINANCE_API_KEY='$BINANCE_API_KEY_INPUT'" >> ~/.bashrc
    if [ $? -ne 0 ]; then
        echo "Error writing BINANCE_API_KEY to ~/.bashrc"
        exit 1
    fi
    export BINANCE_API_KEY="$BINANCE_API_KEY_INPUT"
    echo "✓ BINANCE_API_KEY set"
else
    echo "⊘ Skipping BINANCE_API_KEY"
fi

read -s -p "Binance API Secret (hoặc Enter để bỏ qua): " BINANCE_API_SECRET_INPUT
echo ""  # Add newline after hidden input
if [ ! -z "$BINANCE_API_SECRET_INPUT" ]; then
    # Remove existing entry if present
    grep -v "^export BINANCE_API_SECRET=" ~/.bashrc > ~/.bashrc.tmp && mv ~/.bashrc.tmp ~/.bashrc
    echo "export BINANCE_API_SECRET='$BINANCE_API_SECRET_INPUT'" >> ~/.bashrc || { echo "Error writing to ~/.bashrc"; exit 1; }
    export BINANCE_API_SECRET="$BINANCE_API_SECRET_INPUT"
    echo "✓ Đã set BINANCE_API_SECRET"
else
    echo "⊘ Bỏ qua BINANCE_API_SECRET"
fi

read -p "Google Gemini API Key (or press Enter to skip): " GEMINI_API_KEY_INPUT
if [ ! -z "$GEMINI_API_KEY_INPUT" ]; then
    # Remove existing entry if present
    grep -v "^export GEMINI_API_KEY=" ~/.bashrc > ~/.bashrc.tmp && mv ~/.bashrc.tmp ~/.bashrc
    echo "export GEMINI_API_KEY='$GEMINI_API_KEY_INPUT'" >> ~/.bashrc || { echo "Error writing to ~/.bashrc"; exit 1; }
    export GEMINI_API_KEY="$GEMINI_API_KEY_INPUT"
    echo "✓ GEMINI_API_KEY set"
else
    echo "⊘ Skipping GEMINI_API_KEY"
fi

echo ""
echo "=== All done ==="
echo ""
echo "The environment variables have been added to ~/.bashrc"
echo "To apply immediately, run: source ~/.bashrc"
echo "Or restart your terminal"

