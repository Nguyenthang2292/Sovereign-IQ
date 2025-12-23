#!/bin/bash
# Bash Script để cấu hình API Keys bằng biến môi trường (Linux/Mac)
# Chạy script này và thêm vào ~/.bashrc hoặc ~/.zshrc

echo "=== Cấu hình API Keys cho Crypto Probability ==="
echo ""

# Nhập API Keys
read -p "Binance API Key (hoặc Enter để bỏ qua): " BINANCE_API_KEY_INPUT
if [ ! -z "$BINANCE_API_KEY_INPUT" ]; then
    echo "export BINANCE_API_KEY='$BINANCE_API_KEY_INPUT'" >> ~/.bashrc
    export BINANCE_API_KEY="$BINANCE_API_KEY_INPUT"
    echo "✓ Đã set BINANCE_API_KEY"
else
    echo "⊘ Bỏ qua BINANCE_API_KEY"
fi

read -p "Binance API Secret (hoặc Enter để bỏ qua): " BINANCE_API_SECRET_INPUT
if [ ! -z "$BINANCE_API_SECRET_INPUT" ]; then
    echo "export BINANCE_API_SECRET='$BINANCE_API_SECRET_INPUT'" >> ~/.bashrc
    export BINANCE_API_SECRET="$BINANCE_API_SECRET_INPUT"
    echo "✓ Đã set BINANCE_API_SECRET"
else
    echo "⊘ Bỏ qua BINANCE_API_SECRET"
fi

read -p "Google Gemini API Key (hoặc Enter để bỏ qua): " GEMINI_API_KEY_INPUT
if [ ! -z "$GEMINI_API_KEY_INPUT" ]; then
    echo "export GEMINI_API_KEY='$GEMINI_API_KEY_INPUT'" >> ~/.bashrc
    export GEMINI_API_KEY="$GEMINI_API_KEY_INPUT"
    echo "✓ Đã set GEMINI_API_KEY"
else
    echo "⊘ Bỏ qua GEMINI_API_KEY"
fi

echo ""
echo "=== Hoàn tất ==="
echo ""
echo "Các biến môi trường đã được thêm vào ~/.bashrc"
echo "Chạy lệnh sau để áp dụng ngay: source ~/.bashrc"
echo "Hoặc khởi động lại terminal"

