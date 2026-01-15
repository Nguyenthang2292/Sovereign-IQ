#!/bin/bash

echo ""
echo "========================================"
echo "    ATC Visualizer - Quick Start"
echo "========================================"
echo ""

python3 run_atc_visualizer.py

if [ $? -ne 0 ]; then
    echo ""
    echo "Error occurred. Press Enter to exit..."
    read
fi
