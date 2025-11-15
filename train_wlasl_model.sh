#!/bin/bash
# Script to train WLASL model once - run this after preparing the dataset

set -e  # Exit on error

echo "=== WLASL Model Training Script ==="
echo ""

# Check if dataset exists
if [ ! -d "dataset/wlasl" ]; then
    echo "Error: dataset/wlasl not found!"
    echo "Please prepare the WLASL dataset first:"
    echo "  python src/prepare_wlasl.py --wlasl-dir ~/wlasl --output-dir dataset/wlasl"
    exit 1
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Create models directory
mkdir -p models/wlasl

echo "Starting training..."
echo "Dataset: dataset/wlasl"
echo "Output: models/wlasl"
echo ""

# Train the model
python src/train_asl_model.py \
    --data-dir dataset/wlasl \
    --output-dir models/wlasl \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --train-split 0.8

echo ""
echo "=== Training Complete ==="
echo "Model saved to: models/wlasl/best_model.pt"
echo "Class mapping saved to: models/wlasl/class_mapping.json"
echo ""
echo "You can now run the application:"
echo "  python src/main.py"

