# WLASL Model Setup Guide

This guide will help you set up a pre-trained WLASL model for WaveSL. You only need to do this **once** - after training, the model will be saved and ready to use.

## Overview

1. Download WLASL dataset
2. Prepare the dataset
3. Train the model (one time)
4. Use the pre-trained model (every time you run the app)

## Step-by-Step Setup

### Step 1: Download WLASL Dataset

1. Visit the [WLASL GitHub repository](https://github.com/dxli94/WLASL)
2. Request access to download the dataset (if required)
3. Download:
   - Video files (usually in a `videos/` directory)
   - Annotation files (JSON files with gloss mappings)
4. Extract to a directory, e.g., `~/wlasl/`

**Note**: The dataset is large (~several GB). Make sure you have enough disk space.

### Step 2: Prepare WLASL Dataset

Organize the WLASL videos into the training format:

```bash
# Activate virtual environment
source venv/bin/activate

# Prepare the dataset
python src/prepare_wlasl.py \
    --wlasl-dir ~/wlasl \
    --output-dir dataset/wlasl \
    --class-mapping models/wlasl_class_mapping.json
```

This will:
- Read WLASL JSON annotations
- Organize videos into class directories (by gloss/sign name)
- Create a class mapping file

**Expected output**: `dataset/wlasl/` with subdirectories for each sign class.

### Step 3: Train the Model (One Time)

Train the model using the prepared dataset:

```bash
# Option 1: Use the training script
chmod +x train_wlasl_model.sh
./train_wlasl_model.sh

# Option 2: Train manually
python src/train_asl_model.py \
    --data-dir dataset/wlasl \
    --output-dir models/wlasl \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 0.001
```

**Training time**: Depends on dataset size and hardware:
- WLASL-100: ~30 minutes to 2 hours
- WLASL-300: ~1-4 hours
- WLASL-1000: ~3-8 hours
- WLASL-2000: ~6-12 hours

**What gets saved**:
- `models/wlasl/best_model.pt` - Best model based on validation accuracy
- `models/wlasl/final_model.pt` - Final model after all epochs
- `models/wlasl/class_mapping.json` - Mapping of class indices to sign names

### Step 4: Verify Model is Ready

Check that the model files exist:

```bash
ls -lh models/wlasl/
```

You should see:
- `best_model.pt` (model weights)
- `class_mapping.json` (class mappings)

### Step 5: Use the Pre-trained Model

Once trained, the model is ready to use. The application will automatically load it:

```bash
# Run the application (will auto-detect models/wlasl/best_model.pt)
python src/main.py

# Or specify the model explicitly
python src/main.py --model models/wlasl/best_model.pt
```

**The model is pre-trained** - no training happens when you run the application!

## Training Tips

### Start Small

If you're new to this, start with a subset:

1. **WLASL-100**: Top 100 most common signs
   - Faster training (~30 min - 2 hours)
   - Good accuracy (80-90%)
   - Good for testing the pipeline

2. **WLASL-300**: Top 300 signs
   - Moderate training time (~1-4 hours)
   - Good accuracy (70-85%)
   - Better vocabulary coverage

3. **Full WLASL**: All 2000 signs
   - Long training time (~6-12 hours)
   - Lower accuracy (60-75%)
   - Maximum vocabulary

### Filtering the Dataset

To train on a subset, you can filter the dataset after preparation:

```python
# Example: Keep only top 100 classes by video count
import os
from pathlib import Path

data_dir = Path('dataset/wlasl')
classes = [(d.name, len(list(d.glob('*.mp4')))) for d in data_dir.iterdir() if d.is_dir()]
classes.sort(key=lambda x: x[1], reverse=True)

# Keep top 100
for class_name, _ in classes[100:]:
    shutil.rmtree(data_dir / class_name)
```

### GPU Acceleration

If you have a GPU, training will be much faster:

```bash
# Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available())"
```

The training script automatically uses GPU if available.

## Troubleshooting

### "dataset/wlasl not found"
- Make sure you ran `prepare_wlasl.py` first
- Check that the output directory path is correct

### "No class mapping found"
- The class mapping should be created during dataset preparation
- Check `models/wlasl_class_mapping.json` exists
- Or it will be created during training

### Low Training Accuracy
- Check data quality - ensure videos are clear
- Increase number of epochs
- Adjust learning rate
- Check for class imbalance

### Out of Memory
- Reduce batch size: `--batch-size 16` or `--batch-size 8`
- Use a smaller dataset subset
- Close other applications

### Model Not Loading
- Verify `models/wlasl/best_model.pt` exists
- Check that `class_mapping.json` is in the same directory
- Try specifying model path explicitly: `--model models/wlasl/best_model.pt`

## Next Steps

After training:
1. Test the model: `python src/main.py`
2. Check recognition accuracy in real-time
3. Fine-tune if needed (adjust model architecture, training parameters)
4. Deploy for use in video calls!

## Model Files

After training, you'll have:
```
models/wlasl/
├── best_model.pt          # Best model (use this one)
├── final_model.pt         # Final model after all epochs
└── class_mapping.json    # Class index to sign name mapping
```

**Keep these files** - you'll use them every time you run the application!

