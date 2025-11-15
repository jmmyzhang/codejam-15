# ASL Model Training Guide - WLASL Dataset

This guide explains how to train an ASL recognition model for WaveSL using the **WLASL (Word-Level American Sign Language)** dataset.

## Overview

The training pipeline uses:
- **MediaPipe** to extract hand landmarks from video frames
- **PyTorch** to train a neural network classifier
- **Feature extraction** that matches the inference pipeline for consistency

## Dataset Requirements

### Dataset Structure

Organize your dataset in the following structure:

```
dataset/
├── hello/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
├── thank_you/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
├── yes/
│   └── ...
└── ...
```

Each subdirectory represents a sign class, containing video files of that sign.

### WLASL Dataset

**WaveSL uses the WLASL (Word-Level American Sign Language) dataset.**

- **2,000 common ASL words** - Large vocabulary for production use
- **Download**: https://github.com/dxli94/WLASL
- **Structure**: Videos organized by gloss (sign name) with JSON annotations
- **Format**: MP4 video files with corresponding annotation files

#### Downloading WLASL

1. Visit the [WLASL GitHub repository](https://github.com/dxli94/WLASL)
2. Follow their download instructions (may require request/approval)
3. Download both:
   - Video files (videos directory)
   - Annotation files (JSON files with gloss mappings)
4. Extract to a directory (e.g., `~/wlasl/`)

## Step-by-Step Training with WLASL

### Step 1: Download WLASL Dataset

Download the WLASL dataset following their instructions. You should have:
- A `videos/` directory with all video files
- JSON annotation files mapping video IDs to glosses (sign names)

### Step 2: Prepare WLASL Dataset

Use the preparation script to organize WLASL videos:

```bash
python src/prepare_wlasl.py \
    --wlasl-dir ~/wlasl \
    --output-dir dataset/wlasl \
    --class-mapping models/wlasl_class_mapping.json
```

This script will:
- Read WLASL annotations
- Organize videos into class directories (by gloss/sign name)
- Create a class mapping JSON file

### Step 3: Train the Model

```bash
# Activate virtual environment
source venv/bin/activate

# Train with default settings
python src/train_asl_model.py \
    --data-dir /path/to/dataset \
    --output-dir models \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 0.001
```

### Step 4: Monitor Training

The training script will:
- Print training and validation accuracy each epoch
- Save the best model based on validation accuracy
- Save a final model at the end
- Create a `class_mapping.json` file

### Step 5: Use the Trained Model

After training, update `src/asl_recognizer.py`:

```python
# In __init__, load your model:
self.model = ASLModel(input_size=146, num_classes=YOUR_NUM_CLASSES)
self.model.load_state_dict(torch.load('models/best_model.pt', map_location=self.device))
self.model.eval()

# Update _class_to_text with your class mapping:
with open('models/class_mapping.json', 'r') as f:
    self.class_mapping = json.load(f)

def _class_to_text(self, class_idx: int) -> str:
    # Reverse mapping: class_idx -> sign_name
    for sign_name, idx in self.class_mapping.items():
        if idx == class_idx:
            return sign_name
    return ""
```

## Training Parameters

### Key Parameters

- **`--epochs`**: Number of training epochs (default: 50)
  - Start with 50, increase if model hasn't converged
  - Monitor validation accuracy to avoid overfitting

- **`--batch-size`**: Batch size (default: 32)
  - Increase if you have GPU memory (64, 128)
  - Decrease if you run out of memory (16, 8)

- **`--learning-rate`**: Learning rate (default: 0.001)
  - Try 0.0001 for more stable training
  - Try 0.01 for faster initial learning

- **`--train-split`**: Training/validation split (default: 0.8)
  - 80% training, 20% validation is standard
  - Adjust based on dataset size

### Model Architecture

The default model has:
- Input size: 146 features (2 hands × 73 features each)
- Hidden layers: [256, 128, 64]
- Output: Number of sign classes

You can modify the architecture in `train_asl_model.py`:

```python
model = ASLModel(
    input_size=146,
    num_classes=num_classes,
    hidden_sizes=[512, 256, 128, 64]  # Deeper network
)
```

## Tips for Better Results

### 1. Data Quality
- Ensure videos have clear hand visibility
- Consistent lighting and background
- Multiple signers for robustness
- Balanced classes (similar number of samples per sign)

### 2. Data Augmentation
- Currently not implemented, but you can add:
  - Random frame selection
  - Temporal jittering
  - Hand landmark noise injection

### 3. Handling Temporal Information
- Current model uses single-frame features
- For better accuracy, consider:
  - LSTM/GRU for sequence modeling
  - 3D CNNs for video sequences
  - Transformer models for temporal attention

### 4. Class Imbalance
- If some signs have fewer samples:
  - Use class weights in loss function
  - Oversample minority classes
  - Use data augmentation

### 5. Transfer Learning
- Pre-train on a larger dataset (e.g., WLASL)
- Fine-tune on your specific vocabulary
- Use pre-trained hand pose estimation models

## Evaluation

After training, evaluate on a test set:

```python
# Load model and test
model.eval()
test_loader = DataLoader(test_dataset, batch_size=32)

correct = 0
total = 0
with torch.no_grad():
    for features, labels in test_loader:
        outputs = model(features)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
```

## Troubleshooting

### Low Accuracy
- Check data quality and labeling
- Increase model capacity (more layers/neurons)
- Train for more epochs
- Adjust learning rate
- Add more training data

### Overfitting
- Add dropout (already included)
- Use data augmentation
- Reduce model capacity
- Increase training data
- Use early stopping

### Out of Memory
- Reduce batch size
- Use gradient accumulation
- Process videos in smaller chunks
- Use mixed precision training

### Slow Training
- Use GPU if available
- Reduce batch size
- Use fewer epochs initially
- Profile and optimize data loading

## Next Steps

1. **Start Small**: Train on 10-20 common signs first
2. **Iterate**: Add more signs as you improve the pipeline
3. **Evaluate**: Test on real-world video to identify issues
4. **Refine**: Adjust model architecture and training parameters
5. **Deploy**: Integrate trained model into WaveSL application

## WLASL-Specific Notes

### Dataset Size

WLASL contains up to 2,000 sign classes. For initial training, you may want to:
- Start with a subset (e.g., top 100 or 200 most common signs)
- Filter by minimum number of samples per class
- Use class balancing to handle imbalanced data

### Vocabulary Selection

You can train on:
- **WLASL-100**: Top 100 most common signs
- **WLASL-300**: Top 300 signs
- **WLASL-1000**: Top 1000 signs
- **WLASL-2000**: Full vocabulary

Adjust the dataset preparation to filter classes as needed.

### Performance Expectations

With WLASL dataset:
- **WLASL-100**: Expect 80-90% accuracy with good data
- **WLASL-300**: Expect 70-85% accuracy
- **WLASL-1000+**: Expect 60-75% accuracy (more challenging)

## Resources

- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [MediaPipe Hands Documentation](https://google.github.io/mediapipe/solutions/hands)
- [WLASL Dataset Repository](https://github.com/dxli94/WLASL)
- [WLASL Paper](https://arxiv.org/abs/2004.01988)
- [Sign Language Recognition Papers](https://paperswithcode.com/task/sign-language-recognition)

