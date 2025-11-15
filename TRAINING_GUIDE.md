# ASL Model Training Guide

This guide explains how to train an ASL recognition model for WaveSL using a large dataset.

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

### Recommended Datasets

1. **WLASL (Word-Level American Sign Language)**
   - 2,000 common ASL words
   - Download: https://github.com/dxli94/WLASL
   - Large vocabulary, good for production use

2. **MS-ASL (Microsoft American Sign Language)**
   - Multiple subsets (ASL100, ASL1000)
   - Download: https://www.microsoft.com/en-us/research/project/ms-asl/
   - Well-annotated, good benchmark dataset

3. **ASL Citizen**
   - Community-contributed dataset
   - Download: https://github.com/nesl/asl-citizen
   - Diverse signers, real-world conditions

4. **Custom Dataset**
   - Record your own videos
   - Ensure consistent lighting and camera angle
   - Multiple signers and variations improve robustness

## Step-by-Step Training

### Step 1: Prepare Your Dataset

If your videos aren't organized yet:

```bash
# Create class mapping from existing structure
python src/prepare_dataset.py mapping --data-dir /path/to/dataset --output-file class_mapping.json
```

Or manually organize videos into class directories.

### Step 2: Train the Model

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

### Step 3: Monitor Training

The training script will:
- Print training and validation accuracy each epoch
- Save the best model based on validation accuracy
- Save a final model at the end
- Create a `class_mapping.json` file

### Step 4: Use the Trained Model

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

## Resources

- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [MediaPipe Hands Documentation](https://google.github.io/mediapipe/solutions/hands)
- [WLASL Dataset](https://github.com/dxli94/WLASL)
- [Sign Language Recognition Papers](https://paperswithcode.com/task/sign-language-recognition)

