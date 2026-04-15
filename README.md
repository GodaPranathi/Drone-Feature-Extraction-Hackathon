# Drone-Feature-Extraction-Hackathon

### 1. Install dependencies
```bash
pip install torch torchvision rasterio geopandas opencv-python segmentation-models-pytorch

```

### 2. Load Trained Model & Run Inference

```python
import torch
import segmentation_models_pytorch as smp

NUM_CLASSES = 5


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = smp.FPN(
    encoder_name="resnet50",
    encoder_weights=None,
    in_channels=3,
    classes=NUM_CLASSES
).to(device)


model.load_state_dict(torch.load("model_multiclass.pth", map_location=device))

model.eval()

print(" Model loaded successfully")
