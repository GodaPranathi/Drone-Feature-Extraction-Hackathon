# Drone-Feature-Extraction-Hackathon

### 1. Install dependencies
```bash
pip install torch torchvision rasterio geopandas opencv-python segmentation-models-pytorch

```

### 2. Rasterization Verification and Data Preprocessing Visualization
```bash
import matplotlib.pyplot as plt
import numpy as np
import random

# FIXED CLASS COLORS (same as training)
CLASS_NAMES = [
    "Background",
    "Building",
    "Road",
    "Waterbody",
    "Utility"
]

CLASS_COLORS = [
    (0, 0, 0),        # Background
    (255, 0, 0),      # Building
    (0, 255, 0),      # Road
    (0, 0, 255),      # Waterbody
    (255, 255, 0),    # Utility
]

def colorize(mask):
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for i, color in enumerate(CLASS_COLORS):
        colored[mask == i] = color
    return colored

#  Pick random samples from generated dataset (NOT val_ds)
img_files = sorted(os.listdir(output_img_dir))
indices = random.sample(range(len(img_files)), 5)

for idx in indices:

    img = np.load(os.path.join(output_img_dir, img_files[idx]))
    mask = np.load(os.path.join(output_mask_dir, img_files[idx].replace("img_", "mask_")))

    #  Normalize image for display
    img_disp = (img - img.min()) / (img.max() - img.min() + 1e-8)

    mask_color = colorize(mask)

    #  Overlay
    overlay = (0.7 * img_disp + 0.3 * (mask_color / 255.0)).clip(0,1)

    print("Classes in mask:",
          [CLASS_NAMES[i] for i in np.unique(mask)])

    plt.figure(figsize=(15,4))

    plt.subplot(1,3,1)
    plt.title("Image")
    plt.imshow(img_disp)
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.title("Mask")
    plt.imshow(mask_color)
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.title("Overlay (MOST IMPORTANT)")
    plt.imshow(overlay)
    plt.axis('off')

    plt.show()



```



Classes in mask: ['Building', 'Road']
<img width="822" height="247" alt="download" src="https://github.com/user-attachments/assets/217129c9-cc8d-4b2c-9b4a-3050ebb80d45" />
Classes in mask: ['Background', 'Waterbody']
<img width="822" height="247" alt="download" src="https://github.com/user-attachments/assets/66b8a836-bcaf-4469-ac01-d1376c2317be" />
Classes in mask: ['Background', 'Road']
<img width="822" height="247" alt="download" src="https://github.com/user-attachments/assets/d59a22be-b94d-4a5e-bdca-67a32ea133eb" />
Classes in mask: ['Background', 'Building']
<img width="822" height="247" alt="download" src="https://github.com/user-attachments/assets/7a04ddb0-ed61-4059-9115-84cf9552a4a1" />
Classes in mask: ['Background', 'Building']
<img width="822" height="247" alt="download" src="https://github.com/user-attachments/assets/a9fe0b36-c9cc-4fc7-901d-b78480eb4a2c" />


### 3. Load Trained Model & Run Inference

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


```
### 4. Validation Results
``` python
import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.patches import Patch

model.eval()

# CLASS NAMES
CLASS_NAMES = [
    "Background",
    "Building",
    "Road",
    "Waterbody",
    "Utility"
]

#  COLORS
CLASS_COLORS = [
    (0, 0, 0),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
]

def colorize(mask):
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for i, color in enumerate(CLASS_COLORS):
        colored[mask == i] = color
    return colored

#  RANDOM 10 samples
indices = random.sample(range(len(val_ds)), 100)

print("Selected indices:", indices)

correct_samples = 0

for idx in indices:

    img, mask = val_ds[idx]
    img_input = img.unsqueeze(0).to(device)

    #  TTA
    with torch.no_grad():
        pred1 = model(img_input)

        img_flip = torch.flip(img_input, dims=[3])
        pred2 = model(img_flip)
        pred2 = torch.flip(pred2, dims=[3])

        outputs = (pred1 + pred2) / 2
        pred = torch.argmax(outputs, dim=1).cpu()[0]

    #  IMAGE
    img_disp = img.permute(1,2,0).cpu().numpy()
    img_disp = (img_disp - img_disp.min()) / (img_disp.max() - img_disp.min() + 1e-8)

    mask = mask.cpu().numpy()
    pred = pred.numpy()

    #  CLASS LISTS (IGNORE BACKGROUND OPTIONAL)
    gt_classes = set(np.unique(mask))
    pred_classes = set(np.unique(pred))

    # REMOVE BACKGROUND (optional but better)
    gt_classes.discard(0)
    pred_classes.discard(0)

    #  CHECK MATCH
    if gt_classes == pred_classes:
        is_correct = 1
    elif len(gt_classes)<=len(pred_classes):
            for i in gt_classes:
                if i not in pred_classes:
                    break
            is_correct=1
    else:
        is_correct=0
    
    if is_correct:
        correct_samples += 1

    print("\nGT classes:", [CLASS_NAMES[i] for i in gt_classes])
    print("Pred classes:", [CLASS_NAMES[i] for i in pred_classes])
    print("Match:", "✅ YES" if is_correct else "❌ NO")

    #  COLORIZE
    mask_color = colorize(mask)
    pred_color = colorize(pred)

    overlay = (0.7 * img_disp + 0.3 * (pred_color / 255.0)).clip(0,1)

    legend_elements = [
        Patch(facecolor=np.array(color)/255.0, label=name)
        for color, name in zip(CLASS_COLORS, CLASS_NAMES)
    ]

    plt.figure(figsize=(14,4))

    plt.subplot(1,4,1)
    plt.title("Image")
    plt.imshow(img_disp)
    plt.axis('off')

    plt.subplot(1,4,2)
    plt.title("Ground Truth")
    plt.imshow(mask_color)
    plt.axis('off')

    plt.subplot(1,4,3)
    plt.title("Prediction")
    plt.imshow(pred_color)
    plt.axis('off')

    plt.subplot(1,4,4)
    plt.title("Overlay")
    plt.imshow(overlay)
    plt.axis('off')

    plt.legend(
        handles=legend_elements,
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )

    plt.tight_layout()
    plt.show()

#  FINAL RESULT
print("\n Correct Predictions:", correct_samples, "/ 100")
print(" Accuracy:", correct_samples / 100)
```
Some sample outputs:
GT classes: ['Building']
Pred classes: ['Building']
Match: ✅ YES
<img width="1002" height="241" alt="download" src="https://github.com/user-attachments/assets/da12bc3f-a427-4207-af80-90020b98919f" />
GT classes: ['Building']
Pred classes: ['Building']
Match: ✅ YES
<img width="1002" height="241" alt="download" src="https://github.com/user-attachments/assets/f693e449-caf5-4135-bdd4-9a1b235021ea" />

GT classes: ['Waterbody']
Pred classes: ['Waterbody']
Match: ✅ YES
<img width="1002" height="241" alt="download" src="https://github.com/user-attachments/assets/85e8c3ed-c73e-4661-8567-01c8b4a56ff4" />

GT classes: ['Road']
Pred classes: ['Road']
Match: ✅ YES
<img width="1002" height="241" alt="download" src="https://github.com/user-attachments/assets/508bc86c-6215-4b63-9b61-88d1a0d6a7a2" />



### The results for the above cell is in Results.doc 
The accuracy for 100 images is 91.



### 5. Testing
Testing results are in Test_results.doc
Correct Predictions: 27 / 30
Accuracy: 0.9


### Further still need to be work on COG generation.
