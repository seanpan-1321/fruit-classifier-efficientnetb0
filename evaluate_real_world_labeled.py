from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

MODEL_PATH = Path(r"D:\fruit_project\models\best_efficientnet_b0_fruit_combined_balanced(3).pth")
DATA_DIR = Path(r"D:\fruit_project\clean_fruit_split")
REAL_WORLD_DIR = Path(r"D:\fruit_project\real_world_labeled")
OUTPUT_DIR = Path(r"D:\fruit_project\outputs_real_world")

OUTPUT_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = sorted([p.name for p in (DATA_DIR / "train").iterdir() if p.is_dir()])
class_to_idx = {name: i for i, name in enumerate(class_names)}
num_classes = len(class_names)

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model = model.to(device)
model.eval()

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG"}

y_true = []
y_pred = []
wrong_predictions = []

for class_dir in REAL_WORLD_DIR.iterdir():
    if not class_dir.is_dir():
        continue

    true_class = class_dir.name

    if true_class not in class_to_idx:
        print(f"Skipping unknown class folder: {true_class}")
        continue

    for img_path in class_dir.iterdir():
        if not img_path.is_file() or img_path.suffix.lower() not in image_extensions:
            continue

        image = Image.open(img_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            confidence = probs[pred_idx].item() * 100

        pred_class = class_names[pred_idx]

        y_true.append(class_to_idx[true_class])
        y_pred.append(pred_idx)

        if pred_class != true_class:
            wrong_predictions.append((img_path.name, true_class, pred_class, confidence))

accuracy = np.mean(np.array(y_true) == np.array(y_pred))

print(f"\nReal-world Accuracy: {accuracy:.4f}")
print(f"Correct: {sum(np.array(y_true) == np.array(y_pred))}/{len(y_true)}")

used_labels = sorted(list(set(y_true + y_pred)))
used_class_names = [class_names[i] for i in used_labels]

print("\nClassification Report:")
report = classification_report(
    y_true,
    y_pred,
    labels=used_labels,
    target_names=used_class_names,
    digits=4
)
print(report)

with open(OUTPUT_DIR / "real_world_classification_report.txt", "w", encoding="utf-8") as f:
    f.write(f"Real-world Accuracy: {accuracy:.4f}\n\n")
    f.write(report)

print("\nWrong Predictions:")
for name, true_class, pred_class, confidence in wrong_predictions:
    print(f"{name}: true={true_class}, predicted={pred_class}, confidence={confidence:.2f}%")

with open(OUTPUT_DIR / "wrong_predictions.txt", "w", encoding="utf-8") as f:
    for name, true_class, pred_class, confidence in wrong_predictions:
        f.write(f"{name}: true={true_class}, predicted={pred_class}, confidence={confidence:.2f}%\n")

cm = confusion_matrix(y_true, y_pred, labels=used_labels)

plt.figure(figsize=(10, 8))
plt.imshow(cm)
plt.title("Real-world Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(np.arange(len(used_class_names)), used_class_names, rotation=90)
plt.yticks(np.arange(len(used_class_names)), used_class_names)
plt.colorbar()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "real_world_confusion_matrix.png")
plt.show()