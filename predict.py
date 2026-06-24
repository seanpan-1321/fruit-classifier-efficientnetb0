from pathlib import Path
import argparse

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# =====================
# Settings
# =====================
MODEL_PATH = Path("models/best_efficientnet_b0_fruit_combined_balanced(3).pth")
CLASS_FILE = Path("classes.txt")
DEFAULT_IMAGE_PATH = Path("sample_images/apple.jpg")

IMAGE_SIZE = 224
TOP_K = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================
# Load classes
# =====================
if not CLASS_FILE.exists():
    raise FileNotFoundError(
        f"Missing {CLASS_FILE}. Please create classes.txt with one class name per line."
    )

class_names = CLASS_FILE.read_text(encoding="utf-8").splitlines()
num_classes = len(class_names)


# =====================
# Transforms
# =====================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# =====================
# Load model
# =====================
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Missing model file: {MODEL_PATH}\n"
        "Model weights are not included in this repository. "
        "Please place the .pth file inside the models/ folder."
    )

model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device, weights_only=True)
)

model = model.to(device)
model.eval()


# =====================
# Predict function
# =====================
def predict_image(image_path: Path):
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    top_probs, top_indices = torch.topk(probs, TOP_K)

    print(f"\nImage: {image_path}")
    print("-" * 40)

    for rank, (prob, idx) in enumerate(zip(top_probs, top_indices), start=1):
        class_name = class_names[idx.item()]
        confidence = prob.item() * 100
        print(f"{rank}. {class_name}: {confidence:.2f}%")


# =====================
# Main
# =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict fruit or vegetable class from an image."
    )

    parser.add_argument(
        "image",
        nargs="?",
        default=str(DEFAULT_IMAGE_PATH),
        help="Path to the image file. Example: sample_images/apple.jpg"
    )

    args = parser.parse_args()
    predict_image(Path(args.image))