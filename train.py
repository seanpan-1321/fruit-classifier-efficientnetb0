import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# =====================
# Settings
# =====================
DATA_DIR = Path(r"D:\fruit_project\clean_fruit_split")
MODEL_DIR = Path(r"D:\fruit_project\models")
OUTPUT_DIR = Path(r"D:\fruit_project\outputs")

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 30
LR = 0.001
IMAGE_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =====================
# Transforms
# =====================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =====================
# Dataset
# =====================
train_data = datasets.ImageFolder(DATA_DIR / "train", transform=train_transform)
val_data = datasets.ImageFolder(DATA_DIR / "val", transform=test_transform)
test_data = datasets.ImageFolder(DATA_DIR / "test", transform=test_transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

num_classes = len(train_data.classes)

print("Classes:", num_classes)
print(train_data.classes)
print("Train images:", len(train_data))
print("Val images:", len(val_data))
print("Test images:", len(test_data))

# =====================
# Model: EfficientNet-B0
# =====================
model = models.efficientnet_b0(
    weights=models.EfficientNet_B0_Weights.DEFAULT
)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_classes
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.1,
    patience=2
)

# =====================
# Training
# =====================
train_acc_history = []
val_acc_history = []
train_loss_history = []
val_loss_history = []

best_val_acc = 0.0

patience_counter = 0
EARLY_STOPPING_PATIENCE = 6

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    # ---- Train ----
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    # ---- Validation ----
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / total
    val_acc = correct / total

    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)
    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

    scheduler.step(val_acc)
    current_lr = optimizer.param_groups[0]["lr"]
    print(f"Current LR: {current_lr:.6f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_DIR / "best_efficientnet_b0_fruit_combined_balanced(3).pth")
        print("Saved best model.")
    else:
        patience_counter += 1
        print(f"No improvement count: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print("Early stopping triggered.")
        break
    
print("\nTraining finished.")
print("Best Val Acc:", best_val_acc)

# =====================
# Test
# =====================
model.load_state_dict(torch.load(MODEL_DIR / "best_efficientnet_b0_fruit_combined_balanced(3).pth", weights_only=True))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

test_acc = correct / total
print(f"\nFinal Test Accuracy: {test_acc:.4f}")

# =====================
# Confusion Matrix
# =====================
all_preds = []
all_labels = []

model.eval()

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)

print("\nClassification Report:")
print(classification_report(
    all_labels,
    all_preds,
    target_names=test_data.classes,
    digits=4
))

plt.figure(figsize=(14, 12))
plt.imshow(cm)
plt.title("Efficientnetb0 Fruit Confusion Matrix(3)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(np.arange(len(test_data.classes)), test_data.classes, rotation=90)
plt.yticks(np.arange(len(test_data.classes)), test_data.classes)
plt.colorbar()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "efficientnet_b0_fruit_confusion_matrix(3).png")
plt.show()

# =====================
# Save learning curve
# =====================
plt.figure()
plt.plot(train_acc_history, label="Train Acc")
plt.plot(val_acc_history, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Efficientnetb0 Fruit Accuracy(3)")
plt.savefig(OUTPUT_DIR / "efficientnet_b0_fruit_accuracy(3).png")
plt.show()

plt.figure()
plt.plot(train_loss_history, label="Train Loss")
plt.plot(val_loss_history, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("EfficientnetB0 Fruit Loss(3)")
plt.savefig(OUTPUT_DIR / "efficientnet_b0_fruit_loss(3).png")
plt.show()