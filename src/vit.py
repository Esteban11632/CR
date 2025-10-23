from transformers import ViTForImageClassification, ViTImageProcessor
from torchvision import datasets
from torch.utils.data import DataLoader
import torch
from torch import nn
from tqdm import tqdm

# Load images
image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

def preprocess_fn(img):
    # Convert PIL image to tensor and resize/normalize for ViT
    return image_processor(images=img, return_tensors="pt")['pixel_values'][0]

train_data = datasets.ImageFolder(
    root=r"C:\Users\DELL\Desktop\Pytorch\RF\CR\Cards clash royale.v2i.folder\train",
    transform=preprocess_fn
)

test_data = datasets.ImageFolder(
    root=r"C:\Users\DELL\Desktop\Pytorch\RF\CR\Cards clash royale.v2i.folder\test",
    transform=preprocess_fn
)

BATCH_SIZE = 32

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Model
num_classes = len(train_data.classes)
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=num_classes
).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

epochs = 3

for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch + 1}-------------------------------\n")
    model.train()
    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        outputs = model(X)
        loss = loss_fn(outputs.logits, y)
        train_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1)
        train_acc += (preds == y).sum().item() / len(preds)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}")

    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for batch, (X, y) in enumerate(test_dataloader):
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            test_loss += loss_fn(outputs.logits, y).item()
            preds = torch.argmax(outputs.logits, dim=1)
            test_acc += (preds == y).sum().item() / len(preds)
    test_loss /= len(test_dataloader)
    test_acc /= len(test_dataloader)
    print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}")