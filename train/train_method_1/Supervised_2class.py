import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import timm
import matplotlib.pyplot as plt

# ---------------------- argparse 경로 파싱 ----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', type=str, required=True, help='Path to the training dataset')
parser.add_argument('--test_dir', type=str, required=True, help='Path to the test dataset')
parser.add_argument('--save_dir', type=str, required=True, help='Directory to save models and plots')
args = parser.parse_args()

# ---------------------- 설정 ----------------------
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything()

# ---------------------- Transform ----------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5)
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---------------------- Dataset & Dataloader ----------------------
train_dataset = datasets.ImageFolder(args.train_dir, transform=train_transform)
test_dataset = datasets.ImageFolder(args.test_dir, transform=test_transform)

validation_split = 0.2
train_size = int(len(train_dataset) * (1 - validation_split))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=24)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=24)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=24)

# ---------------------- Model ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=2)
model = nn.DataParallel(model.to(device))

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001)

# ---------------------- Accuracy Function ----------------------
def calculate_accuracy(loader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return 100 * correct / total

# ---------------------- Save Function ----------------------
def save_model(epoch, model, optimizer, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)

# ---------------------- Training Loop ----------------------
train_acc, val_acc, test_acc = [], [], []
best_val_acc = 0
epochs_no_improve = 0
patience = 10

for epoch in range(200):
    model.train()
    start = time.time()
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        progress = (i + 1) / len(train_loader) * 100
        print(f'\rEpoch {epoch+1} [{progress:.2f}%]...', end='')

    train_accuracy = calculate_accuracy(train_loader, model)
    val_accuracy = calculate_accuracy(val_loader, model)
    train_acc.append(train_accuracy)
    val_acc.append(val_accuracy)
    print(f'\nEpoch {epoch+1} - Train: {train_accuracy:.2f}%, Val: {val_accuracy:.2f}%')

    # Save current epoch model
    save_model(epoch, model, optimizer, os.path.join(args.save_dir, f'model_epoch_{epoch+1}.pth'))

    # Save best model
    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        epochs_no_improve = 0
        save_model(epoch, model, optimizer, os.path.join(args.save_dir, 'best_model.pth'))
    else:
        epochs_no_improve += 1
        if epochs_no_improve == patience:
            print('Early stopping triggered.')
            break

# ---------------------- Test ----------------------
checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pth'))
model.module.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

test_accuracy = calculate_accuracy(test_loader, model)
test_acc.append(test_accuracy)
print(f'Test Accuracy: {test_accuracy:.2f}%')

# ---------------------- Plot ----------------------
plt.plot(train_acc, label='Train Acc')
plt.plot(val_acc, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.ylim([0, 100])
plt.savefig(os.path.join(args.save_dir, 'Acc_supervised_2class.png'))
plt.show()
