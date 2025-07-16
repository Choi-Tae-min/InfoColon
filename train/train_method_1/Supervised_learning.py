import argparse
import os
import random
import time
from glob import glob

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
import timm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score

# ---------------------- argparse ----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--train_root_dir', type=str, required=True, help='Path containing train/ val/ unlabeled/')
parser.add_argument('--test_dir', type=str, required=True, help='Test dataset path')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size for dataloaders')
parser.add_argument('--num_workers', type=int, default=24, help='Number of workers for dataloaders')
parser.add_argument('--epochs', type=int, default=200, help='Total training epochs')
parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
parser.add_argument('--model_name', type=str, default='vit_small_patch16_224', help='Model name from timm')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for optimizer')
parser.add_argument('--num_classes', type=int, choices=[2, 6, 7], default=7, help='Number of classes (2, 6, or 7)')
args = parser.parse_args()

CLASS_SETS = {
    6: ['bad_light', 'blurry', 'bubble', 'obstacles', 'tool', 'wall'],
    7: ['informative', 'bad_light', 'blurry', 'bubble', 'obstacles', 'tool', 'wall']
}

# ---------------------- Save Path ----------------------
current_path = os.path.abspath(os.path.dirname(__file__))
train_root = os.path.abspath(os.path.join(current_path, '..'))
save_dir = os.path.join(train_root, 'result', f'supervised_class{args.num_classes}_result')
os.makedirs(save_dir, exist_ok=True)

# ---------------------- Seed ----------------------
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

# ---------------------- Custom Dataset for 2-class ----------------------
class TwoClassDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.samples = []
        self.transform = transform
        for sub in ['train', 'val', 'unlabeled']:
            for class_dir in os.listdir(os.path.join(root_dir, sub)):
                full_path = os.path.join(root_dir, sub, class_dir)
                label = 0 if class_dir == 'informative' else 1
                for img_file in glob(os.path.join(full_path, '*')):
                    self.samples.append((img_file, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = datasets.folder.default_loader(path)
        if self.transform:
            image = self.transform(image)
        return image, label

# ---------------------- Dataset loading ----------------------
if args.num_classes == 2:
    full_dataset = TwoClassDataset(args.train_root_dir, transform=train_transform)
    class_names = ['informative', 'uninformative']
else:
    selected_classes = CLASS_SETS[args.num_classes]
    merged_dir = os.path.join(args.train_root_dir, 'merged_temp')
    os.makedirs(merged_dir, exist_ok=True)
    for sub in ['train', 'val', 'unlabeled']:
        subdir = os.path.join(args.train_root_dir, sub)
        for class_dir in os.listdir(subdir):
            if class_dir not in selected_classes:
                continue
            src = os.path.join(subdir, class_dir)
            dst = os.path.join(merged_dir, class_dir)
            os.makedirs(dst, exist_ok=True)
            for f in glob(os.path.join(src, '*')):
                os.link(f, os.path.join(dst, os.path.basename(f)))
    full_dataset = datasets.ImageFolder(merged_dir, transform=train_transform)
    class_names = full_dataset.classes

train_size = int(len(full_dataset) * 0.8)
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# ---------------------- Test Dataset loading ----------------------
test_dataset = datasets.ImageFolder(args.test_dir, transform=test_transform)
if args.num_classes == 2:
    filtered = []
    for img_path, _ in test_dataset.samples:
        class_name = os.path.basename(os.path.dirname(img_path))
        label = 0 if class_name == 'informative' else 1
        filtered.append((img_path, label))
    test_dataset.samples = filtered
    test_dataset.targets = [label for _, label in filtered]
else:
    test_dataset.samples = [s for s in test_dataset.samples if os.path.basename(os.path.dirname(s[0])) in class_names]
    test_dataset.targets = [class_names.index(os.path.basename(os.path.dirname(s[0]))) for s in test_dataset.samples]

# ---------------------- DataLoader ----------------------
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

# ---------------------- Model ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model(args.model_name, pretrained=False, num_classes=args.num_classes)
model = nn.DataParallel(model.to(device))

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=args.lr)

# ---------------------- Calculate Accuracy ----------------------
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

# ---------------------- Save Model ----------------------
def save_model(epoch, model, optimizer, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)

# ---------------------- Training ----------------------
train_acc, val_acc = [], []
best_val_acc = 0
no_improve = 0

for epoch in range(args.epochs):
    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f'\rEpoch {epoch+1} [{(i+1)/len(train_loader)*100:.2f}%]', end='')

    train_accuracy = calculate_accuracy(train_loader, model)
    val_accuracy = calculate_accuracy(val_loader, model)
    train_acc.append(train_accuracy)
    val_acc.append(val_accuracy)
    print(f'\nEpoch {epoch+1} - Train: {train_accuracy:.2f}%, Val: {val_accuracy:.2f}%')

    save_model(epoch, model, optimizer, os.path.join(save_dir, f'model_epoch_{epoch+1}.pth'))

    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        no_improve = 0
        save_model(epoch, model, optimizer, os.path.join(save_dir, 'best_model.pth'))
    else:
        no_improve += 1
        if no_improve >= args.patience:
            print('Early stopping.')
            break
# ---------------------- Test & Metrics ----------------------
checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'))
model.module.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(labels.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

acc = 100 * np.mean(y_pred == y_true)
f1 = f1_score(y_true, y_pred, average='macro')
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')

# Calculate macro specificity
cm = confusion_matrix(y_true, y_pred)
specificities = []
for i in range(args.num_classes):
    tp = cm[i, i]
    fn = cm[i, :].sum() - tp
    fp = cm[:, i].sum() - tp
    tn = cm.sum() - tp - fp - fn
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    specificities.append(spec)
specificity_macro = np.mean(specificities)

print(f'Test Accuracy: {acc:.2f}%')
print(f'F1 Score (macro): {f1:.4f}')
print(f'Precision (macro): {precision:.4f}')
print(f'Recall (macro): {recall:.4f}')
print(f'Specificity (macro): {specificity_macro:.4f}')

# ---------------------- Confusion Matrix ----------------------
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title(f'Confusion Matrix ({args.num_classes}-Class)')
plt.savefig(os.path.join(save_dir, f'confusion_matrix_{args.num_classes}class.png'))
plt.show()
