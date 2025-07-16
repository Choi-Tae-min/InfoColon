# pseudo_labeling.py (with support for 2, 6, and 7-class settings)

import os
import argparse
import random
import time
import copy
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, f1_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
from timm import create_model

# ---------------------- Argument Parsing ----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', type=str, required=True)
parser.add_argument('--val_dir', type=str, required=True)
parser.add_argument('--test_dir', type=str, required=True)
parser.add_argument('--unlabeled_dir', type=str, required=True)
parser.add_argument('--save_dir', type=str, default='results/pseudo_ssl')
parser.add_argument('--model_name', type=str, default='vit_small_patch16_224')
parser.add_argument('--threshold', type=float, default=0.99)
parser.add_argument('--num_classes', type=int, default=2, choices=[2, 6, 7])
parser.add_argument('--gpus', type=str, default="0,1,2,3")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--rounds', type=int)
parser.add_argument('--samples_per_round', type=int)
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--num_workers', type=int, default=24)
parser.add_argument('--patience', type=int, default=5)
args = parser.parse_args()

# ---------------------- Setup ----------------------
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_count = torch.cuda.device_count()

class_names_all = ['bad_light', 'blurry', 'bubble', 'obstacles', 'tool', 'wall', 'informative']
if args.num_classes == 2:
    class_names = ['informative', 'uninformative']
    args.rounds = args.rounds or 21
    args.samples_per_round = args.samples_per_round or 6000
elif args.num_classes == 6:
    class_names = [c for c in class_names_all if c != 'informative']
    args.rounds = args.rounds or 12
    args.samples_per_round = args.samples_per_round or 4000
else:
    class_names = class_names_all
    args.rounds = args.rounds or 21
    args.samples_per_round = args.samples_per_round or 6000

# ---------------------- Dataset Classes ----------------------
class ImageDataset(Dataset):
    def __init__(self, base_dir, transform=None, num_classes=2):
        self.paths, self.labels = [], []
        self.transform = transform
        for root, _, files in os.walk(base_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    path = os.path.join(root, f)
                    fname = f.lower()
                    if num_classes == 2:
                        label = 0 if 'informative' in fname else 1
                    else:
                        for i, cls in enumerate(class_names):
                            if cls in fname:
                                label = i
                                break
                    self.paths.append(path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

class UnlabeledImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.transform = transform
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(root, file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, -1

class PseudoLabeledDataset(Dataset):
    def __init__(self, original_dataset, pseudo_images, pseudo_labels):
        self.original_dataset = original_dataset
        self.pseudo_images = pseudo_images
        self.pseudo_labels = pseudo_labels

    def __len__(self):
        return len(self.original_dataset) + len(self.pseudo_images)

    def __getitem__(self, idx):
        if idx < len(self.original_dataset):
            return self.original_dataset[idx]
        else:
            pseudo_idx = idx - len(self.original_dataset)
            return self.pseudo_images[pseudo_idx], self.pseudo_labels[pseudo_idx]

# ---------------------- Transforms ----------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---------------------- Dataloaders ----------------------
train_dataset = ImageDataset(args.train_dir, transform=train_transform, num_classes=args.num_classes)
val_dataset = ImageDataset(args.val_dir, transform=val_transform, num_classes=args.num_classes)
test_dataset = ImageDataset(args.test_dir, transform=val_transform, num_classes=args.num_classes)
unlabeled_dataset = UnlabeledImageDataset(args.unlabeled_dir, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

# ---------------------- Model ----------------------
model = create_model(args.model_name, pretrained=True, num_classes=args.num_classes)
if gpu_count > 1:
    model = nn.DataParallel(model)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# ---------------------- Training Utils ----------------------
class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_acc = None
        self.best_model_wts = None
        self.early_stop = False

    def __call__(self, val_acc, model):
        if self.best_acc is None or val_acc > self.best_acc:
            self.best_acc = val_acc
            self.best_model_wts = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

def evaluate_model(model, dataloader, round_num=None):
    model.eval()
    y_true, y_pred, y_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = nn.functional.softmax(outputs, dim=1)
            preds = torch.argmax(probs, 1)

            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    acc = (y_pred == y_true).mean()
    f1 = f1_score(y_true, y_pred, average='macro')
    prec = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')

    try:
        if args.num_classes == 2:
            auroc = roc_auc_score(y_true, y_probs[:, 1])
        else:
            auroc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
    except:
        auroc = 'N/A'
    # Confusion Matrix 저장
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {round_num}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    os.makedirs(args.save_dir, exist_ok=True)
    plt.savefig(os.path.join(args.save_dir, f'confusion_matrix_{round_num}.png'))
    plt.close()

    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"F1-score:  {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"AUROC:     {auroc if auroc != 'N/A' else 'N/A'}")

    return acc, f1, prec, recall, auroc, cm
def compute_accuracy_only(model, dataloader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total
def train_model(model, train_loader, val_loader):
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    for epoch in range(100):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()

        # ✅ train accuracy만 간단히 출력
        train_acc = compute_accuracy_only(model, train_loader)
        # ✅ validation은 full 평가
        val_acc= compute_accuracy_only(model, val_loader)

        print(f"Epoch {epoch+1} - Train Acc: {train_acc*100:.2f}%, Val Acc: {val_acc*100:.2f}%")
        early_stopping(val_acc, model)
        if early_stopping.early_stop:
            print(">>> Early stopping.")
            break

    model.load_state_dict(early_stopping.best_model_wts)
    return model

# ---------------------- Pseudo-labeling ----------------------
def generate_pseudo_labels(model, dataloader, threshold):
    model.eval()
    pseudo_imgs, pseudo_lbls = [], []
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            probs = nn.functional.softmax(model(inputs), dim=1)
            max_probs, preds = torch.max(probs, 1)
            mask = max_probs > threshold
            pseudo_imgs.extend(inputs[mask].cpu())
            pseudo_lbls.extend(preds[mask].cpu())
    return pseudo_imgs, pseudo_lbls

# ---------------------- Main Loop ----------------------
print("==> Start Initial Training")
model = train_model(model, train_loader, val_loader)
evaluate_model(model, test_loader,round_num=0)

used_indices = set()
for r in range(args.rounds):
    print(f"\n===== Round {r+1}/{args.rounds} =====")
    available = list(set(range(len(unlabeled_dataset))) - used_indices)
    sampled = random.sample(available, min(args.samples_per_round, len(available)))
    used_indices.update(sampled)
    subset_loader = DataLoader(Subset(unlabeled_dataset, sampled), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    pseudo_imgs, pseudo_lbls = generate_pseudo_labels(model, subset_loader, args.threshold)
    train_dataset = PseudoLabeledDataset(train_dataset, pseudo_imgs, pseudo_lbls)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    model = train_model(model, train_loader, val_loader)
    evaluate_model(model, test_loader,round_num=r+1)

# ---------------------- Save Model ----------------------
os.makedirs(args.save_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(args.save_dir, f"best_model_{args.num_classes}class.pth"))
