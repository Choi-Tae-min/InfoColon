import os
import argparse
import random
import shutil
import time
import copy
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, datasets

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
parser.add_argument('--flat_unlabeled', action='store_true', help='If set, treat unlabeled_dir as flat (no subfolders)')
parser.add_argument('--save_dir', type=str, default='results/pseudo_ssl')
parser.add_argument('--model_name', type=str, default='vit_small_patch16_224')
parser.add_argument('--threshold', type=float, default=0.95)
parser.add_argument('--num_classes', type=int, default=7)
parser.add_argument('--gpus', type=str, default="0")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--rounds', type=int)
parser.add_argument('--samples_per_round', type=int)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
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
    uninformative_subclasses = ['bad_light', 'blurry', 'bubble', 'obstacles', 'tool', 'wall']
    print(f"INFO: 2-Class setting: [0] informative, [1] uninformative (includes: {', '.join(uninformative_subclasses)})")
    args.rounds = args.rounds or 21
    args.samples_per_round = args.samples_per_round or 6000
elif args.num_classes == 6:
    class_names = [cls for cls in class_names_all if cls != 'informative']
    args.rounds = args.rounds or 12
    args.samples_per_round = args.samples_per_round or 4000
else:
    class_names = class_names_all
    args.rounds = args.rounds or 21
    args.samples_per_round = args.samples_per_round or 6000

# ---------------------- Dataset Classes ----------------------
class UnlabeledImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        if args.flat_unlabeled:
            for img_file in os.listdir(root_dir):
                self.image_paths.append(os.path.join(root_dir, img_file))
        else:
            for cls_folder in os.listdir(root_dir):
                cls_path = os.path.join(root_dir, cls_folder)
                if os.path.isdir(cls_path):
                    for img_file in os.listdir(cls_path):
                        self.image_paths.append(os.path.join(cls_path, img_file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.get_label_from_filename(img_path)
        return image, label

    def get_label_from_filename(self, filepath):
        fname = os.path.basename(filepath)
        if args.num_classes == 2:
            if 'informative' in fname:
                return 0
            elif any(subcls in fname for subcls in uninformative_subclasses):
                return 1
        else:
            for i, cls in enumerate(class_names):
                if cls in fname:
                    return i
        raise ValueError(f"No valid class found in filename: {fname}")

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
            return self.pseudo_images[pseudo_idx], self.pseudo_labels[pseudo_idx]# ---------------------- EarlyStopping ----------------------
class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_acc = None
        self.early_stop = False
        self.best_model_wts = None

    def __call__(self, val_acc, model):
        if self.best_acc is None:
            self.best_acc = val_acc
            self.best_model_wts = copy.deepcopy(model.state_dict())
        elif val_acc > self.best_acc:
            self.best_acc = val_acc
            self.best_model_wts = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

# ---------------------- Transforms ----------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---------------------- Datasets & Dataloaders ----------------------
train_dataset = datasets.ImageFolder(args.train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(args.val_dir, transform=val_transform)
test_dataset = datasets.ImageFolder(args.test_dir, transform=val_transform)
if args.unlabeled_dir:
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

# ---------------------- Evaluation ----------------------
def calculate_specificity(cm):
    FP = cm.sum(axis=0) - np.diag(cm)
    TN = cm.sum() - (FP + cm.sum(axis=1) - np.diag(cm) + np.diag(cm))
    specificity = TN / (TN + FP + 1e-6)
    return np.mean(specificity)

def evaluate(model, dataloader, round_num=None, save_cm=False):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    matrix = confusion_matrix(all_labels, all_preds)
    auroc = roc_auc_score(all_labels, nn.functional.one_hot(torch.tensor(all_preds), num_classes=args.num_classes), multi_class='ovr')
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    specificity = calculate_specificity(matrix)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    if save_cm:
        os.makedirs(args.save_dir, exist_ok=True)
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        cm_name = f'confusion_matrix_round{round_num}_{args.num_classes}class.png' if round_num else f'confusion_matrix_{args.num_classes}class.png'
        plt.savefig(os.path.join(args.save_dir, cm_name))
        plt.close()
    print(f"Accuracy: {acc:.4f}, AUROC: {auroc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Specificity: {specificity:.4f}")
    return report, matrix, auroc, f1, precision, specificity, acc

# ---------------------- Training ----------------------
def evaluate_model(model, dataloader, device, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    acc = 100 * correct / total
    avg_loss = total_loss / len(dataloader)
    return acc, avg_loss

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=20, patience=5):
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        train_acc = 100 * correct / total
        train_loss = total_loss / len(train_loader)
        val_acc, val_loss = evaluate_model(model, val_loader, device, criterion)
        print(f"Epoch {epoch+1} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        early_stopping(val_acc, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
    model.load_state_dict(early_stopping.best_model_wts)
    return model

# ---------------------- Pseudo-labeling ----------------------
def generate_pseudo_labels(model, dataloader, threshold):
    model.eval()
    pseudo_images, pseudo_labels = [], []
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = nn.functional.softmax(outputs, dim=1)
            max_probs, preds = torch.max(probs, dim=1)
            for i in range(len(inputs)):
                if max_probs[i] > threshold:
                    pseudo_images.append(inputs[i].cpu())
                    pseudo_labels.append(preds[i].cpu())
    return pseudo_images, pseudo_labels

# ---------------------- Pseudo-labeling Loop ----------------------
used_indices = set()
model = train_model(model, train_loader, val_loader, criterion, optimizer, device, patience=args.patience)
evaluate(model, test_loader, round_num="initial")

for r in range(args.rounds):
    print(f"\n==== Round {r+1} ====")
    available = list(set(range(len(unlabeled_dataset))) - used_indices)
    sampled = random.sample(available, min(args.samples_per_round, len(available)))
    used_indices.update(sampled)
    subset_loader = DataLoader(Subset(unlabeled_dataset, sampled), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    pseudo_imgs, pseudo_lbls = generate_pseudo_labels(model, subset_loader, args.threshold)
    new_train_dataset = PseudoLabeledDataset(train_dataset, pseudo_imgs, pseudo_lbls)
    train_loader = DataLoader(new_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    model = train_model(model, train_loader, val_loader, criterion, optimizer, device, patience=args.patience)
    evaluate(model, test_loader, round_num=r+1)

# ---------------------- Final Save ----------------------
print("\n==== Final Evaluation ====")
evaluate(model, test_loader, round_num="final", save_cm=True)
torch.save(model.state_dict(), os.path.join(args.save_dir, f"best_model_{args.num_classes}class.pth"))
