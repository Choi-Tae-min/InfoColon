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

# ---------------------- Utility Functions ----------------------
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_specificity(cm):
    FP = cm.sum(axis=0) - np.diag(cm)
    TN = cm.sum() - (FP + cm.sum(axis=1) - np.diag(cm) + np.diag(cm))
    specificity = TN / (TN + FP + 1e-6)
    return np.mean(specificity)

# ---------------------- Dataset Classes ----------------------
class UnlabeledImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
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
            return self.pseudo_images[pseudo_idx], self.pseudo_labels[pseudo_idx]

# ---------------------- Argument Parsing ----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', type=str, required=True)
parser.add_argument('--val_dir', type=str, required=True)
parser.add_argument('--test_dir', type=str, required=True)
parser.add_argument('--unlabeled_dir', type=str, required=True)
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
args = parser.parse_args()

# ---------------------- Setup ----------------------
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
fix_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gpu_count = torch.cuda.device_count()

class_names_all = ['informative', 'bad_light', 'blurry', 'bubble', 'obstacles', 'tool', 'wall']
if args.num_classes == 2:
    class_names = ['informative', 'uninformative']
    if args.rounds is None:
        args.rounds = 21
    if args.samples_per_round is None:
        args.samples_per_round = 6000
elif args.num_classes == 6:
    class_names = [cls for cls in class_names_all if cls != 'informative']
    if args.rounds is None:
        args.rounds = 15
    if args.samples_per_round is None:
        args.samples_per_round = 4000
else:
    class_names = class_names_all
    if args.rounds is None:
        args.rounds = 10
    if args.samples_per_round is None:
        args.samples_per_round = 3000

# ---------------------- Transforms ----------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---------------------- Dataset & Dataloader ----------------------
train_dataset = datasets.ImageFolder(args.train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(args.val_dir, transform=val_transform)
test_dataset = datasets.ImageFolder(args.test_dir, transform=val_transform)
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

# ---------------------- Training Function ----------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=20):
    best_acc = 0
    best_model_wts = copy.deepcopy(model.state_dict())

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

        print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.2f}%, Train Loss: {train_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}%, Val Loss: {val_loss:.4f}")

        # 모델 저장 조건: 검증 정확도 기준
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), 'best_model.pth')

        # 조기 종료 조건
        if train_acc >= 99.0:
            print(f"Early stopping at epoch {epoch+1} as train accuracy reached {train_acc:.2f}%")
            break

    model.load_state_dict(best_model_wts)
    return model
# ---------------------- Evaluation Function ----------------------
def evaluate(model, dataloader, round_num=None):
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

# ---------------------- Pseudo-labeling Function ----------------------
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

# ---------------------- Active Learning Loop ----------------------
used_indices = set()
model = train(model, train_loader, val_loader, criterion, optimizer, device)
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

    model = train(model, train_loader, val_loader, criterion, optimizer, device)
    evaluate(model, test_loader, round_num=r+1)

# Final test evaluation
print("\n==== Final Evaluation ====")
evaluate(model, test_loader)

# Save model
torch.save(model.state_dict(), os.path.join(args.save_dir, f"best_model_{args.num_classes}class.pth"))
