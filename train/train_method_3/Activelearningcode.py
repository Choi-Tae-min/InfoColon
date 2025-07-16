# 전체 실행 스크립트 통합본 with argparse + BALD/AD-BALD 선택
import os
import argparse
import numpy as np
import random
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import timm

# ===========================================
# argparse
# ===========================================
parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, choices=[2, 6, 7], default=2)
parser.add_argument('--method', type=str, choices=['bald', 'ad_bald'], default='ad_bald')
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--rounds', type=int, default=21)
parser.add_argument('--random_sample', type=int, default=None)
parser.add_argument('--topk', type=int, default=None)
parser.add_argument('--threshold', type=float, default=None)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--num_workers', type=int, default=24)
parser.add_argument('--gpus', type=int, choices=[1, 2], default=2)
parser.add_argument('--train_dir', type=str, default='/home/test/colonoscopy/data/class2/train/')
parser.add_argument('--val_dir', type=str, default='/home/test/colonoscopy/data/class2/val/')
parser.add_argument('--test_dir', type=str, default='/home/test/colonoscopy/data/class2/test/')
parser.add_argument('--unlabeled_dir', type=str, default='/home/test/colonoscopy/data/unlabeled/')
args = parser.parse_args()

# 클래스 기본값 설정
class_defaults = {
    2: {'sample_size': 6000, 'topk': 600, 'threshold': 0.3},
    6: {'sample_size': 4000, 'topk': 400, 'threshold': 0.7},
    7: {'sample_size': 6000, 'topk': 4000, 'threshold': 0.7},
}
defs = class_defaults[args.num_classes]
sample_size = args.random_sample if args.random_sample else defs['sample_size']
topk = args.topk if args.topk else defs['topk']
threshold = args.threshold if args.threshold else defs['threshold']

# =============================
# CUDA & SEED
# =============================
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

# =============================
# Dataset 클래스 수정 (2-class일 경우 uninformative 자동 통합)
# =============================
class ImageDataset(Dataset):
    def __init__(self, base_dir, categories, transform=None, labeled=True):
        self.filepaths = []
        self.labels = []
        self.transform = transform or Compose([Resize((224, 224)), ToTensor()])
        self.labeled = labeled

        if labeled:
            if len(categories) == 2 and categories[-1] == 'informative':
                # 2-class: informative 폴더 제외한 모든 폴더 = uninformative
                for category in os.listdir(base_dir):
                    folder_path = os.path.join(base_dir, category)
                    if not os.path.isdir(folder_path):
                        continue
                    label = 1 if category == 'informative' else 0
                    for filename in os.listdir(folder_path):
                        self.filepaths.append(os.path.join(folder_path, filename))
                        self.labels.append(label)
            else:
                for label, category in enumerate(categories):
                    folder_path = os.path.join(base_dir, category)
                    for filename in os.listdir(folder_path):
                        self.filepaths.append(os.path.join(folder_path, filename))
                        self.labels.append(label)
        else:
            for filename in os.listdir(base_dir):
                self.filepaths.append(os.path.join(base_dir, filename))
                self.labels.append(None)  # Placeholder

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img = Image.open(self.filepaths[idx]).convert('RGB')
        img = self.transform(img)
        label = self.labels[idx] if self.labels[idx] is not None else -1
        return img, label, self.filepaths[idx]

# =============================
# Transform 설정
# =============================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =============================
# DataLoader 설정
# =============================
categories = ['bad_light', 'blurry', 'bubble', 'obstacles', 'tool', 'wall', 'informative'][-args.num_classes:]
train_dataset = ImageDataset(args.train_dir, categories, transform=train_transform)
val_dataset = ImageDataset(args.val_dir, categories, transform=train_transform)
test_dataset = ImageDataset(args.test_dir, categories, transform=test_transform)
unlabeled_dataset = ImageDataset(args.unlabeled_dir, categories, labeled=False, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

# =============================
# Model 설정
# =============================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=args.num_classes)
if args.gpus == 2:
    model = nn.DataParallel(model).to(device)
else:
    model = model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

# =============================
# Active Learning Loop
# =============================
print("Starting initial training...")
train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=100, patience=args.patience)
print("Initial training complete.")
initial_test_accuracy, _ = evaluate_model(model, test_loader, device, criterion)
print(initial_test_accuracy)

accuracies = [initial_test_accuracy]
train_sizes = [len(train_dataset)]
used_indices = set()
previous_accuracy = initial_test_accuracy

# 평가 지표 저장용
f1_scores = []
precisions = []
recalls = []
specificities = []
aurocs = []

for idx in range(args.rounds):
    print(f"Round {idx + 1}")

    available_indices = [i for i in range(len(unlabeled_dataset)) if i not in used_indices]
    if len(available_indices) == 0:
        print("No more available data to sample. Exiting loop.")
        break

    current_sample_size = min(sample_size, len(available_indices))
    random_sample_indices = np.random.choice(available_indices, size=current_sample_size, replace=False)
    random_samples = [unlabeled_dataset[i] for i in random_sample_indices]
    samples_loader = DataLoader(Subset(unlabeled_dataset, random_sample_indices), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if args.method == 'bald':
        query_indices = bald_query_topk(model, samples_loader, top_k=topk, device=device)
    else:
        query_indices = bald_query_with_threshold(model, samples_loader, uncertainty_threshold=threshold, device=device)

    queried_data = [random_samples[i] for i in query_indices]
    train_dataset = update_training_data(train_dataset, queried_data)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=100, patience=args.patience)

    current_accuracy, _, _ = evaluate_model(model, test_loader, device, criterion)
    accuracies.append(current_accuracy)
    train_sizes.append(len(train_dataset))
    used_indices.update(random_sample_indices)

    # ====== 추가: 정밀도, 재현율, F1, Specificity, AUROC ======
    all_preds = []
    all_labels = []
    all_probs = []
    model.eval()
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy()) if args.num_classes == 2 else all_probs.extend(probs.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel() if args.num_classes == 2 else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    auroc = roc_auc_score(all_labels, all_probs, multi_class='ovr') if args.num_classes > 2 else roc_auc_score(all_labels, all_probs)

    f1_scores.append(f1)
    precisions.append(precision)
    recalls.append(recall)
    specificities.append(specificity)
    aurocs.append(auroc)

    print(f"Round {idx + 1} F1: {f1:.4f}  Precision: {precision:.4f}  Recall: {recall:.4f}  Specificity: {specificity:.4f}  AUROC: {auroc:.4f}")

    threshold = adjust_threshold(current_accuracy, previous_accuracy, threshold)
    previous_accuracy = current_accuracy

plt.figure(figsize=(13, 8))
plt.plot(accuracies, marker='o', label='Test Accuracies', color='green')
plt.xticks(range(args.rounds + 1), ['0'] + [str(i+1) for i in range(args.rounds)], rotation=0)
plt.xlabel('Round')
plt.ylabel('Accuracy (%)')
plt.ylim([40, 100])
plt.grid(True)
plt.legend()
plt.savefig(f'./AD_Acc_all_split_active_learning_{args.num_classes}class.png')
print(f'Final Accuracy: {accuracies[-1]}%')
