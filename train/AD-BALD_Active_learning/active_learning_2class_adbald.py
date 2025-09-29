# 필요한 모듈 임포트
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import Compose, Resize, ToTensor
from torch import nn
import torch.optim as optim
import timm
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import copy
import torch.nn.functional as F
from torchvision import transforms
import random
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve
import seaborn as sns


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(42)

class ImageDataset(Dataset):
    def __init__(self, base_dir, categories, transform=None, labeled=True):
        self.filepaths = []
        self.labels = []
        self.transform = transform or Compose([Resize((224, 224)), ToTensor()])
        self.labeled = labeled

        if labeled:
            for label, category in enumerate(categories):
                folder_path = os.path.join(base_dir, category)
                for filename in os.listdir(folder_path):
                    img_path = os.path.join(folder_path, filename)
                    self.filepaths.append(img_path)
                    self.labels.append(label)
        else:
            folder_path = base_dir
            for filename in os.listdir(folder_path):
                img_path = os.path.join(folder_path, filename)
                self.filepaths.append(img_path)
                self.labels.append(self.extract_label_from_filename(filename))

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        label = self.labels[idx]
        
        if label is None:
            label = -1  # 기본값, 에러 방지를 위해
        
        return img, label, img_path

    def extract_label_from_filename(self, filename):
        if '_informative' in filename:
            return 0
        elif any(keyword in filename for keyword in ['bad_light', 'blurry', 'bubble', 'obstacles', 'tool', 'wall']):
            return 1
        else:
            return None  # 기본값 설정

class EarlyStopping:
    def __init__(self, patience=10, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_wts = copy.deepcopy(model.state_dict())
        elif val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_model_wts = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def load_best_model(self, model):
        if self.best_model_wts is not None:
            model.load_state_dict(self.best_model_wts)

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, patience=10):
    best_acc = 0
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 레이블을 [batch_size] 형태로 변경
            if len(labels.shape) > 1:
                labels = labels.squeeze()
            
            optimizer.zero_grad()
            outputs = model(inputs)  # outputs은 [batch_size, num_classes] 형태여야 함
            

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        train_acc = 100 * correct / total
        train_loss = total_loss / len(train_loader)
        
        # 이후 코드 유지
        val_acc, val_loss = evaluate_model(model, val_loader, device, criterion)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model_ad0819.pth')
        
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    early_stopping.load_best_model(model)
    return model
class ModelWithDropout(nn.Module):
    def __init__(self, base_model, dropout_p=0.5):
        super(ModelWithDropout, self).__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.base_model.forward_features(x)  # ViT 모델의 feature extraction 부분
        x = x[:, 0, :]  # 첫 번째 토큰 (CLS 토큰)만 선택
        x = self.dropout(x)  # feature에 드롭아웃 추가
        x = self.base_model.head(x)  # 분류 head 부분
        return x
def evaluate_model(model, test_loader, device, criterion):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for images, labels, paths in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    return accuracy, avg_loss

def adjust_threshold(current_accuracy, previous_accuracy, threshold, adjustment_factor=0.1, max_threshold=0.9, min_threshold=0.3, change_factor=0.7):
    # 정확도 변화 계산
    accuracy_change = current_accuracy - previous_accuracy
    if accuracy_change > change_factor:
        threshold += adjustment_factor
    elif accuracy_change < -change_factor:
        threshold -= adjustment_factor

    # threshold를 상한과 하한으로 제한
    threshold = max(min(threshold, max_threshold), min_threshold)
    
    return threshold

def predict_prob_dropout_split(model, data_loader, n_drop, device):
    model.train()
    all_probs = []
    for i in range(n_drop):
        probs = []
        for images, labels, paths in data_loader:
            images = images.to(device)
            with torch.no_grad():
                outputs = model(images)
                probs_batch = F.softmax(outputs, dim=1)
                probs_batch = torch.clamp(probs_batch, min=1e-10, max=1-1e-10)
                probs.append(probs_batch)
        all_probs.append(torch.cat(probs, dim=0))
    return torch.stack(all_probs)


def select_random_samples(dataset, sample_size):
    indices = np.random.choice(len(dataset), size=sample_size, replace=False)
    return [dataset[i] for i in indices]

def create_loader_from_samples(samples, dataset, batch_size=64):
    indices = [dataset.filepaths.index(sample[2]) for sample in samples]
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=24)

def bald_query_with_threshold(model, samples_loader, n_drop=15, uncertainty_threshold=0.3, device='cuda'):
    probs = predict_prob_dropout_split(model, samples_loader, n_drop, device)
    pb = probs.mean(0)
    epsilon = 1e-10
    pb_clamped = torch.clamp(pb, min=epsilon, max=1-epsilon)
    probs_clamped = torch.clamp(probs, min=epsilon, max=1-epsilon)

    entropy1 = -(pb_clamped * torch.log(pb_clamped)).sum(1)
    entropy2 = -(probs_clamped * torch.log(probs_clamped)).sum(2).mean(0)

    uncertainties = entropy1 - entropy2
    print("Entropy1:", entropy1)
    print("Entropy2:", entropy2)
    print("Uncertainties:", uncertainties)

    query_indices = (uncertainties > uncertainty_threshold).nonzero(as_tuple=True)[0]
    print(f"uncertainty_threshold:{uncertainty_threshold}   Number of data points selected: {len(query_indices)}")
    return query_indices

def update_training_data(train_dataset, queried_data):
    label0=0
    label1=0
    initial_size = len(train_dataset)
    new_paths = [x[2] for x in queried_data]
    new_labels = [x[1] for x in queried_data]

    added_count = 0  # 추가된 데이터 수를 추적합니다.
    for path, label in zip(new_paths, new_labels):
        if path not in train_dataset.filepaths:
            train_dataset.filepaths.append(path)
            train_dataset.labels.append(label)
            if label==0:
                label0+=1
            elif label==1:
                label1+=1
            added_count += 1  # 데이터가 추가될 때마다 증가합니다.

    new_size = len(train_dataset)
    print(f"Added {added_count} new data points (unique only).")
    print(f"New size of the training dataset: {new_size}")
    print(f"label_informative {label0} label_uninformative{label1}")
    return train_dataset
def plot_confusion_matrix(cm, classes, round_num, accuracy, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - Round {round_num}, Accuracy: {accuracy:.2f}%')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xticks(ticks=np.arange(len(classes))+0.5, labels=classes, rotation=45)
    plt.yticks(ticks=np.arange(len(classes))+0.5, labels=classes, rotation=0)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def evaluate_model_with_cm(model, test_loader, device, criterion, round_num, categories):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, paths in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, categories, round_num, accuracy, f'2class_confusion_matrix_round_{round_num}.png')
    
    return accuracy, avg_loss, cm
def calculate_metrics_binary(loader, model):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for data in loader:
            images, labels, _ = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())  # Positive 클래스 확률만 가져옴

    # 혼동 행렬
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    # 민감도, 특이도
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # F1-score
    f1 = f1_score(all_labels, all_preds)

    # AUROC
    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auroc = float('nan')  # 예외 처리

    return sensitivity, specificity, f1, auroc
# Model saving function
train_dir = '/home/work/tmchoi/dataset/class2/semi_active_learning/train/train2/'
val_dir='/home/work/tmchoi/dataset/class2/semi_active_learning/train/val2/'
test_dir = '/home/work/tmchoi/dataset/class2/semi_active_learning/test/'
unlabeled_dir = '/home/work/tmchoi/dataset/class2/semi_active_learning/train/unlabeled/'
categories = ['informative', 'uninformative']
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),

])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = ImageDataset(train_dir, categories, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=24,pin_memory=True)
val_dataset=ImageDataset(val_dir, categories, transform=train_transform)
validation_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=24,pin_memory=True)
test_dataset = ImageDataset(test_dir, categories, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=24,pin_memory=True)
# 데이터셋 로드 및 검증
unlabeled_dataset = ImageDataset(unlabeled_dir, categories, labeled=False)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=512, shuffle=True, num_workers=36,pin_memory=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

dropout_p = 0.5
base_model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=2)
model = ModelWithDropout(base_model, dropout_p=dropout_p).to(device)
print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())
_model = model.cuda()
model = nn.DataParallel(_model).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

rounds = 21
train_sizes = []
accuracies = []
remaining_data_points = []
initial_test_accuracy = 0
initial_train_size = len(train_dataset)

print("Starting initial training...")
train_model(model, train_loader, validation_loader, criterion, optimizer, device, epochs=100)
print("Initial training complete.")
initial_test_accuracy, _ = evaluate_model(model, test_loader, device, criterion)
print(initial_test_accuracy)
accuracies.append(initial_test_accuracy)
used_indices = set()
query_num2 = 0
query_num1 = 0
i = 0
thr = 0.2
noncer_data = 0
previous_accuracy = initial_test_accuracy
sample_size = 6000
adjustment_factor = 0.1
for idx in range(rounds):
    print(f'Round no. {idx + 1}')
    
    available_indices = [i for i in range(len(unlabeled_dataset)) if i not in used_indices]

    if len(available_indices) == 0:
        print("No more available data to sample. Exiting loop.")
        break

    current_sample_size = min(sample_size, len(available_indices))
    random_sample_indices = np.random.choice(available_indices, size=current_sample_size, replace=False)
    random_samples = [unlabeled_dataset[i] for i in random_sample_indices]
    print(len(random_samples))
    
    samples_loader = create_loader_from_samples(random_samples, unlabeled_dataset)
    
    query_indices = bald_query_with_threshold(model, samples_loader, uncertainty_threshold=thr, device=device)
    
    queried_data = [random_samples[i] for i in query_indices]
    train_dataset = update_training_data(train_dataset, queried_data)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=24,pin_memory=True)
    train_model(model, train_loader, validation_loader, criterion, optimizer, device, epochs=100)
    
    # Evaluate model and calculate confusion matrix
    current_accuracy, _, cm = evaluate_model_with_cm(model, test_loader, device, criterion, idx + 1, categories)

    # 이진 분류 단계에서 사용
    sensitivity, specificity, f1, auroc = calculate_metrics_binary(test_loader, model)
    print(f'이진 분류 - 민감도: {sensitivity:.4f}')
    print(f'이진 분류 - 특이도: {specificity:.4f}')
    print(f'이진 분류 - F1-score: {f1:.4f}')
    print(f'이진 분류 - AUROC: {auroc:.4f}')
    accuracies.append(current_accuracy)
    train_sizes.append(len(train_dataset))
    print("left data : ", current_sample_size - len(query_indices))
    remaining_data_points.append(current_sample_size - len(query_indices))
    used_indices.update(random_sample_indices)
    print(f'Current training size: {train_sizes[-1]}, Accuracy: {current_accuracy}%')
    thr = adjust_threshold(current_accuracy, previous_accuracy, thr, adjustment_factor)
    previous_accuracy = current_accuracy



plt.figure(figsize=(13, 8))
plt.plot(accuracies, marker='o',label='Test Accuracies',color='red')
plt.xticks(range(rounds + 1), ['0'] + [f'{i+1}' for i in range(rounds)], rotation=0)
plt.xlabel('Round')
plt.ylabel('Accuracy (%)')
plt.ylim([40, 100])
plt.grid(True)
plt.savefig('/home/work/tmchoi/dataset/active_learning_result/class2/Acc_all_split_active_learning_2class_0819.png')

print(f'Final Accuracy: {accuracies[-1]}%')
