import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from torchvision import transforms, datasets
import timm
from torch import nn, optim
import matplotlib.pyplot as plt
import time
import os
import random
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Set the GPUs 2 and 3 to use

def seed_everything(seed=42):
    random.seed(seed)       # Python random module
    np.random.seed(seed)    # Numpy module
    torch.manual_seed(seed) # PyTorch
    torch.cuda.manual_seed(seed) # for using CUDA
    torch.cuda.manual_seed_all(seed) # if using multi-GPU

# Set seed for reproducibility
seed_everything()

# Define transformations
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

def count_files_in_directory(directory):
    total_files = 0
    for root, dirs, files in os.walk(directory):
        total_files += len(files)
        print(f'{root} contains {len(files)} files.')
    return total_files

base_directory = '/home/work/colonoscopy_data/dataset/class7/supervised_learning/train/'  # 'base' 폴더 경로를 지정합니다.
total_files = count_files_in_directory(base_directory)
print(f'Total number of files in the base directory and all subdirectories: {total_files}')

base_directory = '/home/work/colonoscopy_data/dataset/class7/supervised_learning/test/'  # 'base' 폴더 경로를 지정합니다.
total_files = count_files_in_directory(base_directory)
print(f'Total number of files in the base directory and all subdirectories: {total_files}')

train_dataset = datasets.ImageFolder('/home/work/colonoscopy_data/dataset/class7/supervised_learning/train/', transform=train_transform)
test_dataset = datasets.ImageFolder('/home/work/colonoscopy_data/dataset/class7/supervised_learning/test/', transform=test_transform)

# Split train dataset into train and validation
validation_split = 0.2
train_size = int(len(train_dataset) * (1 - validation_split))
validation_size = len(train_dataset) - train_size
train_dataset, validation_dataset = random_split(train_dataset, [train_size, validation_size])

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=24)
validation_loader = DataLoader(validation_dataset, batch_size=512, shuffle=False, num_workers=24)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=24)

# num_workers 값 확인
print("Number of workers in train_loader:", train_loader.num_workers)
print("Number of workers in test_loader:", test_loader.num_workers)

# Initialize the model
model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=7)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())

_model = model.cuda()
model = nn.DataParallel(_model).to(device)

# Function to calculate accuracy
def calculate_accuracy(loader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Model saving function
def save_model(epoch, model, optimizer, path='model.pth'):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

# Early stopping parameters
patience = 10
best_val_accuracy = 0
epochs_no_improve = 0

# Training and evaluation with progress tracking
train_acc = []
validation_acc = []
test_acc = []

for epoch in range(200):
    # Training Phase
    start_time = time.time()
    model.train()
    total_batches_train = len(train_loader)
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Training progress output
        time_elapsed = time.time() - start_time
        time_remaining = time_elapsed / (i + 1) * (total_batches_train - i - 1)
        progress = (i + 1) / total_batches_train * 100
        print(f'\rTraining Epoch {epoch + 1}/200, Batch {i + 1}/{total_batches_train}, Progress: {progress:.2f}%, Time remaining: {time_remaining:.2f}s', end='')

    # Validation Phase
    validation_accuracy = calculate_accuracy(validation_loader, model)
    validation_acc.append(validation_accuracy)

    # Training Accuracy
    train_accuracy = calculate_accuracy(train_loader, model)
    train_acc.append(train_accuracy)

    print(f'\nEpoch {epoch + 1} Complete: Train Accuracy: {train_accuracy}, Validation Accuracy: {validation_accuracy}')

    # Save the model after each epoch
    save_model(epoch, model, optimizer, path=f'/home/work/colonoscopy_data/dataset/supervised_learning/class7_result/model_epoch_{epoch+1}.pth')

    # Check early stopping condition
    if validation_accuracy > best_val_accuracy:
        best_val_accuracy = validation_accuracy
        epochs_no_improve = 0
        save_model(epoch, model, optimizer, path='/home/work/colonoscopy_data/dataset/supervised_learning/class7_result/best_model.pth')
    else:
        epochs_no_improve += 1

    if epochs_no_improve == patience:
        print(f'Early stopping triggered after {epoch + 1} epochs.')
        break

# Load the best model
checkpoint = torch.load('/home/work/colonoscopy_data/dataset/supervised_learning/class7_result/best_model.pth')
model.module.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Test Phase
test_accuracy = calculate_accuracy(test_loader, model)
print(f'Test Accuracy: {test_accuracy}')
test_acc.append(test_accuracy)

# Plotting
plt.plot(range(len(train_acc)), train_acc, label='Train Accuracy')
plt.plot(range(len(validation_acc)), validation_acc, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim([0, 100])
plt.legend()
plt.savefig('Acc_supervised_7class')
plt.show()