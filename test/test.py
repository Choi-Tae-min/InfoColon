# test_inference.py

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from timm import create_model
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------- Argument Parsing ----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--test_dir', type=str, required=True)
parser.add_argument('--checkpoint_path', type=str, required=True)
parser.add_argument('--model_name', type=str, default='vit_small_patch16_224')
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--num_classes', type=int, default=2, choices=[2, 6, 7])
parser.add_argument('--gpus', type=str, default="0")
args = parser.parse_args()

# ---------------------- Label Setup ----------------------
if args.num_classes == 2:
    class_names = ['informative', 'uninformative']
elif args.num_classes == 6:
    class_names = ['bad_light', 'blurry', 'bubble', 'obstacles', 'tool', 'wall']
else:
    class_names = ['bad_light', 'blurry', 'bubble', 'obstacles', 'tool', 'wall', 'informative']

# ---------------------- Dataset ----------------------
class TestImageDataset(Dataset):
    def __init__(self, base_dir, transform=None, num_classes=2):
        self.paths, self.labels = [], []
        self.transform = transform
        for root, _, files in os.walk(base_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
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

# ---------------------- Transforms ----------------------
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---------------------- Load Data ----------------------
test_dataset = TestImageDataset(args.test_dir, transform=test_transform, num_classes=args.num_classes)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

# ---------------------- Load Model ----------------------
def load_model(path):
    ckpt  = torch.load(path, map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt)
    head_key = next(k for k in state.keys() if k.endswith('head.weight'))
    out_dim  = state[head_key].shape[0]

    model = create_model(
        args.model_name,
        pretrained=False,
        num_classes=out_dim
    )
    clean_state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(clean_state, strict=False)
    return model

model = load_model(args.checkpoint_path)
model = model.cuda() if torch.cuda.is_available() else model
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model.eval()

# ---------------------- Evaluation ----------------------
y_true, y_pred, y_probs = [], [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.cuda() if torch.cuda.is_available() else inputs
        outputs = model(inputs)
        probs = nn.functional.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())
        y_probs.extend(probs.cpu().numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_probs = np.array(y_probs)

# ---------------------- Metrics ----------------------
acc = (y_pred == y_true).mean()
f1 = f1_score(y_true, y_pred, average='macro')
try:
    if args.num_classes == 2:
        auroc = roc_auc_score(y_true, y_probs[:, 1])
    else:
        auroc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
except:
    auroc = 'N/A'

print("\nTest Set Evaluation Results:")
print(f"Accuracy:  {acc*100:.2f}%")
print(f"F1-score:  {f1:.4f}")
print(f"AUROC:     {auroc if auroc != 'N/A' else 'N/A'}")
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# ---------------------- Confusion Matrix ----------------------
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/test_confusion_matrix.png")
plt.close()
