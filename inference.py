import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from timm import create_model

# ---------------------- Argument Parsing ----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True)
parser.add_argument('--checkpoint_path', type=str, required=True)
parser.add_argument('--model_name', type=str, default='vit_small_patch16_224')
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--num_classes', type=int, default=7, choices=[2, 6, 7])
parser.add_argument('--output_csv', type=str, default='results/predictions.csv')
args = parser.parse_args()

# ---------------------- Label Setup ----------------------
if args.num_classes == 2:
    class_names = ['informative', 'uninformative']
elif args.num_classes == 6:
    class_names = ['bad_light', 'blurry', 'bubble', 'obstacles', 'tool', 'wall']
else:
    class_names = ['bad_light', 'blurry', 'bubble', 'informative', 'obstacles', 'tool', 'wall']

# ---------------------- Dataset ----------------------
class InferenceDataset(Dataset):
    def __init__(self, base_dir, transform=None):
        self.paths = []
        self.transform = transform
        for root, _, files in os.walk(base_dir):
            for f in sorted(files):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.paths.append(os.path.join(root, f))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.paths[idx]

# ---------------------- Transforms ----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---------------------- Load Data ----------------------
dataset = InferenceDataset(args.input_dir, transform=transform)
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
print(f"Found {len(dataset)} images in {args.input_dir}")

# ---------------------- Load Model ----------------------
def load_model(path):
    ckpt = torch.load(path, map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt)
    head_key = next(k for k in state.keys() if k.endswith('head.weight'))
    out_dim = state[head_key].shape[0]
    model = create_model(args.model_name, pretrained=False, num_classes=out_dim)
    clean_state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(clean_state, strict=False)
    return model

model = load_model(args.checkpoint_path)
model = model.cuda() if torch.cuda.is_available() else model
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model.eval()

# ---------------------- Inference ----------------------
all_paths, all_preds, all_probs = [], [], []

with torch.no_grad():
    for inputs, paths in loader:
        inputs = inputs.cuda() if torch.cuda.is_available() else inputs
        probs = nn.functional.softmax(model(inputs), dim=1)
        preds = torch.argmax(probs, dim=1)

        all_paths.extend(paths)
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# ---------------------- Save Results ----------------------
rows = []
for path, pred, prob in zip(all_paths, all_preds, all_probs):
    row = {
        'filepath': path,
        'filename': os.path.basename(path),
        'predicted_class': class_names[pred],
        'confidence': prob[pred],
    }
    for i, cls in enumerate(class_names):
        row[f'prob_{cls}'] = prob[i]
    rows.append(row)

df = pd.DataFrame(rows)
os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
df.to_csv(args.output_csv, index=False)

print(f"\nPrediction summary:")
print(df['predicted_class'].value_counts().to_string())
print(f"\nSaved to {args.output_csv}")
