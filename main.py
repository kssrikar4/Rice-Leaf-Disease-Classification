import os
import json
import torch
import timm
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch import nn, optim, amp
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

class Config:
    DATA_DIR = "processed_dataset"
    MODEL_NAME = "swin_tiny_patch4_window7_224"
    IMG_SIZE = 224
    BATCH_SIZE = 16
    MAX_LR = 1e-4
    EPOCHS = 15
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    HISTORY_PATH = "history.json"
    WEIGHTS_PATH = "rice_model.pth"
    WARMUP_PCT = 0.05

class RiceDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=None)
        self.albumentations_transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = np.array(self.loader(path))
        if self.albumentations_transform:
            image = self.albumentations_transform(image=image)["image"]
        return image, target

def get_transforms():
    return A.Compose([
        A.RandomResizedCrop(size=(Config.IMG_SIZE, Config.IMG_SIZE)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def train_model():
    dataset = RiceDataset(Config.DATA_DIR, transform=get_transforms())
    train_size = int(0.8 * len(dataset))
    train_ds, val_ds = random_split(dataset, [train_size, len(dataset) - train_size])    
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4)
    model = timm.create_model(Config.MODEL_NAME, pretrained=True, num_classes=len(dataset.classes)).to(Config.DEVICE)    
    optimizer = optim.AdamW(model.parameters(), lr=Config.MAX_LR, weight_decay=0.05)    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=Config.MAX_LR, 
        steps_per_epoch=len(train_loader), 
        epochs=Config.EPOCHS,
        pct_start=Config.WARMUP_PCT,
        anneal_strategy='cos'
    )    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = amp.GradScaler('cuda')
    history = {"train_loss": [], "val_acc": [], "lrs": []}
    for epoch in range(Config.EPOCHS):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")        
        for imgs, labels in pbar:
            imgs, labels = imgs.to(Config.DEVICE), labels.to(Config.DEVICE)
            optimizer.zero_grad()            
            with amp.autocast('cuda'):
                outputs = model(imgs)
                loss = criterion(outputs, labels)            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()            
            running_loss += loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.2e}")
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(Config.DEVICE), labels.to(Config.DEVICE)
                outputs = model(imgs)
                _, pred = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
        val_acc = 100 * correct / total
        history["train_loss"].append(running_loss / len(train_loader))
        history["val_acc"].append(val_acc)
        history["lrs"].append(current_lr)        
        print(f"Validation Accuracy: {val_acc:.2f}% | Final LR: {current_lr:.2e}")
        torch.save(model.state_dict(), Config.WEIGHTS_PATH)
        with open(Config.HISTORY_PATH, "w") as f:
            json.dump(history, f)
def generate_assets():
    if os.path.exists(Config.HISTORY_PATH):
        with open(Config.HISTORY_PATH, "r") as f:
            h = json.load(f)
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax2 = ax1.twinx()
        ax1.plot(h["train_loss"], 'g-', label='Loss')
        ax2.plot(h["val_acc"], 'b-', label='Accuracy')
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss', color='g'); ax2.set_ylabel('Acc (%)', color='b')
        plt.title('Training Metrics'); plt.savefig("metrics.png")
    ds = datasets.ImageFolder(Config.DATA_DIR, transform=transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])    ]))
    indices = np.random.choice(len(ds), 16, replace=False)
    loader = DataLoader(Subset(ds, indices), batch_size=16)
    model = timm.create_model(Config.MODEL_NAME, num_classes=len(ds.classes)).to(Config.DEVICE)
    if os.path.exists(Config.WEIGHTS_PATH):
        model.load_state_dict(torch.load(Config.WEIGHTS_PATH, map_location=Config.DEVICE))
    model.eval()
    imgs, labels = next(iter(loader))
    with torch.no_grad():
        outputs = model(imgs.to(Config.DEVICE))
        _, preds = torch.max(outputs, 1)
    plt.figure(figsize=(12, 12))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        img = imgs[i].cpu().numpy().transpose((1, 2, 0))
        img = np.clip(img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)
        plt.imshow(img)
        color = 'green' if preds[i] == labels[i] else 'red'
        plt.title(f"A: {ds.classes[labels[i]]}\nP: {ds.classes[preds[i]]}", color=color, fontsize=8)
        plt.axis('off')
    plt.tight_layout(); plt.savefig("predictions.png")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--plot":
        generate_assets()
    else:
        train_model()
        generate_assets()
