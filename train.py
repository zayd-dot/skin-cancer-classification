"""Optimized Skin Cancer Classifier v2."""

import os, argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torchvision.transforms as transforms

from src.model import SkinCancerResNet
from configs.config import DataConfig, ModelConfig, LESION_TYPES


class SkinDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(row["cell_type_idx"], dtype=torch.long)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    cfg = DataConfig()

    # Load metadata
    imageid_path = {}
    for d in cfg.image_dirs:
        for p in glob(os.path.join(d, "*.jpg")):
            imageid_path[os.path.splitext(os.path.basename(p))[0]] = p
    print(f"Total images: {len(imageid_path)}")

    df = pd.read_csv(cfg.metadata_path)
    df["path"] = df["image_id"].map(imageid_path.get)
    df["cell_type"] = df["dx"].map(LESION_TYPES.get)
    df["cell_type_idx"] = pd.Categorical(df["cell_type"]).codes
    df["age"] = df["age"].fillna(df["age"].mean())
    df = df.dropna(subset=["path"])
    print(f"Samples: {len(df)}")
    print(f"Classes:\n{df['cell_type'].value_counts()}\n")

    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["cell_type_idx"], random_state=42)

    # Data augmentation (moderate, not extreme)
    train_transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    train_loader = DataLoader(SkinDataset(train_df, train_transform), batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(SkinDataset(test_df, test_transform), batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model with less dropout
    model = SkinCancerResNet(num_classes=7, dropout=0.4).to(device)

    # Mild class weights (square root scaling, not inverse)
    class_counts = train_df["cell_type_idx"].value_counts().sort_index().values.astype(float)
    weights = np.sqrt(class_counts.max() / class_counts)
    weights = torch.FloatTensor(weights).to(device)
    print(f"Class weights: {weights.cpu().numpy().round(2)}")
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(args.epochs):
        model.train()
        run_loss, correct, total = 0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(inputs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
            _, pred = torch.max(out, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

        train_loss = run_loss / len(train_loader)
        train_acc = correct / total

        model.eval()
        vloss, vcorrect, vtotal = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                out = model(inputs)
                vloss += criterion(out, labels).item()
                _, pred = torch.max(out, 1)
                vtotal += labels.size(0)
                vcorrect += (pred == labels).sum().item()

        val_loss = vloss / len(test_loader)
        val_acc = vcorrect / vtotal
        scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        lr = optimizer.param_groups[0]["lr"]
        mark = ""
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            mark = f" *** Best: {best_acc:.4f} ***"

        print(f"Epoch [{epoch+1}/{args.epochs}] Train: {train_loss:.4f}/{train_acc:.4f} | "
              f"Val: {val_loss:.4f}/{val_acc:.4f} | LR: {lr:.6f}{mark}")

    print(f"\nBest Validation Accuracy: {best_acc:.4f}")

    # Plot
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    a1.plot(history["train_acc"], label="Train"); a1.plot(history["val_acc"], label="Val")
    a1.set_title("Accuracy"); a1.legend()
    a2.plot(history["train_loss"], label="Train"); a2.plot(history["val_loss"], label="Val")
    a2.set_title("Loss"); a2.legend()
    plt.savefig("results/training_history.png", dpi=150, bbox_inches="tight"); plt.show()

    # Evaluate best model
    model.load_state_dict(torch.load("checkpoints/best_model.pth"))
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, pred = torch.max(model(inputs), 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())

    names = list(LESION_TYPES.values())
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=names))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=names, yticklabels=names, cmap="Blues")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=150, bbox_inches="tight"); plt.show()


if __name__ == "__main__":
    main()
