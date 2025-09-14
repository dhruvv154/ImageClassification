#!/usr/bin/env python3
import argparse, os, math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models

def build_transforms():
    tfm_train = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.05,0.05), scale=(0.95,1.05), shear=5),
        transforms.ToTensor(),
        # grayscale-friendly norm
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
        transforms.RandomErasing(p=0.5, scale=(0.02,0.12), ratio=(0.3,3.3)),
    ])
    tfm_eval = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
    ])
    return tfm_train, tfm_eval

def make_loaders(patch_root, batch_size=32, num_workers=0):
    tfm_train, tfm_eval = build_transforms()
    train_ds = datasets.ImageFolder(Path(patch_root)/"train", transform=tfm_train)
    val_ds   = datasets.ImageFolder(Path(patch_root)/"validation", transform=tfm_eval)

    # Balanced sampling
    targets = [t for _, t in train_ds.samples]
    class_counts = np.bincount(targets, minlength=len(train_ds.classes))
    inv_freq = 1.0 / np.maximum(class_counts, 1)
    sample_weights = [inv_freq[t] for t in targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                          num_workers=num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)

    print("Class counts (train):", dict(zip(train_ds.classes, class_counts.tolist())))
    return train_dl, val_dl, train_ds.classes, inv_freq

def save_best(model, classes, path):
    torch.save({"state_dict": model.state_dict(), "classes": classes}, path)
    print(f"✅ Saved best to {path}")

def epoch_run(model, loader, device, criterion=None, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, total_correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        if train_mode:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train_mode):
            logits = model(xb)
            loss = criterion(logits, yb) if train_mode else nn.functional.cross_entropy(
                logits, yb, reduction="mean")
            if train_mode:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * xb.size(0)
        total_correct += (logits.argmax(1) == yb).sum().item()
        total += xb.size(0)
    return total_loss/total, total_correct/total

def print_epoch(ep, phase, loss, acc, lr):
    lr_s = f" | lr {lr:.2e}" if lr is not None else ""
    print(f"Epoch {ep:02d} [{phase}] loss={loss:.4f} acc={acc:.3f}{lr_s}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch_root", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3, help="head-only warmup epochs")
    ap.add_argument("--patience", type=int, default=3, help="early stop patience")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=0, help="0 is safest on Windows")
    ap.add_argument("--out_dir", default="runs_quads_bal/cls")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_dl, val_dl, classes, inv_freq = make_loaders(
        args.patch_root, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # Model
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Freeze all
    for p in model.parameters(): p.requires_grad = False
    # Head for warmup
    num_f = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_f, len(classes)))
    for p in model.fc.parameters(): p.requires_grad = True
    model.to(device)

    # Loss with class weights (normalized to mean=1)
    cw = torch.tensor(inv_freq / inv_freq.mean(), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)

    # Warmup: head only
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=3e-4, weight_decay=1e-4)
    best_acc, best_path, stale = 0.0, os.path.join(args.out_dir, "best_patch_cls.pth"), 0

    # ---------- Warmup loop ----------
    for ep in range(1, args.warmup+1):
        tr_loss, tr_acc = epoch_run(model, train_dl, device, criterion, opt)
        va_loss, va_acc = epoch_run(model, val_dl, device)
        lr_now = opt.param_groups[0]["lr"]
        print_epoch(ep, "warmup-tr", tr_loss, tr_acc, lr_now)
        print_epoch(ep, "warmup-va", va_loss, va_acc, None)

        if va_acc > best_acc + 1e-4:
            best_acc = va_acc; stale = 0; save_best(model, classes, best_path)
        else:
            stale += 1
            print(f"No improvement for {stale}/{args.patience} epoch(s).")
            if stale >= args.patience:
                print("⏹ Early stopping during warmup.")
                return

    # Unfreeze layer4, lower LR
    for p in model.layer4.parameters(): p.requires_grad = True
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    stale = 0

    # ---------- Main training ----------
    for ep in range(args.warmup+1, args.epochs+1):
        tr_loss, tr_acc = epoch_run(model, train_dl, device, criterion, opt)
        va_loss, va_acc = epoch_run(model, val_dl, device)
        lr_now = opt.param_groups[0]["lr"]
        print_epoch(ep, "train", tr_loss, tr_acc, lr_now)
        print_epoch(ep, "valid", va_loss, va_acc, None)

        sched.step(va_acc)

        if va_acc > best_acc + 1e-4:
            best_acc = va_acc; stale = 0; save_best(model, classes, best_path)
        else:
            stale += 1
            print(f"No improvement for {stale}/{args.patience} epoch(s).")
            if stale >= args.patience:
                print("⏹ Early stopping: validation accuracy plateaued.")
                break

if __name__ == "__main__":
    main()