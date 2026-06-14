"""搜索 GCS 最优 class_weight —— 修复版"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

from src.data_engine.preprocessor import HeterogeneousDataPreprocessor, CSV_ENCODINGS
from src.data_engine.dataset import IDSStreamDataset
from src.models.hda_1dcnn import HTL_UAV_IDS

CSV_PATH = "data/raw/UAV-NIDD/GSC Case3 Label.csv"

def safe_read_csv(file_path):
    for enc in CSV_ENCODINGS:
        try:
            return pd.read_csv(file_path, low_memory=False, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"无法解码: {file_path}")

def search(device_str):
    dev = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")

    # 1. 加载数据（与 evaluate_v3.py 完全一致的流程）
    print("Loading GCS data...")
    df = safe_read_csv(CSV_PATH)
    preprocessor = HeterogeneousDataPreprocessor()
    X_raw, y, feature_names = preprocessor._clean_dataframe(df, label_col=None)
    input_dim = X_raw.shape[1]
    print(f"Data: {X_raw.shape}, Dim: {input_dim}, Labels: 0={int((y==0).sum())}, 1={int((y==1).sum())}")

    # 2. 划分（用 30% 做搜索验证，加快速度）
    X_train, X_val, y_train, y_val = train_test_split(
        X_raw, y, test_size=0.3, random_state=42, stratify=y
    )
    n_norm = int((y_train == 0).sum())
    n_atk  = int((y_train == 1).sum())
    print(f"Train: {len(y_train)} ({n_norm}N + {n_atk}A)")

    # 3. 标准化
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)

    # 4. 创建 DataLoader
    train_ds = IDSStreamDataset(X_train_s.astype(np.float32), y_train.astype(np.int64))
    val_ds   = IDSStreamDataset(X_val_s.astype(np.float32),   y_val.astype(np.int64))
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False)

    # 5. 搜索 class_weight
    w_norms = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 10.0, 15.0, 20.0]
    w_atks  = [0.5, 0.6, 0.7, 0.8]
    best_f1 = 0.0
    best_w  = (2.0, 0.6)
    total = sum(1 for wn in w_norms for wa in w_atks if not (wn <= 1.0 and wa >= 0.7))
    cnt = 0

    print(f"Testing {total} weight combinations...")
    for wn in w_norms:
        for wa in w_atks:
            if wn <= 1.0 and wa >= 0.7:
                continue
            cnt += 1

            model = HTL_UAV_IDS(input_dim=input_dim, shared_dim=64, num_classes=2).to(dev)
            cw = torch.tensor([wn, wa], dtype=torch.float32).to(dev)
            crit = nn.CrossEntropyLoss(weight=cw)
            opt = torch.optim.Adam(model.parameters(), lr=1e-4)

            model.train()
            for _ in range(5):
                for bx, by in train_loader:
                    bx = bx.to(dev, dtype=torch.float32)
                    by = by.to(dev, dtype=torch.long)
                    opt.zero_grad()
                    crit(model(bx), by).backward()
                    opt.step()

            model.eval()
            preds, labels = [], []
            with torch.no_grad():
                for bx, by in val_loader:
                    bx = bx.to(dev, dtype=torch.float32)
                    logits = model(bx)
                    preds.append(logits.argmax(dim=1).cpu().numpy())
                    labels.append(by.numpy())
            yp = np.concatenate(preds)
            yt = np.concatenate(labels)
            f1m = f1_score(yt, yp, average="macro")
            f1n = f1_score(yt, yp, pos_label=0)
            f1a = f1_score(yt, yp, pos_label=1)

            tag = ""
            if f1m > best_f1:
                best_f1 = f1m
                best_w = (wn, wa)
                tag = " ★"
            print(f"  [{cnt}/{total}] wn={wn:.2f} wa={wa:.2f} -> F1={f1m:.4f} (N:{f1n:.4f} A:{f1a:.4f}){tag}")

    print("=" * 60)
    print(f"  Best: Normal={best_w[0]:.3f}  Attack={best_w[1]:.3f}  F1-Macro={best_f1:.4f}")
    print(f"  To use, edit finetune.py or pass:")
    print(f"    --class_weight {best_w[0]:.3f},{best_w[1]:.3f}")
    print("=" * 60)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    search(args.device)
