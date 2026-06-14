"""
V3 二分类深度学习基线训练脚本
对 UAV / AP / GCS 三个场景分别训练以下基线模型：
  - MLP (VanillaMLP)
  - Standard 1D-CNN
  - LSTM
  - MobileNet1D
  - LightTransformer

用法：
  python scripts/train_dl_baselines.py --scenario uav
  python scripts/train_dl_baselines.py --scenario all
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np
import os
import sys
import argparse
import json
import joblib
import copy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_engine.preprocessor import HeterogeneousDataPreprocessor, CSV_ENCODINGS
from src.data_engine.dataset import IDSStreamDataset
from src.models.hda_1dcnn import HTL_UAV_IDS, DepthwiseSeparableConv1d

# ── 场景定义 ──
SCENARIOS = {
    "uav": {
        "name": "UAV-Case1（无人机节点）",
        "files": ["data/raw/UAV-NIDD/UAV-Case1-Label.csv"],
        "label_col": None,
    },
    "ap": {
        "name": "Access Point Case2（接入点端）",
        "files": ["data/raw/UAV-NIDD/Access Point Case2 Label.csv"],
        "label_col": None,
    },
    "gcs": {
        "name": "GSC Case3（地面控制站）",
        "files": ["data/raw/UAV-NIDD/GSC Case3 Label.csv"],
        "label_col": None,
    },
}


# ── 基线模型定义 ──

class VanillaMLP(nn.Module):
    """纯全连接基线"""
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class Standard1DCNN(nn.Module):
    """标准一维卷积基线（无深度可分离卷积）"""
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.classifier(self.net(x).squeeze(-1))


class LSTMBaseline(nn.Module):
    """单向 LSTM 基线"""
    def __init__(self, input_dim, num_classes=2, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim,
                            num_layers=2, batch_first=True, dropout=0.2)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.unsqueeze(-1)  # (B, D) → (B, D, 1)
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])


class MobileNet1D(nn.Module):
    """轻量级深度可分离卷积基线（无 Domain Projector）"""
    def __init__(self, input_dim, num_classes=2, shared_dim=64):
        super().__init__()
        self.proj = nn.Linear(input_dim, shared_dim)
        self.backbone = nn.Sequential(
            DepthwiseSeparableConv1d(1, 16, 3, 1),
            nn.MaxPool1d(2),
            DepthwiseSeparableConv1d(16, 32, 3, 1),
            nn.MaxPool1d(2),
            DepthwiseSeparableConv1d(32, 64, 3, 1),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, num_classes))

    def forward(self, x):
        x = self.proj(x).unsqueeze(1)
        return self.classifier(self.backbone(x).squeeze(-1))


class LightTransformer(nn.Module):
    """轻量 Transformer 编码器"""
    def __init__(self, input_dim, num_classes=2, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=128, dropout=0.2,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.proj(x).unsqueeze(1)  # (B, D) → (B, 1, d_model)
        x = self.encoder(x)
        return self.classifier(x[:, 0, :])


# ── 模型注册 ──
BASELINE_MODELS = {
    "MLP": VanillaMLP,
    "Standard1DCNN": Standard1DCNN,
    "LSTM": LSTMBaseline,
    "MobileNet1D": MobileNet1D,
    "LightTransformer": LightTransformer,
}


def safe_read_csv(file_path):
    for enc in CSV_ENCODINGS:
        try:
            return pd.read_csv(file_path, low_memory=False, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"无法解码: {file_path}")


def train_one_model(model, train_loader, test_loader, epochs, lr, device, class_weights_tensor, model_name):
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_acc = 0.0
    best_state = None

    print(f"\n{'─'*50}")
    print(f"  训练 {model_name}")
    print(f"{'─'*50}")

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, dtype=torch.float32), labels.to(device, dtype=torch.long)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (preds == labels).sum().item()

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device, dtype=torch.float32), labels.to(device, dtype=torch.long)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                _, preds = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        print(f"   [{model_name}] Epoch {epoch+1:02d}/{epochs}  "
              f"Train Loss: {train_loss/len(train_loader):.4f}  "
              f"Train Acc: {100*train_correct/train_total:.2f}%  "
              f"Val Acc: {val_acc:.2f}%")

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_acc


def run_baselines_for_scenario(scenario_key, epochs=20, lr=1e-4, device="cuda"):
    scenario = SCENARIOS[scenario_key]
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    suffix = scenario_key

    print(f"\n{'='*60}")
    print(f"  场景: {scenario['name']}  —  深度学习基线训练")
    print(f"  设备: {device}  |  Epochs: {epochs}  |  LR: {lr}")
    print(f"{'='*60}")

    # ── 1. 加载数据 ──
    preprocessor = HeterogeneousDataPreprocessor()
    df_list = []
    for f in scenario["files"]:
        if not os.path.exists(f):
            print(f"   ❌ 文件不存在: {f}")
            continue
        df_list.append(safe_read_csv(f) if f.endswith(".csv") else pd.read_parquet(f))
    full_df = pd.concat(df_list, axis=0, ignore_index=True)
    print(f"   数据量: {len(full_df):,}")

    X_raw, y, feature_names = preprocessor._clean_dataframe(
        full_df, label_col=scenario["label_col"]
    )
    input_dim = X_raw.shape[1]
    print(f"   特征维度: {input_dim}")

    # ── 2. 划分 ──
    classes, counts = np.unique(y, return_counts=True)
    stratify_label = y if all(c >= 2 for c in counts) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=stratify_label
    )

    # ── 3. 标准化 ──
    preprocessor.scaler.fit(X_train)
    X_train_scaled = preprocessor.scaler.transform(X_train)
    X_test_scaled = preprocessor.scaler.transform(X_test)

    # 保存基线用的 scaler（与 HTL 模型独立）
    os.makedirs("weights", exist_ok=True)
    joblib.dump(preprocessor.scaler, f"weights/baseline_scaler_{suffix}.pkl")
    with open(f"weights/baseline_feature_names_{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)

    train_loader = DataLoader(IDSStreamDataset(X_train_scaled, y_train), batch_size=128, shuffle=True)
    test_loader  = DataLoader(IDSStreamDataset(X_test_scaled, y_test),   batch_size=128, shuffle=False)

    print(f"   训练集: {len(X_train):,} | 攻击占比: {y_train.mean():.2%}")
    print(f"   测试集: {len(X_test):,}  | 攻击占比: {y_test.mean():.2%}")

    # ── 4. 类别权重 ──
    if scenario_key == "gcs":
        class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)  # 修复：使用 balanced
    else:
        class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"   类别权重: 正常={class_weights[0]:.3f}, 攻击={class_weights[1]:.3f}")

    # ── 5. 逐个训练基线 ──
    results = {}
    for name, ModelCls in BASELINE_MODELS.items():
        model = ModelCls(input_dim=input_dim, num_classes=2).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        best_acc = train_one_model(model, train_loader, test_loader,
                                   epochs=epochs, lr=lr, device=device,
                                   class_weights_tensor=class_weights_tensor,
                                   model_name=name)
        results[name] = {"best_val_acc": best_acc, "params": n_params}
        torch.save(model.state_dict(), f"weights/baseline_{name}_{suffix}.pth")
        print(f"   ✅ {name} 完成，最佳 Val Acc: {best_acc:.2f}%，参数: {n_params:,}")

    return results, input_dim


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V3 二分类深度学习基线训练")
    parser.add_argument("--scenario", type=str, default="all",
                        choices=["uav", "ap", "gcs", "all"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    all_results = {}

    if args.scenario == "all":
        for key in ["uav", "ap", "gcs"]:
            res, dim = run_baselines_for_scenario(key, epochs=args.epochs,
                                                  lr=args.lr, device=args.device)
            all_results[key] = {"results": res, "input_dim": dim}
    else:
        res, dim = run_baselines_for_scenario(args.scenario, epochs=args.epochs,
                                              lr=args.lr, device=args.device)
        all_results[args.scenario] = {"results": res, "input_dim": dim}

    # ── 汇总 ──
    print(f"\n{'='*70}")
    print(f"  深度学习基线训练汇总")
    print(f"{'='*70}")
    for key, data in all_results.items():
        print(f"\n  {SCENARIOS[key]['name']} (dim={data['input_dim']})")
        for name, r in data["results"].items():
            print(f"    {name:<20s}  Val Acc: {r['best_val_acc']:.2f}%  Params: {r['params']:,}")

    print(f"\n  ✅ 全部基线训练完成。")