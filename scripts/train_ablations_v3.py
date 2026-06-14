"""
V3 二分类消融实验脚本
在 UAV / AP / GCS 三个场景上分别训练四个消融变体：
  HTL-noProj   → 移除 Domain Projector，使用零填充对齐维度
  HTL-noFreeze → 不冻结 Backbone，全参数微调
  HTL-stdConv  → 深度可分离卷积替换为标准卷积
  HTL-Scratch  → 不加载源域预训练权重，随机初始化

用法：
  python scripts/train_ablations_v3.py --scenario gcs
  python scripts/train_ablations_v3.py --scenario all
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np
import os, sys, argparse, json, joblib, copy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_engine.preprocessor import HeterogeneousDataPreprocessor, CSV_ENCODINGS
from src.data_engine.dataset import IDSStreamDataset
from src.models.hda_1dcnn import HTL_UAV_IDS, DomainProjector, DepthwiseSeparableConv1d

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

# ── 消融变体模型定义 ──

class HTL_noProj(nn.Module):
    """消融变体：移除 Domain Projector，输入直接通过零填充对齐到 shared_dim"""
    def __init__(self, input_dim, shared_dim=64, num_classes=2):
        super().__init__()
        self.shared_dim = shared_dim
        self.pad_dim = max(0, shared_dim - input_dim)
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1, bias=False), nn.BatchNorm1d(16), nn.ReLU(),
            nn.MaxPool1d(2),
            DepthwiseSeparableConv1d(16, 32, 3, 1), nn.MaxPool1d(2),
            DepthwiseSeparableConv1d(32, 64, 3, 1), nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, num_classes)
        )

    def forward(self, x):
        if x.shape[1] < self.shared_dim:
            x = torch.cat([x, torch.zeros(x.shape[0], self.shared_dim - x.shape[1], device=x.device)], dim=1)
        elif x.shape[1] > self.shared_dim:
            x = x[:, :self.shared_dim]
        x = x.unsqueeze(1)
        return self.classifier(self.backbone(x).squeeze(-1))


class HTL_stdConv(nn.Module):
    """消融变体：深度可分离卷积替换为标准卷积"""
    def __init__(self, input_dim, shared_dim=64, num_classes=2):
        super().__init__()
        self.projector = DomainProjector(input_dim, shared_dim)
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1, bias=False), nn.BatchNorm1d(16), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 3, padding=1, bias=False), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1, bias=False), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.projector(x).unsqueeze(1)
        return self.classifier(self.backbone(x).squeeze(-1))


ABLATION_VARIANTS = {
    "HTL-noProj": {
        "cls": HTL_noProj,
        "name": "HTL-noProj（移除Domain Projector）",
        "use_pretrained": True,
        "freeze_backbone": True,
    },
    "HTL-noFreeze": {
        "cls": HTL_UAV_IDS,
        "name": "HTL-noFreeze（不冻结Backbone）",
        "use_pretrained": True,
        "freeze_backbone": False,
    },
    "HTL-stdConv": {
        "cls": HTL_stdConv,
        "name": "HTL-stdConv（标准卷积）",
        "use_pretrained": True,
        "freeze_backbone": True,
    },
    "HTL-Scratch": {
        "cls": HTL_UAV_IDS,
        "name": "HTL-Scratch（无预训练）",
        "use_pretrained": False,
        "freeze_backbone": True,
    },
}


def safe_read_csv(file_path):
    for enc in CSV_ENCODINGS:
        try: return pd.read_csv(file_path, low_memory=False, encoding=enc)
        except (UnicodeDecodeError, UnicodeError): continue
    raise ValueError(f"无法解码: {file_path}")


def train_one_ablation(model, train_loader, test_loader, epochs, lr, device, class_weights_tensor, variant_name):
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_acc = 0.0
    best_state = None

    print(f"\n  {'─'*45}")
    print(f"    训练 {variant_name}")
    print(f"  {'─'*45}")

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, dtype=torch.float32), labels.to(device, dtype=torch.long)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, preds = torch.max(model(inputs), 1)
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

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"     [{variant_name}] Epoch {epoch+1:02d}/{epochs}  "
                  f"Train Loss: {train_loss/len(train_loader):.4f}  Val Acc: {val_acc:.2f}%")

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_acc


def run_ablations_for_scenario(scenario_key, epochs=20, lr=1e-4, device="cuda"):
    scenario = SCENARIOS[scenario_key]
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    suffix = scenario_key
    base_model_path = "weights/source_base_model.pth"

    print(f"\n{'='*60}")
    print(f"  消融实验：{scenario['name']}")
    print(f"  设备: {device}  |  Epochs: {epochs}  |  LR: {lr}")
    print(f"{'='*60}")

    # ── 加载数据 ──
    preprocessor = HeterogeneousDataPreprocessor()
    df_list = []
    for f in scenario["files"]:
        if not os.path.exists(f): continue
        df_list.append(safe_read_csv(f) if f.endswith(".csv") else pd.read_parquet(f))
    full_df = pd.concat(df_list, axis=0, ignore_index=True)
    X_raw, y, feature_names = preprocessor._clean_dataframe(full_df, label_col=scenario["label_col"])
    input_dim = X_raw.shape[1]

    classes, counts = np.unique(y, return_counts=True)
    stratify_label = y if all(c >= 2 for c in counts) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=stratify_label
    )

    preprocessor.scaler.fit(X_train)
    X_train_scaled = preprocessor.scaler.transform(X_train)
    X_test_scaled = preprocessor.scaler.transform(X_test)

    train_loader = DataLoader(IDSStreamDataset(X_train_scaled, y_train), batch_size=128, shuffle=True)
    test_loader  = DataLoader(IDSStreamDataset(X_test_scaled, y_test),   batch_size=128, shuffle=False)

    print(f"   训练集: {len(X_train):,} | 攻击占比: {y_train.mean():.2%}")
    print(f"   测试集: {len(X_test):,}  | 攻击占比: {y_test.mean():.2%}")

    # 类别权重：GCS 使用 balanced，其余场景也使用 balanced
    class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"   类别权重: 正常={class_weights[0]:.3f}, 攻击={class_weights[1]:.3f}")

    results = {}

    for variant_key, variant_info in ABLATION_VARIANTS.items():
        ModelCls = variant_info["cls"]
        model = ModelCls(input_dim=input_dim, shared_dim=64, num_classes=2).to(device)
        n_params = sum(p.numel() for p in model.parameters())

        # 加载预训练权重（如果变体要求）
        if variant_info["use_pretrained"] and os.path.exists(base_model_path):
            pretrained_dict = torch.load(base_model_path, map_location=device)
            model_dict = model.state_dict()
            filtered_dict = {
                k: v for k, v in pretrained_dict.items()
                if k in model_dict and "projector" not in k and "classifier" not in k
            }
            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict)
            print(f"   [预训练] 已加载 backbone 参数（{len(filtered_dict)} 个）")

        # 冻结策略
        if variant_info["freeze_backbone"] and hasattr(model, "freeze_backbone_for_finetuning"):
            model.freeze_backbone_for_finetuning()
            print(f"   [冻结] Backbone 已冻结")
        elif variant_info["freeze_backbone"] and not hasattr(model, "freeze_backbone_for_finetuning"):
            # 对 HTL_noProj / HTL_stdConv 手动冻结 backbone
            if hasattr(model, "backbone"):
                for p in model.backbone.parameters():
                    p.requires_grad = False
                print(f"   [冻结] Backbone 已冻结")

        best_acc = train_one_ablation(model, train_loader, test_loader, epochs=epochs,
                                      lr=lr, device=device, class_weights_tensor=class_weights_tensor,
                                      variant_name=variant_info["name"])
        results[variant_key] = {"best_val_acc": best_acc, "params": n_params}
        torch.save(model.state_dict(), f"weights/ablation_{variant_key}_{suffix}.pth")
        print(f"   ✅ {variant_key} 完成，最佳 Val Acc: {best_acc:.2f}%，参数: {n_params:,}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V3 消融实验")
    parser.add_argument("--scenario", type=str, default="gcs",
                        choices=["uav", "ap", "gcs", "all"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    all_results = {}

    if args.scenario == "all":
        for key in ["uav", "ap", "gcs"]:
            all_results[key] = run_ablations_for_scenario(key, epochs=args.epochs,
                                                          lr=args.lr, device=args.device)
    else:
        all_results[args.scenario] = run_ablations_for_scenario(args.scenario, epochs=args.epochs,
                                                                lr=args.lr, device=args.device)

    # 汇总
    print(f"\n{'='*70}")
    print(f"  消融实验汇总")
    print(f"{'='*70}")
    for key, results in all_results.items():
        print(f"\n  {SCENARIOS[key]['name']}")
        for variant_key in ["HTL-noProj", "HTL-noFreeze", "HTL-stdConv", "HTL-Scratch"]:
            if variant_key in results:
                r = results[variant_key]
                print(f"    {variant_key:<18s}  Val Acc: {r['best_val_acc']:.2f}%  Params: {r['params']:,}")

    print(f"\n  ✅ 消融实验全部完成。")