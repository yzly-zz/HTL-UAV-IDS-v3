"""
阶段二：目标域微调（HTL-UAV-IDS-V3）
支持三个无人机子场景的独立微调：
  1. uav  — 无人机节点（WiFi帧层，~44维，latin-1）
  2. ap   — 接入点端（WiFi帧层+网络层，~47维，utf-8）
  3. gcs  — 地面控制站（网络流统计层，44维，utf-8）

用法：
  python scripts/finetune.py --scenario uav
  python scripts/finetune.py --scenario ap
  python scripts/finetune.py --scenario gcs
  python scripts/finetune.py --scenario all
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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.data_engine.preprocessor import HeterogeneousDataPreprocessor, CSV_ENCODINGS
from src.data_engine.dataset import IDSStreamDataset
from src.models.hda_1dcnn import HTL_UAV_IDS

# ──────────── 场景定义 ────────────
SCENARIOS = {
    "uav": {
        "name": "UAV-Case1（无人机节点 — WiFi帧层）",
        "files": ["data/raw/UAV-NIDD/UAV-Case1-Label.csv"],
        "label_col": None,
    },
    "ap": {
        "name": "Access Point Case2（接入点端 — 混合层）",
        "files": ["data/raw/UAV-NIDD/Access Point Case2 Label.csv"],
        "label_col": None,
    },
    "gcs": {
        "name": "GSC Case3（地面控制站 — 流统计层）",
        "files": ["data/raw/UAV-NIDD/GSC Case3 Label.csv"],
        "label_col": None,
    },
}


def safe_read_csv(file_path):
    for enc in CSV_ENCODINGS:
        try:
            return pd.read_csv(file_path, low_memory=False, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"无法解码: {file_path}")


def run_finetune(
    scenario_key,
    base_model_path="weights/source_base_model.pth",
    epochs=20,
    class_weight_norm=None,
    lr=1e-4,
    device="cuda",
):
    scenario = SCENARIOS[scenario_key]
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  阶段二：目标域微调 — {scenario['name']}")
    print(f"  设备: {device} | Epochs: {epochs} | LR: {lr}")
    print(f"{'='*60}")

    # ──── 1. 加载 ────
    preprocessor = HeterogeneousDataPreprocessor()
    df_list = []
    for f in scenario["files"]:
        if not os.path.exists(f):
            print(f"   ❌ 文件不存在: {f}")
            continue
        print(f"   📂 {f}")
        df_list.append(safe_read_csv(f) if f.endswith(".csv") else pd.read_parquet(f))

    if not df_list:
        raise FileNotFoundError("没有找到任何目标域数据文件！")

    full_df = pd.concat(df_list, axis=0, ignore_index=True)
    print(f"   ✅ 合并完成: {len(full_df):,} 条")

    # ──── 2. 清洗 ────
    X_raw, y, feature_names = preprocessor._clean_dataframe(
        full_df, label_col=scenario["label_col"]
    )
    target_input_dim = X_raw.shape[1]

    # ──── 3. 划分 ────
    classes, counts = np.unique(y, return_counts=True)
    stratify_label = y if all(c >= 2 for c in counts) else None
    if stratify_label is None:
        print("   ⚠️ 少数类样本不足，已关闭分层抽样。")

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=stratify_label
    )

    print(f"   训练集: {len(X_train):,} | 攻击占比: {y_train.mean():.2%}")
    print(f"   测试集: {len(X_test):,}  | 攻击占比: {y_test.mean():.2%}")

    # ──── 4. 标准化 ────
    X_train_scaled = preprocessor.scaler.fit_transform(X_train)
    X_test_scaled  = preprocessor.scaler.transform(X_test)

    suffix = scenario_key
    os.makedirs("weights", exist_ok=True)
    joblib.dump(preprocessor.scaler, f"weights/target_scaler_{suffix}.pkl")
    with open(f"weights/target_feature_names_{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)

    train_loader = DataLoader(IDSStreamDataset(X_train_scaled, y_train), batch_size=128, shuffle=True)
    test_loader  = DataLoader(IDSStreamDataset(X_test_scaled, y_test),   batch_size=128, shuffle=False)

    # 类别权重：优先使用手动指定，其次使用场景默认，最后使用 balanced
    if class_weight_norm is not None:
        # 手动指定，如 "1.5,0.5" → [1.5, 0.5]
        class_weights = np.array([float(x) for x in class_weight_norm.split(",")], dtype=np.float64)
    elif scenario_key == "gcs":
        class_weights = np.array([2.0, 0.6], dtype=np.float32)
    else:
        class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)

    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"   ⚖️  类别权重: 正常={class_weights[0]:.3f}, 攻击={class_weights[1]:.3f}")

    # ──── 6. 模型 ────
    model = HTL_UAV_IDS(input_dim=target_input_dim, shared_dim=64, num_classes=2).to(device)

    if os.path.exists(base_model_path):
        pretrained_dict = torch.load(base_model_path, map_location=device)
        model_dict = model.state_dict()
        filtered_dict = {
            k: v for k, v in pretrained_dict.items()
            if k in model_dict and "projector" not in k
        }
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
        model.freeze_backbone_for_finetuning()
        print(f"   🔗 基座权重已加载: {base_model_path}")
        print(f"   ❄️  Backbone 已冻结")
    else:
        print(f"   ⚠️ 未找到基座权重，将从头训练（Scratch 模式）")

    # ──── 7. 训练 ────
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    best_acc = 0.0
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

        print(
            f"   Epoch [{epoch+1:02d}/{epochs}]  "
            f"Train Loss: {train_loss/len(train_loader):.4f}  "
            f"Train Acc: {100*train_correct/train_total:.2f}%  "
            f"Val Loss: {val_loss/len(test_loader):.4f}  "
            f"Val Acc: {val_acc:.2f}%"
        )

    # ──── 8. 保存 ────
    save_path = f"weights/target_uav_model_{suffix}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n   ✅ 模型已保存: {save_path}")
    print(f"   🏆 最佳验证准确率: {best_acc:.2f}%")

    return {
        "scenario": scenario_key,
        "input_dim": target_input_dim,
        "best_val_acc": best_acc,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "feature_names": feature_names,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HTL-UAV-IDS 目标域微调")
    parser.add_argument("--scenario", type=str, required=True,
                        choices=["uav", "ap", "gcs", "all"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--class_weight", type=str, default=None,
                        help="手动指定类别权重，格式: normal_weight,attack_weight (如 1.5,0.5)")
    args = parser.parse_args()

    if args.scenario == "all":
        results = {}
        for key in ["uav", "ap", "gcs"]:
            results[key] = run_finetune(key, epochs=args.epochs,
                                        lr=args.lr, device=args.device,
                                        class_weight_norm=args.class_weight)
        print("\n" + "=" * 60)
        print("  全部场景微调完成！汇总：")
        for k, v in results.items():
            print(f"    {SCENARIOS[k]['name']}: dim={v['input_dim']}, acc={v['best_val_acc']:.2f}%")
    else:
        run_finetune(args.scenario, epochs=args.epochs, lr=args.lr, device=args.device,
                     class_weight_norm=args.class_weight)