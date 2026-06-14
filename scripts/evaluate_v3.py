"""
V3 三场景综合评估脚本
对 UAV / AP / GCS 三个场景的微调模型进行完整的二分类评估。
输出：classification_report / confusion_matrix / AUC-ROC / PR-AUC / 汇总对比表。
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import sys
import json
import joblib
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_engine.preprocessor import HeterogeneousDataPreprocessor, CSV_ENCODINGS
from src.data_engine.dataset import IDSStreamDataset
from src.models.hda_1dcnn import HTL_UAV_IDS

# ── 场景定义（与 finetune.py 保持一致） ──
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
    """多编码探测读取 CSV"""
    for enc in CSV_ENCODINGS:
        try:
            return pd.read_csv(file_path, low_memory=False, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"无法解码: {file_path}")


def evaluate_scenario(scenario_key, device="cuda"):
    scenario = SCENARIOS[scenario_key]
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    suffix = scenario_key

    print(f"\n{'='*60}")
    print(f"  评估: {scenario['name']}")
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
    print(f"   数据量: {len(full_df):,} 条")

    X_raw, y, feature_names = preprocessor._clean_dataframe(
        full_df, label_col=scenario["label_col"]
    )
    input_dim = X_raw.shape[1]
    print(f"   特征维度: {input_dim}")

    # ── 2. 划分（与训练时相同的 random_state） ──
    classes, counts = np.unique(y, return_counts=True)
    stratify_label = y if all(c >= 2 for c in counts) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=stratify_label
    )
    print(f"   测试集: {len(X_test):,} 条 | 攻击占比: {y_test.mean():.2%}")

    # ── 3. 加载 scaler 并 transform ──
    scaler_path = f"weights/target_scaler_{suffix}.pkl"
    if not os.path.exists(scaler_path):
        print(f"   ❌ Scaler 不存在: {scaler_path}")
        return None
    scaler = joblib.load(scaler_path)
    X_test_scaled = scaler.transform(X_test)

    # ── 4. 加载模型 ──
    model_path = f"weights/target_uav_model_{suffix}.pth"
    if not os.path.exists(model_path):
        print(f"   ❌ 模型不存在: {model_path}")
        return None
    model = HTL_UAV_IDS(input_dim=input_dim, shared_dim=64, num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # ── 5. 推理 ──
    test_dataset = IDSStreamDataset(X_test_scaled, y_test)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            logits = model(inputs)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    attack_probs = all_probs[:, 1]  # 攻击类概率

    # ── 6. 指标计算 ──
    acc = (all_preds == all_labels).mean()
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n   📊 分类报告:")
    print(classification_report(all_labels, all_preds,
                                target_names=["Normal", "Attack"], digits=4))

    print(f"   混淆矩阵:")
    print(f"               预测Normal  预测Attack")
    print(f"   实际Normal    {cm[0,0]:>8d}    {cm[0,1]:>8d}")
    print(f"   实际Attack    {cm[1,0]:>8d}    {cm[1,1]:>8d}")

    # AUC
    try:
        auc = roc_auc_score(all_labels, attack_probs)
        pr_auc = average_precision_score(all_labels, attack_probs)
        print(f"\n   AUC-ROC: {auc:.4f}  |  PR-AUC: {pr_auc:.4f}")
    except ValueError:
        auc = float("nan")
        pr_auc = float("nan")
        print(f"\n   ⚠️ AUC 无法计算（可能只有一个类别）")

    # ── 7. 置信度分析 ──
    correct_mask = (all_preds == all_labels)
    wrong_mask = ~correct_mask
    print(f"\n   置信度分析:")
    print(f"   正确预测的平均置信度: {attack_probs[correct_mask].mean():.4f}"
          if correct_mask.sum() > 0 else "   N/A")
    print(f"   错误预测的平均置信度: {attack_probs[wrong_mask].mean():.4f}"
          if wrong_mask.sum() > 0 else "   N/A")

    return {
        "scenario": scenario_key,
        "name": scenario["name"],
        "accuracy": acc,
        "auc_roc": auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
        "test_samples": len(y_test),
        "attack_rate": y_test.mean(),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    results = {}
    for key in ["uav", "ap", "gcs"]:
        r = evaluate_scenario(key, device=args.device)
        if r:
            results[key] = r

    # ── 汇总 ──
    print(f"\n{'='*70}")
    print(f"  三场景综合评估汇总")
    print(f"{'='*70}")
    header = f"{'场景':<30s} {'Accuracy':>8s} {'AUC-ROC':>8s} {'PR-AUC':>8s} {'样本数':>8s}"
    print(header)
    print("-" * 70)
    for key in ["uav", "ap", "gcs"]:
        if key not in results:
            continue
        r = results[key]
        print(f"{r['name']:<30s} {r['accuracy']:>7.2%} "
              f"{r['auc_roc']:>8.4f} {r['pr_auc']:>8.4f} {r['test_samples']:>8,d}")

    print(f"\n   ✅ 评估完成。")