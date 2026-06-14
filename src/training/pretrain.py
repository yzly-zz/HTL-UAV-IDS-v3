"""
功能描述：
本脚本执行异构迁移学习 (HTL) 的第一阶段：源域预训练。
它使用海量、通用的泛物联网流量数据集 (ToN-IoT, Parquet格式) 来训练包含源域投影层(Projector_src)和共享主干网络(Backbone)的初始模型。
最终固化并保存一个具有强大通用时序特征提取能力的“基座模型”。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import os
import sys

# 确保能导入自定义模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.data_engine.preprocessor import HeterogeneousDataPreprocessor
from src.data_engine.dataset import IDSStreamDataset
from src.models.hda_1dcnn import HTL_UAV_IDS

def run_pretrain(source_data_path, batch_size=512, epochs=20, lr=1e-3, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"========== [阶段一] 开始源域预训练 | 设备: {device} ==========")

    # 1. 数据管线：加载源域数据 (ToN-IoT)
    preprocessor = HeterogeneousDataPreprocessor()
    X, y, input_dim = preprocessor.process_domain_data(
        file_path=source_data_path,
        domain_name="source",
        is_training=True,
        label_col=None  # ← 增强版会自动找到 'Label'
    )

    # 2. 【核心优化】学术级数据集划分 (80% 训练, 20% 验证)
    print("🛡️ [Data Engine] 正在进行学术级数据集安全划分...")

    # 动态检查极少样本类别，防止 stratify 报错崩溃
    classes, counts = np.unique(y, return_counts=True)
    if any(counts < 2):
        print("⚠️ [警告] 检测到极小样本类别，为保证实验继续，已自动关闭分层抽样 (stratify)。")
        stratify_label = None
    else:
        stratify_label = y

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_label
    )
    print(f"✅ 数据划分完成! 训练集: {X_train.shape[0]} 条, 验证集: {X_val.shape[0]} 条")

    # 3. 封装为 PyTorch Dataset 和 DataLoader
    train_dataset = IDSStreamDataset(X_train, y_train)
    val_dataset = IDSStreamDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 4. 初始化模型、损失函数和优化器
    num_classes = len(np.unique(y))
    model = HTL_UAV_IDS(input_dim=input_dim, shared_dim=64, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 5. 训练与验证循环
    for epoch in range(epochs):
        # --- 训练阶段 ---
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            # 确保数据格式为 Tensor
            inputs = inputs.clone().detach().to(dtype=torch.float32, device=device)
            labels = labels.clone().detach().to(dtype=torch.long, device=device)

            # ❌ 删除或注释掉下面这行，因为模型内部的 forward 会自己做 unsqueeze
            # inputs = inputs.unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # --- 验证阶段 ---
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.clone().detach().to(dtype=torch.float32, device=device)
                labels = labels.clone().detach().to(dtype=torch.long, device=device)

                # ❌ 同样，删除或注释掉验证集里的升维操作
                # inputs = inputs.unsqueeze(1)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(
            f"Epoch [{epoch + 1}/{epochs}] | Train Loss: {running_loss / len(train_loader):.4f} | Val Acc: {100 * correct / total:.2f}%")
    # 6. 固化源域基座模型
    os.makedirs("weights", exist_ok=True)
    base_weight_path = "weights/source_base_model.pth"
    torch.save(model.state_dict(), base_weight_path)
    print(f"✅ [阶段一完成] 源域基座模型已成功保存至: {base_weight_path}")

if __name__ == "__main__":
    # 请确保路径与你的目录结构匹配
    run_pretrain("data/raw/CIC-ToN-IoT-V2.parquet")