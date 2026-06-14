import torch
import torch.nn as nn

"""
    在写论文的实验部分时，必须回答审稿人一个问题：
        “为什么非要用你发明的 HTL-1DCNN，普通的网络不行吗？” 
    这两个模型就是用来做“靶子”的。
"""

class VanillaMLP(nn.Module):
    """
    基线模型 1：纯全连接网络 (MLP)
    用于证明 1D-CNN 在捕捉网络流量时序局部特征上的优越性。
    """
    def __init__(self, input_dim, num_classes=2):
        super(VanillaMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class Standard1DCNN(nn.Module):
    """
    基线模型 2：标准的一维卷积网络 (无深度可分离，无异构投影)
    用于证明你的 HTL 架构以及深度可分离卷积在参数量和自适应能力上的优势。
    """
    def __init__(self, input_dim, num_classes=2):
        super(Standard1DCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        # 强制升维 (Batch, 1, Sequence)
        x = x.unsqueeze(1)
        features = self.net(x).squeeze(-1)
        return self.classifier(features)