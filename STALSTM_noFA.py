import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from components.convlstm import ConvLSTM
from components.transformer import MultiHeadAttention
class STALSTM_noFA(nn.Module):
    """去掉特征注意力，保留时间注意力和双向LSTM"""
    def __init__(self, in_channels, hidden_channels, out_channels, batch_first=True, seq_len=7, drop=0.5, device='cuda:0') -> None:
        super(STALSTM_noFA, self).__init__()
        self.drop = drop
        self.device = device
        self.seq_len = seq_len
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.lstm = nn.LSTM(in_channels, hidden_channels, batch_first=batch_first, bidirectional=True)
        self.liner = nn.Sequential(
            nn.Linear(4 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=drop),
            nn.Linear(hidden_channels, out_channels)
        )
        # 只保留时间注意力
        self.attn1 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, seq_len),
            nn.Sigmoid()
        )
        # 特征注意力被移除（直接跳过）
        # self.attn2 不再使用
        self.attn11 = nn.Sequential(
            nn.Linear(2 * hidden_channels, 2 * hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(2 * hidden_channels, 2 * hidden_channels),
            nn.Sigmoid()
        )
        self.attn22 = nn.Sequential(
            nn.Linear(2 * hidden_channels, 2 * hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(2 * hidden_channels, 2 * hidden_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.float()
        # 时间注意力
        temporal_attn = self.attn1(x)
        temporal_out = torch.matmul(temporal_attn, x)
        # 特征注意力被跳过，直接用原始 x 作为特征增强输出
        spatia_out = x   # 无特征注意力
        x = spatia_out * temporal_out   # 注意：这里原本是 spatia_out * temporal_out，但 spatia_out = x，所以相当于 x * temporal_out
        x, _ = self.lstm(x)
        x1, x2 = x[:, 0], x[:, -1]
        attn11 = self.attn11(x1)
        attn22 = self.attn22(x2)
        x1, x2 = x1 * attn11, x2 * attn22
        x = F.dropout(torch.cat([x1, x2], dim=-1), p=self.drop)
        x = self.liner(x)
        return x.squeeze()