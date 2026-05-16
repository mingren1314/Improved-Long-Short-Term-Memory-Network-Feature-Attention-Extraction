import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from components.convlstm import ConvLSTM
from components.transformer import MultiHeadAttention


# ------------------------------------------------------------------------------------------------------------------------------
# simple lstm model with fully-connect layer
class LSTMModel(nn.Module):
    """single task model"""

    def __init__(self, in_channels, hidden_channels, out_channels, batch_first=True, seq_len=7, drop=0.5, device='cuda:0') -> None:
        super(LSTMModel, self).__init__()
        self.drop = drop
        self.device = device
        self.seq_len = seq_len
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.lstm = nn.LSTM(in_channels, hidden_channels, batch_first=batch_first)
        self.liner = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=drop),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x):
        x = x.float()
        x, _ = self.lstm(x)
        x = F.dropout(x[:, -1], p=self.drop)
        # we only predict the last step
        x = self.liner(x)

        return x.squeeze()


class TemporalBlock(nn.Module):
    """TCN 基本残差块：两层一维卷积 + 残差连接"""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, drop=0.5):
        super(TemporalBlock, self).__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.dropout = nn.Dropout(drop)
        self.relu = nn.ReLU()

        # 如果通道数不一致，用 1x1 卷积对残差分支做升/降维
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        # x: (B, C, T)
        out = self.conv1(x)
        # 因为使用 padding=dilation*(k-1)，前面的 padding 部分对应“未来”，TCN 通常裁掉尾部保持因果性
        out = out[:, :, :x.size(2)]
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = out[:, :, :x.size(2)]
        out = self.relu(out)
        out = self.dropout(out)

        res = x if self.downsample is None else self.downsample(x)
        res = res[:, :, :out.size(2)]
        return self.relu(out + res)


class TCNModel(nn.Module):
    """TCN 时间卷积网络模型（按 LSTM 相同数据接口设计）"""

    def __init__(self, in_channels, hidden_channels, out_channels,
                 batch_first=True, seq_len=7, drop=0.5, device='cuda:0') -> None:
        super(TCNModel, self).__init__()
        assert batch_first, "TCNModel 目前假设输入为 (B, T, C)，即 batch_first=True"
        self.drop = drop
        self.device = device
        self.seq_len = seq_len
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        # 多层膨胀卷积：感受野随层数指数增长，适合时间序列
        num_levels = 3
        kernel_size = 3
        layers = []
        in_c = in_channels
        for i in range(num_levels):
            dilation = 2 ** i
            out_c = hidden_channels
            layers.append(TemporalBlock(in_c, out_c, kernel_size, dilation, drop))
            in_c = out_c
        self.tcn = nn.Sequential(*layers)

        self.liner = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=drop),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x):
        # x: (B, T, C) -> (B, C, T) 供 Conv1d 使用
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        # 只使用最后一个时间步的表示
        x = x[:, :, -1]
        x = self.liner(x)
        return x.squeeze()


class BiLSTMModel(nn.Module):
    """双向LSTM模型"""

    def __init__(self, in_channels, hidden_channels, out_channels, batch_first=True, seq_len=7, drop=0.5, device='cuda:0') -> None:
        super(BiLSTMModel, self).__init__()
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

    def forward(self, x):
        x = x.float()
        x, _ = self.lstm(x)
        x = torch.cat([x[:, 0], x[:, -1]], dim=-1)
        x = F.dropout(x, p=self.drop)
        # we only predict the last step
        x = self.liner(x)

        return x.squeeze()


class AttnLSTMModel(nn.Module):
    """single task model"""

    def __init__(self, in_channels, hidden_channels, out_channels, batch_first=True, seq_len=7, drop=0.5, device='cuda:0') -> None:
        super(AttnLSTMModel, self).__init__()
        self.drop = drop
        self.device = device
        self.seq_len = seq_len
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.lstm = nn.LSTM(in_channels, hidden_channels, batch_first=batch_first)
        self.liner = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=drop),
            nn.Linear(hidden_channels, out_channels)
        )
        self.attn = MultiHeadAttention(d_model=hidden_channels, d_k=hidden_channels, d_v=hidden_channels, n_heads=4, seq_len=seq_len, bias=False, drop=drop)

    def forward(self, x):
        x = x.float()
        x, _ = self.lstm(x)
        x, attn = self.attn(x)
        x = F.dropout(x[:, -1], p=self.drop)
        # we only predict the last step
        x = self.liner(x)

        return x.squeeze()


class CNN(nn.Module):
    """single task model"""

    def __init__(self, in_channels, hidden_channels, out_channels, drop=0.5, cfg=None) -> None:
        super(CNN, self).__init__()
        self.drop = drop
        self.latn = ((2 * cfg["spatial_offset"] + 1) - cfg["kernel_size"]) // cfg["stride_cnn"] + 1
        self.lonn = ((2 * cfg["spatial_offset"] + 1) - cfg["kernel_size"]) // cfg["stride_cnn"] + 1
        self.cnn = nn.Conv2d(in_channels=(in_channels - len(cfg['static_list'])) * cfg['seq_len'] + len(cfg['static_list']), out_channels=hidden_channels, kernel_size=cfg["kernel_size"], stride=cfg["stride_cnn"])
        self.liner = nn.Sequential(
            nn.Linear(int(hidden_channels) * int(self.latn) * int(self.lonn), hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(p=drop),
            nn.Linear(hidden_channels // 2, out_channels)
        )

    def forward(self, x):
        x = self.cnn(x.float())
        x = F.dropout(x).flatten(1)
        x = self.liner(x)
        return x


class ConvLSTMModel(nn.Module):
    """single task model"""

    def __init__(self, in_channels, hidden_channels, out_channels, batch_first=True, num_layers=1, seq_len=7, drop=0.5, kernel_size=3, device='cuda:0', cfg=None) -> None:
        super(ConvLSTMModel, self).__init__()
        self.drop = drop
        self.device = device
        kernels = (kernel_size, kernel_size)
        self.convlstm = ConvLSTM(in_channels, hidden_channels, kernels, num_layers, batch_first).to(device)
        self.liner = nn.Sequential(
            nn.Linear(hidden_channels * int(2 * cfg["spatial_offset"] + 1) * int(2 * cfg["spatial_offset"] + 1), hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(p=drop),
            nn.Linear(hidden_channels // 2, out_channels)
        )

    def forward(self, x):
        x = x.float()
        last_state, encoder_state = self.convlstm(x)
        # 最后一层，最后一个时间步
        x = F.dropout(last_state[-1][:, -1], p=self.drop)
        x = x.flatten(1)
        x = self.liner(x)

        return x.squeeze()


class DARNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, batch_first=False, seq_len=7, drop=0.5, device='cuda:0') -> None:
        super(DARNN, self).__init__()
        self.drop = drop
        self.device = device
        self.seq_len = seq_len
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        # 通道注意力
        self.attn1 = nn.Sequential(
            nn.Linear(2 * hidden_channels + seq_len, seq_len),
            nn.Tanh(),
            nn.Linear(seq_len, 1)
        )
        # 时间注意力
        self.attn2 = nn.Sequential(
            nn.Linear(3 * hidden_channels, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, 1)
        )

        # 编码器
        self.encoder = nn.LSTM(in_channels, hidden_channels, batch_first=batch_first)

        # 输出结果
        self.liner = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, out_channels)
        )

    def forward(self, x):
        seq_len, batch_size, in_channels = x.shape[0], x.shape[1], x.shape[2]

        # x shape is ==> (seq_len, batch_size, channels)
        x = inputs = x.float()

        # 初始化隐藏状态
        h, c = torch.zeros(1, batch_size, self.hidden_channels).to(self.device), \
            torch.zeros(1, batch_size, self.hidden_channels).to(self.device)

        # 保存输出状态
        hidden, state = torch.empty(seq_len, batch_size, self.hidden_channels).to(self.device), \
            torch.empty(seq_len, batch_size, self.hidden_channels).to(self.device)

        # 从每个时间步计算【各个特征】的重要性
        for t in range(seq_len):
            x = torch.concat([
                h.repeat(in_channels, 1, 1),
                c.repeat(in_channels, 1, 1),
                inputs.permute(2, 1, 0)
            ], dim=2)

            # 执行attn函数
            weights = F.softmax(self.attn1(x).squeeze(), dim=0)

            # 权重乘以每个输入向量
            x_weighted = torch.mul(weights.permute(1, 0), inputs[t])

            self.encoder.flatten_parameters()
            _, (h, c) = self.encoder(x_weighted.unsqueeze(0), (h, c))

            hidden[t] = h
            state[t] = c
        # 开始处理时间注意力
        x = torch.concat([
            h.repeat(seq_len, 1, 1),
            c.repeat(seq_len, 1, 1),
            state
        ], dim=2)
        weights = F.softmax(self.attn2(x).squeeze(), dim=0)

        weights = weights.permute(1, 0).unsqueeze(1)
        hidden = hidden.permute(1, 0, 2)
        # 权重乘以每个输入向量，得到输出结果
        output = torch.bmm(weights, hidden).squeeze()
        output = self.liner(output)
        return output


class AttnRNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layer=1, n_heads=3, drop=0.5, n_position=30, bias=False) -> None:
        super(AttnRNN, self).__init__()
        self.encoder = nn.ModuleList([
            MultiHeadAttention(hidden_channels, hidden_channels, hidden_channels, n_heads, bias, drop) for _ in range(n_layer)
        ])
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, in_channels))
        self.liner = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=drop),
            nn.Linear(hidden_channels, out_channels)
        )
        self.conv = nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding='same')

    def forward(self, inputs):
        inputs = inputs.float()
        inputs = F.dropout(inputs + self.pos_table[:, :inputs.size(1)].clone().detach()).float()
        inputs = inputs.permute(0, 2, 1)
        inputs = self.conv(inputs).permute(0, 2, 1)

        for layer in self.encoder:
            inputs, enc_slf_attn = layer(inputs)

        # 只要最后一个时间步骤
        inputs = inputs[:, -1]
        inputs = self.liner(inputs)
        return inputs.squeeze()

    def _get_sinusoid_encoding_table(self, n_position, d_model):
        """
        参数:
        - n_position (int): 位置嵌入的最大位置数。
        - d_model (int): 嵌入维度。
        返回:
        - torch.FloatTensor: 大小为(1, n_position, d_model)的位置嵌入张量。
        """
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_model) for hid_j in range(d_model)]

        # 生成位置嵌入表格
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])

        # 对嵌入表格的偶数维度应用正弦函数，奇数维度应用余弦函数
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class HybridRNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, seq_len=7, n_layer=3, n_heads=8, drop=0.5, n_position=30, bias=False) -> None:
        super(HybridRNN, self).__init__()
        self.encoder = nn.ModuleList([
            MultiHeadAttention(in_channels, hidden_channels // 2, hidden_channels // 2, n_heads, seq_len, bias, drop) for _ in range(n_layer)
        ])
        # self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, in_channels))
        self.liner = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=drop),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, inputs, feature_mask=None):
        inputs = inputs.float()
        # inputs = F.dropout(inputs + self.pos_table[:, :inputs.size(1)].clone().detach()).float()

        for layer in self.encoder:
            # TODO 可以考虑添加feature_mask
            inputs, enc_slf_attn = layer(inputs)
            # inputs, enc_slf_attn = layer(inputs, feature_mask)

        # 只要最后一个时间步骤
        inputs = inputs[:, -1]
        inputs = self.liner(inputs)
        return inputs.squeeze()

    def _get_sinusoid_encoding_table(self, n_position, d_model):
        """
        参数:
        - n_position (int): 位置嵌入的最大位置数。
        - d_model (int): 嵌入维度。
        返回:
        - torch.FloatTensor: 大小为(1, n_position, d_model)的位置嵌入张量。
        """
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_model) for hid_j in range(d_model)]

        # 生成位置嵌入表格
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])

        # 对嵌入表格的偶数维度应用正弦函数，奇数维度应用余弦函数
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class CNNTransformer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, drop=0.5, n_heads=6, cfg=None) -> None:
        super(CNNTransformer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        H, W = 2 * cfg["spatial_offset"] + 1, 2 * cfg["spatial_offset"] + 1

        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden_channels // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels // 2, in_channels, bias=False),
            nn.Sigmoid()
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.attn = MultiHeadAttention(d_model=in_channels, d_k=hidden_channels, d_v=hidden_channels, n_heads=n_heads, bias=False, drop=drop)
        self.conv = nn.Conv2d(in_channels * cfg['seq_len'], in_channels, kernel_size=3, padding='same')
        self.liner = nn.Sequential(
            nn.Linear(in_channels * H * W, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(p=drop),
            nn.Linear(hidden_channels // 2, out_channels)
        )

    def forward(self, x):
        x = x.float()
        B, T, C, H, W = x.shape
        y = self.avg_pool(x.view(B * T, C, H, W)).view(B * T, C)
        y = self.fc(y).view(B, T, C)
        y, attn = self.attn(y)
        y = y.view(B, T, C, 1, 1)
        out = x * y.expand_as(x)
        out = self.conv(out.view(B, T * C, H, W)).squeeze()
        out = self.liner(out.flatten(1))
        return out.squeeze()


class CNNLSTMModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, batch_first=True, seq_len=7, drop=0.5, device='cuda:0') -> None:
        super(CNNLSTMModel, self).__init__()
        self.drop = drop
        self.device = device
        self.seq_len = seq_len
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.lstm = nn.LSTM(hidden_channels, hidden_channels, batch_first=batch_first, bidirectional=True)
        self.liner = nn.Sequential(
            nn.Linear(4 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=drop),
            nn.Linear(hidden_channels, out_channels)
        )
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Dropout(p=drop),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding='same'),
        )

    def forward(self, x):
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.conv(x).permute(0, 2, 1)

        x, _ = self.lstm(x)
        x = F.dropout(torch.cat([x[:, 0], x[:, -1]], dim=-1), p=self.drop)
        # we only predict the last step
        x = self.liner(x)

        return x.squeeze()


class STALSTMModel(nn.Module):
    """时空注意力LSTM"""

    def __init__(self, in_channels, hidden_channels, out_channels, batch_first=True, seq_len=7, drop=0.5, device='cuda:0') -> None:
        super(STALSTMModel, self).__init__()
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
        self.attn1 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, seq_len),
            nn.Sigmoid()
        )
        self.attn2 = nn.Sequential(
            nn.Linear(seq_len, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, in_channels),
            nn.Sigmoid()
        )
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
        temporal_attn = self.attn1(x)
        temporal_out = torch.matmul(temporal_attn, x)

        spatia_attn = self.attn2(x.permute(0, 2, 1))
        spatia_out = torch.matmul(spatia_attn, x.permute(0, 2, 1)).permute(0, 2, 1)
        x = spatia_out * temporal_out
        x, _ = self.lstm(x)
        x1, x2 = x[:, 0], x[:, -1]
        attn11 = self.attn11(x1)
        attn22 = self.attn22(x2)
        x1, x2 = x1 * attn11, x2 * attn22
        x = F.dropout(torch.cat([x1, x2], dim=-1), p=self.drop)
        # we only predict the last step
        x = self.liner(x)

        return x.squeeze()


class MLPModel(nn.Module):
    """简单的MLP模型"""

    def __init__(self, in_channels, hidden_channels, out_channels, batch_first=True, seq_len=7, drop=0.5, device='cuda:0') -> None:
        super(MLPModel, self).__init__()
        self.drop = drop
        self.device = device
        self.seq_len = seq_len
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.liner = nn.Sequential(
            nn.Linear(in_channels * seq_len, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=drop),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(p=drop),
            nn.Linear(hidden_channels // 2, out_channels),
        )

    def forward(self, x):
        x = x.float()
        x = self.liner(x.flatten(1))
        return x.squeeze()


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

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    seq_len = 30
    batch_size = 16
    in_channels = 11
    hidden_channels = 32
    out_channels = 1
    x = torch.randn(size=(batch_size, seq_len, in_channels))
    model = STALSTMModel(in_channels, hidden_channels, out_channels, seq_len=seq_len)
    output = model(x)
    print(output.shape)

    # 可视化STALSTM模型的内部结构

    temporal_attn = model.attn1(x)[0].cpu().detach().numpy()
    spatia_attn = model.attn2(x.permute(0, 2, 1))[0].cpu().detach().numpy()
