Drought-Agriculture: 干旱预测模型

本项目实现了一个综合性的干旱预测系统，专门用于处理ERA5-Land气象数据集。该系统结合了多种深度学习架构，能够有效捕获气象数据中的时空依赖关系，为农业干旱监测提供精准预测。

## 项目特色

✅ **多模型支持**：集成12种先进模型，包括LSTM、BiLSTM、TCN、MLP、CNN、ConvLSTM、CNNTransformer、DARNN、AttnRNN、AttnLSTM、STALSTM、STGNN
✅ **时空建模**：支持纯时间序列、纯空间卷积、时空注意力、时空图神经网络等多种建模方式
✅ **完整数据流水线**：从原始NetCDF数据加载、预处理、归一化到模型训练和评估
✅ **全面评估指标**：R²、KGE、NSE、RMSE、ubRMSE、Bias、PCC、RV、FHV、FLV等10+种评估指标
✅ **丰富可视化**：箱线图、空间分布图、时间序列图、性能对比图等

## 模型架构

### 1. 时序模型 (Time Series Models)
- **LSTM/GRU**: 基础循环神经网络
- **BiLSTM**: 双向LSTM，捕捉前后时间依赖
- **TCN**: 时间卷积网络，感受野随层数指数增长
- **DARNN**: 动态注意力RNN，双层注意力机制
- **AttnRNN/AttnLSTM**: 注意力增强的RNN/LSTM

### 2. 空间模型 (Spatial Models)
- **CNN**: 卷积神经网络，提取局部空间特征
- **ConvLSTM**: 卷积LSTM，同时处理时空特征
- **CNNTransformer**: CNN与Transformer结合

### 3. 时空模型 (Spatio-Temporal Models)
- **STALSTM**: 时空注意力LSTM（FAELSTM）
- **STGNN**: 时空图神经网络，动态图构建 + 图卷积 + 时间注意力

## 数据处理流程

1. **数据加载**: 从ERA5-Land NetCDF文件加载气象强迫数据（温度、降水、风速等）和地表过程数据（辐射、蒸发等）
2. **数据预处理**: 
   - 空间裁剪（使用Shapefile掩膜）
   - 时间对齐和重采样
   - 缺失值插补
3. **特征工程**: 
   - 动态图构建（基于空间邻近性和特征相似性）
   - 位置编码（正弦余弦编码）
   - 静态特征融合（土壤持水量等）
4. **数据标准化**: 支持全局标准化和区域标准化两种模式
5. **数据分割**: 滑动窗口生成训练/验证/测试样本

## 使用方法

### 环境要求

```bash
Python 3.8+
PyTorch 1.10+
PyTorch Geometric 2.0+
Xarray 0.20+
GeoPandas 0.12+
Basemap 1.3+
```

### 安装依赖

```bash
pip install -r requirements.txt
```

### 快速开始

1. **准备数据**: 将ERA5-Land数据放在`Datasets/LandBench/`目录下
2. **配置参数**: 修改`config.py`中的路径和超参数
3. **运行训练**: 
   ```bash
   python main.py --modelname STALSTM --spatial_resolution 1 --seq_len 7
   ```
4. **运行评估**: 
   ```bash
   python eval.py
   ```
5. **生成可视化**: 
   ```bash
   python plot.py
   ```

### 主要配置参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--modelname` | `STALSTM` | 模型类型: LSTM, BiLSTM, TCN, MLP, CNN, ConvLSTM, CNNTransformer, DARNN, AttnRNN, AttnLSTM, STALSTM, STGNN |
| `--spatial_resolution` | `1.0` | 空间分辨率（度） |
| `--seq_len` | `7` | 输入序列长度 |
| `--forecast_time` | `0` | 预测提前期（0=1天，6=7天） |
| `--normalize_type` | `region` | 归一化方式: `global` 或 `region` |
| `--valid_split` | `True` | 是否划分验证集 |

## 评估指标

- **R² (决定系数)**: 衡量模型解释方差的比例
- **KGE (Kling-Gupta效率)**: 综合评估相关性、偏差和变异性的指标
- **NSE (Nash-Sutcliffe效率)**: 水文模型常用评估指标
- **RMSE/ubRMSE**: 均方根误差及其无偏版本
- **Bias**: 平均偏差
- **PCC (皮尔逊相关系数)**: 相关性度量
- **RV (相对变异性)**: 变异性比较
- **FHV/FLV (高峰/低峰流量偏差)**: 极端事件评估

## 可视化功能

- **性能对比**: 箱线图展示各模型在不同指标上的表现
- **空间分布**: 地图可视化预测结果、观测值及误差分布
- **时间序列**: 多站点时间序列对比图
- **模型诊断**: R²、KGE、RMSE等空间分布图

## 项目文件结构

```
├── main.py              # 主程序入口
├── train.py             # 训练脚本
├── eval.py              # 评估脚本
├── plot.py              # 可视化脚本
├── postprocess.py       # 后处理脚本（计算各项指标）
├── config.py            # 配置文件
├── data.py              # 数据处理模块
├── model.py             # 模型定义模块
├── components/          # 模型组件（ConvLSTM, Transformer）
├── utils.py             # 工具函数
├── requirements.txt     # 依赖包列表
└── README.md            # 项目说明文档
```

## 注意事项
代码中有正在开发功能，后续会持续更新，但不会影响目前使用，自主分辨

## 运行示例

```bash
# 训练STALSTM模型（1天预测）
python main.py --modelname STALSTM --forecast_time 0

# 训练STGNN模型（7天预测）
python main.py --modelname STGNN --forecast_time 6

# 生成性能对比表格
python get_table2.py

# 生成可视化图表
python get_three_plots.py
```

## 引用文献

1. Wu, Z., et al. (2020). A comprehensive survey on graph neural networks. IEEE TNNLS.
2. Guo, S., et al. (2019). Attention based spatial-temporal graph convolutional networks for traffic flow forecasting. AAAI.
3. Vaswani, A., et al. (2017). Attention is all you need. NeurIPS.
4. Kling, H., et al. (2012). Comparative evaluation of different efficiency criteria for hydrological model assessment. Hydrology and Earth System Sciences.
5. and so on..

## 许可证

MIT License 