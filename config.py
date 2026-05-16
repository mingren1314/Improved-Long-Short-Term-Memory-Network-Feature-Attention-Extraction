
import argparse


def get_args() -> dict:
    Linux = True
    # Linux = False
    static_list = ['soil_water_capacity'] # 土壤持水量 单位通常为 mm 或 m³/m³

    """
    "2m_temperature",          # 2米高度气温 (单位: K 或 °C)
    "10m_u_component_of_wind", # 10米高度纬向风速 (单位: m/s)
    "10m_v_component_of_wind", # 10米高度经向风速 (单位: m/s)
    "precipitation",           # 降水量 (单位: mm/h 或 kg/m²/s)
    "surface_pressure",        # 地表气压 (单位: Pa)
    "specific_humidity"        # 比湿 (单位: kg/kg)
    """
    forcing = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "precipitation", "surface_pressure", "specific_humidity"]

    """
    "surface_solar_radiation_downwards_w_m2",  # 下行短波辐射 (单位: W/m²)
    "surface_thermal_radiation_downwards_w_m2",# 下行长波辐射 (单位: W/m²)
    "soil_temperature_level_1",               # 表层土壤温度 (单位: K 或 °C)
    "total_evaporation"                       # 总蒸发量 (单位: mm/day)
    """
    land_surface = ["surface_solar_radiation_downwards_w_m2", "surface_thermal_radiation_downwards_w_m2", "soil_temperature_level_1", "total_evaporation"]
    """Parse input arguments"""

    parser = argparse.ArgumentParser()  # 负责定义参数规则、解析用户输入的命令行参数

    parser.add_argument('--device', type=str, default='cuda:0')
#-----------------------------------------------------------------------------------------------------------------
    # 经过处理后，将数据集保存到哪里
    # ! F:/Datasets/server 如果设置为，则处理服务器的文件
    if not Linux:
        sr = 1
        seq_len = 7
        out_path = 'E:/end/all/model/drought-agriculture/Datasets/agriculture/'
    else:
        sr = 1
        seq_len = 30
        # out_path = '/root/autodl-tmp/model/drought-agriculture/Datasets/agriculture/'
        out_path = 'E:/end/all/model/drought-agriculture/Datasets/agriculture/'

    parser.add_argument('--out_path', type=str, default=out_path)
    # 数据输入路径（Landbench路径）
    parser.add_argument('--data_path', type=str, default='E:/end/all/model/drought-agriculture/Datasets/LandBench')
    # parser.add_argument('--data_path', type=str, default='/root/autodl-tmp/Datasets/LandBench')
    # 原始的sf路径   空间范围定义文件（Shapefile），用于裁剪或掩膜数据
    parser.add_argument('--sf', type=str, default='E:/first/project/all/shp/中国大陆/中国大陆.shp')
    # parser.add_argument('--sf', type=str, default='/root/autodl-tmp/Datasets/shp/china/china.shp')

#-----------------------------------------------------------------------------------------------------------------
    # LSTM DARNN CNN ConvLSTM AttnRNN HybridRNN AttnLSTM CNNTransformer STALSTM BiLSTM
    parser.add_argument('--stride', type=float, default=10)
    parser.add_argument('--process', type=str, default='smci')
    parser.add_argument('--modelname', type=str, default='STALSTM')
    # parser.add_argument('--modelname', type=str, default='CNNTransformer')

    # parser.add_argument('--modelname', type=str, default='AttnLSTM')

    # parser.add_argument('--modelname', type=str, default='MLP')
    # parser.add_argument('--modelname', type=str, default='LSTM')
    # parser.add_argument('--modelname', type=str, default='TCN')
    # parser.add_argument('--modelname', type=str, default='BiLSTM')
    parser.add_argument('--data_type', type=str, default='float32')
    # 预测目标变量（土壤湿度层1体积含水量） land_surface文件夹
    parser.add_argument('--label', nargs='+', type=str, default=["volumetric_soil_water_layer_1"])
#-----------------------------------------------------------------------------------------------------------------
    # data
    # 数据时间分辨率（'1D'=逐日，'1H'=逐小时）
    parser.add_argument('--time_resolution', type=str, default='1D')
    #空间分辨率（单位：度/千米），需确保与数据文件实际分辨率一致
    parser.add_argument('--spatial_resolution', type=float, default=sr)
    # 测试集年份列表（时间外验证）
    parser.add_argument('--test_year', nargs='+', type=int, default=[2020])
    # 实际使用的数据年份范围
    parser.add_argument('--selected_year', nargs='+', type=int, default=[2015, 2020])
    # 气象强迫变量列表（如温度、降水）
    parser.add_argument('--forcing_list', nargs='+', type=str, default=forcing)
    # 静态地理变量列表（如土壤类型、高程）
    parser.add_argument('--static_list', nargs='+', type=str, default=static_list)
    # 地表过程变量列表（如辐射、蒸发）
    parser.add_argument('--land_surface_list', nargs='+', type=str, default=land_surface)
    # 是否使用内存映射文件处理大型数据，减少内存占用
    parser.add_argument('--memmap', type=bool, default=True)
    # 是否对输入数据进行归一化
    parser.add_argument('--normalize', type=bool, default=True)
    # 空间滑动窗口偏移量，用于生成空间邻域特征
    parser.add_argument('--spatial_offset', type=int, default=3)
    # 训练集与验证集的划分比例（0.8 表示 80% 训练，20% 验证）
    parser.add_argument('--split_ratio', type=float, default=0.8)
    # 是否从训练集中进一步划分验证集
    parser.add_argument('--valid_split', type=bool, default=True)
    # 归一化方式：'global'（全局统计）或 'region'（按空间区域统计）
    parser.add_argument('--normalize_type', type=str, default='region')
    # 输入特征维度（根据 forcing_list+land_surface_list+static_list 长度自动计算）
    parser.add_argument('--input_size', type=int, default=len(forcing) + len(land_surface) + len(static_list))
    # Early Stopping 耐心值（连续多少轮验证集损失无改善后停止训练）
    parser.add_argument('--patience', type=int, default=999)
# -----------------------------------------------------------------------------------------------------------------
    # model
    parser.add_argument('--niter', type=int, default=100)
    # 时间序列窗口长度（历史步数，需提前定义）
    parser.add_argument('--seq_len', type=int, default=seq_len)
    # 训练轮数
    parser.add_argument('--epochs', type=int, default=50)
    # 实验重复次数（确保结果稳定性）
    parser.add_argument('--num_repeat', type=int, default=1)
    # 批次大小（影响内存消耗和梯度稳定性）
    parser.add_argument('--batch_size', type=int, default=64)
    # 正则化方法Dropout 比率（防止过拟合）
    parser.add_argument('--dropout', type=float, default=0.5)
    # LSTM/RNN 隐含层神经元数（复杂度控制）
    parser.add_argument('--hidden_size', type=int, default=128)
    # 预测未来时间步数（0 表示同时间步预测）
    parser.add_argument('--forecast_time', type=int, default=0)
    # parser.add_argument('--forecast_time', type=int, default=6)
    # parser.add_argument('--forecast_time', type=int, default=7)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    # CNN参数
    # CNN 卷积核尺寸（空间特征提取）
    parser.add_argument('--kernel_size', type=int, default=3)
    # CNN 卷积步长（下采样率）
    parser.add_argument('--stride_cnn', type=int, default=2)
    # CNN 输入特征图尺寸（需与空间分辨率匹配）
    parser.add_argument('--input_size_cnn', type=float, default=64)

    # GNN参数
    # 动态图构建时的邻居数量
    parser.add_argument('--k_neighbors', type=int, default=8)
    # 是否使用空间邻近性构建图
    parser.add_argument('--use_spatial_proximity', type=bool, default=True)
    # GNN层数
    parser.add_argument('--num_gnn_layers', type=int, default=2)
    # 时间注意力层数
    parser.add_argument('--num_temporal_layers', type=int, default=1)
    cfg = vars(parser.parse_args())

    return cfg
