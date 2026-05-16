import os
import torch
import random
import warnings
import numpy as np
from eval import test
from train import train
from data import Dataset
from config import get_args
from postprocess import postprocess
warnings.filterwarnings("ignore")


def prt():
    print("\033[0;31;40m MSG \033[0m")  # 红色
    print("\033[0;32;40m MSG \033[0m")  # 绿色
    print("\033[0;33;40m MSG \033[0m")  # 黄色
    print("\033[0;34;40m MSG \033[0m")  # 蓝色
    print("\033[0;35;40m MSG \033[0m")  # 紫色
    print("\033[0;36;40m MSG \033[0m")  # 青色
    print("\033[0;37;40m MSG \033[0m")  # 灰色


def seed(seed=5201314):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(cfg):
    seed()   # 确保每次运行程序时，所有涉及随机性的操作（如数据打乱、权重初始化等）都能得到相同的结果，从而保证实验的可复现性

    device = torch.device(cfg['device']) if torch.cuda.is_available() else torch.device('cpu')

    # ! Loding data
    print("\033[0;36;40m 1 step loading data:----------------------------------------------------------------------------------------------------------------- \033[0m")  # 青色
    path = cfg['out_path']          # agriculture文件夹
    if not os.path.isdir(path):
        os.makedirs(path)
    if os.path.exists(os.path.join(path, 'x_train_norm.npy')):
        print("\033[0;33;40m Data loaded \033[0m")  # 黄色
        # 从磁盘加载一个.npy文件，文件里存储的是训练数据的形状（shape），并用内存映射（只读）方式加载。
        x_train_shape = np.load(os.path.join(path, 'x_train_norm_shape.npy'), mmap_mode='r')
        # 用np.memmap把训练数据文件映射到内存，适合处理超大数据集（不用一次性全部加载到内存）
        x_train = np.memmap(os.path.join(path, 'x_train_norm.npy'), dtype=cfg['data_type'], mode='r+', shape=(x_train_shape[0], x_train_shape[1], x_train_shape[2], x_train_shape[3]))
        # 测试数据的形状
        x_test_shape = np.load(os.path.join(path, 'x_test_norm_shape.npy'), mmap_mode='r')
        # 内存映射方式加载测试数据
        x_test = np.memmap(os.path.join(path, 'x_test_norm.npy'), dtype=cfg['data_type'], mode='r+', shape=(x_test_shape[0], x_test_shape[1], x_test_shape[2], x_test_shape[3]))
        # 训练集标签，使用内存映射
        y_train = np.load(os.path.join(path, 'y_train_norm.npy'), mmap_mode='r')
        # 测试集标签，使用内存映射
        y_test = np.load(os.path.join(path, 'y_test_norm.npy'), mmap_mode='r')
        # 静态数据（如地理信息等），这里没有用内存映射，数据会全部加载到内存
        static = np.load(os.path.join(path, 'static_norm.npy'))
        # 根据配置中的空间分辨率动态生成掩码文件名，然后加载该掩码文件
        file_name_mask = 'Mask with {sr} spatial resolution.npy'.format(sr=cfg['spatial_resolution'])
        mask = np.load(os.path.join(path, file_name_mask))
        
        # Load lat/lon for GNN
        lat_file_name = f"lat_{cfg['spatial_resolution']}.npy"
        lon_file_name = f"lon_{cfg['spatial_resolution']}.npy"
        if os.path.exists(os.path.join(path, lat_file_name)):
            lat = np.load(os.path.join(path, lat_file_name))
            lon = np.load(os.path.join(path, lon_file_name))
        else:
            # If not found, we might need to rely on Dataset class or handle error
            print("Warning: Lat/Lon files not found. GNN might fail if they are needed.")
            lat, lon = None, None

    else:
        print("not exists x_train_norm.npy,run else..........")
        cls = Dataset(cfg)
        print(cls)
        x_train, y_train, x_test, y_test, static, lat, lon, mask = cls.fit(cfg)
    # load scaler for inverse
    # 归一化是按区域进行的。
    # 这时，scaler_x和scaler_y的形状是(2, ...)，其中2通常代表最小值和最大值（或均值和标准差），后面跟着数据的空间维度-高度、宽度、通道
    if cfg['normalize_type'] in ['region']:
        scaler_x = np.memmap(os.path.join(path, "scaler_x.npy"), dtype=cfg['data_type'], mode='r+', shape=(2, x_train.shape[1], x_train.shape[2], x_train.shape[3]))
        scaler_y = np.memmap(os.path.join(path, "scaler_y.npy"), dtype=cfg['data_type'], mode='r+', shape=(2, y_train.shape[1], y_train.shape[2], y_train.shape[3]))

    # 归一化是全局的（对每个通道整体归一化）。
    # 这时，scaler_x和scaler_y的形状是(2, 通道数)，只对每个通道保存2个参数（如均值和标准差）
    elif cfg['normalize_type'] in ['global']:
        scaler_x = np.memmap(os.path.join(path, "scaler_x.npy"), dtype=cfg['data_type'], mode='r+', shape=(2, x_train.shape[3]))
        scaler_y = np.memmap(os.path.join(path, "scaler_y.npy"), dtype=cfg['data_type'], mode='r+', shape=(2, y_train.shape[3]))

    # ! Model training
    print("\033[0;36;40m 2 step trainng model:----------------------------------------------------------------------------------------------------------------- \033[0m")  # 青色
    out_path = os.path.join(path, cfg['process'])  # agriculture/smci
    if not os.path.isdir(os.path.join(out_path, cfg['modelname'], str(cfg['forecast_time']))):  #如果指定的目录不存在，则创建该目录
        # 这行代码将 out_path、cfg['modelname'] 和 cfg['forecast_time'] 这三个部分拼接成一个完整的目录路径
        # forecast_time-->default=0
        os.makedirs(os.path.join(out_path, cfg['modelname'], str(cfg['forecast_time'])))
    # /path/to/out_path/model_name/20231001/model_name_para.pkl
    if os.path.exists(os.path.join(out_path, cfg['modelname'], str(cfg['forecast_time']), cfg['modelname'] + '_para.pkl')):
        print("\033[0;33;40m Model loaded \033[0m")  # 黄色
        model = torch.load(os.path.join(out_path, cfg['modelname'], str(cfg['forecast_time']), cfg['modelname'] + '_para.pkl'))
    else:
        # train
        for j in range(cfg["num_repeat"]):  # num_repeat-->default=1   实验重复次数
            train(x_train, y_train, static, mask, scaler_x, scaler_y, cfg, j, path, out_path, device, lat=lat, lon=lon)
            model = torch.load(os.path.join(out_path, cfg['modelname'], str(cfg['forecast_time']), cfg['modelname'] + '_para.pkl'))

    # ! Evaluation model
    print("\033[0;36;40m 3 step evaluation model:----------------------------------------------------------------------------------------------------------------- \033[0m")  # 青色
    y_pred, y_test = test(x_test, y_test, static, scaler_y, cfg, model, device)
# ------------------------------------------------------------------------------------------------------------------------------
    # save predicted values and true values
    print(f'Saving predictions by {cfg["modelname"]} Model and we hope to use "postprocess" and "plot_test" codes for detailed analyzing')
    np.save(os.path.join(out_path, cfg['modelname'], str(cfg['forecast_time']), '_predictions.npy'), y_pred)
    np.save(os.path.join(out_path, cfg['modelname'], str(cfg['forecast_time']), 'observations.npy'), y_test)

    # ! Processing data
    print("\033[0;36;40m 3 step processing data:----------------------------------------------------------------------------------------------------------------- \033[0m")  # 青色
    postprocess(cfg)


if __name__ == '__main__':
    cfg = get_args()           # 进入config.py配置程序
    main(cfg)
