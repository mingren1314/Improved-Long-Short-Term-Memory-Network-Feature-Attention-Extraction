import os
import torch
import numpy as np
from tqdm import trange, tqdm
from datetime import datetime, timedelta
from utils import get_sinusoid_encoding_table
from sklearn.ensemble import RandomForestRegressor
from sklearnex import patch_sklearn, unpatch_sklearn


def load_feature_mask(x, y, static, scaler_y, cfg):
    # TODO 修改随机森林参数
    inputs = []
    target = []

    random_forest = RandomForestRegressor()
    print("\033[0;34;40m Now we are processing the feature mask, wait a moment \033[0m")  # 蓝色
    for number_iter in trange(1, cfg['epochs'] * cfg["niter"] + 1):
        x_batch, y_batch, aux_batch, _, _ = load_train_data_for_rnn(cfg, x, y, static, scaler_y)
        aux_batch = torch.unsqueeze(aux_batch, dim=1).repeat(1, x_batch.shape[1], 1)
        x_batch = torch.cat((x_batch, aux_batch), dim=2)
        inputs.append(x_batch)
        target.append(y_batch)
    # inputs = torch.concatenate(inputs, dim=0).cpu().detach().numpy()
    # target = torch.concatenate(target, dim=0).cpu().detach().numpy()
    inputs = torch.cat(inputs, dim=0).cpu().detach().numpy()
    target = torch.cat(target, dim=0).cpu().detach().numpy()

    # 处理成随机森林支持的格式
    inputs = inputs.reshape(inputs.shape[0], inputs.shape[1] * inputs.shape[2])
    # TODO 修改随机森林的拟合数量
    num = 150

    patch_sklearn()
    random_forest.fit(inputs[:num], target[:num])
    feature_importances = random_forest.feature_importances_
    threshold = np.nanmedian(feature_importances) * 5
    feature_mask = (feature_importances > threshold).reshape(cfg['seq_len'], cfg['input_size'])
    np.save(os.path.join(cfg['out_path'], 'feature_mask.npy'), feature_mask)
    print("\033[0;34;40m Processed the feature mask done ! \033[0m")  # 蓝色
    return feature_mask


# 对输入的时空数据（x、y、aux）和掩码（mask）进行维度变换和筛选，只保留掩码为1（有效区域，比如陆地）的数据
def sea_mask_rnn(cfg, x, y, aux, mask):
    # 变换x和y的维度顺序：把通道维（原本在最后）提前到第二维
    x = x.transpose(0, 3, 1, 2)       # (样本数, 通道, 高, 宽)
    y = y.transpose(0, 3, 1, 2)

    # 变换aux的维度顺序：把最后一维提前
    aux = aux.transpose(2, 0, 1)

    # 把空间维度展平成一维
    x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
    y = y.reshape(y.shape[0], y.shape[2] * y.shape[3])
    aux = aux.reshape(aux.shape[0], aux.shape[1] * aux.shape[2])

    # 把mask展平成一维
    mask = mask.reshape(mask.shape[0] * mask.shape[1])

    # 只保留mask==1的空间位置（有效区域）
    x = x[:, :, mask == 1]  # (样本数, 通道, 有效点数)
    y = y[:, mask == 1]
    aux = aux[:, mask == 1]

    return x, y, aux


def sea_mask_cnn(cfg, x, y, aux, mask):
    x = x.transpose(0, 3, 1, 2)
    y = y.transpose(0, 3, 1, 2)
    aux = aux.transpose(2, 0, 1)

    nt, nf, nlat, nlon = x.shape
    ngrid = nlat * nlon
    _index = np.array([i for i in range(0, ngrid, 1)])
    mask = mask.reshape(mask.shape[0] * mask.shape[1])
    mask_index = _index[mask == 1]
    return x, y, aux, mask_index


def load_train_data_for_rnn(cfg, x, y, aux, scaler):
    device = cfg['device']
    nt, nf, ngrid = x.shape
    mean, std = np.array(scaler[0]), np.array(scaler[1])

    idx_time = np.random.randint(0, nt - cfg['seq_len'] - cfg["forecast_time"], 1)[0]
    idx_grid = np.random.randint(0, ngrid, cfg['batch_size'])
    x = np.transpose(x, (2, 0, 1))
    y = np.transpose(y, (1, 0))
    aux = np.transpose(aux, (1, 0))
    x = x[idx_grid, idx_time:idx_time + cfg['seq_len']]
    y = y[idx_grid, idx_time + cfg['seq_len'] + cfg["forecast_time"]]
    aux = aux[idx_grid]
    y[np.isinf(y)] = np.nan
    mask = y == y
    x = x[mask]
    y = y[mask]
    aux = aux[mask]
    x[np.isinf(x)] = np.nan
    x = np.nan_to_num(x)

    # 计算idx_time是当前年的第几天
    start_year = datetime(cfg['selected_year'][0], 1, 1)
    day_of_year = (start_year + timedelta(int(idx_time))).timetuple().tm_yday - 1
    position_encoding = get_sinusoid_encoding_table(cfg['input_size'], config=cfg)
    # TODO 为输入的x变量添加位置编码
    # x = x + position_encoding[:, day_of_year: day_of_year + x.shape[1]]

    x, y, aux = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device), torch.from_numpy(aux).to(device)
    return x, y, aux, mean, std


def load_test_data_for_rnn(cfg, x, y, aux, scaler, stride, i, n):
    device = cfg['device']
    all_days = 365 * (cfg['selected_year'][1] - cfg['selected_year'][0] - len(cfg['test_year']) + 1) + cfg["seq_len"] + cfg["forecast_time"]
    train_days = all_days - 365 * len(cfg['selected_year']) + cfg["seq_len"] + cfg["forecast_time"]
    nt, nf, ngrid = x.shape
    x = np.transpose(x, (2, 0, 1))
    y = np.transpose(y, (1, 0))
    aux = np.transpose(aux, (1, 0))

    mean, std = np.array(scaler[0]), np.array(scaler[1])
    x_new = x[:, i * stride: i * stride + cfg["seq_len"], :][0:ngrid:2 * stride, :, :]
    y_new = y[0:ngrid:2 * stride, i * stride + cfg["seq_len"] + cfg["forecast_time"]]

    aux_new = aux[0:ngrid: 2 * stride]
    # 计算idx_time是当前年的第几天
    start_year = datetime(cfg['selected_year'][0], 1, 1)
    day_of_year = (start_year + timedelta(int(train_days + i))).timetuple().tm_yday
    position_encoding = get_sinusoid_encoding_table(cfg['input_size'], config=cfg)
    # TODO 为输入的x变量添加位置编码
    # x_new = x_new + position_encoding[:, day_of_year: day_of_year + x_new.shape[1]]
    y_new[np.isinf(y_new)] = np.nan
    mask = y_new == y_new
    x_new = x_new[mask]
    y_new = y_new[mask]
    aux_new = aux_new[mask]
    x_new[np.isinf(x_new)] = np.nan
    x_new = np.nan_to_num(x_new)
    x_new, y_new, aux_new = torch.from_numpy(x_new).to(device), torch.from_numpy(y_new).to(device), torch.from_numpy(aux_new).to(device)
    return x_new, y_new, aux_new, np.tile(mean, (1, n, 1)), np.tile(std, (1, n, 1))


# ------------------------------------------------------------------------------------------------------------------------------
def load_train_data_for_cnn(cfg, x, y, aux, scaler, lat_index, lon_index, mask):
    device = cfg['device']
    nt, nf, nlat, nlon = x.shape
    mean, std = np.array(scaler[0]), np.array(scaler[1])
    mask_index = np.random.randint(0, mask.shape[0], cfg['batch_size'])
    idx_grid = mask[mask_index]
    # ngrid convert nlat and nlon
    idx_lon = ((idx_grid + 1) % (nlon + 1)) - 1
    idx_lon[idx_lon == -1] = nlon - 1
    idx_lat = (idx_grid // (nlon + 1))

    idx_time = np.random.randint(0, nt - cfg['seq_len'] - cfg["forecast_time"], 1)[0]

    x_new = np.zeros((idx_lon.shape[0], cfg["seq_len"], nf, 2 * cfg['spatial_offset'] + 1, 2 * cfg['spatial_offset'] + 1)) * np.nan
    y_new = np.zeros((idx_lon.shape[0])) * np.nan
    aux_new = np.zeros((idx_lon.shape[0], aux.shape[0], 2 * cfg['spatial_offset'] + 1, 2 * cfg['spatial_offset'] + 1)) * np.nan

    for i in range(idx_lon.shape[0]):
        lat_index_bias = idx_lat[i] + cfg['spatial_offset']
        lon_index_bias = idx_lon[i] + cfg['spatial_offset']
        x_new[i] = x[idx_time:idx_time + cfg['seq_len'], :, lat_index[lat_index_bias - cfg['spatial_offset']: lat_index_bias + cfg['spatial_offset'] + 1], :][:, :, :, lon_index[lon_index_bias - cfg['spatial_offset']: lon_index_bias + cfg['spatial_offset'] + 1]]
        y_new[i] = y[idx_time + cfg['seq_len'] + cfg["forecast_time"], :, idx_lat[i], idx_lon[i]]
        aux_new[i] = aux[:, lat_index[lat_index_bias - cfg['spatial_offset']: lat_index_bias + cfg['spatial_offset'] + 1], :][:, :, lon_index[lon_index_bias - cfg['spatial_offset']: lon_index_bias + cfg['spatial_offset'] + 1]]

    y_new[np.isinf(y_new)] = np.nan
    mask = y_new == y_new
    x_new = x_new[mask]
    y_new = y_new[mask]
    aux_new = aux_new[mask]
    x_new = np.nan_to_num(x_new)
    aux_new = np.nan_to_num(aux_new)
    x_new, y_new, aux_new = torch.from_numpy(x_new).to(device), torch.from_numpy(y_new).to(device), torch.from_numpy(aux_new).to(device)
    return x_new, y_new, aux_new, mean, std


def load_test_data_for_cnn(cfg, x, y, aux, scaler, slect_list, lat_index, lon_index, z, stride):
    device = cfg['device']
    x = x.transpose(0, 3, 1, 2)
    y = y.transpose(0, 3, 1, 2)
    aux = aux.transpose(2, 0, 1)
    nt, _, nlat, nlon = y.shape
    ny = (2 * nlat // stride) + 1
    nx = (2 * nlon // stride) + 1

    x_new = np.zeros((ny * nx, cfg["seq_len"], x.shape[1], 2 * cfg['spatial_offset'] + 1, 2 * cfg['spatial_offset'] + 1)) * np.nan
    y_new = np.zeros((ny * nx)) * np.nan
    aux_new = np.zeros((ny * nx, aux.shape[0], 2 * cfg['spatial_offset'] + 1, 2 * cfg['spatial_offset'] + 1)) * np.nan
    mean, std = np.array(scaler[0]), np.array(scaler[1])

    count = 0
    for i in range(0, nlon, stride // 2):
        for j in range(0, nlat, stride // 2):
            lat_index_bias = lat_index[j] + cfg['spatial_offset']
            lon_index_bias = lon_index[i] + cfg['spatial_offset']
            x_new[count] = x[z: z + cfg['seq_len'], :, lat_index[lat_index_bias - cfg['spatial_offset']: lat_index_bias + cfg['spatial_offset'] + 1], :][:, :, :, lon_index[lon_index_bias - cfg['spatial_offset']: lon_index_bias + cfg['spatial_offset'] + 1]]
            y_new[count] = y[z + cfg['seq_len'] + cfg["forecast_time"], :, j, i]
            aux_new[count] = aux[:, lat_index[lat_index_bias - cfg['spatial_offset']: lat_index_bias + cfg['spatial_offset'] + 1], :][:, :, lon_index[lon_index_bias - cfg['spatial_offset']: lon_index_bias + cfg['spatial_offset'] + 1]]
            count = count + 1
    y_new[np.isinf(y_new)] = np.nan
    mask = y_new == y_new
    x_new = x_new[mask]
    y_new = y_new[mask]
    aux_new = aux_new[mask]
    x_new = np.nan_to_num(x_new)
    aux_new = np.nan_to_num(aux_new)
    x_new, y_new, aux_new = torch.from_numpy(x_new).to(device), torch.from_numpy(y_new).to(device), torch.from_numpy(aux_new).to(device)
    return x_new, y_new, aux_new, np.tile(mean, (1, ny * nx, 1)), np.tile(std, (1, ny * nx, 1))
# ------------------------------------------------------------------------------------------------------------------------------


def load_train_data_for_co(cfg, x, y, aux, scaler):
    device = cfg['device']
    nt, _, nlat, nlon = y.shape
    print('y.shape is', y)
    ngrid = nlat * nlon
    mean, std = np.array(scaler[0]), np.array(scaler[1])
    idx_grid = np.random.randint(0, ngrid, cfg['batch_size'])
    # ngrid convert nlat and nlon
    idx_lon = ((idx_grid + 1) % (nlon + 1)) - 1
    idx_lon[idx_lon == -1] = nlon
    idx_lat = (idx_grid // (nlon + 1))

    idx_time = np.random.randint(0, nt - cfg['seq_len'], 1)[0]

    x_new = np.zeros((idx_lon.shape[0], cfg["seq_len"] + 1, x.shape[1], 2 * cfg['spatial_offset'], 2 * cfg['spatial_offset'])) * np.nan
    y_new = np.zeros((idx_lon.shape[0])) * np.nan
    aux_new = np.zeros((idx_lon.shape[0], aux.shape[0], 2 * cfg['spatial_offset'], 2 * cfg['spatial_offset'])) * np.nan

    for i in range(idx_lon.shape[0]):
        idx_lat_bias, idx_lon_bias = idx_lat[i] + cfg['spatial_offset'], idx_lon[i] + cfg['spatial_offset']
        x_new[i] = x[idx_time:idx_time + cfg['seq_len'] + 1, :, idx_lat_bias - cfg['spatial_offset']: idx_lat_bias + cfg['spatial_offset'], idx_lon_bias - cfg['spatial_offset']: idx_lon_bias + cfg['spatial_offset']]
        y_new[i] = y[idx_time + cfg['seq_len'] + cfg["forecast_time"], idx_lat[i], idx_lon[i]]
        aux_new[i] = aux[:, idx_lat_bias - cfg['spatial_offset']: idx_lat_bias + cfg['spatial_offset'], idx_lon_bias - cfg['spatial_offset']:idx_lon_bias + cfg['spatial_offset']]
    mask = y_new == y_new
    x_new = x_new[mask]
    y_new = y_new[mask]
    aux_new = aux_new[mask]
    x_new, y_new, aux_new = torch.from_numpy(x_new).to(device), torch.from_numpy(y_new).to(device), torch.from_numpy(aux_new).to(device)
    return x_new, y_new, aux_new, mean, std

# ------------------------------------------------------------------------------------------------------------------------------


def earth_data_transform(cfg, x):
    lat_index = np.array([i for i in range(0, x.shape[1])])
    lon_index = np.array([i for i in range(0, x.shape[2])])

    x_up = lat_index[lat_index.shape[0] - cfg['spatial_offset']:lat_index.shape[0]]
    x_down = lat_index[:cfg['spatial_offset']]
    x_left = lon_index[lon_index.shape[0] - cfg['spatial_offset']:lon_index.shape[0]]
    x_right = lon_index[:cfg['spatial_offset']]
    lat_index_new = np.concatenate((x_up, lat_index), axis=0)
    lat_index_new = np.concatenate((lat_index_new, x_down), axis=0)
    lon_index_new = np.concatenate((x_left, lon_index), axis=0)
    lon_index_new = np.concatenate((lon_index_new, x_right), axis=0)
    return lat_index_new, lon_index_new


def load_train_data_for_gnn(cfg, x, y, aux, scaler):
    """
    Load training data for GNN.
    Samples batch_size time windows, and includes ALL spatial nodes.
    x shape: (nt, nf, num_nodes) (already masked and flattened)
    """
    device = cfg['device']
    nt, nf, num_nodes = x.shape
    mean, std = np.array(scaler[0]), np.array(scaler[1])

    # Randomly select time steps
    # We need seq_len + forecast_time
    idx_time = np.random.randint(0, nt - cfg['seq_len'] - cfg["forecast_time"], cfg['batch_size'])

    x_batch = []
    y_batch = []
    aux_batch = []

    # x is (nt, nf, num_nodes)
    # We want (batch, seq, num_nodes, nf)

    for t in idx_time:
        # x[t:t+seq] -> (seq, nf, num_nodes)
        # permute to (seq, num_nodes, nf)
        xt = x[t : t + cfg['seq_len']]
        xt = np.transpose(xt, (0, 2, 1)) 
        x_batch.append(xt)

        # y[t+seq+forecast] -> (nf, num_nodes) -> we want (num_nodes, output_dim)
        # y is usually (nt, num_nodes) or (nt, 1, num_nodes)?
        # In sea_mask_rnn: y = y.reshape(y.shape[0], y.shape[2] * y.shape[3]) -> (nt, num_nodes)
        yt = y[t + cfg['seq_len'] + cfg["forecast_time"]] # (num_nodes,)
        y_batch.append(yt)

        # aux is (nt, num_nodes) or similar?
        # sea_mask_rnn: aux = aux.reshape... -> (nt, num_nodes*?)
        # Let's assume aux is (nt, num_nodes) for now, similar to y
        auxt = aux[t : t + cfg['seq_len']] # (seq, num_nodes)
        aux_batch.append(auxt)

    x_batch = np.stack(x_batch) # (batch, seq, num_nodes, nf)
    y_batch = np.stack(y_batch) # (batch, num_nodes)
    aux_batch = np.stack(aux_batch) # (batch, seq, num_nodes)

    # Handle NaNs
    y_batch[np.isinf(y_batch)] = np.nan
    # For GNN, we can't easily drop nodes per sample because graph structure is fixed.
    # We replace NaNs with 0 or mean.
    x_batch = np.nan_to_num(x_batch)
    y_batch = np.nan_to_num(y_batch)
    aux_batch = np.nan_to_num(aux_batch)

    x_batch = torch.from_numpy(x_batch).float().to(device)
    y_batch = torch.from_numpy(y_batch).float().to(device)
    aux_batch = torch.from_numpy(aux_batch).float().to(device)

    # Expand y to (batch, num_nodes, 1)
    y_batch = y_batch.unsqueeze(-1)
    
    # Expand aux to (batch, seq, num_nodes, 1)
    aux_batch = aux_batch.unsqueeze(-1)

    return x_batch, y_batch, aux_batch, mean, std


def load_test_data_for_gnn(cfg, x, y, aux, scaler, stride, i, n):
    """
    Load test data for GNN.
    Similar to RNN but returns (1, seq, num_nodes, nf)
    """
    device = cfg['device']
    nt, nf, num_nodes = x.shape
    mean, std = np.array(scaler[0]), np.array(scaler[1])
    
    # i is the index of the sliding window
    # stride is used to skip time steps?
    # In load_test_data_for_rnn: x_new = x[:, i*stride : ...]
    
    t = i * stride
    
    if t + cfg['seq_len'] + cfg['forecast_time'] >= nt:
        # Return dummy or handle end of stream
        # For simplicity, return last valid window or zeros
        return None, None, None, None, None

    xt = x[t : t + cfg['seq_len']] # (seq, nf, num_nodes)
    xt = np.transpose(xt, (0, 2, 1)) # (seq, num_nodes, nf)
    
    yt = y[t + cfg['seq_len'] + cfg["forecast_time"]] # (num_nodes,)
    
    auxt = aux[t : t + cfg['seq_len']] # (seq, num_nodes)
    
    x_batch = np.expand_dims(xt, 0) # (1, seq, num_nodes, nf)
    y_batch = np.expand_dims(yt, 0) # (1, num_nodes)
    aux_batch = np.expand_dims(auxt, 0) # (1, seq, num_nodes)
    
    x_batch = np.nan_to_num(x_batch)
    y_batch = np.nan_to_num(y_batch)
    aux_batch = np.nan_to_num(aux_batch)
    
    x_batch = torch.from_numpy(x_batch).float().to(device)
    y_batch = torch.from_numpy(y_batch).float().to(device)
    aux_batch = torch.from_numpy(aux_batch).float().to(device)
    
    y_batch = y_batch.unsqueeze(-1)
    aux_batch = aux_batch.unsqueeze(-1)
    
    return x_batch, y_batch, aux_batch, mean, std

