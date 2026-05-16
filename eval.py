import os
import time
import torch
import numpy as np
from tqdm import tqdm
from data import Dataset
from utils import r2_score
from data_gen import earth_data_transform
from utils import get_sinusoid_encoding_table


def batcher_lstm(x_test, y_test, aux_test, seq_len, forecast_time, device, cfg=None):
    n_t, n_feat = x_test.shape
    n = n_t - seq_len - forecast_time
    x_new = np.zeros((n, seq_len, n_feat)) * np.nan
    y_new = np.zeros((n, 1)) * np.nan
    aux_new = np.zeros((n, aux_test.shape[0])) * np.nan
    # ? 位置编码
    position_encoding = get_sinusoid_encoding_table(cfg['input_size'], config=cfg)
    for i in range(n):
        x_new[i] = x_test[i: i + seq_len]
        # TODO 为x添加位置编码
        # x_new = x_new + position_encoding[:, i: i + seq_len]
        y_new[i] = y_test[i + seq_len + forecast_time]
        aux_new[i] = aux_test
    x_new = np.nan_to_num(x_new)
    y_new = np.nan_to_num(y_new)
    aux_new = np.nan_to_num(aux_new)
    x_new, y_new, aux_new = torch.from_numpy(x_new).to(device), torch.from_numpy(y_new).to(device), torch.from_numpy(aux_new).to(device)
    return x_new, y_new, aux_new


def batcher_cnn(x_test, y_test, aux_test, seq_len, forecast_time, spatial_offset, i, j, lat_index, lon_index, device):
    x_test = x_test.transpose(0, 3, 1, 2)
    y_test = y_test.transpose(0, 3, 1, 2)
    aux_test = aux_test.transpose(2, 0, 1)
    n_t, n_feat, n_lat, n_lon = x_test.shape

    n = n_t - seq_len - forecast_time
    x_new = (np.zeros((n, seq_len, n_feat, 2 * spatial_offset + 1, 2 * spatial_offset + 1)) * np.nan)
    y_new = np.zeros((n, 1)) * np.nan
    aux_new = (np.zeros((n, aux_test.shape[0], 2 * spatial_offset + 1, 2 * spatial_offset + 1)) * np.nan)
    for ni in range(n):
        lat_index_bias = i + spatial_offset
        lon_index_bias = j + spatial_offset
        x_new[ni] = x_test[ni: ni + seq_len, :, lat_index[lat_index_bias - spatial_offset: lat_index_bias + spatial_offset + 1], :,][:, :, :, lon_index[lon_index_bias - spatial_offset: lon_index_bias + spatial_offset + 1], ]
        y_new[ni] = y_test[ni + seq_len + forecast_time, :, i, j]
        aux_new[ni] = aux_test[:, lat_index[lat_index_bias - spatial_offset: lat_index_bias + spatial_offset + 1], :, ][:, :, lon_index[lon_index_bias - spatial_offset: lon_index_bias + spatial_offset + 1],]
    x_new = np.nan_to_num(x_new)
    y_new = np.nan_to_num(y_new)
    aux_new = np.nan_to_num(aux_new)
    x_new, y_new, aux_new = torch.from_numpy(x_new).to(device), torch.from_numpy(y_new).to(device), torch.from_numpy(aux_new).to(device)
    return x_new, y_new, aux_new


def batcher_convlstm(x_test, y_test, aux_test, seq_len, forecast_time, spatial_offset, i, j, lat_index, lon_index, device):
    x_test = x_test.transpose(0, 3, 1, 2)
    y_test = y_test.transpose(0, 3, 1, 2)
    aux_test = aux_test.transpose(2, 0, 1)
    n_t, n_feat, n_lat, n_lon = x_test.shape

    n = n_t - seq_len - forecast_time
    x_new = np.zeros((n, seq_len, n_feat, 2 * spatial_offset + 1, 2 * spatial_offset + 1)) * np.nan
    y_new = np.zeros((n, 1)) * np.nan
    aux_new = np.zeros((n, aux_test.shape[0], 2 * spatial_offset + 1, 2 * spatial_offset + 1)) * np.nan

    for ni in range(n):
        lat_index_bias = i + spatial_offset
        lon_index_bias = j + spatial_offset
        x_new[ni] = x_test[ni: ni + seq_len, :, lat_index[lat_index_bias - spatial_offset: lat_index_bias + spatial_offset + 1], :][:, :, :, lon_index[lon_index_bias - spatial_offset: lon_index_bias + spatial_offset + 1]]
        y_new[ni] = y_test[ni + seq_len + forecast_time, :, i, j]
        aux_new[ni] = aux_test[:, lat_index[lat_index_bias - spatial_offset: lat_index_bias + spatial_offset + 1], :][:, :, lon_index[lon_index_bias - spatial_offset: lon_index_bias + spatial_offset + 1]]
    x_new = np.nan_to_num(x_new)
    y_new = np.nan_to_num(y_new)
    aux_new = np.nan_to_num(aux_new)
    x_new, y_new, aux_new = torch.from_numpy(x_new).to(device), torch.from_numpy(y_new).to(device), torch.from_numpy(aux_new).to(device)
    return x_new, y_new, aux_new


def test(x, y, static, scaler, cfg, model, device):

    print(f"\033[0;35;40m x_test {x.shape}, y_test {y.shape}, static_test {static.shape} \033[0m")
    model.eval()
    cls = Dataset(cfg)
    feature_mask = np.load(os.path.join(cfg['out_path'], 'feature_mask.npy'))
    feature_mask = torch.from_numpy(feature_mask).to(device)
    if cfg["modelname"] in ["CNN", "ConvLSTM", "CNNTransformer"]:
        lat_index, lon_index = earth_data_transform(cfg, x)
        print("\033[1;31m%s\033[0m" % "Applied Model is {m_n}, we need to transform the data according to the sphere shape".format(m_n=cfg["modelname"]))
    y_pred = np.zeros((y.shape[0] - cfg["seq_len"] - cfg["forecast_time"], y.shape[1], y.shape[2])) * np.nan
    y_true = y[cfg["seq_len"] + cfg["forecast_time"]:, :, :, 0]

    print(f"\033[0;35;40m the true label shape is: {y_true.shape} and the predicton shape is: {y_pred.shape} \033[0m")
    mask = y_true == y_true
    t_begin = time.time()
    # ------------------------------------------------------------------------------------------------------------------------------
    # for each grid by lstm model
    if cfg["modelname"] in ["LSTM", "AttnRNN", "AttnLSTM", "CNNLSTM", "STALSTM", "BiLSTM", "MLP","TCN"]:
        for i, j in tqdm(np.argwhere(mask[-1])):
            x_new, y_new, static_new = batcher_lstm(x[:, i, j], y[:, i, j], static[i, j], cfg["seq_len"], cfg["forecast_time"], device, cfg)
            static_new = static_new.unsqueeze(1).repeat(1, x_new.shape[1], 1)
            x_new = torch.cat([x_new, static_new], 2)
            pred = model(x_new)
            pred = pred.cpu().detach().numpy().squeeze()
            if cfg["normalize"] and cfg["normalize_type"] in ["region"]:
                pred = cls.reverse_normalize(pred, "output", scaler[:, i, j, 0], "minmax", -1)
            elif cfg["normalize"] and cfg["normalize_type"] in ["global"]:
                pred = cls.reverse_normalize(pred, "output", scaler, "minmax", -1)
            y_pred[:, i, j] = pred
    if cfg["modelname"] in ["HybridRNN"]:
        for i, j in tqdm(np.argwhere(mask[-1])):
            x_new, y_new, static_new = batcher_lstm(x[:, i, j], y[:, i, j], static[i, j], cfg["seq_len"], cfg["forecast_time"], device, cfg)
            static_new = static_new.unsqueeze(1).repeat(1, x_new.shape[1], 1)
            x_new = torch.cat([x_new, static_new], 2)
            pred = model(x_new, feature_mask)
            pred = pred.cpu().detach().numpy().squeeze()
            if cfg["normalize"] and cfg["normalize_type"] in ["region"]:
                pred = cls.reverse_normalize(pred, "output", scaler[:, i, j, 0], "minmax", -1)
            elif cfg["normalize"] and cfg["normalize_type"] in ["global"]:
                pred = cls.reverse_normalize(pred, "output", scaler, "minmax", -1)
            y_pred[:, i, j] = pred
    if cfg["modelname"] in ["DARNN"]:
        for i, j in tqdm(np.argwhere(mask[-1])):
            x_new, y_new, static_new = batcher_lstm(x[:, i, j], y[:, i, j], static[i, j], cfg["seq_len"], cfg["forecast_time"], device, cfg)
            static_new = static_new.unsqueeze(1)
            static_new = static_new.repeat(1, x_new.shape[1], 1)
            x_new = torch.cat([x_new, static_new], 2)
            x_new = x_new.permute(1, 0, 2)
            pred = model(x_new).squeeze()
            pred = pred.cpu().detach().numpy()
            if cfg["normalize"] and cfg["normalize_type"] in ["region"]:
                pred = cls.reverse_normalize(pred, "output", scaler[:, i, j, 0], "minmax", -1)
            elif cfg["normalize"] and cfg["normalize_type"] in ["global"]:
                pred = cls.reverse_normalize(pred, "output", scaler, "minmax", -1)
            y_pred[:, i, j] = pred
    # ------------------------------------------------------------------------------------------------------------------------------
    # for each grid by cnn model
    if cfg["modelname"] in ["CNN"]:
        for i, j in tqdm(np.argwhere(mask[-1])):
            x_new, y_new, static_new = batcher_convlstm(x, y, static, cfg["seq_len"], cfg["forecast_time"], cfg["spatial_offset"], i, j, lat_index, lon_index, device)
            x_new = x_new.squeeze(1)
            x_new = x_new.reshape(x_new.shape[0], x_new.shape[1] * x_new.shape[2], x_new.shape[3], x_new.shape[4],)
            x_new = torch.cat([x_new, static_new], 1)
            pred = model(x_new)
            # pred = pred*std[i, j, :]+mean[i, j, :] #(nsample,1,1)
            pred = pred.cpu().detach().numpy()
            pred = np.squeeze(pred)
            if cfg["normalize"] and cfg["normalize_type"] in ["region"]:
                pred = cls.reverse_normalize(pred, "output", scaler[:, i, j, 0], "minmax", -1)
            elif cfg["normalize"] and cfg["normalize_type"] in ["global"]:
                pred = cls.reverse_normalize(pred, "output", scaler, "minmax", -1)
            y_pred[:, i, j] = pred
    # ------------------------------------------------------------------------------------------------------------------------------
    # for each grid by convlstm model
    if cfg["modelname"] in ["ConvLSTM", "CNNTransformer"]:
        for i, j in tqdm(np.argwhere(mask[-1])):
            x_new, y_new, static_new = batcher_convlstm(x, y, static, cfg["seq_len"], cfg["forecast_time"], cfg["spatial_offset"], i, j, lat_index, lon_index, device)
            static_new = static_new.unsqueeze(1)
            static_new = static_new.repeat(1, x_new.shape[1], 1, 1, 1)
            x_new = torch.cat([x_new, static_new], 2)
            pred = model(x_new)
            pred = pred.cpu().detach().numpy()
            pred = np.squeeze(pred)
            if cfg["normalize"] and cfg["normalize_type"] in ["region"]:
                pred = cls.reverse_normalize(pred, "output", scaler[:, i, j, 0], "minmax", -1)
            elif cfg["normalize"] and cfg["normalize_type"] in ["global"]:
                pred = cls.reverse_normalize(pred, "output", scaler, "minmax", -1)
            y_pred[:, i, j] = pred
    # ----------------------------------------------------------------------------------------------------------------------------
    t_end = time.time()
    y_true_mask = y_true[mask]
    y_pred_mask = y_pred[mask]
    # log
    R = np.zeros(y_true.shape[0])
    R2 = r2_score(y_true_mask, y_pred_mask)
    for i in range(y_true.shape[0]):
        obs = np.squeeze(y_true[i, :])
        pre = np.squeeze(y_pred[i, :])
        msk = (obs == obs) * (pre == pre)
        R[i] = np.corrcoef(obs[msk], pre[msk])[0, 1]
    print("\033[1;31m%s\033[0m" % "Median R  {:.3f} time cost {:.2f}".format(np.nanmedian(R), t_end - t_begin))
    print("\033[1;31m%s\033[0m" % "ALLLLL R  {:.3f} time cost {:.2f}".format(np.corrcoef(y_true_mask, y_pred_mask)[0, 1], t_end - t_begin))
    print("\033[1;31m%s\033[0m" % "ALLLLL R2 {:.3f} time cost {:.2f}".format(R2, t_end - t_begin))
    return y_pred, y_true
