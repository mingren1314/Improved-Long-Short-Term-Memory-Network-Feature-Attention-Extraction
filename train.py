import os
import sys
import time
import tqdm
import torch
import torch.nn
import numpy as np
from tqdm import trange
from loss import NaNMSELoss
from loss import AutomaticWeightedLoss
from model import LSTMModel, CNN, ConvLSTMModel, DARNN, AttnRNN, HybridRNN, AttnLSTMModel, CNNTransformer, CNNLSTMModel, \
    STALSTMModel, BiLSTMModel, MLPModel, TCNModel
from data_gen import load_test_data_for_rnn, load_train_data_for_rnn, \
    load_test_data_for_cnn, load_train_data_for_cnn, earth_data_transform, \
    sea_mask_rnn, sea_mask_cnn, load_feature_mask, load_train_data_for_gnn, load_test_data_for_gnn


def load_iter(model, x, y, static, scaler_y, lat_index, lon_index, mask_index, cfg, feature_mask=None, grid_pos=None):
    if cfg["modelname"] in ['LSTM', 'AttnRNN', 'AttnLSTM', 'CNNLSTM', 'STALSTM', 'BiLSTM', 'MLP','TCN']:
        # generate batch data for Recurrent Neural Network
        x_batch, y_batch, aux_batch, _, _ = load_train_data_for_rnn(cfg, x, y, static, scaler_y)
        aux_batch = aux_batch.unsqueeze(1)
        aux_batch = aux_batch.repeat(1, x_batch.shape[1], 1)
        x_batch = torch.cat([x_batch, aux_batch], 2)
        pred = model(x_batch)
    if cfg["modelname"] in ['HybridRNN']:
        # generate batch data for Recurrent Neural Network
        x_batch, y_batch, aux_batch, _, _ = load_train_data_for_rnn(cfg, x, y, static, scaler_y)
        aux_batch = aux_batch.unsqueeze(1)
        aux_batch = aux_batch.repeat(1, x_batch.shape[1], 1)
        x_batch = torch.cat([x_batch, aux_batch], 2)
        pred = model(x_batch, feature_mask)
    if cfg["modelname"] in ['DARNN']:
        # generate batch data for Recurrent Neural Network
        x_batch, y_batch, aux_batch, _, _ = load_train_data_for_rnn(cfg, x, y, static, scaler_y)
        aux_batch = aux_batch.unsqueeze(1).repeat(1, x_batch.shape[1], 1)
        x_batch = torch.cat([x_batch, aux_batch], 2)
        # 转换维度
        x_batch = x_batch.permute(1, 0, 2)
        # 回归 分类
        pred = model(x_batch).squeeze()
    #  train way for CNN model
    elif cfg['modelname'] in ['CNN']:
        # generate batch data for Convolutional Neural Network
        x_batch, y_batch, aux_batch, _, _ = load_train_data_for_cnn(cfg, x, y, static, scaler_y, lat_index, lon_index, mask_index)
        x_batch = x_batch.squeeze(dim=1)
        x_batch = x_batch.reshape(x_batch.shape[0], x_batch.shape[1] * x_batch.shape[2], x_batch.shape[3], x_batch.shape[4])
        x_batch = torch.cat([x_batch, aux_batch], 1)
        pred = model(x_batch)
    elif cfg['modelname'] in ['ConvLSTM', 'CNNTransformer']:
        # generate batch data for Convolutional LSTM
        x_batch, y_batch, aux_batch, _, _ = load_train_data_for_cnn(cfg, x, y, static, scaler_y, lat_index, lon_index, mask_index)  # same as Convolutional Neural Network
        aux_batch = aux_batch.unsqueeze(1)
        aux_batch = aux_batch.repeat(1, x_batch.shape[1], 1, 1, 1)
        x_batch = x_batch.squeeze(dim=1)
        x_batch = torch.cat([x_batch, aux_batch], 2)
        pred = model(x_batch)
    elif cfg['modelname'] == 'STGNN':
        # generate batch data for GNN
        x_batch, y_batch, aux_batch, _, _ = load_train_data_for_gnn(cfg, x, y, static, scaler_y)
        # x_batch: (batch, seq, num_nodes, nf)
        # aux_batch: (batch, seq, num_nodes, 1)
        # Concatenate aux to x
        x_batch = torch.cat([x_batch, aux_batch], dim=-1)
        pred = model(x_batch, grid_pos)
    return pred, y_batch


def load_valid(model, wait, criterion, x_valid, y_valid, static_valid, scaler_y, lat_index, lon_index, device, cfg, feature_mask=None, grid_pos=None):
    wait += 1
    valid_batches = 0
    MSE_valid_loss = 0
    # NOTE: We used grids-mean NSE as valid metrics.
    # ------------------------------------------------------------------------------------------------------------------------------
    #  validate way for LSTM model
    if cfg["modelname"] in ['LSTM', 'AttnRNN', 'AttnLSTM', 'CNNLSTM', 'STALSTM', 'BiLSTM', 'MLP','TCN']:
        gt_list = [i for i in range(0, x_valid.shape[0] - cfg['seq_len'], cfg["stride"])]
        n = (x_valid.shape[0] - cfg["seq_len"]) // cfg["stride"]
        for i in range(0, n):
            # mask
            x_valid_batch, y_valid_batch, aux_valid_batch, _, _ = load_test_data_for_rnn(cfg, x_valid, y_valid, static_valid, scaler_y, cfg["stride"], i, n)
            aux_valid_batch = aux_valid_batch.unsqueeze(1)
            aux_valid_batch = aux_valid_batch.repeat(1, x_valid_batch.shape[1], 1)
            x_valid_batch = torch.cat([x_valid_batch, aux_valid_batch], 2)
            with torch.no_grad():
                pred_valid = model(x_valid_batch)
            mse_valid_loss = criterion.fit(pred_valid, y_valid_batch)
            MSE_valid_loss += mse_valid_loss.item()
            valid_batches += pred_valid.shape[0]
    if cfg["modelname"] in ['DARNN']:
        gt_list = [i for i in range(0, x_valid.shape[0] - cfg['seq_len'], cfg["stride"])]
        n = (x_valid.shape[0] - cfg["seq_len"]) // cfg["stride"]
        for i in range(0, n):
            # mask
            x_valid_batch, y_valid_batch, aux_valid_batch, _, _ = load_test_data_for_rnn(cfg, x_valid, y_valid, static_valid, scaler_y, cfg["stride"], i, n)
            aux_valid_batch = aux_valid_batch.unsqueeze(1)
            aux_valid_batch = aux_valid_batch.repeat(1, x_valid_batch.shape[1], 1)
            x_valid_batch = torch.cat([x_valid_batch, aux_valid_batch], 2)
            with torch.no_grad():
                # 转换维度
                x_valid_batch = x_valid_batch.permute(1, 0, 2)
                pred_valid = model(x_valid_batch)
            mse_valid_loss = criterion.fit(pred_valid, y_valid_batch)
            MSE_valid_loss += mse_valid_loss.item()
            valid_batches += pred_valid.shape[0]
    if cfg["modelname"] in ['HybridRNN']:
        gt_list = [i for i in range(0, x_valid.shape[0] - cfg['seq_len'], cfg["stride"])]
        n = (x_valid.shape[0] - cfg["seq_len"]) // cfg["stride"]
        for i in range(0, n):
            # mask
            x_valid_batch, y_valid_batch, aux_valid_batch, _, _ = load_test_data_for_rnn(cfg, x_valid, y_valid, static_valid, scaler_y, cfg["stride"], i, n)
            aux_valid_batch = aux_valid_batch.unsqueeze(1)
            aux_valid_batch = aux_valid_batch.repeat(1, x_valid_batch.shape[1], 1)
            x_valid_batch = torch.cat([x_valid_batch, aux_valid_batch], 2)
            with torch.no_grad():
                pred_valid = model(x_valid_batch, feature_mask)
            mse_valid_loss = criterion.fit(pred_valid, y_valid_batch)
            MSE_valid_loss += mse_valid_loss.item()
            valid_batches += pred_valid.shape[0]
    #  validate way for CNN model
    elif cfg['modelname'] in ['CNN']:
        gt_list = [i for i in range(0, x_valid.shape[0] - cfg['seq_len'] - cfg['forecast_time'], cfg["stride"])]
        for i in gt_list:
            x_valid_batch, y_valid_batch, aux_valid_batch, _, _ = load_test_data_for_cnn(cfg, x_valid, y_valid, static_valid, scaler_y, gt_list, lat_index, lon_index, i, cfg["stride"])  # same as Convolutional Neural Network
            # x_valid_temp = torch.cat([x_valid_temp, static_valid_temp], 2)
            x_valid_batch = x_valid_batch.squeeze(1)
            x_valid_batch = x_valid_batch.reshape(x_valid_batch.shape[0], x_valid_batch.shape[1] * x_valid_batch.shape[2], x_valid_batch.shape[3], x_valid_batch.shape[4])
            x_valid_batch = torch.cat([x_valid_batch, aux_valid_batch], axis=1)
            with torch.no_grad():
                pred_valid = model(x_valid_batch)
            mse_valid_loss = criterion.fit(pred_valid, y_valid_batch)
            MSE_valid_loss += mse_valid_loss.item()
            valid_batches += pred_valid.shape[0]
    #  validate way for ConvLSTM model，same as CNN model
    elif cfg['modelname'] in ['ConvLSTM', 'CNNTransformer']:
        gt_list = [i for i in range(0, x_valid.shape[0] - cfg['seq_len'] - cfg['forecast_time'], cfg["stride"])]
        for i in gt_list:
            x_valid_batch, y_valid_batch, aux_valid_batch, _, _ = load_test_data_for_cnn(cfg, x_valid, y_valid, static_valid, scaler_y, gt_list, lat_index, lon_index, i, cfg["stride"])  # same as Convolutional Neural Network
            aux_valid_batch = aux_valid_batch.unsqueeze(1)
            aux_valid_batch = aux_valid_batch.repeat(1, x_valid_batch.shape[1], 1, 1, 1)
            x_valid_batch = torch.cat([x_valid_batch, aux_valid_batch], 2)
            with torch.no_grad():
                pred_valid = model(x_valid_batch)
            mse_valid_loss = criterion.fit(pred_valid, y_valid_batch)
            MSE_valid_loss += mse_valid_loss.item()
            valid_batches += pred_valid.shape[0]
    elif cfg['modelname'] == 'STGNN':
        gt_list = [i for i in range(0, x_valid.shape[0] - cfg['seq_len'] - cfg['forecast_time'], cfg["stride"])]
        for i in gt_list:
            x_valid_batch, y_valid_batch, aux_valid_batch, _, _ = load_test_data_for_gnn(cfg, x_valid, y_valid, static_valid, scaler_y, cfg["stride"], i, len(gt_list))
            if x_valid_batch is None: continue
            x_valid_batch = torch.cat([x_valid_batch, aux_valid_batch], dim=-1)
            with torch.no_grad():
                pred_valid = model(x_valid_batch, grid_pos)
            mse_valid_loss = criterion.fit(pred_valid, y_valid_batch)
            MSE_valid_loss += mse_valid_loss.item()
            valid_batches += pred_valid.shape[0]
    mse_valid_loss = MSE_valid_loss / len(gt_list)
    return mse_valid_loss, wait


def load_model(cfg):
    out_channels = 1
    drop = cfg['dropout']
    device = cfg['device']
    seq_len = cfg['seq_len']
    in_channels, hidden_channels = cfg['input_size'], cfg['hidden_size']

    if cfg['modelname'] in ['MLP']:
        model = MLPModel(in_channels, hidden_channels, out_channels, batch_first=True, seq_len=seq_len, drop=drop, device=device).to(device)
    if cfg['modelname'] in ['LSTM']:
        model = LSTMModel(in_channels, hidden_channels, out_channels, batch_first=True, seq_len=seq_len, drop=drop, device=device).to(device)
    if cfg['modelname'] in ['BiLSTM']:
        model = BiLSTMModel(in_channels, hidden_channels, out_channels, batch_first=True, seq_len=seq_len, drop=drop, device=device).to(device)
    if cfg['modelname'] in ['TCN']:
        model = TCNModel(in_channels, hidden_channels, out_channels, batch_first=True, seq_len=seq_len, drop=drop, device=device).to(device)
    elif cfg['modelname'] in ['CNN']:
        model = CNN(in_channels, hidden_channels, out_channels, drop=drop, cfg=cfg).to(device)
    elif cfg['modelname'] in ['ConvLSTM']:
        model = ConvLSTMModel(in_channels, hidden_channels, out_channels, batch_first=True, num_layers=1, seq_len=seq_len, drop=drop, kernel_size=cfg['kernel_size'], device=device, cfg=cfg).to(device)
    elif cfg['modelname'] in ['CNNTransformer']:
        model = CNNTransformer(in_channels, hidden_channels, out_channels, drop, n_heads=4, cfg=cfg).to(device)
    elif cfg['modelname'] in ['DARNN']:
        model = DARNN(cfg['input_size'], cfg['hidden_size'], out_channels, batch_first=False, seq_len=cfg['seq_len'], drop=drop, device=device).to(device)
    elif cfg['modelname'] in ['AttnRNN']:
        model = AttnRNN(in_channels, hidden_channels, out_channels, drop=drop).to(device)
    elif cfg['modelname'] in ['HybridRNN']:
        model = HybridRNN(in_channels, hidden_channels, out_channels, seq_len, drop=drop).to(device)
    elif cfg['modelname'] in ['AttnLSTM']:
        model = AttnLSTMModel(in_channels, hidden_channels, out_channels, batch_first=True, seq_len=seq_len, drop=drop, device=device).to(device)
    elif cfg['modelname'] in ['CNNLSTM']:
        model = CNNLSTMModel(in_channels, hidden_channels, out_channels, batch_first=True, seq_len=seq_len, drop=drop, device=device).to(device)
    elif cfg['modelname'] in ['STALSTM']:
        model = STALSTMModel(in_channels, hidden_channels, out_channels, batch_first=True, seq_len=seq_len, drop=drop, device=device).to(device)

    elif cfg['modelname'] == 'STGNN':
        # input_dim should include aux features (1 dim)
        # x_train has (forcing + land_surface) features.
        # In load_iter, we concat aux (1 dim).
        # So input_dim = cfg['input_size'] + 1
        model = SpatioTemporalGNN(
            input_dim=cfg['input_size'] + 1,
            hidden_dim=hidden_channels,
            output_dim=out_channels,
            seq_len=seq_len,
            num_gnn_layers=cfg['num_gnn_layers'],
            num_temporal_layers=cfg['num_temporal_layers'],
            dropout=drop,
            k_neighbors=cfg['k_neighbors'],
            use_spatial_proximity=cfg['use_spatial_proximity']
        ).to(device)
    criterion = NaNMSELoss(cfg)
    optim = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])

    return model, optim, criterion


def train(x, y, static, mask, scaler_x, scaler_y, cfg, num_repeat, PATH, out_path, device, num_task=None, valid_split=True, lat=None, lon=None):
    wait = 0
    best = 9999
    patience = cfg['patience']    # Early Stopping 耐心值（连续多少轮验证集损失无改善后停止训练）
    valid_split = cfg['valid_split']    # 是否从训练集中进一步划分验证集   false/true

    lat_index, lon_index, mask_index = None, None, None
    if cfg['modelname'] in ['CNN', 'ConvLSTM', 'CNNTransformer']:
        #  Splice x according to the sphere shape
        lat_index, lon_index = earth_data_transform(cfg, x)
        print('\033[1;31m%s\033[0m' % "Applied Model is {m_n}, we need to transform the data according to the sphere shape".format(m_n=cfg['modelname']))
    if valid_split:
        nt, nf, nlat, nlon = x.shape  # x shape :nt,nf,nlat,nlon 8/15/10/1820
        # Partition validation set and training set
        # . nt, nf, nlat, nlon = x.shape
        # 这行代码将张量 x 的四个维度分别赋值给变量：
        # nt：时间步数（样本数），这里是 8
        # nf：特征数，这里是 15
        # nlat：纬度数，这里是 10
        # nlon：经度数，这里是 1820

        # N = int(nt * cfg['split_ratio'])
        # 这行代码的作用是根据配置文件中的 split_ratio，计算训练集的样本数量。
        # cfg['split_ratio'] 通常是一个小于 1 的小数，比如 0.75，表示 75% 的数据用于训练。
        # N 就是训练集的样本数，剩下的（nt - N）就是验证集的样本数。
        N = int(nt * cfg['split_ratio'])   # 训练集与验证集的划分比例（0.8 表示 80% 训练，20% 验证）-->0.8
        x_valid, y_valid, static_valid = x[N:], y[N:], static
        x, y = x[:N], y[:N]

    # filter Antatctica
    print(f"\033[0;34;40m x_train {x.shape}, y_trian {y.shape}, static_train {static.shape}, mask {mask.shape}\033[0m")  # 蓝色

    # mask see regions
    # Determine the land boundary
    if cfg['modelname'] in ['LSTM', 'DARNN', 'AttnRNN', 'HybridRNN', 'AttnLSTM', 'CNNLSTM', 'STALSTM', 'BiLSTM', 'MLP', 'TCN']:
        # 调用sea_mask_rnn函数，对验证集数据（x_valid, y_valid, static_valid）和掩码（mask）进行处理。
        # cfg：配置参数
        # x_valid：验证集输入数据
        # y_valid：验证集标签
        # static_valid：验证集的静态特征
        # mask：掩码（通常用于屏蔽无效区域，比如海洋、缺失值等）
        if valid_split:
            x_valid, y_valid, static_valid = sea_mask_rnn(cfg, x_valid, y_valid, static_valid, mask)
        x, y, static = sea_mask_rnn(cfg, x, y, static, mask)
    elif cfg['modelname'] in ['CNN', 'ConvLSTM', 'CNNTransformer']:
        x, y, static, mask_index = sea_mask_cnn(cfg, x, y, static, mask)

    if not os.path.exists(os.path.join(cfg['out_path'], 'feature_mask.npy')):
        feature_mask = load_feature_mask(x, y, static, scaler_y, cfg)
    else:
        feature_mask = np.load(os.path.join(cfg['out_path'], 'feature_mask.npy'))

    # 将一个 NumPy 数组 feature_mask 转换为 PyTorch 的张量（Tensor），并移动到指定的设备
    feature_mask = torch.from_numpy(feature_mask).to(device)

    # 调用 load_model(cfg) 函数，返回模型（model）、优化器（optim）和损失函数
    model, optim, criterion = load_model(cfg)

    # Prepare grid_pos for STGNN
    grid_pos = None
    if cfg['modelname'] == 'STGNN':
        if lat is not None and lon is not None:
            # Apply mask to lat/lon
            # mask is (nlat, nlon) with 1s and 0s
            # lat, lon are (nlat, nlon)
            mask_flat = mask.flatten()
            lat_flat = lat.flatten()
            lon_flat = lon.flatten()
            
            valid_lat = lat_flat[mask_flat == 1]
            valid_lon = lon_flat[mask_flat == 1]
            
            # Stack to (num_nodes, 2)
            grid_pos = np.stack([valid_lat, valid_lon], axis=1)
            grid_pos = torch.from_numpy(grid_pos).float().to(device)
            print(f"STGNN Grid Pos Shape: {grid_pos.shape}")
        else:
            raise ValueError("Lat/Lon data missing for STGNN model!")

    # 创建一个余弦退火学习率调度器（Cosine Annealing Learning Rate Scheduler）。
    # 参数说明：
    # optim：要调整学习率的优化器。
    # T_max=cfg['epochs']：一个周期的迭代次数（通常设为总训练轮数）。
    # eta_min=0：最小学习率。
    # last_epoch=-1：从头开始训练。
    # 用途：在训练过程中动态调整学习率，使其按照余弦函数逐渐减小，有助于模型收敛。
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg['epochs'], eta_min=0, last_epoch=-1)

    for epoch in range(1, cfg['epochs'] + 1):
        # train
        MSELoss = 0
        pbar = trange(1, cfg["niter"] + 1, desc=f'Epoch:{epoch} / {cfg["epochs"]}', file=sys.stdout)
        for number_iter in pbar:
            y_pred, y_true = load_iter(model, x, y, static, scaler_y, lat_index, lon_index, mask_index, cfg, feature_mask, grid_pos)
            loss = criterion.fit(y_pred.float(), y_true.float())
            optim.zero_grad()
            loss.backward()
            optim.step()
            MSELoss += loss.item()
            pbar.set_postfix(loss='{:.3f}'.format(MSELoss / number_iter))

        scheduler.step()
        if valid_split:
            if (epoch % 5 == 0):
                val_save_acc, wait = load_valid(model, wait, criterion, x_valid, y_valid, static_valid, scaler_y, lat_index, lon_index, device, cfg, feature_mask, grid_pos)
                # get loss log
                loss_str = '\033[1;31m%s\033[0m' % "Epoch {} valid loss {:.3f} \t\t\t\t\t [wait = {}]".format(epoch, val_save_acc, wait)
                tqdm.tqdm.write(loss_str)
                # save best model by val loss
                # NOTE: save best MSE results get `single_task` better than `multi_tasks`
                #       save best NSE results get `multi_tasks` better than `single_task`
                if val_save_acc < best:
                    # if MSE_valid_loss < best:
                    wait = 0  # release wait
                    best = val_save_acc  # MSE_valid_loss
                    torch.save(model, os.path.join(out_path, cfg['modelname'], str(cfg['forecast_time']), cfg['modelname'] + '_para.pkl'))
                    tqdm.tqdm.write('\033[1;31m%s\033[0m' % f'Save Epoch {epoch} Model With Valid Loss {val_save_acc} , Reset [wait = 0]')
        else:
            # save best model by train loss
            if MSELoss < best:
                best = MSELoss
                wait = 0
                torch.save(model, os.path.join(out_path, cfg['modelname'], str(cfg['forecast_time']), cfg['modelname'] + '_para.pkl'))
                tqdm.tqdm.write('\033[1;31m%s\033[0m' % f'Save Epoch {epoch} Model With Loss {MSELoss / cfg["niter"]}')
        # early stopping
        if wait >= patience:
            return
    return
