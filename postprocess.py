import os
import warnings
import numpy as np
from tqdm import tqdm
from config import get_args
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils import unbiased_rmse, _rmse, _bias, GetKGE, r2_score, GetPCC, GetNSE, _rv, _fhv, _flv
warnings.filterwarnings("ignore")


def lon_transform(x):
    x_new = np.zeros(x.shape)
    x_new[:, :, :int(x.shape[2] / 2)] = x[:, :, int(x.shape[2] / 2):]
    x_new[:, :, int(x.shape[2] / 2):] = x[:, :, :int(x.shape[2] / 2)]
    return x_new


def postprocess(cfg):
    path = os.path.join(cfg['out_path'], cfg['process'], cfg['modelname'], str(cfg['forecast_time']))
    file_name_mask = 'Mask with {sr} spatial resolution.npy'.format(sr=cfg['spatial_resolution'])
    mask = np.load(os.path.join(cfg['out_path'], file_name_mask))
    # ------------------------------------------------------------------------------------------------------------------------------
    y_pred = np.load(os.path.join(path, '_predictions.npy'))
    y_test = np.load(os.path.join(path, 'observations.npy'))

    # get shape
    nt, nlat, nlon = y_test.shape
    # cal perf
    r2 = np.full((nlat, nlon), np.nan)
    kge = np.full((nlat, nlon), np.nan)
    pcc = np.full((nlat, nlon), np.nan)
    nse = np.full((nlat, nlon), np.nan)
    urmse = np.full((nlat, nlon), np.nan)
    r = np.full((nlat, nlon), np.nan)
    rmse = np.full((nlat, nlon), np.nan)
    bias = np.full((nlat, nlon), np.nan)
    rv = np.full((nlat, nlon), np.nan)
    fhv = np.full((nlat, nlon), np.nan)
    flv = np.full((nlat, nlon), np.nan)
    for i, j in tqdm(np.argwhere(mask)):
        urmse[i, j] = unbiased_rmse(y_test[:, i, j], y_pred[:, i, j])
        kge[i, j] = GetKGE(y_test[:, i, j], y_pred[:, i, j])
        pcc[i, j] = GetPCC(y_test[:, i, j], y_pred[:, i, j])
        nse[i, j] = GetNSE(y_test[:, i, j], y_pred[:, i, j])
        r2[i, j] = r2_score(y_test[:, i, j], y_pred[:, i, j])
        rv[i, j] = _rv(y_test[:, i, j], y_pred[:, i, j])
        fhv[i, j] = _fhv(y_test[:, i, j], y_pred[:, i, j])
        flv[i, j] = _flv(y_test[:, i, j], y_pred[:, i, j])
        r[i, j] = np.corrcoef(y_test[:, i, j], y_pred[:, i, j])[0, 1]
        # 如果当前网格点的r值空，置为0.999
        if np.isnan(r[i, j].sum()) > 0:
            r[i, j] = 0.999
        rmse[i, j] = _rmse(y_test[:, i, j], y_pred[:, i, j])
        bias[i, j] = _bias(y_test[:, i, j], y_pred[:, i, j])
    np.save(os.path.join(path, 'r2_' + cfg['modelname'] + '.npy'), r2)
    np.save(os.path.join(path, 'KGE_' + cfg['modelname'] + '.npy'), kge)
    np.save(os.path.join(path, 'NSE_' + cfg['modelname'] + '.npy'), nse)
    np.save(os.path.join(path, 'rv_' + cfg['modelname'] + '.npy'), rv)
    np.save(os.path.join(path, 'fhv_' + cfg['modelname'] + '.npy'), fhv)
    np.save(os.path.join(path, 'flv_' + cfg['modelname'] + '.npy'), flv)
    np.save(os.path.join(path, 'r_' + cfg['modelname'] + '.npy'), r)
    np.save(os.path.join(path, 'rmse_' + cfg['modelname'] + '.npy'), rmse)
    np.save(os.path.join(path, 'bias_' + cfg['modelname'] + '.npy'), bias)
    np.save(os.path.join(path, 'urmse_' + cfg['modelname'] + '.npy'), urmse)
    print('postprocess ove, please go on')


if __name__ == '__main__':
    cfg = get_args()
    postprocess(cfg)
