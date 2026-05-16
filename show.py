import os
import torch
import random
import warnings
import numpy as np
import xarray as xr
import pandas as pd
import seaborn as sns
from eval import test
from train import train
from config import get_args
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


'''
 figure 1
'''

time = 1000
cfg = get_args()
path = cfg['out_path']
x_train_shape = np.load(os.path.join(path, 'x_train_norm_shape.npy'), mmap_mode='r')
x_train = np.memmap(os.path.join(path, 'x_train_norm.npy'), dtype=cfg['data_type'], mode='r+', shape=(x_train_shape[0], x_train_shape[1], x_train_shape[2], x_train_shape[3]))
x_test_shape = np.load(os.path.join(path, 'x_test_norm_shape.npy'), mmap_mode='r')
x_test = np.memmap(os.path.join(path, 'x_test_norm.npy'), dtype=cfg['data_type'], mode='r+', shape=(x_test_shape[0], x_test_shape[1], x_test_shape[2], x_test_shape[3]))
y_train = np.load(os.path.join(path, 'y_train_norm.npy'), mmap_mode='r')
y_test = np.load(os.path.join(path, 'y_test_norm.npy'), mmap_mode='r')
static = np.load(os.path.join(path, 'static_norm.npy'))
file_name_mask = 'Mask with {sr} spatial resolution.npy'.format(sr=cfg['spatial_resolution'])
mask = np.load(os.path.join(path, file_name_mask))
static = np.load(os.path.join(path, "static_norm.npy"))
static = np.expand_dims(static, 0).repeat(x_train_shape[0], axis=0)

msk = np.expand_dims(mask, 0).repeat(x_train_shape[0], axis=0)
msk = np.expand_dims(msk, -1).repeat(x_train.shape[-1] + static.shape[-1] + y_train.shape[-1], axis=-1)
mask = mask.astype(np.bool_)
# data = np.concatenate([x_train, static, y_train], axis=-1)[:, mask]
data = np.concatenate([x_train, static, y_train], axis=-1).reshape(x_train_shape[0] * x_train_shape[1] * x_train_shape[2], -1)
column = ['t2m', 'u', 'v', 'pre', 'ssr', 'spec', 'ssrd', 'strd', 'stl1', 'e', 'swc', 'smci']
data = pd.DataFrame(data, columns=column)
matrix = data.corr()
sns.heatmap(matrix, annot=True, cmap='RdBu_r')
plt.show()
