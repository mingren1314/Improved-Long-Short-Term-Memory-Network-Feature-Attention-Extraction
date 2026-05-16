"""
检查 mask 文件和数据维度
诊断为什么只有 140 个有效格点
"""
import numpy as np
import os

# 数据路径
data_path = '/root/autodl-tmp/model/drought-agriculture/Datasets/agriculture/'

print("=" * 60)
print("Mask 文件诊断")
print("=" * 60)

# 1. 检查 mask 文件
mask_file = os.path.join(data_path, 'Mask with 0.5 spatial resolution.npy')
if os.path.exists(mask_file):
    mask = np.load(mask_file)
    print(f"\nMask 文件: {mask_file}")
    print(f"  Shape: {mask.shape}")
    print(f"  有效格点 (=1): {np.sum(mask == 1)}")
    print(f"  无效格点 (=0): {np.sum(mask == 0)}")
    print(f"  NaN 格点: {np.sum(np.isnan(mask))}")
    print(f"  唯一值: {np.unique(mask[~np.isnan(mask)])}")
else:
    print(f"\n❌ Mask 文件不存在: {mask_file}")

# 2. 检查经纬度文件
print("\n" + "-" * 60)
print("经纬度文件检查")
print("-" * 60)

lat_file = os.path.join(data_path, 'lat_0.5.npy')
lon_file = os.path.join(data_path, 'lon_0.5.npy')

if os.path.exists(lat_file):
    lat = np.load(lat_file)
    print(f"\nLat 文件: {lat_file}")
    print(f"  Shape: {lat.shape}")
    print(f"  Range: {lat.min():.2f} - {lat.max():.2f}")
else:
    print(f"\n❌ Lat 文件不存在: {lat_file}")

if os.path.exists(lon_file):
    lon = np.load(lon_file)
    print(f"\nLon 文件: {lon_file}")
    print(f"  Shape: {lon.shape}")
    print(f"  Range: {lon.min():.2f} - {lon.max():.2f}")
else:
    print(f"\n❌ Lon 文件不存在: {lon_file}")

# 3. 检查预测数据维度
print("\n" + "-" * 60)
print("预测数据维度检查")
print("-" * 60)

pred_path = os.path.join(data_path, 'test/STALSTM/0/')
obs_file = os.path.join(pred_path, 'observations.npy')
pred_file = os.path.join(pred_path, '_predictions.npy')

if os.path.exists(obs_file):
    obs = np.load(obs_file)
    print(f"\nObservations: {obs_file}")
    print(f"  Shape: {obs.shape}")
    print(f"  有效值数量: {np.sum(~np.isnan(obs))}")
else:
    print(f"\n❌ Observations 文件不存在: {obs_file}")

if os.path.exists(pred_file):
    pred = np.load(pred_file)
    print(f"\nPredictions: {pred_file}")
    print(f"  Shape: {pred.shape}")
else:
    print(f"\n❌ Predictions 文件不存在: {pred_file}")

# 4. 检查指标文件
print("\n" + "-" * 60)
print("指标文件维度检查")
print("-" * 60)

r_file = os.path.join(pred_path, 'r_STALSTM.npy')
if os.path.exists(r_file):
    r = np.load(r_file)
    print(f"\nR 指标文件: {r_file}")
    print(f"  Shape: {r.shape}")
    print(f"  有效值 (非NaN): {np.sum(~np.isnan(r))}")
    print(f"  NaN 数量: {np.sum(np.isnan(r))}")
else:
    print(f"\n❌ R 指标文件不存在: {r_file}")

# 5. 列出目录下所有 .npy 文件
print("\n" + "-" * 60)
print(f"目录下所有 .npy 文件: {data_path}")
print("-" * 60)

for f in os.listdir(data_path):
    if f.endswith('.npy'):
        fpath = os.path.join(data_path, f)
        arr = np.load(fpath)
        print(f"  {f}: shape={arr.shape}")

print("\n" + "=" * 60)
print("诊断完成")
print("=" * 60)
