"""
检查原始 1° 数据的实际覆盖范围
"""
import numpy as np
import os

data_path = '/root/autodl-tmp/model/drought-agriculture/Datasets/agriculture/'

print("=" * 60)
print("检查原始 1° 数据")
print("=" * 60)

# 检查 1° 的 lat/lon
lat_file = os.path.join(data_path, 'lat_1.npy')
lon_file = os.path.join(data_path, 'lon_1.npy')

if os.path.exists(lat_file):
    lat = np.load(lat_file)
    print(f"\n1° Lat: shape={lat.shape}")
    print(f"  范围: {lat.min():.1f}° - {lat.max():.1f}°")
    print(f"  值: {lat}")

if os.path.exists(lon_file):
    lon = np.load(lon_file)
    print(f"\n1° Lon: shape={lon.shape}")
    print(f"  范围: {lon.min():.1f}° - {lon.max():.1f}°")
    print(f"  值: {lon}")

# 检查 1° mask
mask_file = os.path.join(data_path, 'Mask with 1 spatial resolution.npy')
if os.path.exists(mask_file):
    mask = np.load(mask_file)
    print(f"\n1° Mask: shape={mask.shape}")
    print(f"  有效格点: {np.sum(mask == 1)}")

# 检查是否有 LandBench 原始数据
print("\n" + "-" * 60)
print("检查 LandBench 原始数据路径")
print("-" * 60)

possible_paths = [
    '/root/autodl-tmp/datasets/LandBench/',
    '/root/autodl-tmp/LandBench/',
    '/root/autodl-tmp/model/drought-agriculture/Datasets/LandBench/',
]

for p in possible_paths:
    if os.path.exists(p):
        print(f"\n✅ 找到: {p}")
        # 列出内容
        for item in os.listdir(p)[:10]:
            print(f"    {item}")
        if len(os.listdir(p)) > 10:
            print(f"    ... 共 {len(os.listdir(p))} 个文件/文件夹")
    else:
        print(f"❌ 不存在: {p}")

# 检查 config 中的 data_path
print("\n" + "-" * 60)
print("检查 config 中的原始数据路径")
print("-" * 60)

try:
    import sys
    sys.path.insert(0, '/root/autodl-tmp/model/drought-agriculture/')
    from config import get_args
    config = get_args()
    print(f"data_path: {config.get('data_path', 'N/A')}")
    if os.path.exists(config.get('data_path', '')):
        print("  ✅ 路径存在")
        for item in os.listdir(config['data_path'])[:10]:
            print(f"    {item}")
    else:
        print("  ❌ 路径不存在")
except Exception as e:
    print(f"无法加载 config: {e}")

print("\n" + "=" * 60)
print("结论")
print("=" * 60)
print("""
如果你有 LandBench 原始数据（通常是 0.1° 或更高分辨率），
可以重新处理生成真正的 0.5° 数据。

如果原始数据本身就是 1°（8×15），那么插值到 0.5° 
只会得到 (16×30)，不会增加实际信息量。
""")
