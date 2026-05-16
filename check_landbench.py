"""
检查 LandBench 原始数据的分辨率和内容
"""
import numpy as np
import os

landbench_path = '/root/autodl-tmp/model/drought-agriculture/Datasets/LandBench/'

print("=" * 60)
print("检查 LandBench 原始数据")
print("=" * 60)

def explore_dir(path, depth=0, max_depth=3):
    """递归探索目录结构"""
    if depth > max_depth:
        return
    
    indent = "  " * depth
    try:
        items = os.listdir(path)
        for item in items[:20]:  # 限制显示数量
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                print(f"{indent}📁 {item}/")
                explore_dir(item_path, depth + 1, max_depth)
            else:
                # 获取文件大小
                size = os.path.getsize(item_path)
                size_str = f"{size / 1024 / 1024:.1f}MB" if size > 1024*1024 else f"{size / 1024:.1f}KB"
                print(f"{indent}📄 {item} ({size_str})")
                
                # 如果是 .npy 或 .nc 文件，尝试读取 shape
                if item.endswith('.npy'):
                    try:
                        arr = np.load(item_path, allow_pickle=True)
                        if hasattr(arr, 'shape'):
                            print(f"{indent}   → shape: {arr.shape}")
                    except:
                        pass
                elif item.endswith('.nc'):
                    try:
                        import xarray as xr
                        ds = xr.open_dataset(item_path)
                        print(f"{indent}   → dims: {dict(ds.dims)}")
                        print(f"{indent}   → vars: {list(ds.data_vars)[:5]}")
                        ds.close()
                    except:
                        print(f"{indent}   → (需要 xarray 读取)")
        
        if len(items) > 20:
            print(f"{indent}... 共 {len(items)} 个项目")
    except Exception as e:
        print(f"{indent}❌ 无法读取: {e}")

explore_dir(landbench_path)

# 特别检查是否有高分辨率数据
print("\n" + "=" * 60)
print("搜索可能的高分辨率数据文件")
print("=" * 60)

for root, dirs, files in os.walk(landbench_path):
    for f in files:
        if any(x in f.lower() for x in ['0.1', '0.25', '0.5', 'lat', 'lon', 'mask']):
            fpath = os.path.join(root, f)
            print(f"\n找到: {fpath}")
            if f.endswith('.npy'):
                try:
                    arr = np.load(fpath, allow_pickle=True)
                    if hasattr(arr, 'shape'):
                        print(f"  shape: {arr.shape}")
                        if 'lat' in f.lower() and len(arr.shape) == 1:
                            print(f"  范围: {arr.min():.2f} - {arr.max():.2f}")
                            print(f"  分辨率: {abs(arr[1]-arr[0]) if len(arr)>1 else 'N/A'}°")
                except Exception as e:
                    print(f"  读取失败: {e}")

print("\n" + "=" * 60)
