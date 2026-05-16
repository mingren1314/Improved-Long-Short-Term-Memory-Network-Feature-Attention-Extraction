"""
将 1° 分辨率数据插值到 0.5° 分辨率
"""
import os
import numpy as np
from scipy.ndimage import zoom

# 路径配置
# Windows 路径
# input_path = 'E:/end/all/model/drought-agriculture/Datasets/agriculture/'
# Linux (AutoDL) 路径
input_path = '/root/autodl-tmp/model/drought-agriculture/Datasets/agriculture/'
output_path = input_path  # 保存到同一目录

# 插值倍数：1° -> 0.5° 需要放大2倍
scale_factor = 2

def interpolate_data(data, scale_factor):
    """
    对数据进行双线性插值
    data shape: (time, height, width, channels) 或 (height, width, channels) 或 (height, width)
    """
    if len(data.shape) == 4:
        # (time, height, width, channels)
        zoom_factors = (1, scale_factor, scale_factor, 1)
    elif len(data.shape) == 3:
        # (height, width, channels)
        zoom_factors = (scale_factor, scale_factor, 1)
    elif len(data.shape) == 2:
        # (height, width)
        zoom_factors = (scale_factor, scale_factor)
    else:
        raise ValueError(f"Unsupported data shape: {data.shape}")
    
    return zoom(data, zoom_factors, order=1)  # order=1 表示双线性插值


def main():
    years = range(2015, 2021)
    
    # 1. 处理 forcing 数据
    print("处理 forcing 数据...")
    for year in years:
        input_file = f'ERA5-Land_forcing 1 spatial resolution {year}.npy'
        output_file = f'ERA5-Land_forcing 0.5 spatial resolution {year}.npy'
        
        if os.path.exists(os.path.join(input_path, input_file)):
            print(f"  处理 {year} 年 forcing 数据...")
            data = np.load(os.path.join(input_path, input_file))
            print(f"    原始形状: {data.shape}")
            data_interp = interpolate_data(data, scale_factor)
            print(f"    插值后形状: {data_interp.shape}")
            np.save(os.path.join(output_path, output_file), data_interp.astype(np.float32))
            print(f"    已保存: {output_file}")
    
    # 2. 处理 land_surface 数据
    print("\n处理 land_surface 数据...")
    for year in years:
        input_file = f'ERA5-Land_land_surface 1 spatial resolution {year}.npy'
        output_file = f'ERA5-Land_land_surface 0.5 spatial resolution {year}.npy'
        
        if os.path.exists(os.path.join(input_path, input_file)):
            print(f"  处理 {year} 年 land_surface 数据...")
            data = np.load(os.path.join(input_path, input_file))
            print(f"    原始形状: {data.shape}")
            data_interp = interpolate_data(data, scale_factor)
            print(f"    插值后形状: {data_interp.shape}")
            np.save(os.path.join(output_path, output_file), data_interp.astype(np.float32))
            print(f"    已保存: {output_file}")
    
    # 3. 处理 label 数据
    print("\n处理 label 数据...")
    for year in years:
        input_file = f'ERA5_LAND_label_1_{year}.npy'
        output_file = f'ERA5_LAND_label_0.5_{year}.npy'
        
        if os.path.exists(os.path.join(input_path, input_file)):
            print(f"  处理 {year} 年 label 数据...")
            data = np.load(os.path.join(input_path, input_file))
            print(f"    原始形状: {data.shape}")
            data_interp = interpolate_data(data, scale_factor)
            print(f"    插值后形状: {data_interp.shape}")
            np.save(os.path.join(output_path, output_file), data_interp.astype(np.float32))
            print(f"    已保存: {output_file}")
    
    # 4. 处理 Mask
    print("\n处理 Mask...")
    mask_input = 'Mask with 1 spatial resolution.npy'
    mask_output = 'Mask with 0.5 spatial resolution.npy'
    if os.path.exists(os.path.join(input_path, mask_input)):
        mask = np.load(os.path.join(input_path, mask_input))
        print(f"  原始形状: {mask.shape}")
        # Mask 使用最近邻插值保持 0/1 值
        mask_interp = zoom(mask, scale_factor, order=0)
        print(f"  插值后形状: {mask_interp.shape}")
        np.save(os.path.join(output_path, mask_output), mask_interp)
        print(f"  已保存: {mask_output}")
    
    # 5. 处理 lat/lon
    print("\n处理 lat/lon...")
    lat_input = 'lat_1.npy'
    lon_input = 'lon_1.npy'
    if os.path.exists(os.path.join(input_path, lat_input)):
        lat = np.load(os.path.join(input_path, lat_input))
        lon = np.load(os.path.join(input_path, lon_input))
        
        # 生成新的 0.5° 经纬度网格
        lat_new = np.linspace(lat.min(), lat.max(), len(lat) * scale_factor)
        lon_new = np.linspace(lon.min(), lon.max(), len(lon) * scale_factor)
        
        np.save(os.path.join(output_path, 'lat_0.5.npy'), lat_new.astype(np.float32))
        np.save(os.path.join(output_path, 'lon_0.5.npy'), lon_new.astype(np.float32))
        print(f"  lat: {lat.shape} -> {lat_new.shape}")
        print(f"  lon: {lon.shape} -> {lon_new.shape}")
        print("  已保存: lat_0.5.npy, lon_0.5.npy")
    
    # 6. 处理 static 数据
    print("\n处理 static 数据...")
    static_input = 'static_norm.npy'
    if os.path.exists(os.path.join(input_path, static_input)):
        static = np.load(os.path.join(input_path, static_input))
        print(f"  原始形状: {static.shape}")
        static_interp = interpolate_data(static, scale_factor)
        print(f"  插值后形状: {static_interp.shape}")
        np.save(os.path.join(output_path, 'static_norm_0.5.npy'), static_interp.astype(np.float32))
        print("  已保存: static_norm_0.5.npy")

    print("\n✅ 所有数据插值完成！")
    print(f"数据保存在: {output_path}")


if __name__ == '__main__':
    main()
