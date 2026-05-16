from netCDF4 import Dataset
import h5py

# 测试 HDF5 支持
with h5py.File("test.h5", "w") as f:
    f.create_dataset("data", data=[1,2,3])

# 测试 NetCDF 支持
with Dataset("test.nc", "w") as ds:
    ds.createDimension("time", None)
    print("NetCDF 文件创建成功！")