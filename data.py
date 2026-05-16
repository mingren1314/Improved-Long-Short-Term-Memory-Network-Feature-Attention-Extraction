import os
import sys
import geopandas
import numpy as np
import xarray as xr
from tqdm import tqdm, trange
from shapely.geometry import mapping
from scipy.interpolate import griddata

import rioxarray  # 必须导入以激活 xarray 的 .rio 扩展

class Dataset:
    def __init__(self, cfg: dict):
        self.sf = cfg['sf']
        self.label = cfg["label"]
        self.seq_len = cfg["seq_len"]
        self.test_year = cfg["test_year"]
        self.normalize = cfg["normalize"]
        self.static_list = cfg["static_list"]
        self.forcing_list = cfg["forcing_list"]
        self.selected_year = cfg["selected_year"]
        self.s_resolution = cfg["spatial_resolution"]
        self.land_surface_list = cfg["land_surface_list"]
        self.s_namedict = {
            "2m_temperature": "t2m",
            "10m_u_component_of_wind": "u10",
            "10m_v_component_of_wind": "v10",
            "precipitation": "tp",
            "snow_depth_water_equivalent": "sd",
            "surface_sensible_heat_flux": "sshf",
            "soil_temperature_level_1": "stl1",
            "soil_temperature_level_2": "stl2",
            "soil_temperature_level_3": "stl3",
            "soil_temperature_level_4": "stl4",
            "surface_pressure": "sp",
            "specific_humidity": "Q",
            "surface_thermal_radiation_downwards_w_m2": "strd",
            "surface_solar_radiation_downwards_w_m2": "ssrd",
            "total_runoff": "ro",
            "volumetric_soil_water_layer_1": "swvl1",
            "volumetric_soil_water_layer_2": "swvl2",
            "volumetric_soil_water_layer_3": "swvl3",
            "volumetric_soil_water_layer_4": "swvl4",
            "clay_0-5cm_mean": "Band1",
            "sand_0-5cm_mean": "Band1",
            "silt_0-5cm_mean": "Band1",
            "soil_water_capacity": "SC",
            "total_evaporation": "e"
        }

    def fit(self, cfg):
        print("data.py--------fit.........")
        out_path = cfg['out_path']
        end_year = self.selected_year[1]
        begin_year = self.selected_year[0]
        data_path = os.path.join(cfg['data_path'], str(cfg['spatial_resolution']))

        # ? Loading forcing data
        day_list = []
        forcing_list = []
        print(f"\033[0;32;40m Loading forcing data from {begin_year} to {end_year} \033[0m")  # 绿色
        print(f"\033[0;32;40m Forcing data {cfg['forcing_list']} \033[0m")  # 绿色
        for year in range(begin_year, end_year + 1):
            file_name_forcing = "ERA5-Land_forcing {sr} spatial resolution {year}.npy".format(sr=self.s_resolution, year=year)
            if not os.path.exists(os.path.join(out_path, file_name_forcing)):
                latitude, longitude = self._load_forcing_or_land_surface(data_path, self.forcing_list, self.s_resolution, self.s_namedict, year, cfg, out_path, file_name_forcing, category="atmosphere")
            data = np.load(os.path.join(out_path, file_name_forcing), mmap_mode="r")
            forcing_list.append(data)
            day_list.append(data.shape[0])
            print(f"\033[0;37;40m Loading {year} forcing data\033[0m")  # 灰色
        lat_file_name = f"lat_{self.s_resolution}.npy"
        lon_file_name = f"lon_{self.s_resolution}.npy"
        if not os.path.exists(os.path.join(out_path, lat_file_name)):
            np.save(os.path.join(out_path, lat_file_name), latitude)
            np.save(os.path.join(out_path, lon_file_name), longitude)
        else:
            latitude = np.load(os.path.join(out_path, lat_file_name))
            longitude = np.load(os.path.join(out_path, lon_file_name))
        if cfg["memmap"]:
            forcing = np.memmap(os.path.join(out_path, 'forcing_memmap.npy'), dtype=cfg["data_type"], mode="w+", shape=(np.sum(day_list, axis=0), forcing_list[0].shape[1], forcing_list[0].shape[2], forcing_list[0].shape[3]))
            start = 0
            end = 0
            for i in range(len(forcing_list)):
                if i == 0:
                    start = 0
                    end = day_list[i]
                else:
                    start = end
                    end = start + day_list[i]
                forcing[start:end] = forcing_list[i]
            forcing.flush()
            del forcing
        forcing = np.memmap(os.path.join(out_path, "forcing_memmap.npy"), dtype=cfg["data_type"], mode="r", shape=(np.sum(day_list, axis=0), forcing_list[0].shape[1], forcing_list[0].shape[2], forcing_list[0].shape[3]))
        print("\033[0;33;40m Loading forcing data done! \033[0m")  # 黄色

        # ? Loading land surface data
        land_surface_list = []
        for year in range(begin_year, end_year + 1):
            file_name_land_surface = "ERA5-Land_land_surface {sr} spatial resolution {year}.npy".format(sr=self.s_resolution, year=year)
            if not os.path.exists(os.path.join(out_path, file_name_land_surface)):
                latitude, longitude = self._load_forcing_or_land_surface(data_path, self.land_surface_list, self.s_resolution, self.s_namedict, year, cfg, out_path, file_name_land_surface, category="land_surface")
            data = np.load(os.path.join(out_path, file_name_land_surface), mmap_mode="r")
            land_surface_list.append(data)
            print(f"\033[0;37;40m Loading {year} land_surface data\033[0m")  # 灰色
        if cfg["memmap"]:
            land_surface = np.memmap(os.path.join(out_path, "land_surface_memmap.npy"), dtype=cfg["data_type"], mode="w+", shape=(np.sum(day_list, axis=0), land_surface_list[0].shape[1], land_surface_list[0].shape[2], land_surface_list[0].shape[3]))

            for i in range(len(land_surface_list)):
                if i == 0:
                    start = 0
                    end = day_list[i]
                else:
                    start = end
                    end = start + day_list[i]
                land_surface[start:end] = land_surface_list[i]
            land_surface.flush()
            del land_surface
        land_surface = np.memmap(os.path.join(out_path, "land_surface_memmap.npy"), dtype=cfg["data_type"], mode="r", shape=(np.sum(day_list, axis=0), land_surface_list[0].shape[1], land_surface_list[0].shape[2], land_surface_list[0].shape[3]))
        print("\033[0;33;40m Loading land_surface data done! \033[0m")  # 黄色

        # ? Loading label
        label = []
        for year in range(begin_year, end_year + 1):
            file_name_label = "ERA5_LAND_label_{sr}_{year}.npy".format(sr=self.s_resolution, year=year)
            if not os.path.exists(os.path.join(out_path, file_name_label)):
                latitude, longitude = self._load_forcing_or_land_surface(data_path, self.label, self.s_resolution, self.s_namedict, year, cfg, out_path, file_name_label, category="land_surface",)
            data = np.load(os.path.join(out_path, file_name_label), mmap_mode="r")
            label.append(data)
            print(f"\033[0;37;40m Loading {year} label data\033[0m")  # 灰色
        label = np.concatenate(label, axis=0).astype(np.float32)
        print("\033[0;33;40m Loading label data done! \033[0m")  # 黄色

        # ? process label
        min_val = np.nanmin(label, axis=(0), keepdims=True)
        max_val = np.nanmax(label, axis=(0), keepdims=True)
        theta = np.nanmean(label, axis=(0), keepdims=True)
        label = (label - min_val) / (max_val - min_val) if cfg['process'] in ['smci'] else (label - theta) / theta

        file_name_mask = "Mask with {sr} spatial resolution.npy".format(sr=self.s_resolution)
        mask = np.ones(label[0, :, :, 0].shape)
        mask[np.isnan(label[0, :, :, 0])] = 0

        if len(cfg['static_list']) > 0 and not os.path.exists(os.path.join(out_path, "static_norm.npy")):
            static = []
            for i in range(len(self.static_list)):
                file_static = os.path.join(data_path, 'constants', self.static_list[i] + ".nc")
                with xr.open_dataset(file_static) as f:
                    f.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
                    f.rio.write_crs("EPSG:4326", inplace=True)
                    geodf = geopandas.read_file(self.sf)
                    clipped = f.rio.clip(geodf.geometry.apply(mapping), geodf.crs)
                    static_data = clipped[self.s_namedict[self.static_list[i]]].to_numpy()
                    static_data = self._interp(static_data, mask)
                static.append(static_data)
            static = np.stack(static, axis=-1)
            if cfg["data_type"] == "float32":
                static = static.astype(np.float32)
            static = self._spatial_normalize(static)
            np.save(os.path.join(out_path, "static_norm.npy"), static)
        elif len(cfg['static_list']) > 0:
            static = np.load(os.path.join(out_path, "static_norm.npy"))

        self.time_length_f, self.nlat_f, self.nlon_f, self.num_features_f = forcing.shape
        self.time_length_l, self.nlat_l, self.nlon_l, self.num_features_l = land_surface.shape

        # TODO 修改天数
        days = 365 if cfg['time_resolution'] in ['1D'] else 12
        N = days * len(self.test_year) + cfg["seq_len"] + cfg["forecast_time"]
        print("\033[0;32;40m The following process are all time-consuming, especially for the high-resolution data. Please wait patiently \033[0m")  # 绿色
        if not os.path.exists(os.path.join(out_path, "x_train.npy")):
            x_train = np.memmap(os.path.join(out_path, "x_train.npy"), dtype=cfg["data_type"], mode="w+", shape=(self.time_length_f - N, self.nlat_f, self.nlon_f, self.num_features_f + self.num_features_l,))
            x_train[:, :, :, : self.num_features_f] = forcing[: self.time_length_f - N]
            x_train[:, :, :, self.num_features_f:] = land_surface[: self.time_length_f - N]
            x_train.flush()
            del x_train
        x_train = np.memmap(os.path.join(out_path, "x_train.npy"), dtype=cfg["data_type"], mode="r+", shape=(self.time_length_f - N, self.nlat_f, self.nlon_f, self.num_features_f + self.num_features_l))
        # ------------------------------------------------------------------------------------------------------------------------------
        y_train = label[: self.time_length_f - N]
        np.save(os.path.join(out_path, "y_train.npy"), y_train)

        # Create a test dataset ；The default for the test dataset is only 2020
        if not os.path.exists(os.path.join(out_path, "x_test.npy")):
            x_test = np.memmap(os.path.join(out_path, "x_test.npy"), dtype=cfg["data_type"], mode="w+", shape=(N, self.nlat_f, self.nlon_f, self.num_features_f + self.num_features_l))
            x_test[:, :, :, : self.num_features_f], x_test[:, :, :, self.num_features_f:] = forcing[self.time_length_f - N:], land_surface[self.time_length_f - N:]
            x_test.flush()
            del x_test
        x_test = np.memmap(os.path.join(out_path, "x_test.npy"), dtype=cfg["data_type"], mode="r+", shape=(N, self.nlat_f, self.nlon_f, self.num_features_f + self.num_features_l))
        y_test = label[self.time_length_f - N:]
        del forcing, label, land_surface

        if self.normalize:
            if not os.path.exists(os.path.join(out_path, "y_train_norm.npy")):
                if cfg["normalize_type"] in ["region"]:
                    scaler_x = np.memmap(os.path.join(out_path, "scaler_x.npy"), dtype=cfg["data_type"], mode="w+", shape=(2, x_train.shape[1], x_train.shape[2], x_train.shape[3]))
                    scaler_y = np.memmap(os.path.join(out_path, "scaler_y.npy"), dtype=cfg["data_type"], mode="w+", shape=(2, y_train.shape[1], y_train.shape[2], y_train.shape[3]))
                    for i in range(x_train.shape[2]):
                        out_x, out_y = self._get_minmax_scaler(x_train[:, :, i, :], y_train[:, :, i, :], scaler_x[:, :, i, :], scaler_y[:, :, i, :], "region")
                        scaler_x[:, :, i, :], scaler_y[:, :, i, :] = out_x, out_y
                    scaler_x.flush()
                    scaler_y.flush()
                    del scaler_x
                    del scaler_y
                    scaler_x = np.memmap(os.path.join(out_path, "scaler_x.npy"), dtype=cfg["data_type"], mode="r", shape=(2, x_train.shape[1], x_train.shape[2], x_train.shape[3]))
                    scaler_y = np.memmap(os.path.join(out_path, "scaler_y.npy"), dtype=cfg["data_type"], mode="r", shape=(2, y_train.shape[1], y_train.shape[2], y_train.shape[3]))

                    for i in range(x_train.shape[1]):
                        out_x_train = self._normalize(x_train[:, i, :, :], "input", scaler_x[:, i, :, :], "minmax")
                        x_train[:, i, :, :] = out_x_train
                        out_y_train = self._normalize(y_train[:, i, :, :], "output", scaler_y[:, i, :, :], "minmax",)
                        y_train[:, i, :, :] = out_y_train
                        out_x_test = self._normalize(x_test[:, i, :, :], "input", scaler_x[:, i, :, :], "minmax")
                        x_test[:, i, :, :] = out_x_test

                elif cfg["normalize_type"] in ["global"]:
                    scaler_x = np.memmap(os.path.join(out_path, "scaler_x.npy"), dtype=cfg["data_type"], mode="w+", shape=(2, x_train.shape[3]))
                    scaler_y = np.memmap(os.path.join(out_path, "scaler_y.npy"), dtype=cfg["data_type"], mode="w+", shape=(2, y_train.shape[3]))
                    scaler_y_t = {}
                    for i in range(x_train.shape[3]):
                        out_x, _ = self._get_minmax_scaler(x_train[:, :, :, i], x_train[:, :, :, i], scaler_x[:, i], scaler_y_t, "global")
                        scaler_x[:, i] = np.squeeze(out_x)
                    for i in range(y_train.shape[3]):
                        out_y, _ = self._get_minmax_scaler(y_train[:, :, :, i], y_train[:, :, :, i], scaler_y[:, i], scaler_y_t, "global")
                        scaler_y[:, i] = np.squeeze(out_y)
                    scaler_x.flush()
                    scaler_y.flush()
                    del scaler_x
                    del scaler_y
                    scaler_x = np.memmap(os.path.join(out_path, "scaler_x.npy"), dtype=cfg["data_type"], mode="r", shape=(2, x_train.shape[3]))
                    scaler_y = np.memmap(os.path.join(out_path, "scaler_y.npy"), dtype=cfg["data_type"], mode="r", shape=(2, y_train.shape[3]))
                    print("processed: x_train shape is: {x_s}, y_train shape is: {y_s}, x_test shape is: {x_ts_s}".format(x_s=x_train.shape, y_s=y_train.shape, x_ts_s=x_test.shape))

                    for i in range(y_train.shape[3]):
                        scaler_y_in = np.expand_dims(scaler_y[:, i], axis=1)
                        scaler_y_in = np.expand_dims(scaler_y_in, axis=2)
                        scaler_y_in = np.repeat(scaler_y_in, y_train.shape[1], axis=1)
                        scaler_y_in = np.repeat(scaler_y_in, y_train.shape[2], axis=2)
                        out_y_train = self._normalize(y_train[:, :, :, i], "output", scaler_y_in, "minmax")
                        y_train[:, :, :, i] = out_y_train

                    for i in range(x_train.shape[3]):
                        scaler_x_in = np.expand_dims(scaler_x[:, i], axis=1)
                        scaler_x_in = np.expand_dims(scaler_x_in, axis=2)
                        scaler_x_in = np.repeat(scaler_x_in, x_train.shape[1], axis=1)
                        scaler_x_in = np.repeat(scaler_x_in, x_train.shape[2], axis=2)
                        out_x_train = self._normalize(x_train[:, :, :, i], "input", scaler_x_in, "minmax")
                        x_train[:, :, :, i] = out_x_train
                        out_x_test = self._normalize(x_test[:, :, :, i], "input", scaler_x_in, "minmax")
                        x_test[:, :, :, i] = out_x_test

        np.save(os.path.join(out_path, "x_train_norm_shape.npy"), x_train.shape)
        np.save(os.path.join(out_path, "x_test_norm_shape.npy"), x_test.shape)
        x_train_norm = np.memmap(os.path.join(out_path, "x_train_norm.npy"), dtype=cfg["data_type"], mode="w+", shape=(x_train.shape))
        x_train_norm[:] = x_train[:]

        x_train_norm.flush()
        del x_train_norm

        x_test_norm = np.memmap(os.path.join(out_path, "x_test_norm.npy"), dtype=cfg["data_type"], mode="w+", shape=(x_test.shape))
        x_test_norm[:] = x_test[:]
        x_test_norm.flush()
        del x_test_norm

        y_train_norm = y_train
        y_test_norm = y_test
        np.save(os.path.join(out_path, "y_test_norm.npy"), y_test_norm)
        np.save(os.path.join(out_path, "y_train_norm.npy"), y_train_norm)
        np.save(os.path.join(out_path, file_name_mask), np.squeeze(mask))

        x_train_norm = np.memmap(os.path.join(out_path, "x_train_norm.npy"), dtype=cfg["data_type"], mode="r", shape=(x_train.shape))
        x_test_norm = np.memmap(os.path.join(out_path, "x_test_norm.npy"), dtype=cfg["data_type"], mode="r", shape=(x_test.shape),)
        return (x_train_norm, y_train_norm, x_test_norm, y_test_norm, static, latitude, longitude, mask)

    # Convert nc data into npy data
    def _load_forcing_or_land_surface(self, root, _list, s_resolution, s_namedict, year, cfg, PATH, file_name, category):
        tmp = []
        time = None
        for i in range(len(_list)):
            file_ = os.path.join(root, category, str(year), _list[i] + ".nc" )
            print(f"尝试打开文件：{file_}")
            # file_str = str(file_)
            with xr.open_dataset(file_) as f:
                print("dir(f): ",dir(f))
                print("f.dims: ",f.dims)
                print("f.coords: ",f.coords)
                f.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
                f.rio.write_crs("EPSG:4326", inplace=True)
                geodf = geopandas.read_file(self.sf)
                clipped = f.rio.clip(geodf.geometry.apply(mapping), geodf.crs)
                if time is None:
                    time = clipped.time
                if _list[i] in ['total_evaporation']:
                    clipped = clipped.swap_dims({'day': 'time'})  # 将维度名称从latitude换成lat
                    del clipped['day']
                    clipped['time'] = time

                # TODO 修改时间分辨率
                # clipped = clipped.resample(time='1M').mean()
                tmp.append(clipped[s_namedict[_list[i]]])
                lat, lon = np.array(clipped.latitude), np.array(clipped.longitude)
        tmp = np.stack(tmp, axis=-1)
        np.save(os.path.join(PATH, file_name), tmp)
        return lat, lon

    # ------------------------------------------------------------------------------------------------------------------------------
    def _normalize(self, feature, variable, scaler, scaler_type):
        if scaler_type == "standard":
            if variable == "input":
                feature = (feature - np.array(scaler[0])) / np.array(scaler[1])
            elif variable == "output":
                feature = (feature - np.array(scaler[0])) / np.array(scaler[1])
            else:
                raise RuntimeError(f"Unknown variable type {variable}")
        elif scaler_type == "minmax":
            if variable == "input":
                feature = (feature - np.array(scaler[0])) / (np.array(scaler[1]) - np.array(scaler[0]))  # ?
            elif variable == "output":
                feature = (feature - np.array(scaler[0])) / (np.array(scaler[1]) - np.array(scaler[0]))
            else:
                raise RuntimeError(f"Unknown variable type {variable}")
        return feature

    # ------------------------------------------------------------------------------------------------------------------------------
    # Reverse normalization, which is used in prediction
    def reverse_normalize(self, feature, variable: str, scaler, scaler_method: str, is_multivars: int) -> np.ndarray:
        """reverse normalized features using pre-computed statistics"""
        if variable == "input":
            a, b = np.array(scaler[0]), np.array(scaler[1])
        elif variable == "output":
            c, d = np.array(scaler[0]), np.array(scaler[1])
        if is_multivars != -1:
            a, b = a[:, :, is_multivars: is_multivars + 1], b[:, :, is_multivars: is_multivars + 1]
            c, d = c[:, :, is_multivars: is_multivars + 1], d[:, :, is_multivars: is_multivars + 1]
        if variable == "input":
            if scaler_method == "standard":
                feature = feature * b + a
            else:
                feature = feature * (b - a) + a
        elif variable == "output":
            if scaler_method == "standard":
                feature = feature * d + c  # ?
            else:
                feature = feature * (d - c) + c
        else:
            raise RuntimeError(f"Unknown variable type {variable}")
        return feature

    # ------------------------------------------------------------------------------------------------------------------------------
    def _get_minmax_scaler(self, X, y, scaler_x, scaler_y, type: str) -> dict:
        if type == "global":
            scaler_x[0] = np.squeeze(np.nanmin(X, axis=(0, 1, 2), keepdims=True).tolist())
            scaler_x[1] = np.squeeze(np.nanmax(X, axis=(0, 1, 2), keepdims=True).tolist())
            scaler_y = {}
        elif type == "region":
            scaler_x[0] = np.nanmin(X, axis=(0), keepdims=True)
            scaler_x[1] = np.nanmax(X, axis=(0), keepdims=True)
            scaler_y[0] = np.nanmin(y, axis=(0), keepdims=True)
            scaler_y[1] = np.nanmax(y, axis=(0), keepdims=True)
        else:
            raise IOError(f"Unknown variable type {type}")
        return scaler_x, scaler_y

    # ------------------------------------------------------------------------------------------------------------------------------
    def _spatial_normalize(self, static):
        # (ngrid, nfeat) for static data
        mean = np.nanmean(static, axis=(0, 1), keepdims=True)
        std = np.nanstd(static, axis=(0, 1), keepdims=True)
        return (static - mean) / std

    # ------------------------------------------------------------------------------------------------------------------------------
    def _lon_transform(self, x):
        x_new = np.zeros(x.shape)
        x_new[:, : int(x.shape[1] / 2)] = x[:, int(x.shape[1] / 2):]
        x_new[:, int(x.shape[1] / 2):] = x[:, : int(x.shape[1] / 2)]
        return x_new

    # ------------------------------------------------------------------------------------------------------------------------------
    def _interp(self, x, mask):
        x_ = np.ma.masked_invalid(x)
        arrange_lat = np.arange(0, x_.shape[0])
        arrange_lon = np.arange(0, x_.shape[1])
        lon_, lat_ = np.meshgrid(arrange_lon, arrange_lat)
        lat11_ = lat_[~x_.mask]
        lon11_ = lon_[~x_.mask]
        new_x = x_[~x_.mask].data
        inter_mean = np.nanmean(x_)
        out = griddata((lon11_, lat11_), new_x.ravel(), (lon_, lat_), method="linear", fill_value=inter_mean)
        mask_value = x == x
        out[mask_value] = x[mask_value]
        out[mask == 0] = np.nan

        return out
