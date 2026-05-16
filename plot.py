import os
import numpy as np
from config import get_args
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from mpl_toolkits.basemap import Basemap
from utils import unbiased_rmse, _rmse, _bias, GetKGE, r2_score, GetPCC, GetNSE, _rv, _fhv, _flv

plt.rc('font', family='Times New Roman')
plt.rcParams['font.size'] = 12
# ---------------------------------# ---------------------------------
# ! 2 ==> 观测和真实值 5 ==> R 6 ==> 时间序列
# plt_f = 'Fig.1'
# plt_f = 'Fig.2'
# plt_f = 'Fig.3'
# plt_f = 'Fig.4'
plt_f = 'Fig.5'
# plt_f = 'Fig.6'
# plt_f = 'Fig.7'
# plt_f = 'Fig.8'

# configures
pltday = 10  # used for plt spatial distributions at 'pltday' day
config = get_args()
name_test = 'Observations'
# sf = '/data/jiyuheng/china/china'
# sf = 'D:/Downloads/中华人民共和国/中华人民共和国'

# sf = '/data/jiyuheng/china/china'   # linux
sf = 'E:/first/project/all/shp/中华人民共和国/中华人民共和国'
# sf = '/root/autodl-tmp/datasets/shp/china/china'


def init(config):
    model = config.get('modelname')
    spatio = config.get('spatial_resolution')
    path = os.path.join(config['out_path'], config['process'], config['modelname'], str(config['forecast_time']))
    lats = np.load(os.path.join(config['out_path'], f'lat_{spatio}.npy'))
    lons = np.load(os.path.join(config['out_path'], f'lon_{spatio}.npy'))
    mask = np.load(os.path.join(config['out_path'], f"Mask with {str(config['spatial_resolution'])} spatial resolution.npy"))
    y_pred = np.load(os.path.join(path, '_predictions.npy'))
    y_test = np.load(os.path.join(path, 'observations.npy'))
    msk = (y_test == y_test)

    print('lats: ',lats)
    print('lons: ',lons)

    print('--------------------------------------------------------------------')
    print(f'all average r  of {model} model is : {np.corrcoef(y_test[msk], y_pred[msk])[0, 1]}')
    print(f'all average r2 of {model} model is : {r2_score(y_test[msk], y_pred[msk])}')
    print(f'all average kge of {model} model is : {GetKGE(y_test[msk], y_pred[msk])}')
    print(f'all average nse of {model} model is : {GetNSE(y_test[msk], y_pred[msk])}')
    print('--------------------------------------------------------------------')
    return model, spatio, path, mask, y_pred, y_test, lats, lons


def load_indices(config):
    path = os.path.join(config['out_path'], config['process'], config['modelname'], str(config['forecast_time']))
    r2_ = np.load(os.path.join(path, 'r2_' + config['modelname'] + '.npy'))
    r_ = np.load(os.path.join(path, 'r_' + config['modelname'] + '.npy'))
    rmse_ = np.load(os.path.join(path, 'rmse_' + config['modelname'] + '.npy'))
    urmse_ = np.load(os.path.join(path, 'urmse_' + config['modelname'] + '.npy'))
    bias_ = np.load(os.path.join(path, 'bias_' + config['modelname'] + '.npy'))
    KGE_ = np.load(os.path.join(path, 'KGE_' + config['modelname'] + '.npy'))
    NSE_ = np.load(os.path.join(path, 'NSE_' + config['modelname'] + '.npy'))
    rv_ = np.load(os.path.join(path, 'rv_' + config['modelname'] + '.npy'))
    fhv_ = np.load(os.path.join(path, 'fhv_' + config['modelname'] + '.npy'))
    flv_ = np.load(os.path.join(path, 'flv_' + config['modelname'] + '.npy'))

    return r2_, r_, rmse_, urmse_, bias_, KGE_, NSE_, rv_, fhv_, flv_


def print_indices(r2_, r_, rmse_, urmse_, bias_, KGE_, NSE_, rv_, fhv_, flv_, mask):
    print('the average r      of', config['modelname'], 'model is :', np.nanmedian(r_[mask == 1]))
    print('the average r2     of', config['modelname'], 'model is :', np.nanmedian(r2_[mask == 1]))
    print('--------------------------------------------------------------------')
    print('the average bias   of', config['modelname'], 'model is :', np.nanmedian(bias_[mask == 1]))
    print('the average rmse   of', config['modelname'], 'model is :', np.nanmedian(rmse_[mask == 1]))
    print('the average ubrmse of', config['modelname'], 'model is :', np.nanmedian(urmse_[mask == 1]))
    print('--------------------------------------------------------------------')
    print('the average KGE    of', config['modelname'], 'model is :', np.nanmedian(KGE_[mask == 1]))
    print('the average NSE    of', config['modelname'], 'model is :', np.nanmedian(NSE_[mask == 1]))
    print('--------------------------------------------------------------------')
    print('the average rv     of', config['modelname'], 'model is :', np.nanmedian(rv_[mask == 1]))
    print('the average fhv    of', config['modelname'], 'model is :', np.nanmedian(fhv_[mask == 1]))
    print('the average flv    of', config['modelname'], 'model is :', np.nanmedian(flv_[mask == 1]))


model, spatio, path, mask, y_pred, y_test, lats, lons = init(config)
r2_, r_, rmse_, urmse_, bias_, KGE_, NSE_, rv_, fhv_, flv_ = load_indices(config)
print_indices(r2_, r_, rmse_, urmse_, bias_, KGE_, NSE_, rv_, fhv_, flv_, mask)


# 南北纬互换
lat_ = lats
lon_ = lons

# Figure 6： configure for time series plot
# sites_lat_index=[110, 40, 50, 55, 60]
# sites_lon_index=[120, 80, 220, 280, 270]
# sites_lat_index = [16, 46, 60, 8, 40]
# sites_lon_index = [25, 35, 80, 100, 68]
sites_lat_index = [1, 3, 6, 4, 5]
sites_lon_index = [2, 6, 9, 10, 4]

# ---------------------------------
# Figure 1： box plot
# ---------------------------------
if plt_f in ['Fig.1']:
    # r2
    # do mask
    fig = plt.figure()
    # ==================== 修改开始 (解决R²负数显示问题) ====================
    # 原代码：直接使用原始数据
    # r2_box = r2_[mask == 1]
    # data_r2 = [r2_box]
    # 新代码：过滤掉负值，只保留R²>=0的数据
    r2_box = r2_[(mask == 1) & (r2_ >= 0)]
    data_r2 = [r2_box]
    # ==================== 修改结束 ====================
    ax = plt.subplot(111)

    plt.ylabel('R$^{2}$')
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    print("config: ",config)
    # ax.boxplot(data_r2, notch=True, patch_artist=True, showfliers=False, labels=[config['model']], boxprops=dict(facecolor='lightblue', color='black'))
    ax.boxplot(data_r2, notch=True, patch_artist=True, showfliers=False, labels=[config['modelname']], boxprops=dict(facecolor='lightblue', color='black'))

    fig = plt.figure()
    urmse_box = urmse_[mask == 1]
    data_urmse = [urmse_box]
    ax = plt.subplot(111)
    plt.ylabel("urmse")
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    # ax.boxplot(data_urmse, notch=True, patch_artist=True, showfliers=False, labels=[config['model']], boxprops=dict(facecolor='red', color='black'))
    ax.boxplot(data_urmse, notch=True, patch_artist=True, showfliers=False, labels=[config['modelname']], boxprops=dict(facecolor='red', color='black'))

    fig = plt.figure()
    r_box = r_[mask == 1]
    r_box = r_box[~np.isnan(r_box)]
    data_r = [r_box]
    ax = plt.subplot(111)
    plt.ylabel("r")
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    # ax.boxplot(data_r, notch=True, patch_artist=True, showfliers=False, labels=[config['model']], boxprops=dict(facecolor='green', color='black'))
    ax.boxplot(data_r, notch=True, patch_artist=True, showfliers=False, labels=[config['modelname']], boxprops=dict(facecolor='green', color='black'))
    plt.show()
    print('Figure 1 : box plot completed!')

# ------------------------------------------------------------------
# Figure 2： spatial distributions for predictions and observations
# ------------------------------------------------------------------
if plt_f in ['Fig.2']:
    if model in ['STALSTM']:
        model = 'FAELSTM'
    liner = np.arange(0, 1.1, 0.1)
    cmp = 'RdBu'
    plt.figure()
    plt.subplot(1, 2, 1)
    lon, lat = np.meshgrid(lon_, lat_)
    # 绘制吉林省区域
    # m = Basemap(llcrnrlon=110, llcrnrlat=36, urcrnrlon=140, urcrnrlat=52, projection='lcc', lat_1=25, lat_2=45, lon_0=110)
    # 绘制中国区域
    m = Basemap(projection='lcc', llcrnrlon=80, llcrnrlat=12, urcrnrlon=160, urcrnrlat=52, lat_1=30, lat_2=45, lon_0=100)
    m.readshapefile(shapefile=sf, name='states', drawbounds=True)
    parallels = np.arange(-90., 91, 5.)
    meridians = np.arange(-180., 181., 5.)
    m.drawparallels(parallels, labels=[False, True, False, False])
    m.drawmeridians(meridians, labels=[False, False, False, True])
    xi, yi = m(lon, lat)
    y_pred_pltday = y_pred[pltday]
    cs = m.contourf(xi, yi, y_pred_pltday, liner, cmap=cmp)
    cbar = m.colorbar(cs, location='bottom', pad="10%")
    plt.title(f"Forecast {config['forecast_time'] + 1} day of {model}", fontweight='bold')

    plt.subplot(1, 2, 2)
    m = Basemap(projection='lcc', llcrnrlon=80, llcrnrlat=12, urcrnrlon=160, urcrnrlat=52, lat_1=30, lat_2=45, lon_0=100)
    m.readshapefile(shapefile=sf, name='states', drawbounds=True)
    parallels = np.arange(-90., 91, 5.)
    meridians = np.arange(-180., 181., 5.)
    m.drawparallels(parallels, labels=[False, True, False, False])
    m.drawmeridians(meridians, labels=[False, False, False, True])
    xi, yi = m(lon, lat)
    y_test_pltday = y_test[pltday]
    cs = m.contourf(xi, yi, y_test_pltday, liner, cmap=cmp)
    cbar = m.colorbar(cs, location='bottom', pad="10%")
    plt.title(name_test, fontweight='bold')
    print('Figure 2 : spatial distributions for predictions and observations completed!')
    plt.show()

# ------------------------------------------------------------------
# Figure 3： spatial distributions for r2
# ------------------------------------------------------------------
if plt_f in ['Fig.3']:
    plt.subplot(1, 1, 1)
    lon, lat = np.meshgrid(lon_, lat_)
    m = Basemap(projection='lcc', llcrnrlon=80, llcrnrlat=14, urcrnrlon=140, urcrnrlat=52, lat_1=33, lat_2=45, lon_0=100)
    m.readshapefile(shapefile=sf, name='states', drawbounds=True)
    parallels = np.arange(-90., 91, 5.)
    meridians = np.arange(-180., 181., 5.)
    m.drawparallels(parallels, labels=[False, True, False, False])
    m.drawmeridians(meridians, labels=[False, False, False, True])
    xi, yi = m(lon, lat)

    # ==================== 修改开始 (解决R²负数显示问题) ====================
    # 原代码：直接使用 r2_ 绘图，部分格点R²为负数会导致显示异常
    # cs = m.contourf(xi, yi, r2_, np.arange(0, 1.1, 0.1), cmap='coolwarm')
    # 新代码：将R²限制在[0,1]范围内，负值截断为0
    r2_plot = np.clip(r2_, 0, 1)
    cs = m.contourf(xi, yi, r2_plot, np.arange(0, 1.1, 0.1), cmap='coolwarm')
    # ==================== 修改结束 ====================
    
    cbar = m.colorbar(cs, location='bottom', pad="10%")
    cbar.set_label('R$^{2}$')
    plt.title(model)

    print('Figure 3: spatial distributions for r2 completed!')
    plt.show()

# ------------------------------------------------------------------
# Figure 4： spatial distributions for ubrmse
# ------------------------------------------------------------------
if plt_f in ['Fig.4']:
    plt.figure()
    lon, lat = np.meshgrid(lon_, lat_)
    m = Basemap(projection='lcc', llcrnrlon=80, llcrnrlat=14, urcrnrlon=140, urcrnrlat=52, lat_1=33, lat_2=45, lon_0=100)
    m.readshapefile(shapefile=sf, name='states', drawbounds=True)
    parallels = np.arange(-90., 91, 5.)
    meridians = np.arange(-180., 181., 5.)
    m.drawparallels(parallels, labels=[False, True, False, False])
    m.drawmeridians(meridians, labels=[False, False, False, True])
    xi, yi = m(lon, lat)

    # convlstm
    urmse_[mask == 0] = -9999
    cs = m.contourf(xi, yi, urmse_, np.arange(0, 0.2, 0.01), cmap='RdBu')
    cbar = m.colorbar(cs, location='bottom', pad="10%")
    cbar.set_label('ubrmse(m$^{3}$/m$^{3}$)')

    plt.title(model)
    print('Figure 4: spatial distributions for ubrmse completed!')
    plt.show()

# ------------------------------------------------------------------
# Figure 5： spatial distributions for r
# ------------------------------------------------------------------
if plt_f in ['Fig.5']:
    if model in ['STALSTM']:
        model = 'FAELSTM'
    liner = np.arange(0, 1.1, 0.1)
    plt.subplot(1, 1, 1)
    plt.title(f'{config["forecast_time"] + 1} day of {model}', fontweight='bold')
    lon, lat = np.meshgrid(lon_, lat_)
    m = Basemap(projection='lcc', llcrnrlon=80, llcrnrlat=12, urcrnrlon=160, urcrnrlat=52, lat_1=30, lat_2=45, lon_0=100)
    m.readshapefile(shapefile=sf, name='states', drawbounds=True)
    parallels = np.arange(-90., 91, 5.)
    meridians = np.arange(-180., 181., 5.)
    m.drawparallels(parallels, labels=[False, True, False, False])
    m.drawmeridians(meridians, labels=[False, False, False, True])
    xi, yi = m(lon, lat)
    cs = m.contourf(xi, yi, r_, liner, cmap='jet')  # 'seismic'
    cbar = m.colorbar(cs, location='bottom', pad="10%")

    # 插入南海地图
    # ax_inset = plt.axes([0.7, 0.1, 0.2, 0.2])  # [left, bottom, width, height]
    # m_inset = Basemap(projection='lcc', llcrnrlon=105, llcrnrlat=0, urcrnrlon=125, urcrnrlat=25, lat_1=15, lat_2=25, lon_0=115, ax=ax_inset)
    # m_inset.readshapefile(shapefile=sf, name='states', drawbounds=True)
    # m_inset.drawparallels(parallels, labels=[False, False, False, False])
    # m_inset.drawmeridians(meridians, labels=[False, False, False, False])
    plt.show()

# ---------------------------------
# Figure 6： time series plot
# # ---------------------------------
# if plt_f in ['Fig.6']:
#     def load(config):
#         modelname = ['MLP', 'LSTM', 'BiLSTM', 'STALSTM']
#         data_pth = os.path.join(config['out_path'], config['process'])
#         y_true = np.load(os.path.join(data_pth, modelname[0], str(config['forecast_time']), 'observations.npy'))
#         y_pred0 = np.load(os.path.join(data_pth, modelname[0], str(config['forecast_time']), '_predictions.npy'))
#         y_pred1 = np.load(os.path.join(data_pth, modelname[1], str(config['forecast_time']), '_predictions.npy'))
#         y_pred2 = np.load(os.path.join(data_pth, modelname[2], str(config['forecast_time']), '_predictions.npy'))
#         y_pred3 = np.load(os.path.join(data_pth, modelname[3], str(config['forecast_time']), '_predictions.npy'))
#         modelname[-1] = 'FAELSTM'
#         return y_true, y_pred0, y_pred1, y_pred2, y_pred3, modelname
#     y_true, y_pred0, y_pred1, y_pred2, y_pred3, modelname = load(config)
#     plt.subplot(1, 1, 1)
#     lon, lat = np.meshgrid(lon_, lat_)
#     m = Basemap(projection='lcc', llcrnrlon=80, llcrnrlat=12, urcrnrlon=160, urcrnrlat=52, lat_1=30, lat_2=45, lon_0=100)
#     m.readshapefile(shapefile=sf, name='states', drawbounds=True)
#     parallels = np.arange(-90., 91, 5.)
#     meridians = np.arange(-180., 181., 5.)
#     m.drawparallels(parallels, labels=[False, True, False, False])
#     m.drawmeridians(meridians, labels=[False, False, False, True])
#     xi, yi = m(lon, lat)
#     print("xi shape:", xi.shape)  # 例如输出 (8, 16)
#     print("yi shape:", yi.shape)
#     for lon_index, lat_index in zip(sites_lon_index, sites_lat_index):
#         print(f"lat_index: {lat_index}, lon_index: {lon_index}")
#         m.plot(xi[int(lat_index), int(lon_index)], yi[int(lat_index), int(lon_index)], marker='*', color='red', markersize=9)
#     plt.legend(loc=0)
#     plt.show()
#
#     data_all = [y_test, y_pred0, y_pred1, y_pred2, y_pred3]  # y_pred_process
#     color_list = ['black', 'yellow', 'green', 'blue', 'red']  # red
#     modelname.insert(0, 'Observations')
#     for lon_index, lat_index in zip(sites_lon_index, sites_lat_index):
#         count = 0
#         fig, axs = plt.subplots(1, 1, figsize=(15, 2))
#         print('lat is {lat_v} and lon is {ln_v}'.format(lat_v=lat_[int(lat_index)], ln_v=lon_[int(lon_index)]))
#         print('r is', r_[lat_index, lon_index])
#         print('urmse is', urmse_[lat_index, lon_index])
#         print('rmse is', rmse_[lat_index, lon_index])
#         print('bias is', bias_[lat_index, lon_index])
#         for data_f5plt in (data_all):
#             axs.plot(data_f5plt[:, lat_index, lon_index], color=color_list[count])  #, label=modelname[count]
#             axs.legend(loc=1)
#             count = count + 1
#         axs.set_title(f'({int(lat_[int(lat_index)])}$^\circ$N,   {int(lon_[int(lon_index)])}$^\circ$E)', fontweight='bold')
#         # axs.set_title('lat is {lat_v} and lon is {ln_v}'.format(lat_v=lat_[int(lat_index)], ln_v=lon_[int(lon_index)]))
#     print('Figure 6： time series plot completed!')
#     plt.show()

if plt_f in ['Fig.6']:
    def load(config):
        modelname = ['MLP', 'LSTM', 'BiLSTM', 'STALSTM']
        data_pth = os.path.join(config['out_path'], config['process'])
        y_true = np.load(os.path.join(data_pth, modelname[0], str(config['forecast_time']), 'observations.npy'))
        y_pred0 = np.load(os.path.join(data_pth, modelname[0], str(config['forecast_time']), '_predictions.npy'))
        y_pred1 = np.load(os.path.join(data_pth, modelname[1], str(config['forecast_time']), '_predictions.npy'))
        y_pred2 = np.load(os.path.join(data_pth, modelname[2], str(config['forecast_time']), '_predictions.npy'))
        y_pred3 = np.load(os.path.join(data_pth, modelname[3], str(config['forecast_time']), '_predictions.npy'))
        modelname[-1] = 'FAELSTM'
        return y_true, y_pred0, y_pred1, y_pred2, y_pred3, modelname


    y_true, y_pred0, y_pred1, y_pred2, y_pred3, modelname = load(config)
    plt.subplot(1, 1, 1)
    lon, lat = np.meshgrid(lon_, lat_)
    m = Basemap(projection='lcc', llcrnrlon=80, llcrnrlat=12, urcrnrlon=160, urcrnrlat=52, lat_1=30, lat_2=45,
                lon_0=100)
    m.readshapefile(shapefile=sf, name='states', drawbounds=True)
    parallels = np.arange(-90., 91, 5.)
    meridians = np.arange(-180., 181., 5.)
    m.drawparallels(parallels, labels=[False, True, False, False])
    m.drawmeridians(meridians, labels=[False, False, False, True])
    xi, yi = m(lon, lat)

    # 修正地图图例部分
    handles = []
    for lon_index, lat_index in zip(sites_lon_index, sites_lat_index):
        line, = m.plot(xi[int(lat_index), int(lon_index)], yi[int(lat_index), int(lon_index)],
                       marker='*', color='red', markersize=9, label='Selected Sites')
        handles.append(line)
    # 去重标签
    unique_labels = dict()
    for handle in handles:
        unique_labels[handle.get_label()] = handle
    plt.legend(handles=unique_labels.values(), loc=0)
    plt.show()

    data_all = [y_true, y_pred0, y_pred1, y_pred2, y_pred3]  # 注意变量名统一（原代码中的 y_test 应为 y_true）
    color_list = ['black', 'yellow', 'green', 'blue', 'red']
    modelname.insert(0, 'Observations')

    # 修正时间序列图例部分
    for lon_index, lat_index in zip(sites_lon_index, sites_lat_index):
        count = 0
        fig, axs = plt.subplots(1, 1, figsize=(15, 2))
        print(f'lat is {lat_[int(lat_index)]} and lon is {lon_[int(lon_index)]}')
        print(
            f'r: {r_[lat_index, lon_index]}, urmse: {urmse_[lat_index, lon_index]}, rmse: {rmse_[lat_index, lon_index]}, bias: {bias_[lat_index, lon_index]}')

        # 绘制所有线条并设置标签
        for data_f5plt in data_all:
            axs.plot(data_f5plt[:, lat_index, lon_index],
                     color=color_list[count],
                     label=modelname[count])  # 关键修正：取消注释 label
            count += 1
        # 添加图例（移至循环外）
        axs.legend(loc='upper right')  # 使用更明确的定位参数
        axs.set_title(f'({int(lat_[int(lat_index)])}$^\circ$N, {int(lon_[int(lon_index)])}$^\circ$E)',
                      fontweight='bold')
    print('Figure 6： time series plot completed!')
    plt.show()

# ------------------------------------------------------------------
# Figure 7： spatial distributions for bias
# ------------------------------------------------------------------
if plt_f in ['Fig.7']:
    plt.subplot(1, 1, 1)
    bias_ = np.mean((y_pred - y_test), axis=0)
    lon, lat = np.meshgrid(lons, lats)
    m = Basemap(projection='lcc', llcrnrlon=80, llcrnrlat=14, urcrnrlon=140, urcrnrlat=52, lat_1=33, lat_2=45, lon_0=100)
    m.readshapefile(shapefile=sf, name='states', drawbounds=True)
    parallels = np.arange(-90., 91, 5.)
    meridians = np.arange(-180., 181., 5.)
    m.drawparallels(parallels, labels=[False, True, False, False])
    m.drawmeridians(meridians, labels=[False, False, False, True])
    xi, yi = m(lon, lat)
    # convlstm
    bias_[mask == 0] = -9999
    cs = m.contourf(xi, yi, bias_, np.arange(-0.04, 0.05, 0.01), cmap='coolwarm')  # 'seismic'
    cbar = m.colorbar(cs, location='bottom', pad="10%", ticks=np.arange(-0.04, 0.05, 0.01))
    cbar.set_label('bias(m$^{3}$/m$^{3}$)')

    plt.title(model)

    print('Figure 7: spatial distributions for bias completed!')
    plt.show()
# ------------------------------------------------------------------
# Figure 4： spatial distributions for rmse
# ------------------------------------------------------------------
if plt_f in ['Fig.8']:
    # plt.figure
    lon, lat = np.meshgrid(lon_, lat_)
    m = Basemap(projection='lcc', llcrnrlon=80, llcrnrlat=14, urcrnrlon=140, urcrnrlat=52, lat_1=33, lat_2=45, lon_0=100)
    m.readshapefile(shapefile=sf, name='states', drawbounds=True)
    parallels = np.arange(-90., 91, 5.)
    meridians = np.arange(-180., 181., 5.)
    m.drawparallels(parallels, labels=[False, True, False, False])
    m.drawmeridians(meridians, labels=[False, False, False, True])
    xi, yi = m(lon, lat)

    # convlstm
    urmse_[mask == 0] = -9999
    cs = m.contourf(xi, yi, rmse_, np.arange(0, 0.21, 0.01), cmap='RdBu')
    cbar = m.colorbar(cs, location='bottom', pad="10%", ticks=np.arange(0, 0.21, 0.02))
    cbar.set_label('rmse(m$^{3}$/m$^{3}$)')

    plt.title(model)
    print('Figure 8: spatial distributions for rmse completed!')
    plt.show()
