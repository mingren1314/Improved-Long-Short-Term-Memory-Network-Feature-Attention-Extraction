import os
import numpy as np
import pandas as pd
import seaborn as sns
from config import get_args
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from mpl_toolkits.basemap import Basemap
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

import platform

config = get_args()
name_test = 'Observations'
# sf = 'D:/Downloads/中华人民共和国/中华人民共和国'

sf = 'E:/first/project/all/shp/中华人民共和国/中华人民共和国'
# sf = '/root/autodl-tmp/Datasets/shp/china'
# sf = '/root/autodl-tmp/shp/中国大陆/中国大陆'  # 不需要 .shp 后缀

def init(config):
    is_linux = platform.system().lower() == "linux"
    print(f"当前系统是否为 Linux: {is_linux}")

    model = config.get('modelname')
    spatio = config.get('spatial_resolution')
    print("out_path: ",config['out_path'])
    print("seq_len: ",config['seq_len'])
    print("spatial_resolution: ",config['spatial_resolution'])


    path = os.path.join(config['out_path'], config['process'], config['modelname'], str(config['forecast_time']))
    lats = np.load(os.path.join(config['out_path'], f'lat_{spatio}.npy'))
    lons = np.load(os.path.join(config['out_path'], f'lon_{spatio}.npy'))
    mask = np.load(os.path.join(config['out_path'], f"Mask with {str(config['spatial_resolution'])} spatial resolution.npy"))
    y_pred = np.load(os.path.join(path, '_predictions.npy'))
    y_test = np.load(os.path.join(path, 'observations.npy'))
    msk = (y_test == y_test)
    print('--------------------------------------------------------------------')
    print(f'all average r  of {model} model is : {np.corrcoef(y_test[msk], y_pred[msk])[0, 1]}')
    print(f'all average r2 of {model} model is : {r2_score(y_test[msk], y_pred[msk])}')
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
# print_indices(r2_, r_, rmse_, urmse_, bias_, KGE_, NSE_, rv_, fhv_, flv_, mask)


# 绘制相关性矩阵
def plotcorr():
    # path = os.path.join('F:/Datasets/server/')
    # out_path = 'E:/ILRoad/临时文件/LandBench2.0/图像'

    path = os.path.join('/root/autodl-tmp/datasets/agriculture/')
    out_path = '/root/autodl-tmp/agriculture/temporary/LandBench2.0/image'

    data = np.load(os.path.join(path, 'data.npy'))

    plt.figure(figsize=(15, 10))
    column = ['t2m', 'u', 'v', 'pre', 'ssr', 'spec', 'ssrd', 'strd', 'stl1', 'e', 'swc', 'smci']
    data = pd.DataFrame(data, columns=column)
    matrix = data.corr()
    sns.heatmap(matrix, annot=True, cmap='RdBu_r')
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, '1-相关性矩阵.png'))


def load(config):
    # ==================== 修改开始 (扩展为7个模型) ====================
    # 原代码：4个模型
    # modelname = ['MLP', 'LSTM', 'BiLSTM', 'STALSTM']
    # 新代码：7个模型 (MLP, LSTM, BiLSTM, TCN, AttnLSTM, CNNTransformer, STALSTM)
    modelname = ['MLP', 'LSTM', 'BiLSTM', 'TCN', 'AttnLSTM', 'CNNTransformer', 'STALSTM']
    # ==================== 修改结束 ====================
    data_pth = os.path.join(config['out_path'], config['process'])
    y_true = np.load(os.path.join(data_pth, modelname[0], str(config['forecast_time']), 'observations.npy'))
    y_pred0 = np.load(os.path.join(data_pth, modelname[0], str(config['forecast_time']), '_predictions.npy'))
    y_pred1 = np.load(os.path.join(data_pth, modelname[1], str(config['forecast_time']), '_predictions.npy'))
    y_pred2 = np.load(os.path.join(data_pth, modelname[2], str(config['forecast_time']), '_predictions.npy'))
    y_pred3 = np.load(os.path.join(data_pth, modelname[3], str(config['forecast_time']), '_predictions.npy'))
    # ==================== 修改开始 (新增3个模型的预测结果) ====================
    y_pred4 = np.load(os.path.join(data_pth, modelname[4], str(config['forecast_time']), '_predictions.npy'))
    y_pred5 = np.load(os.path.join(data_pth, modelname[5], str(config['forecast_time']), '_predictions.npy'))
    y_pred6 = np.load(os.path.join(data_pth, modelname[6], str(config['forecast_time']), '_predictions.npy'))
    return y_true, y_pred0, y_pred1, y_pred2, y_pred3, y_pred4, y_pred5, y_pred6, modelname
    # ==================== 修改结束 ====================


# 绘制箱型图
# ==================== 修改开始 (扩展为7个模型) ====================
# 原代码：def plotbox(y_true, y_pred0, y_pred1, y_pred2, y_pred3, modelname, config):
def plotbox(y_true, y_pred0, y_pred1, y_pred2, y_pred3, y_pred4, y_pred5, y_pred6, modelname, config):
    # 原代码：4个模型
    # modelname = ['MLP', 'LSTM', 'BiLSTM', 'STALSTM']
    # 新代码：7个模型
    modelname = ['MLP', 'LSTM', 'BiLSTM', 'TCN', 'AttnLSTM', 'CNNTransformer', 'STALSTM']
    modelname_display = ['MLP', 'LSTM', 'BiLSTM', 'TCN', 'AttnLSTM', 'CNNTrans', 'FAELSTM']  # 显示用名称
    # ==================== 修改结束 ====================
    data_pth = os.path.join(config['out_path'], config['process'])
    r2_0 = np.load(os.path.join(data_pth, modelname[0], str(config['forecast_time']), 'r2_' + modelname[0] + '.npy'))
    r2_1 = np.load(os.path.join(data_pth, modelname[1], str(config['forecast_time']), 'r2_' + modelname[1] + '.npy'))
    r2_2 = np.load(os.path.join(data_pth, modelname[2], str(config['forecast_time']), 'r2_' + modelname[2] + '.npy'))
    r2_3 = np.load(os.path.join(data_pth, modelname[3], str(config['forecast_time']), 'r2_' + modelname[3] + '.npy'))
    # ==================== 修改开始 (新增3个模型的指标) ====================
    r2_4 = np.load(os.path.join(data_pth, modelname[4], str(config['forecast_time']), 'r2_' + modelname[4] + '.npy'))
    r2_5 = np.load(os.path.join(data_pth, modelname[5], str(config['forecast_time']), 'r2_' + modelname[5] + '.npy'))
    r2_6 = np.load(os.path.join(data_pth, modelname[6], str(config['forecast_time']), 'r2_' + modelname[6] + '.npy'))
    # ==================== 修改结束 ====================

    kge_0 = np.load(os.path.join(data_pth, modelname[0], str(config['forecast_time']), 'KGE_' + modelname[0] + '.npy'))
    kge_1 = np.load(os.path.join(data_pth, modelname[1], str(config['forecast_time']), 'KGE_' + modelname[1] + '.npy'))
    kge_2 = np.load(os.path.join(data_pth, modelname[2], str(config['forecast_time']), 'KGE_' + modelname[2] + '.npy'))
    kge_3 = np.load(os.path.join(data_pth, modelname[3], str(config['forecast_time']), 'KGE_' + modelname[3] + '.npy'))
    # ==================== 修改开始 (新增3个模型的KGE) ====================
    kge_4 = np.load(os.path.join(data_pth, modelname[4], str(config['forecast_time']), 'KGE_' + modelname[4] + '.npy'))
    kge_5 = np.load(os.path.join(data_pth, modelname[5], str(config['forecast_time']), 'KGE_' + modelname[5] + '.npy'))
    kge_6 = np.load(os.path.join(data_pth, modelname[6], str(config['forecast_time']), 'KGE_' + modelname[6] + '.npy'))
    # ==================== 修改结束 ====================

    ubrmse_0 = np.load(os.path.join(data_pth, modelname[0], str(config['forecast_time']), 'urmse_' + modelname[0] + '.npy'))
    ubrmse_1 = np.load(os.path.join(data_pth, modelname[1], str(config['forecast_time']), 'urmse_' + modelname[1] + '.npy'))
    ubrmse_2 = np.load(os.path.join(data_pth, modelname[2], str(config['forecast_time']), 'urmse_' + modelname[2] + '.npy'))
    ubrmse_3 = np.load(os.path.join(data_pth, modelname[3], str(config['forecast_time']), 'urmse_' + modelname[3] + '.npy'))
    # ==================== 修改开始 (新增3个模型的ubrmse) ====================
    ubrmse_4 = np.load(os.path.join(data_pth, modelname[4], str(config['forecast_time']), 'urmse_' + modelname[4] + '.npy'))
    ubrmse_5 = np.load(os.path.join(data_pth, modelname[5], str(config['forecast_time']), 'urmse_' + modelname[5] + '.npy'))
    ubrmse_6 = np.load(os.path.join(data_pth, modelname[6], str(config['forecast_time']), 'urmse_' + modelname[6] + '.npy'))
    # ==================== 修改结束 ====================
    
    # r2
    # do mask - 扩展为7个模型的mask
    mask = (r2_0 == r2_0) * (r2_1 == r2_1) * (r2_2 == r2_2) * (r2_3 == r2_3) * (r2_4 == r2_4) * (r2_5 == r2_5) * (r2_6 == r2_6)
    plt.figure(figsize=(12, 6))  # 加宽图表以容纳7个模型
    ax = plt.subplot(111)
    # 过滤掉负值，只保留R²>=0的数据
    data_r2 = [
        r2_0[(mask == 1) & (r2_0 >= 0)],
        r2_1[(mask == 1) & (r2_1 >= 0)],
        r2_2[(mask == 1) & (r2_2 >= 0)],
        r2_3[(mask == 1) & (r2_3 >= 0)],
        r2_4[(mask == 1) & (r2_4 >= 0)],
        r2_5[(mask == 1) & (r2_5 >= 0)],
        r2_6[(mask == 1) & (r2_6 >= 0)]
    ]
    plt.ylabel('R$^{2}$')
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.boxplot(data_r2, notch=True, patch_artist=True, showfliers=False, labels=modelname_display, boxprops=dict(facecolor='lightblue', color='black'))
    plt.xticks(rotation=15)  # 旋转标签避免重叠
    plt.title(f"Forecast time {config['forecast_time'] + 1} day of R$^{2}$")
    plt.tight_layout()
    plt.show()

    # kge - 扩展为7个模型
    mask = (kge_0 == kge_0) * (kge_1 == kge_1) * (kge_2 == kge_2) * (kge_3 == kge_3) * (kge_4 == kge_4) * (kge_5 == kge_5) * (kge_6 == kge_6)
    plt.figure(figsize=(12, 6))
    ax = plt.subplot(111)
    # ==================== 修改开始 (解决KGE负数显示问题) ====================
    # 原代码：直接使用原始数据
    # data_kge = [kge_0[mask == 1], kge_1[mask == 1], kge_2[mask == 1], kge_3[mask == 1], kge_4[mask == 1], kge_5[mask == 1], kge_6[mask == 1]]
    # 新代码：过滤掉负值，只保留KGE>=0的数据
    data_kge = [
        kge_0[(mask == 1) & (kge_0 >= 0)],
        kge_1[(mask == 1) & (kge_1 >= 0)],
        kge_2[(mask == 1) & (kge_2 >= 0)],
        kge_3[(mask == 1) & (kge_3 >= 0)],
        kge_4[(mask == 1) & (kge_4 >= 0)],
        kge_5[(mask == 1) & (kge_5 >= 0)],
        kge_6[(mask == 1) & (kge_6 >= 0)]
    ]
    # ==================== 修改结束 ====================

    plt.ylabel('KGE')
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.boxplot(data_kge, notch=True, patch_artist=True, showfliers=False, labels=modelname_display, boxprops=dict(facecolor='lightblue', color='black'))
    plt.xticks(rotation=15)
    plt.title(f"Forecast time {config['forecast_time'] + 1} day of KGE")
    plt.tight_layout()
    plt.show()

    # ubrmse - 扩展为7个模型
    mask = (ubrmse_0 == ubrmse_0) * (ubrmse_1 == ubrmse_1) * (ubrmse_2 == ubrmse_2) * (ubrmse_3 == ubrmse_3) * (ubrmse_4 == ubrmse_4) * (ubrmse_5 == ubrmse_5) * (ubrmse_6 == ubrmse_6)
    plt.figure(figsize=(12, 6))
    ax = plt.subplot(111)
    data_ubrmse = [ubrmse_0[mask == 1], ubrmse_1[mask == 1], ubrmse_2[mask == 1], ubrmse_3[mask == 1], ubrmse_4[mask == 1], ubrmse_5[mask == 1], ubrmse_6[mask == 1]]

    plt.ylabel('ubrmse')
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.boxplot(data_ubrmse, notch=True, patch_artist=True, showfliers=False, labels=modelname_display, boxprops=dict(facecolor='lightblue', color='black'))
    plt.xticks(rotation=15)
    plt.title(f"Forecast time {config['forecast_time'] + 1} day of ubrmse")
    plt.tight_layout()
    plt.show()
    print('Figure 1 : box plot completed!')


def plotayler_7models(y_true, y_pred0, y_pred1, y_pred2, y_pred3, y_pred4, y_pred5, y_pred6, modelname, config):
    """7个模型的泰勒图"""
    plt.rc('font', family='DejaVu Serif')
    plt.rcParams['font.size'] = 9
    
    modelname_display = ['MLP', 'LSTM', 'BiLSTM', 'TCN', 'AttnLSTM', 'CNNTrans', 'FAELSTM']
    colors = ['orange', 'green', 'blue', 'purple', 'cyan', 'brown', 'red']
    
    from matplotlib.projections import PolarAxes
    from mpl_toolkits.axisartist import grid_finder
    from mpl_toolkits.axisartist import floating_axes
    import matplotlib.lines as mlines

    def set_tayloraxes(fig, location):
        trans = PolarAxes.PolarTransform()
        r1_locs = np.hstack((np.arange(1, 10) / 10.0, [0.95, 0.99]))
        t1_locs = np.arccos(r1_locs)
        gl1 = grid_finder.FixedLocator(t1_locs)
        tf1 = grid_finder.DictFormatter(dict(zip(t1_locs, map(str, r1_locs))))
        # 扩大标准差范围到 2.5，以容纳表现较差的模型（如CNNTransformer）
        r2_locs = np.arange(0, 2.75, 0.25)
        r2_labels = ['0 ', '0.25 ', '0.50 ', '0.75 ', 'obs ', '1.25 ', '1.50 ', '1.75 ', '2.00 ', '2.25 ', '2.50 ']
        gl2 = grid_finder.FixedLocator(r2_locs)
        tf2 = grid_finder.DictFormatter(dict(zip(r2_locs, map(str, r2_labels))))
        ghelper = floating_axes.GridHelperCurveLinear(trans, extremes=(0, np.pi / 2, 0, 2.5), grid_locator1=gl1, tick_formatter1=tf1, grid_locator2=gl2, tick_formatter2=tf2)
        ax = floating_axes.FloatingSubplot(fig, location, grid_helper=ghelper)
        fig.add_subplot(ax)
        # ax.axis["top"].set_axis_direction("bottom")
        # ax.axis["top"].toggle(ticklabels=True, label=True)
        # ax.axis["top"].major_ticklabels.set_axis_direction("top")
        # ax.axis["top"].label.set_axis_direction("top")
        # ax.axis["top"].label.set_text("Correlation")
        # ax.axis["top"].label.set_fontsize(14)
        # ax.axis["left"].set_axis_direction("bottom")
        # ax.axis["left"].label.set_text("Standard Deviation")
        # ax.axis["left"].label.set_fontsize(14)
        # ax.axis["right"].set_axis_direction("top")
        # ax.axis["right"].toggle(ticklabels=True)
        # ax.axis["right"].major_ticklabels.set_axis_direction("left")
        # ax.axis["bottom"].set_visible(False)
        ax.axis["top"].set_axis_direction("bottom")
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text("Correlation Coefficient")  # 明确为相关系数
        ax.axis["top"].label.set_fontsize(14)

        ax.axis["left"].set_axis_direction("left")  # 改为 left，使标签出现在左侧
        ax.axis["left"].label.set_text("Normalized Standard Deviation (σ_model / σ_obs)")
        ax.axis["left"].label.set_fontsize(14)
        ax.axis["left"].label.set_rotation(90)  # 使文字垂直，更符合泰勒图习惯

        ax.axis["right"].set_axis_direction("top")
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].major_ticklabels.set_axis_direction("left")

        ax.axis["bottom"].set_visible(False)
        ax.grid(True)
        polar_ax = ax.get_aux_axes(trans)
        rs, ts = np.meshgrid(np.linspace(0, 2.5, 100), np.linspace(0, np.pi / 2, 100))
        rms = np.sqrt(1 + rs ** 2 - 2 * rs * np.cos(ts))
        CS = polar_ax.contour(ts, rs, rms, colors='gray', linestyles='--')
        plt.clabel(CS, inline=1, fontsize=10)
        t = np.linspace(0, np.pi / 2)
        r = np.zeros_like(t) + 1
        polar_ax.plot(t, r, 'k--')
        polar_ax.text(np.pi / 2 + 0.032, 1.02, " 1.00", ha="right", va="top", bbox=dict(boxstyle="square", ec='w', fc='w'))
        return polar_ax

    def plot_taylor(axes, refsample, sample, *args, **kwargs):
        std = np.std(refsample) / np.std(sample)
        corr = np.corrcoef(refsample, sample)
        theta = np.arccos(corr[0, 1])
        t, r = theta, std
        d = axes.plot(t, r, *args, **kwargs)
        return d

    mask = (y_true == y_true)
    y_true_m = y_true[mask]
    preds = [y_pred0[mask], y_pred1[mask], y_pred2[mask], y_pred3[mask], y_pred4[mask], y_pred5[mask], y_pred6[mask]]

    fig = plt.figure(dpi=200, figsize=(12, 9))
    ax1 = set_tayloraxes(fig, 111)
    
    # 绘制7个模型
    for i, (pred, name, color) in enumerate(zip(preds, modelname_display, colors)):
        plot_taylor(ax1, y_true_m, pred, 'o', markersize=10, alpha=0.8, color=color)
    
    # 绘制观测点
    plot_taylor(ax1, y_true_m, y_true_m, 'o', markersize=10, color='black')
    
    # 手动创建图例 - 放在右上角
    legend_handles = [mlines.Line2D([], [], color=c, marker='o', linestyle='None', markersize=8, label=n) 
                      for c, n in zip(colors, modelname_display)]
    legend_handles.append(mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=8, label='Observation'))
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1.0), loc='upper left', fontsize=10, frameon=True)

    # 标题放在图的上方中间
    if config['forecast_time'] == 0:
        num = 'a'
    else:
        num = 'b'
    fig.suptitle(f"({num}) Forecast {config['forecast_time'] + 1} day", fontsize=16, fontweight='normal', y=0.98)
    
    plt.subplots_adjust(right=0.82, top=0.92)  # 留出右侧图例空间
    plt.show()
    print('Taylor diagram completed!')


def plotayler(y_true, y_pred0, y_pred1, y_pred2, y_pred3, modelname, config):
    # plt.rc('font', family='Times New Roman')  # 如果没有该字体，使用下面的替代
    plt.rc('font', family='DejaVu Serif')  # Linux 默认可用的衬线字体
    plt.rcParams['font.size'] = 7
    modelname[-1] = 'FAELSTM'
    from matplotlib.projections import PolarAxes
    from matplotlib.patches import ConnectionPatch
    from mpl_toolkits.axisartist import grid_finder
    from mpl_toolkits.axisartist import floating_axes
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes

    def set_tayloraxes(fig, location):
        trans = PolarAxes.PolarTransform()
        r1_locs = np.hstack((np.arange(1, 10) / 10.0, [0.95, 0.99]))
        t1_locs = np.arccos(r1_locs)
        gl1 = grid_finder.FixedLocator(t1_locs)
        tf1 = grid_finder.DictFormatter(dict(zip(t1_locs, map(str, r1_locs))))
        r2_locs = np.arange(0, 2, 0.25)
        r2_labels = ['0 ', '0.25 ', '0.50 ', '0.75 ', 'obs ', '1.25 ', '1.50 ', '1.75 ']
        gl2 = grid_finder.FixedLocator(r2_locs)
        tf2 = grid_finder.DictFormatter(dict(zip(r2_locs, map(str, r2_labels))))
        ghelper = floating_axes.GridHelperCurveLinear(trans, extremes=(0, np.pi / 2, 0, 1.75), grid_locator1=gl1, tick_formatter1=tf1, grid_locator2=gl2, tick_formatter2=tf2)
        ax = floating_axes.FloatingSubplot(fig, location, grid_helper=ghelper)
        fig.add_subplot(ax)

        ax.axis["top"].set_axis_direction("bottom")
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text("Correlation")
        ax.axis["top"].label.set_fontsize(14)
        ax.axis["left"].set_axis_direction("bottom")
        ax.axis["left"].label.set_text("Standard deviation")
        ax.axis["left"].label.set_fontsize(14)
        ax.axis["right"].set_axis_direction("top")
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].major_ticklabels.set_axis_direction("left")
        ax.axis["bottom"].set_visible(False)
        ax.grid(True)
        polar_ax = ax.get_aux_axes(trans)

        rs, ts = np.meshgrid(np.linspace(0, 1.75, 100), np.linspace(0, np.pi / 2, 100))
        rms = np.sqrt(1 + rs ** 2 - 2 * rs * np.cos(ts))
        CS = polar_ax.contour(ts, rs, rms, colors='gray', linestyles='--')
        plt.clabel(CS, inline=1, fontsize=10)
        t = np.linspace(0, np.pi / 2)
        r = np.zeros_like(t) + 1
        polar_ax.plot(t, r, 'k--')
        polar_ax.text(np.pi / 2 + 0.032, 1.02, " 1.00", ha="right", va="top", bbox=dict(boxstyle="square", ec='w', fc='w'))

        return polar_ax

    def plot_taylor(axes, refsample, sample, *args, **kwargs):
        std = np.std(refsample) / np.std(sample)
        corr = np.corrcoef(refsample, sample)
        theta = np.arccos(corr[0, 1])
        t, r = theta, std
        d = axes.plot(t, r, *args, **kwargs)
        return d

    def add_contours(polar_ax):
        rs, ts = np.meshgrid(np.linspace(0, 1.75, 100), np.linspace(0, np.pi / 2, 100))
        rms = np.sqrt(1 + rs ** 2 - 2 * rs * np.cos(ts))
        CS = polar_ax.contour(ts, rs, rms, colors='gray', linestyles='--')
        plt.clabel(CS, inline=1, fontsize=10)
        t = np.linspace(0, np.pi / 2)
        r = np.zeros_like(t) + 1
        polar_ax.plot(t, r, 'k--')
        polar_ax.text(np.pi / 2 + 0.052, 1.02, " 1.00", size=10.3, ha="right", va="top", bbox=dict(boxstyle="square", ec='w', fc='w'))
    mask = (y_true == y_true)
    y_true = y_true[mask]
    y_pred0 = y_pred0[mask]
    y_pred1 = y_pred1[mask]
    y_pred2 = y_pred2[mask]
    y_pred3 = y_pred3[mask]

    fig = plt.figure(dpi=200)
    # fig = plt.figure(figsize=(8, 8))
    ax1 = set_tayloraxes(fig, 111)
    d1 = plot_taylor(ax1, y_true, y_pred0, 'o', markersize=8, alpha=0.7, color='orange', label=f'{modelname[0]}')
    d2 = plot_taylor(ax1, y_true, y_pred1, 'o', markersize=8, alpha=0.7, color='green', label=f'{modelname[1]}')
    d3 = plot_taylor(ax1, y_true, y_pred2, 'o', markersize=8, alpha=0.7, color='blue', label=f'{modelname[2]}')
    d4 = plot_taylor(ax1, y_true, y_pred3, 'o', markersize=8, alpha=0.7, color='red', label=f'{modelname[3]}')
    d5 = plot_taylor(ax1, y_true, y_true, 'o', markersize=8, color='black', label='Observation')
    plt.legend(bbox_to_anchor=(-0.2, 1))

    if config['forecast_time'] == 0:
        num = 'a'
    else:
        num = 'b'
    fig.text(0.05, 0.95, f"({num}) Forecast {config['forecast_time'] + 1} day", fontdict=dict(fontsize=16, color='black', family='Times New Roman', weight='light'))

    # 局部放大图
    axins = zoomed_inset_axes(ax1, 14, bbox_to_anchor=(1.7, 1.15), bbox_transform=ax1.transAxes)

    # 计算  数据点的相关性和标准差
    std_2 = np.std(y_true) / np.std(y_pred2)
    corr_2 = np.corrcoef(y_true, y_pred2)[0, 1]
    theta_2 = np.arccos(corr_2)
    t_2, r_2 = theta_2, std_2

    # 设置放大区域的边界，这里我们围绕 1 数据点放大一个小的区域
    x_margin = 0.05  # x方向上的边界余量
    y_margin = 0.05  # y方向上的边界余量
    x1 = t_2 - x_margin
    x2 = t_2 + x_margin
    y1 = r_2 - y_margin
    y2 = r_2 + y_margin

    # 更新 axins 的视图限制
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    plot_taylor(axins, y_true, y_pred2, 'o', markersize=8, color='green')

    # 计算 1 数据点的相关性和标准差
    std_1 = np.std(y_true) / np.std(y_pred1)
    corr_1 = np.corrcoef(y_true, y_pred1)[0, 1]
    theta_1 = np.arccos(corr_1)
    t_1, r_1 = theta_1, std_1

    # 设置放大区域的边界，这里我们围绕 1 数据点放大一个小的区域
    x_margin = 0.05  # x方向上的边界余量
    y_margin = 0.05  # y方向上的边界余量
    x1 = t_1 - x_margin
    x2 = t_1 + x_margin
    y1 = r_1 - y_margin
    y2 = r_1 + y_margin

    # 更新 axins 的视图限制
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    # 在局部放大子图中重新绘制等值线并确保虚线平行
    add_contours(axins)
    plot_taylor(axins, y_true, y_pred1, 'o', markersize=8, color='blue')

    std_3 = np.std(y_true) / np.std(y_pred3)
    corr_3 = np.corrcoef(y_true, y_pred3)[0, 1]
    theta_3 = np.arccos(corr_3)
    t_3, r_3 = theta_3, std_3

    # 设置放大区域的边界，这里我们围绕 1 数据点放大一个小的区域
    x_margin = 0.05  # x方向上的边界余量
    y_margin = 0.05  # y方向上的边界余量
    x1 = t_3 - x_margin
    x2 = t_3 + x_margin
    y1 = r_3 - y_margin
    y2 = r_3 + y_margin

    # 更新 axins 的视图限制
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    plot_taylor(axins, y_true, y_pred3, 'o', markersize=8, color='red')

    # 建立父坐标系与子坐标系的连接线
    # 原图中画方框
    sx = [x1, x2, x2, x1, x1]
    sy = [y1, y1, y2, y2, y1]
    ax1.plot(sx, sy, "black")

    # 画两条线
    xy = (x1, y2)
    xy2 = (x1, y1)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data", axesA=axins, axesB=ax1)
    axins.add_artist(con)

    xy = (x2, y2)
    xy2 = (x1, y2)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data", axesA=axins, axesB=ax1)
    axins.add_artist(con)

    plt.show()


def plotscatter(y_true, y_pred0, y_pred1, y_pred2, y_pred3, modelname, config):
    modelname[-1] = 'FAELSTM'
    pred = np.array([y_pred0, y_pred1, y_pred2, y_pred3])
    mask = y_true == y_true
    for idx, mdn in enumerate(modelname):
        a = y_true[mask == 1]
        b = pred[idx][mask == 1]
        # 绘制散点图
        plt.scatter(a, b, color='blue', marker='o', label='Data Points')

        # 使用线性回归拟合直线
        fit = np.polyfit(a, b, 1)
        fit_fn = np.poly1d(fit)

        # 绘制拟合直线
        plt.plot(a, fit_fn(a), color='red', label='Linear Fit')

        # 添加标签和标题
        plt.xlabel('a')
        plt.ylabel('b')
        plt.title('Scatter Plot with Linear Fit')

        # 显示图例
        plt.legend()

        # 显示图像
        plt.show()


def r(modelname, config):
    lats = np.load(os.path.join(config['out_path'], f'lat_{spatio}.npy'))
    lons = np.load(os.path.join(config['out_path'], f'lon_{spatio}.npy'))
    r0 = np.load(os.path.join(config['out_path'], config['process'], modelname[0], str(config['forecast_time']), 'r_' + modelname[0] + '.npy'))
    r1 = np.load(os.path.join(config['out_path'], config['process'], modelname[1], str(config['forecast_time']), 'r_' + modelname[1] + '.npy'))
    r2 = np.load(os.path.join(config['out_path'], config['process'], modelname[2], str(config['forecast_time']), 'r_' + modelname[2] + '.npy'))
    r3 = np.load(os.path.join(config['out_path'], config['process'], modelname[3], str(config['forecast_time']), 'r_' + modelname[3] + '.npy'))

    r_0 = (r3 - r0) / r3
    r_1 = (r3 - r1) / r3
    r_2 = (r3 - r2) / r3
    r_3 = (r3 - r3) / r3
    r = [r_0, r_1, r_2, r_3]
    modelname[-1] = 'FAELSTM'
    for idx, model in enumerate(modelname):
        liner = np.arange(0, 1.1, 0.1)
        plt.subplot(1, 1, 1)
        plt.title(f'Forecast {config["forecast_time"] + 1} day of {model}')
        # plt.title(f'Forecast {config["forecast_time"] + 1} day of FAELSTM')
        lon, lat = np.meshgrid(lons, lats)
        m = Basemap(projection='lcc', llcrnrlon=80, llcrnrlat=12, urcrnrlon=160, urcrnrlat=52, lat_1=30, lat_2=45, lon_0=100)
        m.readshapefile(shapefile=sf, name='states', drawbounds=True)
        parallels = np.arange(-90., 91, 5.)
        meridians = np.arange(-180., 181., 5.)
        m.drawparallels(parallels, labels=[False, True, False, False])
        m.drawmeridians(meridians, labels=[False, False, False, True])
        xi, yi = m(lon, lat)
        cs = m.contourf(xi, yi, r[idx], liner, cmap='jet')  # 'seismic'
        cbar = m.colorbar(cs, location='bottom', pad="10%")
        plt.show()


# def spatial_comparison_7models(config, day_index=100, forecast_day=None, save_path=None):
#     """
#     Fig 11: 各模型在每个网格点的 R 值空间分布图
#     第一个子图为观测值，后面7个为各模型的R值
#     左下角显示中值 (median)
#
#     参数:
#         forecast_day: 预测天数 (1 或 7)，用于标题显示
#         save_path: 保存路径，如果提供则保存图片
#     """
#     modelnames = ['MLP', 'LSTM', 'BiLSTM', 'TCN', 'AttnLSTM', 'CNNTransformer', 'STALSTM']
#     modelname_display = ['MLP', 'LSTM', 'BiLSTM', 'TCN', 'AttnLSTM', 'CNNTrans', 'FAELSTM']
#
#     # 确定预测天数
#     if forecast_day is None:
#         forecast_day = config['forecast_time'] + 1  # forecast_time 是索引，从0开始
#
#     spatio = config['spatial_resolution']
#     lats = np.load(os.path.join(config['out_path'], f'lat_{spatio}.npy'))
#     lons = np.load(os.path.join(config['out_path'], f'lon_{spatio}.npy'))
#     mask = np.load(os.path.join(config['out_path'], f"Mask with {spatio} spatial resolution.npy"))
#
#     # 打印数据范围，便于调试
#     print(f"数据范围: 纬度 {lats.min():.2f} - {lats.max():.2f}, 经度 {lons.min():.2f} - {lons.max():.2f}")
#
#     # 加载观测值（用于第一个子图）
#     data_pth = os.path.join(config['out_path'], config['process'])
#     y_true = np.load(os.path.join(data_pth, modelnames[0], str(config['forecast_time']), 'observations.npy'))
#     y_true_day = y_true[day_index]
#     if len(y_true_day.shape) == 3:
#         y_true_day = y_true_day[:, :, 0]
#
#     # 加载所有模型的 R 值（相关系数，不是R²）
#     r_list = []
#     for model in modelnames:
#         r_path = os.path.join(config['out_path'], config['process'], model, str(config['forecast_time']), f'r_{model}.npy')
#         r = np.load(r_path)
#         r_list.append(r)
#
#     lon, lat = np.meshgrid(lons, lats)
#
#     # 根据数据实际范围设置地图边界（留一点边距）
#     lon_min, lon_max = lons.min() - 2, lons.max() + 2
#     lat_min, lat_max = lats.min() - 2, lats.max() + 2
#
#     # 创建 2x4 子图（观测值 + 7个模型）
#     fig, axes = plt.subplots(2, 4, figsize=(20, 10), dpi=150)
#     axes = axes.flatten()
#
#     # 添加总标题：预测天数
#     fig.suptitle(f'Forecast {forecast_day} Day', fontsize=20, fontweight='bold', y=0.98)
#
#     # 第一个子图：观测值
#     ax = axes[0]
#     m = Basemap(projection='cyl', llcrnrlon=lon_min, llcrnrlat=lat_min,
#                 urcrnrlon=lon_max, urcrnrlat=lat_max, ax=ax)
#     try:
#         m.readshapefile(shapefile=sf, name='states', drawbounds=True)
#     except:
#         m.drawcoastlines(linewidth=0.5)
#         m.drawcountries(linewidth=0.5)
#     m.drawparallels(np.arange(-90., 91, 5.), labels=[False, True, False, False], fontsize=8)
#     m.drawmeridians(np.arange(-180., 181., 5.), labels=[False, False, False, True], fontsize=8)
#     xi, yi = m(lon, lat)
#     # 观测值使用 SMCI 范围
#     obs_levels = np.linspace(0, 0.6, 13)
#     cs_obs = m.contourf(xi, yi, y_true_day, obs_levels, cmap='RdYlBu', extend='both')
#     ax.set_title('Observation', fontsize=14, fontweight='bold')
#
#     # 加载各模型的预测值用于计算指标
#     from sklearn.metrics import r2_score, mean_squared_error
#     pred_list = []
#     for model in modelnames:
#         y_pred = np.load(os.path.join(data_pth, model, str(config['forecast_time']), '_predictions.npy'))
#         pred_list.append(y_pred)
#
#     # 后面7个子图：各模型的R值
#     for idx, (r_val, name) in enumerate(zip(r_list, modelname_display)):
#         ax = axes[idx + 1]  # 从第2个子图开始
#         # 使用数据实际范围作为地图边界
#         m = Basemap(projection='cyl', llcrnrlon=lon_min, llcrnrlat=lat_min,
#                     urcrnrlon=lon_max, urcrnrlat=lat_max, ax=ax)
#         # 尝试加载shapefile，如果不存在则跳过
#         try:
#             m.readshapefile(shapefile=sf, name='states', drawbounds=True)
#         except:
#             m.drawcoastlines(linewidth=0.5)
#             m.drawcountries(linewidth=0.5)
#         m.drawparallels(np.arange(-90., 91, 5.), labels=[False, True, False, False], fontsize=8)
#         m.drawmeridians(np.arange(-180., 181., 5.), labels=[False, False, False, True], fontsize=8)
#
#         xi, yi = m(lon, lat)
#         # R 值范围设为 0 到 1
#         levels = np.linspace(0, 1, 11)
#         cs = m.contourf(xi, yi, r_val, levels, cmap='RdYlBu', extend='both')
#         ax.set_title(f'{name}', fontsize=14, fontweight='bold')
#
#         # 计算整个测试集的指标
#         y_true_flat = y_true.flatten()
#         pred_flat = pred_list[idx].flatten()
#         valid_mask = ~np.isnan(y_true_flat) & ~np.isnan(pred_flat)
#
#         r_metric = np.corrcoef(y_true_flat[valid_mask], pred_flat[valid_mask])[0, 1]
#         r2_metric = r2_score(y_true_flat[valid_mask], pred_flat[valid_mask])
#         bias_metric = np.mean(pred_flat[valid_mask] - y_true_flat[valid_mask])
#         rmse_metric = np.sqrt(mean_squared_error(y_true_flat[valid_mask], pred_flat[valid_mask]))
#
#         # 左上角显示指标
#         metrics_text = f'R={r_metric:.3f}\nR²={r2_metric:.3f}\nBias={bias_metric:.3f}\nRMSE={rmse_metric:.3f}'
#         ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
#                 fontsize=8, color='black', fontweight='bold',
#                 verticalalignment='top', horizontalalignment='left',
#                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
#
#         # 左下角显示中值（红色字体）
#         median_r = np.nanmedian(r_val[mask == 1])
#         ax.text(0.05, 0.05, f'Median: {median_r:.3f}', transform=ax.transAxes,
#                 fontsize=10, color='red', fontweight='bold',
#                 verticalalignment='bottom', horizontalalignment='left',
#                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
#
#     # 添加统一的颜色条（R值）
#     cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
#     cbar = fig.colorbar(cs, cax=cbar_ax)
#     cbar.set_label('R', fontsize=12)
#
#     plt.subplots_adjust(left=0.05, right=0.9, top=0.92, bottom=0.05, wspace=0.15, hspace=0.2)
#
#     # 保存图片
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
#         print(f'图片已保存: {save_path}')
#
#     plt.show()
#     print(f'Fig 11 (Forecast {forecast_day} Day) completed!')

# 修改后fig11生成函数
def spatial_comparison_7models(config, day_index=100, forecast_day=None, save_path=None):
    """
    Fig 11: 各模型在每个网格点的 R 值空间分布图
    修改说明：左上角指标改为计算 Median 值，以匹配 Table 3
    """
    modelnames = ['MLP', 'LSTM', 'BiLSTM', 'TCN', 'AttnLSTM', 'CNNTransformer', 'STALSTM']
    modelname_display = ['MLP', 'LSTM', 'BiLSTM', 'TCN', 'AttnLSTM', 'CNNTrans', 'FAELSTM']

    if forecast_day is None:
        forecast_day = config['forecast_time'] + 1

    spatio = config['spatial_resolution']
    lats = np.load(os.path.join(config['out_path'], f'lat_{spatio}.npy'))
    lons = np.load(os.path.join(config['out_path'], f'lon_{spatio}.npy'))
    mask = np.load(os.path.join(config['out_path'], f"Mask with {spatio} spatial resolution.npy"))

    # 加载观测值
    data_pth = os.path.join(config['out_path'], config['process'])
    y_true = np.load(os.path.join(data_pth, modelnames[0], str(config['forecast_time']), 'observations.npy'))
    y_true_day = y_true[day_index]
    if len(y_true_day.shape) == 3:
        y_true_day = y_true_day[:, :, 0]

    # 加载 R 值列表 (用于绘图)
    r_list = []
    for model in modelnames:
        r_path = os.path.join(data_pth, model, str(config['forecast_time']), f'r_{model}.npy')
        r = np.load(r_path)
        r_list.append(r)

    lon, lat = np.meshgrid(lons, lats)
    lon_min, lon_max = lons.min() - 2, lons.max() + 2
    lat_min, lat_max = lats.min() - 2, lats.max() + 2

    fig, axes = plt.subplots(2, 4, figsize=(20, 10), dpi=150)
    axes = axes.flatten()
    fig.suptitle(f'Forecast {forecast_day} Day', fontsize=20, fontweight='bold', y=0.98)

    # 1. 绘制观测值
    ax = axes[0]
    m = Basemap(projection='cyl', llcrnrlon=lon_min, llcrnrlat=lat_min, urcrnrlon=lon_max, urcrnrlat=lat_max, ax=ax)
    try:
        m.readshapefile(shapefile=sf, name='states', drawbounds=True)
    except:
        pass
    m.drawparallels(np.arange(-90., 91, 5.), labels=[False, True, False, False], fontsize=8)
    m.drawmeridians(np.arange(-180., 181., 5.), labels=[False, False, False, True], fontsize=8)
    xi, yi = m(lon, lat)
    obs_levels = np.linspace(0, 0.6, 13)
    cs_obs = m.contourf(xi, yi, y_true_day, obs_levels, cmap='RdYlBu', extend='both')
    ax.set_title('Observation', fontsize=14, fontweight='bold')

    # 2. 绘制各模型
    for idx, (r_val, name) in enumerate(zip(r_list, modelname_display)):
        model_real_name = modelnames[idx]  # 对应的文件夹名
        ax = axes[idx + 1]

        m = Basemap(projection='cyl', llcrnrlon=lon_min, llcrnrlat=lat_min, urcrnrlon=lon_max, urcrnrlat=lat_max, ax=ax)
        try:
            m.readshapefile(shapefile=sf, name='states', drawbounds=True)
        except:
            pass
        m.drawparallels(np.arange(-90., 91, 5.), labels=[False, True, False, False], fontsize=8)
        m.drawmeridians(np.arange(-180., 181., 5.), labels=[False, False, False, True], fontsize=8)

        xi, yi = m(lon, lat)
        levels = np.linspace(0, 1, 11)
        cs = m.contourf(xi, yi, r_val, levels, cmap='RdYlBu', extend='both')
        ax.set_title(f'{name}', fontsize=14, fontweight='bold')

        # ==================== 修改核心部分开始 ====================
        # 不再使用 flatten 计算整体指标，而是加载空间分布文件计算中位数

        # 1. 加载该模型的其他指标的空间分布文件
        base_path = os.path.join(data_pth, model_real_name, str(config['forecast_time']))
        r2_map = np.load(os.path.join(base_path, f'r2_{model_real_name}.npy'))
        bias_map = np.load(os.path.join(base_path, f'bias_{model_real_name}.npy'))
        rmse_map = np.load(os.path.join(base_path, f'rmse_{model_real_name}.npy'))

        # 2. 计算中位数 (Median)，需应用 mask
        median_r = np.nanmedian(r_val[mask == 1])
        median_r2 = np.nanmedian(r2_map[mask == 1])
        median_bias = np.nanmedian(bias_map[mask == 1])
        median_rmse = np.nanmedian(rmse_map[mask == 1])

        # 3. 更新左上角文字
        metrics_text = f'R={median_r:.3f}\nR²={median_r2:.3f}\nBias={median_bias:.3f}\nRMSE={median_rmse:.3f}'

        # 左上角显示 (现在是 Median 值了)
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
                fontsize=8, color='black', fontweight='bold',
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 左下角 (为了不重复，可以去掉或者保留作为强调)
        ax.text(0.05, 0.05, f'Median R: {median_r:.3f}', transform=ax.transAxes,
                fontsize=10, color='red', fontweight='bold',
                verticalalignment='bottom', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        # ==================== 修改核心部分结束 ====================

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(cs, cax=cbar_ax)
    cbar.set_label('R', fontsize=12)

    plt.subplots_adjust(left=0.05, right=0.9, top=0.92, bottom=0.05, wspace=0.15, hspace=0.2)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f'图片已保存: {save_path}')

    plt.show()
    print(f'Fig 11 (Forecast {forecast_day} Day) completed!')

def fig12_prediction_comparison(config, day_index=0, forecast_day=None, save_path=None):
    """
    Fig 12: 不同模型在预测第1天和第7天时，真实值与预测值之间的对比
    第一列为观测值，后面列依次为各模型预测值
    左上角显示 R, R², Bias, RMSE 四个指标
    
    参数:
        config: 配置字典
        day_index: 选择测试集中的第几天进行可视化 (默认0表示第一天)
        forecast_day: 预测天数 (1 或 7)，用于标题显示
        save_path: 保存路径，如果提供则保存图片
    """
    from sklearn.metrics import r2_score, mean_squared_error
    
    modelnames = ['MLP', 'LSTM', 'BiLSTM', 'TCN', 'AttnLSTM', 'CNNTransformer', 'STALSTM']
    modelname_display = ['MLP', 'LSTM', 'BiLSTM', 'TCN', 'AttnLSTM', 'CNNTrans', 'FAELSTM']
    
    # 确定预测天数
    if forecast_day is None:
        forecast_day = config['forecast_time'] + 1
    
    spatio = config['spatial_resolution']
    lats = np.load(os.path.join(config['out_path'], f'lat_{spatio}.npy'))
    lons = np.load(os.path.join(config['out_path'], f'lon_{spatio}.npy'))
    mask = np.load(os.path.join(config['out_path'], f"Mask with {spatio} spatial resolution.npy"))
    
    # 加载观测值和各模型预测值
    data_pth = os.path.join(config['out_path'], config['process'])
    y_true = np.load(os.path.join(data_pth, modelnames[0], str(config['forecast_time']), 'observations.npy'))
    
    pred_list = []
    for model in modelnames:
        y_pred = np.load(os.path.join(data_pth, model, str(config['forecast_time']), '_predictions.npy'))
        pred_list.append(y_pred)
    
    # 选择某一天的数据进行可视化
    y_true_day = y_true[day_index]  # shape: (lat, lon, 1) 或 (lat, lon)
    if len(y_true_day.shape) == 3:
        y_true_day = y_true_day[:, :, 0]
    
    pred_day_list = []
    for pred in pred_list:
        pred_day = pred[day_index]
        if len(pred_day.shape) == 3:
            pred_day = pred_day[:, :, 0]
        pred_day_list.append(pred_day)
    
    lon, lat = np.meshgrid(lons, lats)
    lon_min, lon_max = lons.min() - 2, lons.max() + 2
    lat_min, lat_max = lats.min() - 2, lats.max() + 2
    
    # 创建子图：2行4列（观测值 + 7个模型）
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), dpi=150)
    axes = axes.flatten()
    
    # 添加总标题：预测天数
    fig.suptitle(f'Forecast {forecast_day} Day', fontsize=20, fontweight='bold', y=0.98)
    
    # SMCI 值范围 (土壤湿度通常 0-0.6)
    levels = np.linspace(0, 0.6, 13)
    
    # 绘制观测值
    ax = axes[0]
    m = Basemap(projection='cyl', llcrnrlon=lon_min, llcrnrlat=lat_min,
                urcrnrlon=lon_max, urcrnrlat=lat_max, ax=ax)
    try:
        m.readshapefile(shapefile=sf, name='states', drawbounds=True)
    except:
        m.drawcoastlines(linewidth=0.5)
        m.drawcountries(linewidth=0.5)
    xi, yi = m(lon, lat)
    cs = m.contourf(xi, yi, y_true_day, levels, cmap='RdYlBu', extend='both')
    ax.set_title('Observation', fontsize=12, fontweight='bold')
    
    # 绘制各模型预测值
    for idx, (pred_day, name) in enumerate(zip(pred_day_list, modelname_display)):
        ax = axes[idx + 1]
        m = Basemap(projection='cyl', llcrnrlon=lon_min, llcrnrlat=lat_min,
                    urcrnrlon=lon_max, urcrnrlat=lat_max, ax=ax)
        try:
            m.readshapefile(shapefile=sf, name='states', drawbounds=True)
        except:
            m.drawcoastlines(linewidth=0.5)
            m.drawcountries(linewidth=0.5)
        xi, yi = m(lon, lat)
        cs = m.contourf(xi, yi, pred_day, levels, cmap='RdYlBu', extend='both')
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        
        # 计算整个测试集的指标（不只是这一天）
        y_true_flat = y_true.flatten()
        pred_flat = pred_list[idx].flatten()
        valid_mask = ~np.isnan(y_true_flat) & ~np.isnan(pred_flat)
        
        r_val = np.corrcoef(y_true_flat[valid_mask], pred_flat[valid_mask])[0, 1]
        r2_val = r2_score(y_true_flat[valid_mask], pred_flat[valid_mask])
        bias_val = np.mean(pred_flat[valid_mask] - y_true_flat[valid_mask])
        rmse_val = np.sqrt(mean_squared_error(y_true_flat[valid_mask], pred_flat[valid_mask]))
        
        # 左上角显示指标
        metrics_text = f'R={r_val:.3f}\nR²={r2_val:.3f}\nBias={bias_val:.3f}\nRMSE={rmse_val:.3f}'
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
                fontsize=9, color='black', fontweight='bold',
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 添加统一的颜色条
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(cs, cax=cbar_ax)
    cbar.set_label('SMCI', fontsize=12)
    
    plt.subplots_adjust(left=0.05, right=0.9, top=0.92, bottom=0.05, wspace=0.15, hspace=0.2)
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f'图片已保存: {save_path}')
    
    plt.show()
    print(f'Fig 12 (Forecast {forecast_day} Day) completed!')


def fig13_timeseries_comparison(config):
    """
    Fig 13: 随机选取5个点，展示真实值与各模型预测值的时间序列对比
    5个点分别代表：西北、西南、东南沿海、东北、中部
    分为3个图：1.点位置图  2.前3个点时间序列  3.后2个点时间序列
    """
    modelnames = ['MLP', 'LSTM', 'BiLSTM', 'TCN', 'AttnLSTM', 'CNNTransformer', 'STALSTM']
    modelname_display = ['MLP', 'LSTM', 'BiLSTM', 'TCN', 'AttnLSTM', 'CNNTrans', 'FAELSTM']
    
    # 7个模型的颜色（FAELSTM用红色突出显示）
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#17becf', '#d62728']
    
    spatio = config['spatial_resolution']
    lats = np.load(os.path.join(config['out_path'], f'lat_{spatio}.npy'))
    lons = np.load(os.path.join(config['out_path'], f'lon_{spatio}.npy'))
    mask = np.load(os.path.join(config['out_path'], f"Mask with {spatio} spatial resolution.npy"))
    
    # 加载观测值和各模型预测值
    data_pth = os.path.join(config['out_path'], config['process'])
    y_true = np.load(os.path.join(data_pth, modelnames[0], str(config['forecast_time']), 'observations.npy'))
    
    pred_list = []
    for model in modelnames:
        y_pred = np.load(os.path.join(data_pth, model, str(config['forecast_time']), '_predictions.npy'))
        pred_list.append(y_pred)
    
    # 定义5个代表性点的大致位置（纬度，经度）
    target_regions = [
        ('Northwest', 40, 90),       # 西北干旱区
        ('Southwest', 28, 100),      # 西南湿润区
        ('Southeast Coast', 25, 118),  # 东南沿海
        ('Northeast', 45, 125),      # 东北地区
        ('Central', 32, 112)         # 中部地区
    ]
    
    # 点的标记符号
    point_markers = ['A', 'B', 'C', 'D', 'E']
    point_colors = ['red', 'blue', 'green', 'purple', 'orange']
    
    # 找到最接近目标位置的有效格点
    selected_points = []
    for region_name, target_lat, target_lon in target_regions:
        lat_idx = np.argmin(np.abs(lats - target_lat))
        lon_idx = np.argmin(np.abs(lons - target_lon))
        
        # 如果该点无效，寻找附近的有效点
        if mask[lat_idx, lon_idx] != 1:
            found = False
            for d in range(1, 15):
                for di in range(-d, d+1):
                    for dj in range(-d, d+1):
                        ni, nj = lat_idx + di, lon_idx + dj
                        if 0 <= ni < len(lats) and 0 <= nj < len(lons):
                            if mask[ni, nj] == 1:
                                lat_idx, lon_idx = ni, nj
                                found = True
                                break
                    if found:
                        break
                if found:
                    break
        
        selected_points.append({
            'name': region_name,
            'lat_idx': lat_idx,
            'lon_idx': lon_idx,
            'lat': lats[lat_idx],
            'lon': lons[lon_idx]
        })
    
    # ==================== 图1：点位置图 ====================
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8), dpi=150)
    
    lon_min, lon_max = lons.min() - 2, lons.max() + 2
    lat_min, lat_max = lats.min() - 2, lats.max() + 2
    
    m = Basemap(projection='cyl', llcrnrlon=lon_min, llcrnrlat=lat_min,
                urcrnrlon=lon_max, urcrnrlat=lat_max, ax=ax1)
    try:
        m.readshapefile(shapefile=sf, name='states', drawbounds=True)
    except:
        m.drawcoastlines(linewidth=0.5)
        m.drawcountries(linewidth=0.5)
    m.drawparallels(np.arange(-90., 91, 10.), labels=[True, False, False, False], fontsize=10)
    m.drawmeridians(np.arange(-180., 181., 10.), labels=[False, False, False, True], fontsize=10)
    
    # 标记5个点
    for idx, point in enumerate(selected_points):
        x, y = m(point['lon'], point['lat'])
        ax1.plot(x, y, 'o', markersize=15, color=point_colors[idx], markeredgecolor='black', markeredgewidth=2)
        ax1.text(x + 1.5, y + 1.5, f'{point_markers[idx]}', fontsize=14, fontweight='bold', color=point_colors[idx])
    
    # 添加图例
    legend_text = '\n'.join([f'{point_markers[i]}: {selected_points[i]["name"]}' for i in range(5)])
    ax1.text(0.02, 0.98, legend_text, transform=ax1.transAxes, fontsize=10, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.show()
    print('Fig 13a (点位置图) completed!')
    
    # ==================== 图2：前3个点的时间序列 ====================
    n_days = y_true.shape[0]
    days = np.arange(1, n_days + 1)
    
    fig2, axes2 = plt.subplots(3, 1, figsize=(14, 10), dpi=150)
    
    for idx in range(3):
        ax = axes2[idx]
        point = selected_points[idx]
        lat_i, lon_i = point['lat_idx'], point['lon_idx']
        
        # 提取该点的时间序列
        obs_series = y_true[:, lat_i, lon_i]
        if len(obs_series.shape) > 1:
            obs_series = obs_series[:, 0]
        
        # 绘制观测值（黑色粗线）
        ax.plot(days, obs_series, 'k-', linewidth=2.5, label='Observation', zorder=10)
        
        # 绘制各模型预测值
        for m_idx, (pred, name, color) in enumerate(zip(pred_list, modelname_display, colors)):
            pred_series = pred[:, lat_i, lon_i]
            if len(pred_series.shape) > 1:
                pred_series = pred_series[:, 0]
            lw = 2.0 if name == 'FAELSTM' else 1.2
            ax.plot(days, pred_series, color=color, linewidth=lw, label=name, alpha=0.8)
        
        ax.set_title(f'{point_markers[idx]}: {point["name"]} (Lat: {point["lat"]:.1f}°, Lon: {point["lon"]:.1f}°)', 
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('SMCI', fontsize=10)
        ax.set_xlim(1, n_days)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        
        if idx == 2:
            ax.set_xlabel('Day of Year', fontsize=10)
    
    handles, labels = axes2[0].get_legend_handles_labels()
    fig2.legend(handles, labels, loc='upper center', ncol=8, fontsize=9, 
                bbox_to_anchor=(0.5, 0.99), frameon=True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    print('Fig 13b (前3个点时间序列) completed!')
    
    # ==================== 图3：后2个点的时间序列 ====================
    fig3, axes3 = plt.subplots(2, 1, figsize=(14, 7), dpi=150)
    
    for idx in range(2):
        ax = axes3[idx]
        point = selected_points[idx + 3]  # 第4、5个点
        lat_i, lon_i = point['lat_idx'], point['lon_idx']
        
        obs_series = y_true[:, lat_i, lon_i]
        if len(obs_series.shape) > 1:
            obs_series = obs_series[:, 0]
        
        ax.plot(days, obs_series, 'k-', linewidth=2.5, label='Observation', zorder=10)
        
        for m_idx, (pred, name, color) in enumerate(zip(pred_list, modelname_display, colors)):
            pred_series = pred[:, lat_i, lon_i]
            if len(pred_series.shape) > 1:
                pred_series = pred_series[:, 0]
            lw = 2.0 if name == 'FAELSTM' else 1.2
            ax.plot(days, pred_series, color=color, linewidth=lw, label=name, alpha=0.8)
        
        ax.set_title(f'{point_markers[idx + 3]}: {point["name"]} (Lat: {point["lat"]:.1f}°, Lon: {point["lon"]:.1f}°)', 
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('SMCI', fontsize=10)
        ax.set_xlim(1, n_days)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        
        if idx == 1:
            ax.set_xlabel('Day of Year', fontsize=10)
    
    handles, labels = axes3[0].get_legend_handles_labels()
    fig3.legend(handles, labels, loc='upper center', ncol=8, fontsize=9, 
                bbox_to_anchor=(0.5, 0.99), frameon=True)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    print('Fig 13c (后2个点时间序列) completed!')


def fig14_confusion_matrix(config):
    """
    Fig 14: 7个模型对不同SMCI等级的混淆矩阵
    每个混淆矩阵下方展示宏平均和微平均的F1 Score
    SMCI分为5个等级: ND(>=0.4), MD(0.3-0.4), MOD(0.2-0.3), SD(0.1-0.2), ED(<0.1)
    """
    from sklearn.metrics import f1_score, confusion_matrix
    import seaborn as sns
    
    modelnames = ['MLP', 'LSTM', 'BiLSTM', 'TCN', 'AttnLSTM', 'CNNTransformer', 'STALSTM']
    modelname_display = ['MLP', 'LSTM', 'BiLSTM', 'TCN', 'AttnLSTM', 'CNNTrans', 'FAELSTM']
    
    # 加载观测值和各模型预测值
    data_pth = os.path.join(config['out_path'], config['process'])
    y_true = np.load(os.path.join(data_pth, modelnames[0], str(config['forecast_time']), 'observations.npy'))
    
    pred_list = []
    for model in modelnames:
        y_pred = np.load(os.path.join(data_pth, model, str(config['forecast_time']), '_predictions.npy'))
        pred_list.append(y_pred)
    
    # SMCI分类函数
    def classify_smci(data):
        data = data.copy()
        result = np.zeros_like(data)
        result[data >= 0.4] = 1      # ND (No Drought)
        result[(data >= 0.3) & (data < 0.4)] = 2  # MD (Mild Drought)
        result[(data >= 0.2) & (data < 0.3)] = 3  # MOD (Moderate Drought)
        result[(data >= 0.1) & (data < 0.2)] = 4  # SD (Severe Drought)
        result[data < 0.1] = 5       # ED (Extreme Drought)
        return result
    
    # 分类
    y_true_class = classify_smci(y_true)
    pred_class_list = [classify_smci(pred) for pred in pred_list]
    
    # 有效数据掩膜
    valid_mask = ~np.isnan(y_true)
    y_true_flat = y_true_class[valid_mask].flatten()
    
    class_names = ['ND', 'MD', 'MOD', 'SD', 'ED']
    
    # ==================== 第一组：前4个模型 (MLP, LSTM, BiLSTM, TCN) ====================
    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 10), dpi=150)
    axes1 = axes1.flatten()
    
    for idx in range(4):
        ax = axes1[idx]
        pred_class = pred_class_list[idx]
        name = modelname_display[idx]
        pred_flat = pred_class[valid_mask].flatten()
        
        cm = confusion_matrix(y_true_flat, pred_flat, labels=[1, 2, 3, 4, 5])
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        f1_macro = f1_score(y_true_flat, pred_flat, average='macro')
        f1_micro = f1_score(y_true_flat, pred_flat, average='micro')
        
        annot_matrix = np.array([[f"{value:.1f}%" for value in row] for row in cm_percent])
        sns.heatmap(cm_percent, annot=annot_matrix, fmt='', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names, 
                    ax=ax, annot_kws={"fontsize": 9, "fontweight": "bold"},
                    cbar=True, vmin=0, vmax=100)
        
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('Actual', fontsize=10)
        ax.set_title(f'{name}\nF1(macro)={f1_macro:.3f}, F1(micro)={f1_micro:.3f}', 
                     fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    print('Fig 14a (MLP, LSTM, BiLSTM, TCN) completed!')
    
    # ==================== 第二组：后3个模型 (AttnLSTM, CNNTrans, FAELSTM) ====================
    # 使用 GridSpec 实现 FAELSTM 居中
    from matplotlib.gridspec import GridSpec
    
    fig2 = plt.figure(figsize=(12, 12), dpi=150)
    gs = GridSpec(2, 2, figure=fig2, height_ratios=[1, 1], hspace=0.35)
    
    # 第一行：AttnLSTM, CNNTrans
    for idx, pos in enumerate([4, 5]):
        ax = fig2.add_subplot(gs[0, idx])
        pred_class = pred_class_list[pos]
        name = modelname_display[pos]
        pred_flat = pred_class[valid_mask].flatten()
        
        cm = confusion_matrix(y_true_flat, pred_flat, labels=[1, 2, 3, 4, 5])
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        f1_macro = f1_score(y_true_flat, pred_flat, average='macro')
        f1_micro = f1_score(y_true_flat, pred_flat, average='micro')
        
        annot_matrix = np.array([[f"{value:.1f}%" for value in row] for row in cm_percent])
        sns.heatmap(cm_percent, annot=annot_matrix, fmt='', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names, 
                    ax=ax, annot_kws={"fontsize": 9, "fontweight": "bold"},
                    cbar=True, vmin=0, vmax=100)
        
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('Actual', fontsize=10)
        ax.set_title(f'{name}\nF1(macro)={f1_macro:.3f}, F1(micro)={f1_micro:.3f}', 
                     fontsize=11, fontweight='bold')
    
    # 第二行：FAELSTM 居中（跨两列）
    ax_faelstm = fig2.add_subplot(gs[1, :])
    
    pred_class = pred_class_list[6]
    name = modelname_display[6]
    pred_flat = pred_class[valid_mask].flatten()
    
    cm = confusion_matrix(y_true_flat, pred_flat, labels=[1, 2, 3, 4, 5])
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    f1_macro = f1_score(y_true_flat, pred_flat, average='macro')
    f1_micro = f1_score(y_true_flat, pred_flat, average='micro')
    
    annot_matrix = np.array([[f"{value:.1f}%" for value in row] for row in cm_percent])
    sns.heatmap(cm_percent, annot=annot_matrix, fmt='', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, 
                ax=ax_faelstm, annot_kws={"fontsize": 10, "fontweight": "bold"},
                cbar=True, vmin=0, vmax=100)
    
    ax_faelstm.set_xlabel('Predicted', fontsize=11)
    ax_faelstm.set_ylabel('Actual', fontsize=11)
    ax_faelstm.set_title(f'{name}\nF1(macro)={f1_macro:.3f}, F1(micro)={f1_micro:.3f}', 
                         fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    print('Fig 14b (AttnLSTM, CNNTrans, FAELSTM) completed!')
    
    # 打印F1 Score汇总
    print('\n========== F1 Score Summary ==========')
    print(f'{"Model":<15} {"F1(macro)":<12} {"F1(micro)":<12}')
    print('-' * 40)
    for idx, name in enumerate(modelname_display):
        pred_flat = pred_class_list[idx][valid_mask].flatten()
        f1_macro = f1_score(y_true_flat, pred_flat, average='macro')
        f1_micro = f1_score(y_true_flat, pred_flat, average='micro')
        print(f'{name:<15} {f1_macro:<12.4f} {f1_micro:<12.4f}')
    print('=' * 40)
    print('Fig 14 completed!')


# 分类指标对比
def clsscom(y_true, y_pred0, y_pred1, y_pred2, y_pred3, modelname, config):
    from sklearn.metrics import f1_score
    modelname[-1] = 'FAELSTM'
    plt.rc('font', family='Times New Roman')
    plt.rcParams['font.size'] = 12
    y_true[y_true >= 0.4] = 1
    y_true[(y_true >= 0.3) & (y_true < 0.4)] = 2
    y_true[(y_true >= 0.2) & (y_true < 0.3)] = 3
    y_true[(y_true >= 0.1) & (y_true < 0.2)] = 4
    y_true[(y_true < 0.1)] = 5

    y_pred0[y_pred0 >= 0.4] = 1
    y_pred0[(y_pred0 >= 0.3) & (y_pred0 < 0.4)] = 2
    y_pred0[(y_pred0 >= 0.2) & (y_pred0 < 0.3)] = 3
    y_pred0[(y_pred0 >= 0.1) & (y_pred0 < 0.2)] = 4
    y_pred0[(y_pred0 < 0.1)] = 5

    y_pred1[y_pred1 >= 0.4] = 1
    y_pred1[(y_pred1 >= 0.3) & (y_pred1 < 0.4)] = 2
    y_pred1[(y_pred1 >= 0.2) & (y_pred1 < 0.3)] = 3
    y_pred1[(y_pred1 >= 0.1) & (y_pred1 < 0.2)] = 4
    y_pred1[(y_pred1 < 0.1)] = 5

    y_pred2[y_pred2 >= 0.4] = 1
    y_pred2[(y_pred2 >= 0.3) & (y_pred2 < 0.4)] = 2
    y_pred2[(y_pred2 >= 0.2) & (y_pred2 < 0.3)] = 3
    y_pred2[(y_pred2 >= 0.1) & (y_pred2 < 0.2)] = 4
    y_pred2[(y_pred2 < 0.1)] = 5

    y_pred3[y_pred3 >= 0.4] = 1
    y_pred3[(y_pred3 >= 0.3) & (y_pred3 < 0.4)] = 2
    y_pred3[(y_pred3 >= 0.2) & (y_pred3 < 0.3)] = 3
    y_pred3[(y_pred3 >= 0.1) & (y_pred3 < 0.2)] = 4
    y_pred3[(y_pred3 < 0.1)] = 5

    mask = y_true == y_true
    avg = 'macro'
    print('--------------------------------------------')
    print(f'The {avg} of MLP is ', f1_score(y_true[mask == 1], y_pred0[mask == 1], average=avg))
    print(f'The {avg} of LSTM is ', f1_score(y_true[mask == 1], y_pred1[mask == 1], average=avg))
    print(f'The {avg} of BiLSTM is ', f1_score(y_true[mask == 1], y_pred2[mask == 1], average=avg))
    print(f'The {avg} of FAELSTM is ', f1_score(y_true[mask == 1], y_pred3[mask == 1], average=avg))
    print('--------------------------------------------')
    avg = 'micro'
    print('--------------------------------------------')
    print(f'The {avg} of MLP is ', f1_score(y_true[mask == 1], y_pred0[mask == 1], average=avg))
    print(f'The {avg} of LSTM is ', f1_score(y_true[mask == 1], y_pred1[mask == 1], average=avg))
    print(f'The {avg} of BiLSTM is ', f1_score(y_true[mask == 1], y_pred2[mask == 1], average=avg))
    print(f'The {avg} of FAELSTM is ', f1_score(y_true[mask == 1], y_pred3[mask == 1], average=avg))
    print('--------------------------------------------')

    for idx, pred in enumerate([y_pred0, y_pred1, y_pred2, y_pred3]):

        matrix = confusion_matrix(y_true[mask == 1], pred[mask == 1])
        # 将混淆矩阵的值转换为百分比
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis] * 100
        class_names = ['ND', 'MD', 'MOD', 'SD', 'ED']

        # 格式化矩阵中的值为字符串，保留两位小数并添加百分号
        annot_matrix = np.array([[f"{value:.2f}%" for value in row] for row in matrix])

        # 绘制混淆矩阵
        plt.figure(figsize=(16, 10), dpi=200)
        heatmap = sns.heatmap(matrix, annot=annot_matrix, fmt='', cmap='Blues', xticklabels=class_names, yticklabels=class_names, annot_kws={"weight": "bold"})

        # 获取颜色条对象
        colorbar = heatmap.collections[0].colorbar
        # 设置颜色条标签格式
        colorbar.set_ticks(np.linspace(matrix.min(), matrix.max(), 6))
        colorbar.set_ticklabels([f'{tick:.2f}%' for tick in np.linspace(0, 100, 6)])

        plt.title(f"{modelname[idx]}", fontweight='bold')
        plt.show()


# 可视化内部结构
def vis(config):
    import torch
    path = os.path.join(config['out_path'], config['process'])
    model = torch.load(os.path.join(path, config['modelname'], str(config['forecast_time']), config['modelname'] + '_para.pkl'))

    x = torch.randn(size=(config['batch_size'], config['seq_len'], config['input_size'])).to(config['device'])
    temporal_attn = model.attn1(x)[0].cpu().detach().numpy()
    spatia_attn = model.attn2(x.permute(0, 2, 1))[0].cpu().detach().numpy()
    plt.imshow(temporal_attn)

    # 审稿人让加的变量权重图


def visualize_attention_mechanism(config):
    """
    可视化注意力机制：特征重要性 (Feature Attention) 和 时间依赖 (Temporal Attention)
    """
    import torch
    import seaborn as sns
    import matplotlib.pyplot as plt

    print("正在生成注意力机制可视化图...")

    # 1. 加载模型
    path = os.path.join(config['out_path'], config['process'])
    # 确保加载的是 FAELSTM (即代码中的 STALSTM)
    real_model_name = 'STALSTM'
    model_path = os.path.join(path, real_model_name, str(config['forecast_time']), real_model_name + '_para.pkl')

    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {model_path}")
        return

    # 加载模型
    device = config['device']
    model = torch.load(model_path, map_location=device)
    model.eval()

    # 2. 加载真实的测试集数据：复用 main.py/data.py 的数据流程
    # 从 config['out_path'] 读取归一化后的 x_test_norm
    try:
        print("加载 x_test_norm 用于注意力可视化...")
        out_path = config['out_path']

        # 读取测试集形状
        x_test_shape_path = os.path.join(out_path, 'x_test_norm_shape.npy')
        x_test_norm_path = os.path.join(out_path, 'x_test_norm.npy')

        if (not os.path.exists(x_test_shape_path)) or (not os.path.exists(x_test_norm_path)):
            print(f"错误：找不到 x_test_norm 相关文件: {x_test_shape_path} 或 {x_test_norm_path}")
            return

        x_test_shape = np.load(x_test_shape_path, mmap_mode='r')
        # x_test_norm 的形状通常为 (N_time, nlat, nlon, n_features_total)
        x_test_mem = np.memmap(x_test_norm_path, dtype=config['data_type'], mode='r',
                               shape=(x_test_shape[0], x_test_shape[1], x_test_shape[2], x_test_shape[3]))

        N_time, nlat, nlon, nfeat_total = x_test_mem.shape
        print(f"x_test_norm shape: {x_test_mem.shape}")

        # 读取空间掩膜，选择有效格点
        mask_path = os.path.join(out_path, f"Mask with {config['spatial_resolution']} spatial resolution.npy")
        if not os.path.exists(mask_path):
            print(f"警告：找不到掩膜文件 {mask_path}，将使用全部格点")
            valid_indices = [(i, j) for i in range(nlat) for j in range(nlon)]
        else:
            mask = np.load(mask_path)
            valid_indices = list(zip(*np.where(mask == 1)))

        if len(valid_indices) == 0:
            print("错误：掩膜中没有有效格点，无法构造时间序列样本")
            return

        # 从有效格点中选取若干个代表点
        max_grids = 20
        if len(valid_indices) > max_grids:
            valid_indices = valid_indices[:max_grids]

        seq_len = config['seq_len']
        # 模型结构是按 config['input_size'] 定义的，比如 Linear(11, 128)，
        # 因此这里以 config['input_size'] 为准；如果实际特征数不足，则在最后一维做零填充
        input_size = config['input_size']

        # 从 x_test_norm 中按时间滑窗构造 (batch, seq_len, input_size)
        samples = []
        max_samples = 256  # 控制可视化样本数量，避免过大

        for (lat_idx, lon_idx) in valid_indices:
            # 取该格点的时间序列，形状 (N_time, nfeat_total)
            series = x_test_mem[:, lat_idx, lon_idx, :]

            # 若实际特征数 >= input_size：直接截取前 input_size 个特征
            if series.shape[1] >= input_size:
                series = series[:, :input_size]
            else:
                # 若实际特征数 < input_size：在最后一维补零，使其达到 input_size
                pad_dim = input_size - series.shape[1]
                pad_zeros = np.zeros((series.shape[0], pad_dim), dtype=series.dtype)
                series = np.concatenate([series, pad_zeros], axis=1)

            # 时间滑窗
            for t in range(0, N_time - seq_len + 1):
                window = series[t:t + seq_len, :]
                samples.append(window)
                if len(samples) >= max_samples:
                    break
            if len(samples) >= max_samples:
                break

        if len(samples) == 0:
            print("错误：未能从 x_test_norm 构造任何样本，请检查 N_time 与 seq_len 的设置")
            return

        X_test = np.stack(samples, axis=0)  # (batch, seq_len, input_size)
        print("构造用于注意力可视化的 X_test 形状:", X_test.shape)

    except Exception as e:
        print(f"数据加载出错: {e}")
        return

    # 转换为 Tensor
    x_tensor = torch.from_numpy(X_test).float().to(device)

    # 3. 获取真实注意力权重
    temporal_weights = []
    feature_weights = []

    with torch.no_grad():
        # FAELSTM (STALSTMModel) 的 attn1 和 attn2 是公开属性，可以直接调用

        # --- 时间注意力 ---
        # model.attn1 返回 (Batch, Seq_Len, Seq_Len)
        t_attn_out = model.attn1(x_tensor)
        # 取平均值降维 -> (Batch, Seq_Len)
        temporal_weights = torch.mean(t_attn_out, dim=2).cpu().numpy()

        # --- 特征注意力 ---
        # model.attn2 需要输入 (Batch, Channels, Seq_Len)
        # model.attn2 返回 (Batch, Channels, Channels)
        f_attn_out = model.attn2(x_tensor.permute(0, 2, 1))
        # 取平均值降维 -> (Batch, Channels)
        feature_weights = torch.mean(f_attn_out, dim=2).cpu().numpy()

    # 4. 绘图 1: 特征重要性 (Feature Importance)
    # 计算所有样本的平均特征权重
    avg_feat_imp = np.mean(feature_weights, axis=0)
    # 归一化 (Min-Max Scaling)
    avg_feat_imp = (avg_feat_imp - avg_feat_imp.min()) / (avg_feat_imp.max() - avg_feat_imp.min() + 1e-9)

    feature_names = ['t2m', 'u', 'v', 'pre', 'ssr', 'spec', 'ssrd', 'strd', 'stl1', 'e', 'swc']

    # 确保特征名数量匹配
    if len(avg_feat_imp) != len(feature_names):
        print(f"警告：特征权重数量 ({len(avg_feat_imp)}) 与 特征名数量 ({len(feature_names)}) 不匹配，截断或填充")
        feature_names = feature_names[:len(avg_feat_imp)]

    plt.figure(figsize=(10, 6), dpi=300)
    # 排序
    sorted_idx = np.argsort(avg_feat_imp)[::-1]
    sns.barplot(x=np.array(feature_names)[sorted_idx], y=avg_feat_imp[sorted_idx], palette="viridis")
    plt.title("Variable Importance (Feature Attention)", fontsize=14, fontweight='bold')
    plt.ylabel("Normalized Attention Weight")
    plt.xlabel("Meteorological Variables")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(config['out_path'], 'Fig15_Feature_Importance.png'))
    plt.show()

    # 5. 绘图 2: 时间依赖热力图 (Temporal Attention Heatmap)
    # 选取前 30 个样本画热力图
    subset_temporal = temporal_weights[:30, :]

    plt.figure(figsize=(10, 8), dpi=300)
    sns.heatmap(subset_temporal, cmap="Reds", cbar_kws={'label': 'Attention Weight'})
    plt.title("Temporal Attention Weights (Lag Impact)", fontsize=14, fontweight='bold')
    plt.xlabel("Time Lag (Days into the past)")
    plt.ylabel("Sample Index")
    # 设置X轴标签
    ticks = np.arange(0, subset_temporal.shape[1], 5)
    labels = np.arange(subset_temporal.shape[1], 0, -5)
    plt.xticks(ticks=ticks + 0.5, labels=labels)

    plt.tight_layout()
    plt.savefig(os.path.join(config['out_path'], 'Fig16_Temporal_Attention.png'))
    plt.show()
    print("注意力机制图表已生成完毕！")

    
if __name__ == '__main__':
    # ==================== 修改开始 (扩展为7个模型) ====================
    # 原代码：
    # y_true, y_pred0, y_pred1, y_pred2, y_pred3, modelname = load(config)
    # 新代码：
    y_true, y_pred0, y_pred1, y_pred2, y_pred3, y_pred4, y_pred5, y_pred6, modelname = load(config)
    # ==================== 修改结束 ====================
    
    # plotcorr()
    # ==================== 修改开始 (扩展为7个模型) ====================
    # 原代码：
    # plotbox(y_true, y_pred0, y_pred1, y_pred2, y_pred3, modelname, config)
    # 新代码：
    plotbox(y_true, y_pred0, y_pred1, y_pred2, y_pred3, y_pred4, y_pred5, y_pred6, modelname, config)
    # ==================== 修改结束 ====================
    
    # 泰勒图（7个模型）
    # plotayler_7models(y_true, y_pred0, y_pred1, y_pred2, y_pred3, y_pred4, y_pred5, y_pred6, modelname, config)
    
    # Fig 11: 空间对比图（观测值 + 7个模型的R值空间分布）
    # 生成预测1天的图
    config['forecast_time'] = 0  # 0 表示预测第1天
    spatial_comparison_7models(config, day_index=100, forecast_day=1,
                               save_path=r'E:\end\all\imgs\imgs\fig11_forecast_1day.png')

    # # 生成预测7天的图
    config['forecast_time'] = 6  # 6 表示预测第7天
    spatial_comparison_7models(config, day_index=100, forecast_day=7,
                               save_path=r'E:\end\all\imgs\imgs\fig11_forecast_7day.png')
    #
    # Fig 12: 预测值与真实值对比图
    # 生成预测1天的图
    config['forecast_time'] = 0
    fig12_prediction_comparison(config, day_index=100, forecast_day=1,
                                save_path=r'E:\end\all\imgs\imgs\fig12_forecast_1day.png')
    #
    # # 生成预测7天的图
    config['forecast_time'] = 6
    fig12_prediction_comparison(config, day_index=100, forecast_day=7,
                                save_path=r'E:\end\all\imgs\imgs\fig12_forecast_7day.png')
    #
    # Fig 13: 5个代表性点的时间序列对比图
    fig13_timeseries_comparison(config)
    
    # Fig 14: 7个模型的混淆矩阵（SMCI分类精度）
    fig14_confusion_matrix(config)
    
    # 注意：以下函数如需使用，也需要相应修改为7个模型
    # plotayler(y_true, y_pred0, y_pred1, y_pred2, y_pred3, modelname, config)
    # plotscatter(y_true, y_pred0, y_pred1, y_pred2, y_pred3, modelname, config)
    # clsscom(y_true, y_pred0, y_pred1, y_pred2, y_pred3, modelname, config)
    # r(modelname, config)
    # vis(config)

    # 在 main 中调用
    visualize_attention_mechanism(config)
