import os
import numpy as np
import matplotlib.pyplot as plt
from config import get_args

# 加载配置，获取路径和分辨率
cfg = get_args()
out_path = cfg['out_path']          # /root/.../agriculture/
process = cfg['process']            # 'smci'
spatial_res = cfg['spatial_resolution']
mask_file = os.path.join(out_path, f"Mask with {spatial_res} spatial resolution.npy")
mask = np.load(mask_file)
valid_mask = (mask == 1)            # 有效格点布尔掩码

# 模型名称及显示名
models = ['LSTM', 'BiLSTM', 'STALSTM']
display_names = ['LSTM', 'BiLSTM', 'FAELSTM']
metrics = ['r2', 'KGE', 'urmse']    # 我们关注 R², KGE, unbiased RMSE
forecast_times = [0, 6]             # 0=1天, 6=7天
forecast_labels = ['1 day', '7 days']

# 存储数据： dict[metric][forecast][model] = list of values
data = {metric: {ft: {model: [] for model in models} for ft in forecast_times} for metric in metrics}

# 读取数据
for model in models:
    for ft in forecast_times:
        ft_dir = os.path.join(out_path, process, model, str(ft))
        for metric in metrics:
            # 指标文件名
            if metric == 'r2':
                fname = f'r2_{model}.npy'
            elif metric == 'KGE':
                fname = f'KGE_{model}.npy'
            elif metric == 'urmse':
                fname = f'urmse_{model}.npy'
            else:
                continue
            filepath = os.path.join(ft_dir, fname)
            if os.path.exists(filepath):
                arr = np.load(filepath)
                # 提取有效格点的值（去掉 NaN 和 inf）
                vals = arr[valid_mask]
                vals = vals[np.isfinite(vals)]
                # 对于 R² 和 KGE，只取 <=1 的值（过滤异常高值，若有）
                if metric in ['r2', 'KGE']:
                    vals = vals[vals <= 1.0]
                data[metric][ft][model] = vals.tolist()
            else:
                print(f"Warning: {filepath} not found, skipping.")
                data[metric][ft][model] = []

# 定义绘图函数
def plot_boxplot(metric, ft_idx, ft_label, models, display_names, data, out_dir):
    """
    metric: 'r2', 'KGE', 'urmse'
    ft_idx: 0 or 1 对应 forecast_times
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    values_to_plot = []
    for model in models:
        vals = data[metric][forecast_times[ft_idx]][model]
        if len(vals) == 0:
            values_to_plot.append([np.nan])
        else:
            values_to_plot.append(vals)
    bp = ax.boxplot(values_to_plot, labels=display_names, patch_artist=True,
                    showfliers=False, notch=True,
                    boxprops=dict(facecolor='lightblue', color='black'),
                    medianprops=dict(color='red', linewidth=2))
    # 设置 y 轴标签
    ylabel_dict = {'r2': '$R^2$', 'KGE': 'KGE', 'urmse': 'ubRMSE (m³/m³)'}
    ax.set_ylabel(ylabel_dict.get(metric, metric), fontsize=12)
    ax.set_title(f'{metric.upper()} – {ft_label}', fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    # 保存图片
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    save_path = os.path.join(out_dir, f'boxplot_{metric}_{ft_label.replace(" ", "_")}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

# 创建输出目录
out_dir = '/tmp/pycharm_project_721/model/drought-agriculture/output_plots'
# 生成所有箱线图
for metric in metrics:
    for i, ft_label in enumerate(forecast_labels):
        plot_boxplot(metric, i, ft_label, models, display_names, data, out_dir)

print("All box plots generated successfully!")