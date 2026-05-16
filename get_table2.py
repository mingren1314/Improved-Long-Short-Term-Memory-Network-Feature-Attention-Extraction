"""
批量获取 Table 2 的所有模型指标
运行此脚本可一次性输出所有模型在不同预测提前期的中值指标
"""
import os
import numpy as np
from config import get_args

# 获取基础配置
config = get_args()

# ==================== 配置区域 ====================
# 选择要对比的模型（可以切换）
# 原始4个模型
modelnames_4 = ['MLP', 'LSTM', 'BiLSTM', 'STALSTM']
display_names_4 = ['MLP', 'LSTM', 'BiLSTM', 'FAELSTM']

# 7个对比模型
modelnames_7 = ['MLP', 'LSTM', 'BiLSTM', 'TCN', 'AttnLSTM', 'CNNTransformer', 'STALSTM']
display_names_7 = ['MLP', 'LSTM', 'BiLSTM', 'TCN', 'AttnLSTM', 'CNNTrans', 'FAELSTM']

# ! 切换这里选择用4个还是7个模型
USE_7_MODELS = True  # True=7个模型, False=原始4个模型

if USE_7_MODELS:
    modelnames = modelnames_7
    display_names = display_names_7
else:
    modelnames = modelnames_4
    display_names = display_names_4

# 预测提前期 (0=1天, 6=7天)
forecast_times = [0, 6]  # 可以添加更多: [0, 2, 4, 6] 表示1天、3天、5天、7天
# ==================== 配置区域结束 ====================


def load_metrics(config, modelname, forecast_time):
    """加载单个模型的所有指标"""
    data_pth = os.path.join(config['out_path'], config['process'], modelname, str(forecast_time))
    
    metrics = {}
    try:
        metrics['r'] = np.load(os.path.join(data_pth, f'r_{modelname}.npy'))
        metrics['r2'] = np.load(os.path.join(data_pth, f'r2_{modelname}.npy'))
        metrics['rmse'] = np.load(os.path.join(data_pth, f'rmse_{modelname}.npy'))
        metrics['urmse'] = np.load(os.path.join(data_pth, f'urmse_{modelname}.npy'))
        metrics['bias'] = np.load(os.path.join(data_pth, f'bias_{modelname}.npy'))
        metrics['KGE'] = np.load(os.path.join(data_pth, f'KGE_{modelname}.npy'))
        metrics['NSE'] = np.load(os.path.join(data_pth, f'NSE_{modelname}.npy'))
    except FileNotFoundError as e:
        print(f"警告: {modelname} 的指标文件不存在，请先运行 postprocess.py")
        print(f"缺失文件: {e}")
        return None
    
    return metrics


def get_median_metrics(metrics, mask):
    """计算各指标的中值"""
    if metrics is None:
        return None
    
    result = {}
    for key, value in metrics.items():
        result[key] = np.nanmedian(value[mask == 1])
    return result


def print_table(results, forecast_time, mask_count):
    """打印表格格式的结果"""
    print("\n" + "=" * 100)
    print(f"Table 2: Comparison of the median results - Forecast {forecast_time + 1} day")
    print(f"(有效格点数: {mask_count})")
    print("=" * 100)
    
    # 表头
    header = f"{'Model':<15} {'R':>8} {'R²':>8} {'Bias':>8} {'RMSE':>8} {'ubrmse':>8} {'KGE':>8} {'NSE':>8}"
    print(header)
    print("-" * 100)
    
    # 数据行
    for model_name, display_name in zip(modelnames, display_names):
        if model_name in results and results[model_name] is not None:
            m = results[model_name]
            row = f"{display_name:<15} {m['r']:>8.3f} {m['r2']:>8.3f} {m['bias']:>8.3f} {m['rmse']:>8.3f} {m['urmse']:>8.3f} {m['KGE']:>8.3f} {m['NSE']:>8.3f}"
            print(row)
        else:
            print(f"{display_name:<15} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8}")
    
    print("=" * 100)


def main():
    # 加载 mask
    mask_file = f"Mask with {config['spatial_resolution']} spatial resolution.npy"
    base_mask = np.load(os.path.join(config['out_path'], mask_file))
    
    print(f"\n使用 {len(modelnames)} 个模型进行对比: {display_names}")
    print(f"基础 mask 有效格点数: {np.sum(base_mask == 1)}")
    
    for ft in forecast_times:
        print(f"\n正在处理 Forecast {ft + 1} day...")
        results = {}
        all_r2 = []
        
        # 加载所有模型的指标，并诊断每个模型的有效格点数
        print("\n  各模型有效格点数诊断:")
        for model_name in modelnames:
            metrics = load_metrics(config, model_name, ft)
            results[model_name] = metrics
            if metrics is not None:
                r2 = metrics['r2']
                valid_count = np.sum((base_mask == 1) & (r2 == r2))  # 非NaN的格点数
                nan_count = np.sum((base_mask == 1) & (r2 != r2))   # NaN的格点数
                print(f"    {model_name}: 有效={valid_count}, NaN={nan_count}")
                all_r2.append(r2)
        
        # 方案选择：True=每个模型单独计算，False=所有模型交集
        USE_INDIVIDUAL_MASK = True
        
        if USE_INDIVIDUAL_MASK:
            # 每个模型单独计算，不受其他模型NaN影响
            print("\n  使用单独mask模式（每个模型独立计算）")
            median_results = {}
            for model_name in modelnames:
                if results[model_name] is not None:
                    r2 = results[model_name]['r2']
                    individual_mask = base_mask * (r2 == r2)
                    mask_count = np.sum(individual_mask == 1)
                    median_results[model_name] = get_median_metrics(results[model_name], individual_mask)
                    print(f"    {model_name}: 使用 {mask_count} 个格点")
                else:
                    median_results[model_name] = None
            mask_count = int(np.mean([np.sum(base_mask * (results[m]['r2'] == results[m]['r2'])) 
                                      for m in modelnames if results[m] is not None]))
        else:
            # 所有模型的有效格点交集
            if all_r2:
                mask = base_mask.copy()
                for r2 in all_r2:
                    mask = mask * (r2 == r2)  # 排除 NaN
                mask_count = np.sum(mask == 1)
                print(f"\n  所有模型交集后有效格点数: {mask_count}")
            else:
                mask = base_mask
                mask_count = np.sum(base_mask == 1)
            
            # 计算中值
            median_results = {}
            for model_name in modelnames:
                if results[model_name] is not None:
                    median_results[model_name] = get_median_metrics(results[model_name], mask)
                else:
                    median_results[model_name] = None
        
        print_table(median_results, ft, mask_count)
    
    print("\n" + "=" * 100)
    print("完成！")
    print("如果7天预测数据异常，请尝试设置 USE_7_MODELS = False 只用原始4个模型")
    print("=" * 100)


if __name__ == '__main__':
    main()
