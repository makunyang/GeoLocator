import json
import numpy as np
import pandas as pd
import math
import os

# 导入原有代码中的类（假设原有代码文件名为original_fuzzy_geolocator.py，您需调整路径）
from fuzzy_geolocator import FuzzyGeoLocator

# 北京地区的经纬度到米转换系数（参考原代码）
LONGITUDE_TO_METERS = 85000
LATITUDE_TO_METERS = 111000


def haversine_distance(lon1, lat1, lon2, lat2):
    """
    计算两个经纬度点之间的距离（米）
    """
    # 转换为弧度
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # Haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371000  # 地球平均半径（米）
    return c * r


def load_ground_truth(excel_path, sheet_name='Sheet2', start_row=2, column='C', expected_count=25):
    """
    从Excel加载真实坐标（格式: 'Point (116.40771673 39.96112695)'）
    动态读取从start_row开始的非空单元格，直到收集足够或到文件末尾。
    返回列表[(lon1, lat1), (lon2, lat2), ...]
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    num_rows = len(df)
    print(f"Sheet '{sheet_name}' 有 {num_rows} 行数据")

    gt_coords = []
    col_idx = ord(column) - ord('A')  # C列为2
    current_row = start_row - 1  # 0-based, Excel行3 -> iloc[2]

    while len(gt_coords) < expected_count and current_row < num_rows:
        cell_value = df.iloc[current_row, col_idx]
        if pd.isna(cell_value):  # 空单元格，停止
            print(f"遇到空单元格于行 {current_row + 1}，停止读取")
            break
        if isinstance(cell_value, str) and cell_value.startswith('Point (') and cell_value.endswith(')'):
            try:
                coords_str = cell_value[7:-1].strip()  # 去除'Point (' 和 ')', 并清理空格
                lon, lat = map(float, coords_str.split())  # 以空格分割
                gt_coords.append((lon, lat))
                print(f"解析成功: 行 {current_row + 1} -> ({lon}, {lat})")
            except ValueError:
                print(f"警告: 行 {current_row + 1} 格式无效，无法解析坐标，跳过")
        else:
            print(f"警告: 行 {current_row + 1} 不是预期格式，跳过")

        current_row += 1

    print(f"加载真实坐标: {len(gt_coords)} 个")
    if len(gt_coords) < expected_count:
        print(f"警告: 只找到 {len(gt_coords)} 个坐标，预期 {expected_count} 个。请检查Excel文件。")
    return gt_coords


def calculate_metrics(pred_coords, gt_coords, error_threshold=50):
    """
    计算平均定位误差和误差阈值召回率
    pred_coords和gt_coords均为[(lon, lat)]列表
    """
    if len(pred_coords) != len(gt_coords):
        raise ValueError("预测坐标和真实坐标数量不匹配")

    errors = []
    for pred, gt in zip(pred_coords, gt_coords):
        error = haversine_distance(pred[0], pred[1], gt[0], gt[1])
        errors.append(error)

    avg_error = np.mean(errors)
    recall = np.sum(np.array(errors) < error_threshold) / len(errors) * 100  # 百分比

    return avg_error, recall


def run_geolocator(output_dir, excel_file, fuzziness_levels=None, fusion_method='product'):
    """
    运行FuzzyGeoLocator，获取预测坐标
    fuzziness_levels: None为默认，否则自定义（用于消融）
    fusion_method: 'product'为默认，None为无融合（为每个参照物独立计算坐标，然后取平均）
    """
    locator = FuzzyGeoLocator(grid_resolution=80, output_dir=output_dir, fuzziness_levels=fuzziness_levels)
    data = locator.load_excel_data(excel_file)
    results = locator.process_all_points(data, save_results=True)

    # 如果无融合，修改逻辑：为每个参照物独立计算坐标，然后取平均作为最终坐标
    if fusion_method is None:
        for result in results:
            if 'distributions' in result and len(result['distributions']) > 0:
                x_grid, y_grid = result['grid_info'][:2]
                individual_coords = []
                for dist, _ in result['distributions']:
                    coord, confidence = locator.defuzzify(dist, x_grid, y_grid)
                    if coord is not None:
                        individual_coords.append(coord)

                if individual_coords:
                    # 取多个坐标的平均作为最终坐标
                    avg_coord = np.mean(individual_coords, axis=0)
                    result['final_coordinate'] = (float(avg_coord[0]), float(avg_coord[1]))
                    result['confidence'] = np.mean(
                        [locator.defuzzify(dist, x_grid, y_grid)[1] for dist, _ in result['distributions']])  # 平均置信度
                else:
                    # 如果所有独立defuzzify失败，使用fallback
                    print(f"警告: {result['location_id']} 无有效独立坐标，使用边界点平均")
                    if 'boundary_points' in result and len(result['boundary_points']) > 0:
                        avg_boundary = np.mean(result['boundary_points'], axis=0)
                        result['final_coordinate'] = (float(avg_boundary[0]), float(avg_boundary[1]))
                        result['confidence'] = 0.1
                    else:
                        result['final_coordinate'] = (0.0, 0.0)
                        result['confidence'] = 0.0
            else:
                print(f"警告: {result['location_id']} 无分布，使用fallback")
                if 'boundary_points' in result and len(result['boundary_points']) > 0:
                    avg_boundary = np.mean(result['boundary_points'], axis=0)
                    result['final_coordinate'] = (float(avg_boundary[0]), float(avg_boundary[1]))
                    result['confidence'] = 0.1
                else:
                    result['final_coordinate'] = (0.0, 0.0)
                    result['confidence'] = 0.0

    # 提取预测坐标，确保无None
    pred_coords = []
    for result in results:
        coord = result.get('final_coordinate')
        if coord is None:
            print(f"警告: {result['location_id']} 坐标为None，使用(0,0)")
            coord = (0.0, 0.0)
        pred_coords.append(coord)

    return pred_coords, results


def main():
    # 参数设置（参考论文实验参数）
    excel_file = 'D:\\TTT\\大模型位置推理实验\\beijingtest2\\MCP-Loc-test2\\data-pre\\1\\DeepseekAPI-miaoshu3.xlsx'
    output_base_dir = 'D:\\TTT\\大模型位置推理实验\\beijingtest2\\MCP-Loc-test2\\data-pre\\1'
    gt_excel = 'D:\\TTT\\大模型位置推理实验\\beijingtest2\\实验记录-0.xlsx'
    error_threshold = 50  # 米，参考论文误差阈值召回率

    # 加载真实坐标（假设25个点）
    gt_coords = load_ground_truth(gt_excel)

    # 原模型运行
    print("=== 原模型运行 ===")
    original_dir = os.path.join(output_base_dir, 'original_results')
    pred_original, _ = run_geolocator(original_dir, excel_file)
    avg_error_orig, recall_orig = calculate_metrics(pred_original, gt_coords, error_threshold)
    print(f"原模型 - 平均定位误差: {avg_error_orig:.2f} 米, 误差<{error_threshold}米召回率: {recall_orig:.2f}%")

    # 消融1: 没有模糊建模（delta=0）
    print("=== 消融: 没有模糊建模 ===")
    no_fuzzy_levels = {
        'high': {'delta_theta': 0.0, 'delta_distance': 0.0},
        'medium': {'delta_theta': 0.0, 'delta_distance': 0.0},
        'low': {'delta_theta': 0.0, 'delta_distance': 0.0}
    }
    no_fuzzy_dir = os.path.join(output_base_dir, 'no_fuzzy_results')
    pred_no_fuzzy, _ = run_geolocator(no_fuzzy_dir, excel_file, fuzziness_levels=no_fuzzy_levels)
    avg_error_no_fuzzy, recall_no_fuzzy = calculate_metrics(pred_no_fuzzy, gt_coords, error_threshold)
    print(
        f"无模糊建模 - 平均定位误差: {avg_error_no_fuzzy:.2f} 米, 误差<{error_threshold}米召回率: {recall_no_fuzzy:.2f}%")

    # 消融2: 没有多约束融合（只用第一个参照物）
    print("=== 消融: 没有多约束融合 ===")
    no_fusion_dir = os.path.join(output_base_dir, 'no_fusion_results')
    pred_no_fusion, _ = run_geolocator(no_fusion_dir, excel_file, fusion_method=None)
    avg_error_no_fusion, recall_no_fusion = calculate_metrics(pred_no_fusion, gt_coords, error_threshold)
    print(
        f"无多约束融合 - 平均定位误差: {avg_error_no_fusion:.2f} 米, 误差<{error_threshold}米召回率: {recall_no_fusion:.2f}%")

    # 保存所有指标到CSV
    metrics_df = pd.DataFrame({
        '模型变体': ['原模型', '无模糊建模', '无多约束融合'],
        '平均定位误差 (米)': [avg_error_orig, avg_error_no_fuzzy, avg_error_no_fusion],
        f'误差<{error_threshold}米召回率 (%)': [recall_orig, recall_no_fuzzy, recall_no_fusion]
    })
    metrics_csv = os.path.join(output_base_dir, 'evaluation_metrics.csv')
    metrics_df.to_csv(metrics_csv, index=False, encoding='utf-8-sig')
    print(f"指标保存到: {metrics_csv}")


if __name__ == "__main__":
    main()
