import numpy as np
from shapely.wkt import loads
from shapely.geometry import LineString, Point
import os
import pandas as pd

# -------------------------- 1. 核心参数配置（用户可按需调整） --------------------------
# 最终结果Excel保存路径
RESULT_OUTPUT_DIR = 'D:\TTT\大模型位置推理实验/beijingtest2\MCP-Loc-test2\jihe-results'
# Excel结果文件名
RESULT_FILENAME = '最小二乘坐标结果（精简版）.xlsx'
# Excel列名映射（关键！需与你的Excel实际列名匹配）
EXCEL_COL_MAP = {
    '点ID': 'point_id',  # 对应每个待计算点的唯一ID
    'WKT': 'wkt',  # 参考区域的WKT格式（如Polygon）
    '距离': 'distance',  # 目标点到参考区域的距离（单位：米）
    '方位角': 'bearing'  # 目标点相对于参考区域的方位角（单位：度）
}


# -------------------------- 2. 核心工具函数（仅保留必要逻辑） --------------------------
def parse_wkt(wkt_str):
    """解析WKT字符串，返回shapely几何对象（处理MultiPolygon兼容）"""
    if pd.isna(wkt_str) or str(wkt_str).strip() == '':
        return None
    try:
        geom = loads(str(wkt_str).strip())
        # 若为MultiPolygon，取第一个Polygon（默认参考区域为单个多边形）
        if geom.geom_type == 'MultiPolygon':
            return geom.geoms[0]
        return geom
    except Exception as e:
        print(f"⚠️ WKT解析失败：{wkt_str}，错误：{str(e)}")
        return None


def calculate_ideal_points(point):
    """
    核心：计算每个参考区域对应的理想点（无可视化）
    返回：理想点坐标列表 [(经度, 纬度), ...]
    """
    ideal_points = []
    for ref in point['references']:
        # 1. 解析参考区域WKT
        ref_geom = parse_wkt(ref['wkt'])
        if not ref_geom:
            print(f"⚠️ 点 {point['id']} 的某个参考区域WKT无效，跳过该区域")
            continue

        # 2. 计算参考区域质心
        centroid = ref_geom.centroid
        exterior_coords = list(ref_geom.exterior.coords)

        # 3. 单位转换：米→经纬度差（111000米≈1度经纬度），角度→弧度
        distance_deg = ref['distance'] / 111000
        bearing_rad = np.radians(ref['bearing'])

        # 4. 计算方位线（延长线，提升交点检测成功率）
        line_length = distance_deg * 10  # 延长线长度
        line_end_x = centroid.x + line_length * np.sin(bearing_rad)
        line_end_y = centroid.y + line_length * np.cos(bearing_rad)

        # 5. 计算质心到参考区域边界的交点（取最远交点确保方向正确）
        boundary_line = LineString(exterior_coords)
        bearing_line = LineString([(centroid.x, centroid.y), (line_end_x, line_end_y)])
        intersection = bearing_line.intersection(boundary_line)

        if intersection.is_empty:
            # 无交点时：取质心到边界的最近点
            nearest_point = boundary_line.interpolate(boundary_line.project(Point(centroid.x, centroid.y)))
            start_x, start_y = nearest_point.x, nearest_point.y
        else:
            # 多交点时：取与质心距离最远的点
            if intersection.geom_type == 'MultiPoint':
                centroid_point = Point(centroid.x, centroid.y)
                max_dist_point = max(intersection.geoms, key=lambda p: centroid_point.distance(p))
                start_x, start_y = max_dist_point.x, max_dist_point.y
            else:
                start_x, start_y = intersection.x, intersection.y

        # 6. 计算理想点（沿方位角方向，距离=输入距离）
        ideal_x = start_x + distance_deg * np.sin(bearing_rad)
        ideal_y = start_y + distance_deg * np.cos(bearing_rad)
        ideal_points.append((ideal_x, ideal_y))

    return ideal_points


# -------------------------- 3. 数据加载与结果保存 --------------------------
def load_excel_input(file_path='D:\TTT\大模型位置推理实验/beijingtest2\MCP-Loc-test2\data-pre/1/Gemini3-miaoshu3.xlsx'):
    """加载Excel输入数据，转换为计算所需结构"""
    # 读取Excel文件
    try:
        df = pd.read_excel(file_path)
        print(f"📥 成功读取Excel文件：{file_path}（原始行数：{len(df)}）")
    except FileNotFoundError:
        raise Exception(f"❌ Excel文件未找到，请检查路径：{file_path}")
    except Exception as e:
        raise Exception(f"❌ 读取Excel失败：{str(e)}")

    # 检查必要列是否存在
    required_raw_cols = list(EXCEL_COL_MAP.keys())
    missing_cols = [col for col in required_raw_cols if col not in df.columns]
    if missing_cols:
        raise Exception(f"❌ Excel缺少必要列：{missing_cols}\n请确保包含列：{required_raw_cols}")

    # 数据清洗：过滤无效行
    df_clean = df.rename(columns=EXCEL_COL_MAP)  # 统一列名
    # 1. 去除空值
    df_clean = df_clean.dropna(subset=['point_id', 'wkt', 'distance', 'bearing'])
    # 2. 确保距离和方位角为数值
    df_clean['distance'] = pd.to_numeric(df_clean['distance'], errors='coerce')
    df_clean['bearing'] = pd.to_numeric(df_clean['bearing'], errors='coerce')
    # 3. 过滤不合理数值（距离≥0，方位角0-360度）
    df_clean = df_clean[(df_clean['distance'] >= 0) &
                        (df_clean['bearing'] >= 0) &
                        (df_clean['bearing'] <= 360)]
    # 4. 点ID转为字符串，避免数字ID格式问题
    df_clean['point_id'] = df_clean['point_id'].astype(str).str.strip()

    # 按点ID分组，构建计算所需结构
    data = {'points': []}
    for point_id, group in df_clean.groupby('point_id'):
        references = []
        for _, row in group.iterrows():
            references.append({
                'wkt': row['wkt'].strip(),
                'distance': float(row['distance']),
                'bearing': float(row['bearing'])
            })
        data['points'].append({
            'id': point_id,
            'references': references
        })

    # 输出清洗结果
    print(f"✅ 数据清洗完成：有效行数 {len(df_clean)} → 有效点数 {len(data['points'])}")
    if len(data['points']) == 0:
        raise Exception("❌ 无有效数据，请检查Excel中的数值和格式")
    return data


def save_wkt_excel(ls_coords):
    """将最小二乘坐标转换为WKT格式，保存到Excel"""
    # 创建结果目录
    os.makedirs(RESULT_OUTPUT_DIR, exist_ok=True)
    result_path = os.path.join(RESULT_OUTPUT_DIR, RESULT_FILENAME)

    # 构建结果数据框
    result_list = []
    for point_id, (lon, lat) in ls_coords.items():
        # 生成WKT格式（保留6位小数，确保地理坐标精度）
        wkt = f"POINT({lon:.6f} {lat:.6f})"
        result_list.append({
            '点ID': point_id,
            '最小二乘经度': round(lon, 6),
            '最小二乘纬度': round(lat, 6),
            'WKT坐标': wkt
        })

    # 保存到Excel
    df_result = pd.DataFrame(result_list)
    df_result.to_excel(result_path, index=False, engine='openpyxl')
    print(f"\n📤 结果已保存至：{result_path}")
    print(f"   - 共 {len(result_list)} 个点的WKT坐标")
    return result_path


# -------------------------- 4. 核心计算流程（无离群点检测） --------------------------
def calculate_least_squares_coords(data):
    """
    精简流程：加载数据 → 计算理想点 → 最小二乘求解
    返回：最小二乘坐标字典 {点ID: (经度, 纬度)}
    """
    ls_coords = {}

    for point in data['points']:
        point_id = point['id']
        ref_count = len(point['references'])
        print(f"\n" + "-" * 40)
        print(f"处理点：{point_id}（参考区域数：{ref_count}）")

        # 1. 计算理想点
        ideal_points = calculate_ideal_points(point)
        if len(ideal_points) == 0:
            print(f"⚠️ 点 {point_id} 无有效理想点，跳过计算")
            continue
        print(f"   有效理想点数：{len(ideal_points)}/{ref_count}")

        # 2. 最小二乘计算（直接取理想点坐标均值，无离群点过滤）
        ls_lon = np.mean([coord[0] for coord in ideal_points])  # 经度均值
        ls_lat = np.mean([coord[1] for coord in ideal_points])  # 纬度均值
        ls_coords[point_id] = (ls_lon, ls_lat)

        # 输出单点点结果
        print(f"   最小二乘坐标：({ls_lon:.6f}, {ls_lat:.6f})")
        print(f"   WKT格式：POINT({ls_lon:.6f} {ls_lat:.6f})")

    return ls_coords


# -------------------------- 5. 主函数（执行入口） --------------------------
def main():
    try:
        print("=" * 50)
        print("  最小二乘地理坐标计算（精简版）")
        print("  功能：Excel输入 → 理想点计算 → WKT Excel输出")
        print("=" * 50)

        # 步骤1：加载Excel数据
        print("\n【步骤1/3】加载并清洗Excel数据...")
        data = load_excel_input()

        # 步骤2：计算最小二乘坐标
        print("\n【步骤2/3】计算各点最小二乘坐标...")
        ls_coords = calculate_least_squares_coords(data)
        if not ls_coords:
            raise Exception("❌ 未计算出任何有效坐标")

        # 步骤3：保存WKT结果到Excel
        print("\n【步骤3/3】保存WKT坐标到Excel...")
        save_wkt_excel(ls_coords)

        print("\n" + "=" * 50)
        print("  所有计算完成！")
        print("=" * 50)
        return ls_coords

    except Exception as e:
        print(f"\n❌ 程序执行失败：{str(e)}")
        return None


# -------------------------- 6. 执行代码 --------------------------
if __name__ == "__main__":
    final_results = main()