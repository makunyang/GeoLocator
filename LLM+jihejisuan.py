import numpy as np
from shapely.wkt import loads
from shapely.geometry import LineString, Point
import os
import pandas as pd

# -------------------------- 1. Core Parameter Configuration --------------------------

RESULT_OUTPUT_DIR = 'OUTPUT_DIR'
# 
RESULT_FILENAME = 'Minimum Two-Seat Coordinate Results .xlsx'
# 
EXCEL_COL_MAP = {
    '点ID': 'point_id',  
    'WKT': 'wkt',  
    '距离': 'distance',  
    '方位角': 'bearing'  
}


# -------------------------- 2. Core utility functions --------------------------
def parse_wkt(wkt_str):
    
    if pd.isna(wkt_str) or str(wkt_str).strip() == '':
        return None
    try:
        geom = loads(str(wkt_str).strip())
        
        if geom.geom_type == 'MultiPolygon':
            return geom.geoms[0]
        return geom
    except Exception as e:
        print(f"⚠️ WKT解析失败：{wkt_str}，错误：{str(e)}")
        return None


def calculate_ideal_points(point):
    ideal_points = []
    for ref in point['references']:
        ref_geom = parse_wkt(ref['wkt'])
        if not ref_geom:
            print(f"⚠️ 点 {point['id']} 的某个参考区域WKT无效，跳过该区域")
            continue

    
        centroid = ref_geom.centroid
        exterior_coords = list(ref_geom.exterior.coords)


        distance_deg = ref['distance'] / 111000
        bearing_rad = np.radians(ref['bearing'])


        line_length = distance_deg * 10 
        line_end_x = centroid.x + line_length * np.sin(bearing_rad)
        line_end_y = centroid.y + line_length * np.cos(bearing_rad)


        boundary_line = LineString(exterior_coords)
        bearing_line = LineString([(centroid.x, centroid.y), (line_end_x, line_end_y)])
        intersection = bearing_line.intersection(boundary_line)

        if intersection.is_empty:
  
            nearest_point = boundary_line.interpolate(boundary_line.project(Point(centroid.x, centroid.y)))
            start_x, start_y = nearest_point.x, nearest_point.y
        else:
 
            if intersection.geom_type == 'MultiPoint':
                centroid_point = Point(centroid.x, centroid.y)
                max_dist_point = max(intersection.geoms, key=lambda p: centroid_point.distance(p))
                start_x, start_y = max_dist_point.x, max_dist_point.y
            else:
                start_x, start_y = intersection.x, intersection.y

 
        ideal_x = start_x + distance_deg * np.sin(bearing_rad)
        ideal_y = start_y + distance_deg * np.cos(bearing_rad)
        ideal_points.append((ideal_x, ideal_y))

    return ideal_points


# -------------------------- 3. Data loading and result saving --------------------------
def load_excel_input(file_path=''):

    try:
        df = pd.read_excel(file_path)
        print(f"📥 成功读取Excel文件：{file_path}（原始行数：{len(df)}）")
    except FileNotFoundError:
        raise Exception(f"❌ Excel文件未找到，请检查路径：{file_path}")
    except Exception as e:
        raise Exception(f"❌ 读取Excel失败：{str(e)}")


    required_raw_cols = list(EXCEL_COL_MAP.keys())
    missing_cols = [col for col in required_raw_cols if col not in df.columns]
    if missing_cols:
        raise Exception(f"❌ Excel缺少必要列：{missing_cols}\n请确保包含列：{required_raw_cols}")


    df_clean = df.rename(columns=EXCEL_COL_MAP)  

    df_clean = df_clean.dropna(subset=['point_id', 'wkt', 'distance', 'bearing'])

    df_clean['distance'] = pd.to_numeric(df_clean['distance'], errors='coerce')
    df_clean['bearing'] = pd.to_numeric(df_clean['bearing'], errors='coerce')

    df_clean = df_clean[(df_clean['distance'] >= 0) &
                        (df_clean['bearing'] >= 0) &
                        (df_clean['bearing'] <= 360)]

    df_clean['point_id'] = df_clean['point_id'].astype(str).str.strip()


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


    print(f"✅ 数据清洗完成：有效行数 {len(df_clean)} → 有效点数 {len(data['points'])}")
    if len(data['points']) == 0:
        raise Exception("❌ 无有效数据，请检查Excel中的数值和格式")
    return data


def save_wkt_excel(ls_coords):

    os.makedirs(RESULT_OUTPUT_DIR, exist_ok=True)
    result_path = os.path.join(RESULT_OUTPUT_DIR, RESULT_FILENAME)


    result_list = []
    for point_id, (lon, lat) in ls_coords.items():

        wkt = f"POINT({lon:.6f} {lat:.6f})"
        result_list.append({
            '点ID': point_id,
            '最小二乘经度': round(lon, 6),
            '最小二乘纬度': round(lat, 6),
            'WKT坐标': wkt
        })


    df_result = pd.DataFrame(result_list)
    df_result.to_excel(result_path, index=False, engine='openpyxl')
    print(f"\n📤 结果已保存至：{result_path}")
    print(f"   - 共 {len(result_list)} 个点的WKT坐标")
    return result_path


# -------------------------- 4. Core computational process --------------------------
def calculate_least_squares_coords(data):

    ls_coords = {}

    for point in data['points']:
        point_id = point['id']
        ref_count = len(point['references'])
        print(f"\n" + "-" * 40)
        print(f"处理点：{point_id}（参考区域数：{ref_count}）")

        ideal_points = calculate_ideal_points(point)
        if len(ideal_points) == 0:
            print(f"⚠️ 点 {point_id} 无有效理想点，跳过计算")
            continue
        print(f"   有效理想点数：{len(ideal_points)}/{ref_count}")


        ls_lon = np.mean([coord[0] for coord in ideal_points])  # 经度均值
        ls_lat = np.mean([coord[1] for coord in ideal_points])  # 纬度均值
        ls_coords[point_id] = (ls_lon, ls_lat)

  
        print(f"   最小二乘坐标：({ls_lon:.6f}, {ls_lat:.6f})")
        print(f"   WKT格式：POINT({ls_lon:.6f} {ls_lat:.6f})")

    return ls_coords


# -------------------------- 5. main function --------------------------
def main():
    try:
        print("=" * 50)
        print("  最小二乘地理坐标计算（精简版）")
        print("  功能：Excel输入 → 理想点计算 → WKT Excel输出")
        print("=" * 50)

        print("\n【步骤1/3】加载并清洗Excel数据...")
        data = load_excel_input()

        print("\n【步骤2/3】计算各点最小二乘坐标...")
        ls_coords = calculate_least_squares_coords(data)
        if not ls_coords:
            raise Exception("❌ 未计算出任何有效坐标")

        print("\n【步骤3/3】保存WKT坐标到Excel...")
        save_wkt_excel(ls_coords)

        print("\n" + "=" * 50)
        print("  所有计算完成！")
        print("=" * 50)
        return ls_coords

    except Exception as e:
        print(f"\n❌ 程序执行失败：{str(e)}")
        return None


if __name__ == "__main__":
    final_results = main()
