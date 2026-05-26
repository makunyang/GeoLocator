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
        print(f"⚠️ WKT parsing failed：{str(e)}")
        return None


def calculate_ideal_points(point):
    ideal_points = []
    for ref in point['references']:
        ref_geom = parse_wkt(ref['wkt'])
        if not ref_geom:
            print(f"⚠️  {point['id']} The WKT of a certain reference area is invalid; skipping this area.")
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
        print(f"📥 Successfully read the Excel file：{file_path}（{len(df)}）")
    except FileNotFoundError:
        raise Exception(f"❌ Excel file not found, please check the path：{file_path}")
    except Exception as e:
        raise Exception(f"❌ Failed to read Excel：{str(e)}")


    required_raw_cols = list(EXCEL_COL_MAP.keys())
    missing_cols = [col for col in required_raw_cols if col not in df.columns]
    if missing_cols:
        raise Exception(f"❌ Excel is missing required columns：{missing_cols}\n")


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


    print(f"✅ Data cleaning completed")
    if len(data['points']) == 0:
        raise Exception("❌ No valid data available; please check the values and formatting in Excel.")
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
    print(f"\n📤 Results have been saved to：{result_path}")
    print(f"   - WKT coordinates of {len(result_list)} points")
    return result_path


# -------------------------- 4. Core computational process --------------------------
def calculate_least_squares_coords(data):

    ls_coords = {}

    for point in data['points']:
        point_id = point['id']
        ref_count = len(point['references'])
        print(f"\n" + "-" * 40)
        print(f"Processing point：{point_id}")

        ideal_points = calculate_ideal_points(point)
        if len(ideal_points) == 0:
            print(f"⚠️ {point_id} No valid ideal point, skipping calculation")
            continue
        print(f"   Effective ideal points：{len(ideal_points)}/{ref_count}")


        ls_lon = np.mean([coord[0] for coord in ideal_points])  
        ls_lat = np.mean([coord[1] for coord in ideal_points])  
        ls_coords[point_id] = (ls_lon, ls_lat)

  
        print(f"   Minimum two-seat coordinates：({ls_lon:.6f}, {ls_lat:.6f})")
        print(f"   WKT format：POINT({ls_lon:.6f} {ls_lat:.6f})")

    return ls_coords


# -------------------------- 5. main function --------------------------
def main():
    try:
        print("=" * 50)
        print("  Least Squares Geodetic Coordinate Calculation")

        print("=" * 50)


        data = load_excel_input()


        ls_coords = calculate_least_squares_coords(data)
        if not ls_coords:
            raise Exception("❌ No valid coordinates calculated")


        save_wkt_excel(ls_coords)

        print("\n" + "=" * 50)

        print("=" * 50)
        return ls_coords

    except Exception as e:
        print(f"\n❌ Program execution failed：{str(e)}")
        return None


if __name__ == "__main__":
    final_results = main()
