import json
import numpy as np
from shapely.wkt import loads
from shapely.geometry import LineString, Point
import os
import matplotlib.pyplot as plt
from scipy.optimize import minimize

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.formatter.useoffset'] = False  
plt.rcParams['axes.formatter.limits'] = (-6, 6)  
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['figure.dpi'] = 100  


def visualize_ideal_points_and_references(point, output_dir='', outliers=None):
 
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    reference_geoms = []
    ideal_points = []  
    intersection_points = []
    centroid_points = []

    for i, ref in enumerate(point['references']):
        geom = loads(ref['wkt'])
        ref_geom = geom.geoms[0] if geom.geom_type == 'MultiPolygon' else geom
        reference_geoms.append(ref_geom)
        coords = list(ref_geom.exterior.coords)
        x_coords, y_coords = zip(*coords)

        ax.fill(x_coords, y_coords, alpha=0.5, edgecolor='black', linewidth=1.5, facecolor=f'C{i}',
                label=f'参考区域 {i + 1}')

        centroid = ref_geom.centroid
        centroid_points.append((centroid.x, centroid.y))
        ax.plot(centroid.x, centroid.y, 'ks', markersize=8, label=f'质心 {i + 1}' if i == 0 else "")

        distance_deg = ref['distance'] / 111000
        line_bearing = ref['bearing']
        line_bearing_rad = np.radians(line_bearing)

        line_dx = np.sin(line_bearing_rad)
        line_dy = np.cos(line_bearing_rad)
        line_length = distance_deg * 10  
        line_end_x = centroid.x + line_length * line_dx
        line_end_y = centroid.y + line_length * line_dy
        line = LineString([(centroid.x, centroid.y), (line_end_x, line_end_y)])
        ax.plot([centroid.x, line_end_x], [centroid.y, line_end_y], 'k--', alpha=0.5)

        boundary_intersection = line.intersection(ref_geom.boundary)
        if boundary_intersection.is_empty:
            start_point = Point(centroid.x, centroid.y)
            nearest_dist = float('inf')
            nearest_x, nearest_y = centroid.x, centroid.y
            for j in range(len(coords) - 1):
                seg = LineString([coords[j], coords[j + 1]])
                dist = start_point.distance(seg)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_point = seg.interpolate(seg.project(start_point))
                    nearest_x, nearest_y = nearest_point.x, nearest_point.y
            start_x, start_y = nearest_x, nearest_y
        else:
            if boundary_intersection.geom_type == 'MultiPoint':
                centroid_point = Point(centroid.x, centroid.y)
                max_dist = max(
                    (centroid_point.distance(p), p) for p in boundary_intersection.geoms
                )
                start_x, start_y = max_dist[1].x, max_dist[1].y
            else:
                start_x, start_y = boundary_intersection.x, boundary_intersection.y

        intersection_points.append((start_x, start_y))
        ax.plot(start_x, start_y, 'go', markersize=10, label=f'边界交点 {i + 1}' if i == 0 else "")

        original_bearing_rad = np.radians(ref['bearing'])
        dx = np.sin(original_bearing_rad)
        dy = np.cos(original_bearing_rad)
        x_prime = start_x + distance_deg * dx
        y_prime = start_y + distance_deg * dy
        ideal_points.append(((x_prime, y_prime), i + 1))

        ax.plot([start_x, x_prime], [start_y, y_prime], 'r-', alpha=0.8, linewidth=2)
        ax.plot(x_prime, y_prime, 'ro', markersize=10, label=f'理想点 {i + 1}' if i == 0 else "")

    if outliers:
        outlier_x, outlier_y = zip(*outliers)
        ax.plot(outlier_x, outlier_y, 'mo', markersize=12, markerfacecolor='none', label='离群点')

    all_x = []
    all_y = []
    for geom in reference_geoms:
        x, y = zip(*list(geom.exterior.coords))
        all_x.extend(x)
        all_y.extend(y)
    for (x, y), _ in ideal_points:
        all_x.append(x)
        all_y.append(y)
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    margin_x = max((x_max - x_min) * 0.3, 0.0001)
    margin_y = max((y_max - y_min) * 0.3, 0.0001)
    ax.set_xlim(x_min - margin_x, x_max + margin_x)
    ax.set_ylim(y_min - margin_y, y_max + margin_y)

    ax.set_aspect('equal')
    ax.set_title(f'点 {point["id"]} 的理想点和参考多边形可视化', fontsize=12)
    ax.set_xlabel('经度', fontsize=10)
    ax.set_ylabel('纬度', fontsize=10)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=9)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f'{point["id"]}_ideal_points_visualization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"可视化结果已保存至: {output_path}")
    return ideal_points


def remove_outliers(ideal_points, z_threshold=1.5):
    if len(ideal_points) <= 2:
        return ideal_points, []
    points_array = np.array([p[0] for p in ideal_points])
    center = np.mean(points_array, axis=0)
    distances = np.sqrt(np.sum((points_array - center) ** 2, axis=1))
    z_scores = np.abs((distances - np.mean(distances)) / np.std(distances))
    non_outliers = [ideal_points[i] for i in range(len(ideal_points)) if z_scores[i] <= z_threshold]
    outliers = [ideal_points[i] for i in range(len(ideal_points)) if z_scores[i] > z_threshold]
    if len(non_outliers) < 2:
        return ideal_points, []
    return non_outliers, outliers


def huber_robust_estimation(ideal_points, delta=0.001):
    if not ideal_points:
        return None, None
    ideal_x = np.array([p[0][0] for p in ideal_points])
    ideal_y = np.array([p[0][1] for p in ideal_points])

    def huber_loss(params):
        x, y = params
        residuals_x = x - ideal_x
        residuals_y = y - ideal_y
        loss_x = np.where(np.abs(residuals_x) <= delta, 0.5 * residuals_x ** 2,
                          delta * (np.abs(residuals_x) - 0.5 * delta))
        loss_y = np.where(np.abs(residuals_y) <= delta, 0.5 * residuals_y ** 2,
                          delta * (np.abs(residuals_y) - 0.5 * delta))
        return np.sum(loss_x + loss_y)

    x0 = np.mean(ideal_x)
    y0 = np.mean(ideal_y)
    result = minimize(huber_loss, [x0, y0], method='L-BFGS-B')
    return result.x[0], result.x[1]


def fermat_point(ideal_points):
    if not ideal_points:
        return None, None

    def distance_sum(params):
        x, y = params
        return sum(np.sqrt((x - px) ** 2 + (y - py) ** 2) for (px, py), _ in ideal_points)

    ideal_x = np.array([p[0][0] for p in ideal_points])
    ideal_y = np.array([p[0][1] for p in ideal_points])
    x0 = np.mean(ideal_x)
    y0 = np.mean(ideal_y)
    result = minimize(distance_sum, [x0, y0], method='L-BFGS-B')
    return result.x[0], result.x[1]


def calculate_optimal_coordinates(data):
    ls_coords = {}
    huber_coords = {}
    fermat_coords = {}
    for point in data['points']:
        point_id = point['id']
        print(f"\n处理点: {point_id}")
        ideal_points_with_regions = visualize_ideal_points_and_references(point)
        filtered_points, outliers = remove_outliers(ideal_points_with_regions)

        if outliers:
            outlier_regions = [r for (p, r) in outliers]
            print(f"点 {point_id} 检测到 {len(outliers)} 个离群点，来自区域: {outlier_regions}")
            outlier_coords = [p for (p, r) in outliers]
            visualize_ideal_points_and_references(point, outliers=outlier_coords)

        target_points = filtered_points if filtered_points else ideal_points_with_regions
        target_coords = [p for (p, r) in target_points]

        if target_coords:
            ls_x = np.mean([p[0] for p in target_coords])
            ls_y = np.mean([p[1] for p in target_coords])
            ls_coords[point_id] = (ls_x, ls_y)
            huber_x, huber_y = huber_robust_estimation(target_points)
            huber_coords[point_id] = (huber_x, huber_y)
            fermat_x, fermat_y = fermat_point(target_points)
            fermat_coords[point_id] = (fermat_x, fermat_y)

            print(f"最小二乘坐标: ({ls_x:.6f}, {ls_y:.6f})")
            print(f"鲁棒估计坐标: ({huber_x:.6f}, {huber_y:.6f})")
            print(f"费马点坐标: ({fermat_x:.6f}, {fermat_y:.6f})")
        else:
            first_ref = point['references'][0] if point['references'] else None
            if first_ref:
                geom = loads(first_ref['wkt'])
                centroid = geom.geoms[0].centroid if geom.geom_type == 'MultiPolygon' else geom.centroid
                centroid_x, centroid_y = centroid.x, centroid.y
                ls_coords[point_id] = (centroid_x, centroid_y)
                huber_coords[point_id] = (centroid_x, centroid_y)
                fermat_coords[point_id] = (centroid_x, centroid_y)
                print(f"无有效理想点，使用参考质心: ({centroid_x:.6f}, {centroid_y:.6f})")
    return ls_coords, huber_coords, fermat_coords


def save_results(results, output_dir='', filename=''):
    os.makedirs(output_dir, exist_ok=True)
    results_serializable = {k: list(v) for k, v in results.items()}
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, ensure_ascii=False, indent=2)
    print(f"\n计算结果已保存至: {output_path}")
    return output_path


def load_input_data(file_path=''):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    data = load_input_data()
    ls_results, huber_results, fermat_results = calculate_optimal_coordinates(data)
    save_results(ls_results, filename='least_squares_results.json')
    save_results(huber_results, filename='huber_robust_results.json')
    save_results(fermat_results, filename='fermat_point_results.json')
    print("\n所有处理完成！")
    return ls_results, huber_results, fermat_results


if __name__ == "__main__":
    main()
