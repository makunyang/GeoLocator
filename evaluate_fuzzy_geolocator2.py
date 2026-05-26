import json
import numpy as np
import pandas as pd
import math
import os


from fuzzy_geolocator import FuzzyGeoLocator


LONGITUDE_TO_METERS = 85000
LATITUDE_TO_METERS = 111000


def haversine_distance(lon1, lat1, lon2, lat2):
   
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

 
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371000  
    return c * r


def load_ground_truth(excel_path, sheet_name='Sheet2', start_row=2, column='C', expected_count=25):
    
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    num_rows = len(df)
    print(f"Sheet '{sheet_name}' has {num_rows} data")

    gt_coords = []
    col_idx = ord(column) - ord('A')  
    current_row = start_row - 1  # 0-based, Excel行3 -> iloc[2]

    while len(gt_coords) < expected_count and current_row < num_rows:
        cell_value = df.iloc[current_row, col_idx]
        if pd.isna(cell_value):  

            break
        if isinstance(cell_value, str) and cell_value.startswith('Point (') and cell_value.endswith(')'):
            try:
                coords_str = cell_value[7:-1].strip()  
                lon, lat = map(float, coords_str.split())  
                gt_coords.append((lon, lat))
                print(f"Parsing successful")
            except ValueError:
                print(f"Row {current_row + 1} has invalid format, coordinate cannot be parsed, skipping")
        else:
            print(f" Row {current_row + 1} is not in the expected format, skipping")

        current_row += 1

    print(f"Load real coordinates: {len(gt_coords)} 个")
    if len(gt_coords) < expected_count:
        print(f"Warning: Only  {len(gt_coords)} coordinates found, expected {expected_count} Please check the Excel file.")
    return gt_coords


def calculate_metrics(pred_coords, gt_coords, error_threshold=50):
  
    if len(pred_coords) != len(gt_coords):
        raise ValueError("The number of predicted coordinates does not match the number of true coordinates.")

    errors = []
    for pred, gt in zip(pred_coords, gt_coords):
        error = haversine_distance(pred[0], pred[1], gt[0], gt[1])
        errors.append(error)

    avg_error = np.mean(errors)
    recall = np.sum(np.array(errors) < error_threshold) / len(errors) * 100  

    return avg_error, recall


def run_geolocator(output_dir, excel_file, fuzziness_levels=None, fusion_method='product'):
    
    locator = FuzzyGeoLocator(grid_resolution=80, output_dir=output_dir, fuzziness_levels=fuzziness_levels)
    data = locator.load_excel_data(excel_file)
    results = locator.process_all_points(data, save_results=True)


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
               
                    avg_coord = np.mean(individual_coords, axis=0)
                    result['final_coordinate'] = (float(avg_coord[0]), float(avg_coord[1]))
                    result['confidence'] = np.mean(
                        [locator.defuzzify(dist, x_grid, y_grid)[1] for dist, _ in result['distributions']])  
                else:
           
                    print(f"No valid independent coordinates; using average of boundary points")
                    if 'boundary_points' in result and len(result['boundary_points']) > 0:
                        avg_boundary = np.mean(result['boundary_points'], axis=0)
                        result['final_coordinate'] = (float(avg_boundary[0]), float(avg_boundary[1]))
                        result['confidence'] = 0.1
                    else:
                        result['final_coordinate'] = (0.0, 0.0)
                        result['confidence'] = 0.0
            else:
                print(f" {result['location_id']} No distribution, using fallback")
                if 'boundary_points' in result and len(result['boundary_points']) > 0:
                    avg_boundary = np.mean(result['boundary_points'], axis=0)
                    result['final_coordinate'] = (float(avg_boundary[0]), float(avg_boundary[1]))
                    result['confidence'] = 0.1
                else:
                    result['final_coordinate'] = (0.0, 0.0)
                    result['confidence'] = 0.0

   
    pred_coords = []
    for result in results:
        coord = result.get('final_coordinate')
        if coord is None:

            coord = (0.0, 0.0)
        pred_coords.append(coord)

    return pred_coords, results


def main():
   
    excel_file = 'FromLLM.xlsx'
    output_base_dir = ''
    gt_excel = ''
    error_threshold = 50  

  
    gt_coords = load_ground_truth(gt_excel)



    original_dir = os.path.join(output_base_dir, 'original_results')
    pred_original, _ = run_geolocator(original_dir, excel_file)
    avg_error_orig, recall_orig = calculate_metrics(pred_original, gt_coords, error_threshold)
    print(f"Original model - average positioning error: {avg_error_orig:.2f} m, error<{error_threshold}m,recall rate: {recall_orig:.2f}%")


    print("=== No fuzzy modeling ===")
    no_fuzzy_levels = {
        'high': {'delta_theta': 0.0, 'delta_distance': 0.0},
        'medium': {'delta_theta': 0.0, 'delta_distance': 0.0},
        'low': {'delta_theta': 0.0, 'delta_distance': 0.0}
    }
    no_fuzzy_dir = os.path.join(output_base_dir, 'no_fuzzy_results')
    pred_no_fuzzy, _ = run_geolocator(no_fuzzy_dir, excel_file, fuzziness_levels=no_fuzzy_levels)
    avg_error_no_fuzzy, recall_no_fuzzy = calculate_metrics(pred_no_fuzzy, gt_coords, error_threshold)
    print(
        f"No blur modeling - average localization error: {avg_error_no_fuzzy:.2f} m, erro<{error_threshold}m,recall rate: {recall_no_fuzzy:.2f}%")


    print("===  No multi-constraint fusion ===")
    no_fusion_dir = os.path.join(output_base_dir, 'no_fusion_results')
    pred_no_fusion, _ = run_geolocator(no_fusion_dir, excel_file, fusion_method=None)
    avg_error_no_fusion, recall_no_fusion = calculate_metrics(pred_no_fusion, gt_coords, error_threshold)
    print(
        f"Unconstrained Fusion - Average Positioning Error: {avg_error_no_fusion:.2f} 米, error<{error_threshold}m,recall rate: {recall_no_fusion:.2f}%")

    metrics_df = pd.DataFrame({
       Model Variants: ['Original Model', 'No Blur Modeling', 'No Multi-Constraint Fusion'],
        'Average positioning error (meters)': [avg_error_orig, avg_error_no_fuzzy, avg_error_no_fusion],
        f'erro<{error_threshold}m,recall rate': [recall_orig, recall_no_fuzzy, recall_no_fusion]
    })
    metrics_csv = os.path.join(output_base_dir, 'evaluation_metrics.csv')
    metrics_df.to_csv(metrics_csv, index=False, encoding='utf-8-sig')
    print(f"Save the indicator to: {metrics_csv}")


if __name__ == "__main__":
    main()
