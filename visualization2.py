import json
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, Polygon as ShapelyPolygon, MultiPolygon
from shapely import wkt, ops
import pandas as pd
import os
import warnings
from mpl_toolkits.mplot3d import Axes3D
import math
import matplotlib.cm as cm
from matplotlib.patches import Polygon as MplPolygon

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False


class ImprovedVisualizer:
  
    def __init__(self, output_dir='results'):
   
        self.output_dir = output_dir
        self.img_dir = os.path.join(output_dir, 'images')

        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

       
        self.sci_colors = {
            'background': 'white',
            'text': 'black',
            'building_fill': '#F8F9FA', 
            'building_edge': '#495057',  
            'boundary_point': '#1E88E5', 
            'centroid': '#D32F2F',  
            'final_point': '#388E3C',  
            'confidence_region': '#7B1FA2',  
            'direction_arrow': '#FF9800',  
            'fuzzy_high': '#C62828',  
            'fuzzy_low': '#1565C0',  
            'text_box': '#FAFAFA'  
        }


        self.ref_colors = ['#E53935', '#43A047', '#FB8C00', '#039BE5',
                           '#8E24AA', '#F4511E', '#00ACC1', '#5E35B1']

        self.longitude_to_meters = 85000
        self.latitude_to_meters = 111000

    def parse_wkt_to_polygons(self, wkt_string):
      
        try:
            geom = wkt.loads(wkt_string)
            polygons = []

            if geom.geom_type == 'MultiPolygon':
                for polygon in geom.geoms:
                    if hasattr(polygon, 'exterior'):
                        polygons.append(np.array(polygon.exterior.coords))
                    elif hasattr(polygon, 'geoms'):
                        for poly in polygon.geoms:
                            if hasattr(poly, 'exterior'):
                                polygons.append(np.array(poly.exterior.coords))
            elif geom.geom_type == 'Polygon':
                polygons.append(np.array(geom.exterior.coords))

            return polygons
        except Exception as e:
            print(f"WKT parsing error: {e}")
            return []

    def create_sci_visualization(self, result, point_index=None):
    
        if result is None:
            return

        
        self._create_2d_plot(result, point_index)

        self._create_3d_plot(result, point_index)

        self._create_statistics_plot(result, point_index)

    def _create_2d_plot(self, result, point_index=None):
      
        if result is None:
            return

    
        fig = plt.figure(figsize=(10, 8), facecolor=self.sci_colors['background'])

     
        location_id = result['location_id']
        main_title = f'2D Fuzzy Distribution - {location_id}'
        if point_index is not None:
            main_title += f' (Point {point_index + 1})'

        ax = fig.add_subplot(111)
        ax.set_facecolor(self.sci_colors['background'])

  
        self._plot_2d_fused_result(ax, result)

     
        fig.suptitle(main_title, fontsize=14, fontweight='bold',
                     color=self.sci_colors['text'], y=0.95)
        plt.tight_layout(rect=[0, 0, 1, 0.93])

    
        img_filename = f"{location_id}_2d_plot.png"
        img_path = os.path.join(self.img_dir, img_filename)
        plt.savefig(img_path, dpi=300, bbox_inches='tight',
                    facecolor=self.sci_colors['background'])
        print(f"  2D visualization has been saved: {img_path}")
        plt.close(fig)

    def _create_3d_plot(self, result, point_index=None):
    
        if result is None:
            return

  
        fig = plt.figure(figsize=(10, 8), facecolor=self.sci_colors['background'])

    
        location_id = result['location_id']
        main_title = f'3D Fuzzy Distribution - {location_id}'
        if point_index is not None:
            main_title += f' (Point {point_index + 1})'

    
        ax = fig.add_subplot(111, projection='3d')

    
        self._plot_3d_distribution(ax, result)

  
        fig.suptitle(main_title, fontsize=14, fontweight='bold',
                     color=self.sci_colors['text'], y=0.95)
        plt.tight_layout(rect=[0, 0, 1, 0.93])

    
        img_filename = f"{location_id}_3d_plot.png"
        img_path = os.path.join(self.img_dir, img_filename)
        plt.savefig(img_path, dpi=300, bbox_inches='tight',
                    facecolor=self.sci_colors['background'])
        print(f"  3D visualization has been saved: {img_path}")
        plt.close(fig)

    def _create_statistics_plot(self, result, point_index=None):
   
        if result is None:
            return

 
        fig = plt.figure(figsize=(8, 10), facecolor=self.sci_colors['background'])

     
        location_id = result['location_id']
        main_title = f'Location Statistics - {location_id}'
        if point_index is not None:
            main_title += f' (Point {point_index + 1})'

    
        ax = fig.add_subplot(111)
        ax.set_facecolor(self.sci_colors['background'])

 
        self._plot_statistics(ax, result)

     
        fig.suptitle(main_title, fontsize=14, fontweight='bold',
                     color=self.sci_colors['text'], y=0.97)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

    
        img_filename = f"{location_id}_statistics.png"
        img_path = os.path.join(self.img_dir, img_filename)
        plt.savefig(img_path, dpi=300, bbox_inches='tight',
                    facecolor=self.sci_colors['background'])
        print(f"  The statistical infographic has been saved: {img_path}")
        plt.close(fig)

    def _plot_2d_fused_result(self, ax, result):
       
        ax.set_title("2D Fuzzy Distribution with Reference Buildings",
                     fontsize=12, fontweight='bold', pad=10)

     
        x_grid, y_grid, X, Y, avg_latitude = result['grid_info']
        fused_dist = result['fused_distribution']

    
        if fused_dist is not None:
   
            im = ax.contourf(X, Y, fused_dist, levels=20,
                             cmap='viridis', alpha=0.7, zorder=1)

     
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.15)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label('Membership Degree', fontsize=9)
            cbar.ax.tick_params(labelsize=12)


        if 'reference_polygons' in result and result['reference_polygons']:
            patches = []

            for i, polygons in enumerate(result['reference_polygons']):
                color_idx = i % len(self.ref_colors)
                ref_color = self.ref_colors[color_idx]

                for polygon_coords in polygons:
                    if len(polygon_coords) > 0:
                        polygon_patch = MplPolygon(
                            polygon_coords,
                            closed=True,
                            facecolor=ref_color,
                            edgecolor='black',
                            linewidth=1.5,
                            alpha=0.4,  
                            zorder=2,
                        )
                        ax.add_patch(polygon_patch)
                        patches.append(polygon_patch)

                        if len(polygon_coords) > 0:
                            center_x = np.mean(polygon_coords[:, 0])
                            center_y = np.mean(polygon_coords[:, 1])

                            label_text = f'R{i + 1}'
                            ax.text(center_x, center_y, label_text,
                                    fontsize=9, fontweight='bold',
                                    ha='center', va='center',
                                    color='black',
                                    bbox=dict(boxstyle='round,pad=0.2',
                                              facecolor='white', alpha=0.7),
                                    zorder=3)

        final_coord = result['final_coordinate']
        if final_coord:
            ax.plot(final_coord[0], final_coord[1], marker='*',
                    markersize=15, color=self.sci_colors['final_point'],
                    markeredgecolor='black', markeredgewidth=1.5,
                    zorder=5)

        for i, (centroid, boundary) in enumerate(zip(result['centroids'],
                                                     result['boundary_points'])):
            color_idx = i % len(self.ref_colors)
            ref_color = self.ref_colors[color_idx]

            ax.plot(centroid[0], centroid[1], marker='o', markersize=7,
                    color=ref_color, alpha=0.9, markeredgecolor='black',
                    markeredgewidth=1, zorder=4)

            ax.plot(boundary[0], boundary[1], marker='s', markersize=9,
                    color=self.sci_colors['boundary_point'], alpha=0.9,
                    markeredgecolor='black', markeredgewidth=1, zorder=4)


            ax.plot([centroid[0], boundary[0]], [centroid[1], boundary[1]],
                    'k--', linewidth=1, alpha=0.5, zorder=3)


            if i < len(result['references_info']):
                bearing = result['references_info'][i]['bearing']
                distance = result['references_info'][i]['distance']


                bearing_rad = math.radians(bearing)
                distance_degrees = distance / (self.longitude_to_meters *
                                               math.cos(math.radians(avg_latitude)))

                dx = distance_degrees * math.sin(bearing_rad)
                dy = distance_degrees * math.cos(bearing_rad)

                arrow = ax.arrow(boundary[0], boundary[1], dx, dy,
                                 head_width=distance_degrees * 0.08,
                                 head_length=distance_degrees * 0.12,
                                 fc=self.sci_colors['direction_arrow'],
                                 ec=self.sci_colors['direction_arrow'],
                                 alpha=0.7, width=distance_degrees * 0.01,
                                 length_includes_head=True, zorder=3)

    
                offset_x = 0.00005  
                offset_y = 0.00005

  
                if dx > 0: 
                    text_x = centroid[0] - offset_x * 2
                else:  
                    text_x = centroid[0] + offset_x * 2

                if dy > 0:  
                    text_y = centroid[1] - offset_y * 2
                else: 
                    text_y = centroid[1] + offset_y * 2

                ax.text(text_x, text_y,
                        f'{bearing}°\n{distance}m',
                        fontsize=7.5, ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.2',
                                  facecolor='white', alpha=0.8,
                                  edgecolor=ref_color, linewidth=1),
                        zorder=4)

        conf_region = result['confidence_region']
        if conf_region:
            rect = plt.Rectangle((conf_region['x_min'], conf_region['y_min']),
                                 conf_region['x_max'] - conf_region['x_min'],
                                 conf_region['y_max'] - conf_region['y_min'],
                                 fill=False, edgecolor=self.sci_colors['confidence_region'],
                                 linewidth=2, linestyle='--', alpha=0.8,
                                 zorder=2)
            ax.add_patch(rect)

        # ax.set_xlabel('Longitude', fontsize=10)
        # ax.set_ylabel('Latitude', fontsize=10)
        ax.set_xticks([])  
        ax.set_yticks([])  
        ax.grid(True, alpha=0.2, linestyle='--', zorder=1)

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='*', color='w', markerfacecolor=self.sci_colors['final_point'],
                   markersize=12, markeredgecolor='black', label='Final Position'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor=self.sci_colors['boundary_point'],
                   markersize=8, markeredgecolor='black', label='Boundary Point'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=self.ref_colors[0],
                   markersize=6, markeredgecolor='black', label='Centroid'),
            Line2D([0], [0], color=self.sci_colors['direction_arrow'], lw=2,
                   label='Direction'),
            Line2D([0], [0], color=self.sci_colors['confidence_region'],
                   linestyle='--', lw=2, label='Confidence Region'),
        ]

        ref_count = min(len(result.get('reference_polygons', [])), 4)
        for i in range(ref_count):
            color_idx = i % len(self.ref_colors)
            legend_elements.append(
                plt.Rectangle((0, 0), 1, 1, facecolor=self.ref_colors[color_idx],
                              alpha=0.4, edgecolor='black', label=f'Building {i + 1}')
            )

        # ax.legend(handles=legend_elements, fontsize=8, loc='upper right',
        #           framealpha=0.9, ncol=1, bbox_to_anchor=(1.4, 1.0))

        if fused_dist is not None:  
            cax_pos = cax.get_position()
            fig = ax.get_figure()
            fig.legend(
                handles=legend_elements,
                fontsize=8,
                framealpha=0.9,
                ncol=1,
                loc='upper left',  
                bbox_to_anchor=(cax_pos.x1 + 0.1, cax_pos.y1),
                frameon=True,
                edgecolor='gray'
            )
        else:
            ax.legend(handles=legend_elements, fontsize=8, loc='upper right',
                      framealpha=0.9, ncol=1, bbox_to_anchor=(1.5, 1.0))


        ax.set_aspect('equal', adjustable='box')

        all_points = []

        if 'reference_polygons' in result:
            for polygons in result['reference_polygons']:
                for polygon in polygons:
                    if len(polygon) > 0:
                        all_points.extend(polygon)

        if 'boundary_points' in result:
            all_points.extend(result['boundary_points'])

        if 'final_coordinate' in result:
            all_points.append(result['final_coordinate'])

        if all_points:
            all_points = np.array(all_points)
            x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
            y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()

            x_margin = (x_max - x_min) * 0.15
            y_margin = (y_max - y_min) * 0.15

            ax.set_xlim(x_min - x_margin, x_max + x_margin)
            ax.set_ylim(y_min - y_margin, y_max + y_margin)

    def _plot_3d_distribution(self, ax, result):
    
        ax.set_title("3D Fuzzy Distribution Surface",
                     fontsize=12, fontweight='bold', pad=1)

    
        x_grid, y_grid, X, Y, avg_latitude = result['grid_info']
        Z = result['fused_distribution']

        max_z = np.max(Z)

        if max_z > 0:
            surf = ax.plot_surface(X, Y, Z, cmap='plasma', alpha=0.8,
                                   rstride=2, cstride=2, edgecolor='none',
                                   linewidth=0.5, antialiased=True, zorder=1)

            offset = np.min(Z) - 0.1 * (np.max(Z) - np.min(Z))
            ax.contourf(X, Y, Z, zdir='z', offset=offset,
                        cmap='plasma', alpha=0.6, levels=10, zorder=2)

            max_pos = np.unravel_index(np.argmax(Z), Z.shape)
            ax.scatter(X[max_pos], Y[max_pos], Z[max_pos] + 0.02,
                       c='red', s=100, marker='*', edgecolors='black',
                       label='Max Membership', zorder=3)

            final_coord = result['final_coordinate']
            if final_coord:
                x_idx = np.argmin(np.abs(x_grid - final_coord[0]))
                y_idx = np.argmin(np.abs(y_grid - final_coord[1]))
                if 0 <= x_idx < len(x_grid) and 0 <= y_idx < len(y_grid):
                    z_value = Z[y_idx, x_idx]
                    ax.scatter([final_coord[0]], [final_coord[1]], [z_value],
                               c='green', s=150, marker='*', edgecolors='black',
                               label='Final Position', zorder=3)
        else:
            ax.plot_surface(X, Y, np.zeros_like(Z), color='gray', alpha=0.5, zorder=1)

        ax.set_xlabel('Longitude', fontsize=10, labelpad=10)
        ax.set_ylabel('Latitude', fontsize=10, labelpad=10)
        ax.set_zlabel('Membership', fontsize=10, labelpad=10)

        ax.view_init(elev=25, azim=45)

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='*', color='w', markerfacecolor='green',
                   markersize=10, markeredgecolor='black', label='Final Position'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
                   markersize=10, markeredgecolor='black', label='Max Membership'),
        ]
        ax.legend(handles=legend_elements, fontsize=8, loc='upper right')

    def _plot_statistics(self, ax, result):
    
        ax.set_title("Location Statistics", fontsize=12, fontweight='bold', pad=10, y=1)
        ax.axis('off')

        final_coord = result['final_coordinate']
        conf_region = result['confidence_region']
        boundary_dists = result.get('boundary_distances', [])


        stats_text = f"""Location Information:

Location ID: {result['location_id']}
Number of References: {result['distribution_count']}

Final Coordinate:
  Longitude: {final_coord[0]:.6f}
  Latitude: {final_coord[1]:.6f}

Confidence Score: {result['confidence']:.3f}
"""

        if conf_region:
            stats_text += f"""
Confidence Region (50%):

  Longitude: [{conf_region['x_min']:.6f}, {conf_region['x_max']:.6f}]
  Latitude: [{conf_region['y_min']:.6f}, {conf_region['y_max']:.6f}]
  Area: {conf_region['area_m2']:.1f} m²
"""

        if boundary_dists:
            avg_boundary_dist = np.mean(boundary_dists)
            stats_text += f"""
Boundary Information:

  Avg. Centroid-Boundary: {avg_boundary_dist:.2f} m
  Max: {max(boundary_dists):.2f} m
  Min: {min(boundary_dists):.2f} m
"""


        if 'references_info' in result and result['references_info']:
            stats_text += f"\nReference Information:\n"
            for i, ref_info in enumerate(result['references_info'][:4]):  
                stats_text += f"  R{i + 1}: {ref_info['bearing']}°, "
                stats_text += f"{ref_info['distance']} m\n"

            if len(result['references_info']) > 4:
                stats_text += f"  ... and {len(result['references_info']) - 4} more\n"


        stats_text += f"""

Algorithm Parameters:

  Grid Resolution: 100×100
  Fusion Method: Product
  Defuzzification: Centroid
"""


        ax.text(0.05, 0.95, stats_text, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round',
                          facecolor=self.sci_colors['text_box'],
                          alpha=0.9, edgecolor='gray', linewidth=1),
                fontfamily='monospace')


class EnhancedFuzzyGeoLocator:

    def __init__(self, fuzziness_levels=None, grid_resolution=100, output_dir='sci_results'):

        self.output_dir = output_dir

        self.visualizer = ImprovedVisualizer(output_dir)

        self.results = {}

        self.raw_data = None

        try:
            from fuzzy_geolocator import FuzzyGeoLocator

            self.locator = FuzzyGeoLocator(fuzziness_levels, grid_resolution, output_dir)
            self.has_original_locator = True
        except ImportError:
            print("Warning: Unable to import the original fuzzy_geolocator; only visualization features are supported.")
            self.has_original_locator = False

    def load_data_from_json(self, json_file):

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            print(f"JSON file loaded")

            self.raw_data = data

            if self.has_original_locator:
                self.locator.data = data

            return data

        except Exception as e:
            print(f"Failed to load JSON data: {e}")
            return None

    def extract_polygons_from_references(self, references):
    
        polygons_list = []

        for ref in references:
            if 'wkt' in ref:
                polygons = self.visualizer.parse_wkt_to_polygons(ref['wkt'])
                polygons_list.append(polygons)

        return polygons_list

    def process_single_point_with_original(self, point_data, point_index=None):
     
        if not self.has_original_locator:
            print("Error: Original locator is unavailable")
            return None

        result = self.locator.process_location_point(point_data, point_index, save_images=False)

        if result:
            if 'references' in point_data:
                polygons_list = self.extract_polygons_from_references(point_data['references'])
                result['reference_polygons'] = polygons_list

            self.visualizer.create_sci_visualization(result, point_index)

            self.results[result['location_id']] = result

        return result

    def simulate_processing_for_visualization(self, point_data, point_index=None):

        polygons_list = []
        centroids = []
        boundary_points = []
        references_info = []

        if 'references' in point_data:
            polygons_list = self.extract_polygons_from_references(point_data['references'])

            for i, ref in enumerate(point_data['references']):
                if 'wkt' in ref and 'bearing' in ref:
                    try:
                        geom = wkt.loads(ref['wkt'])
                        centroid_geom = geom.centroid
                        centroid = (centroid_geom.x, centroid_geom.y)
                        centroids.append(centroid)

                        bearing_rad = math.radians(ref['bearing'])
                        boundary_distance_m = 10  
                        boundary_distance_deg = boundary_distance_m / (self.visualizer.longitude_to_meters *
                                                                       math.cos(math.radians(39.9)))

                        boundary_point = (
                            centroid[0] + boundary_distance_deg * math.sin(bearing_rad),
                            centroid[1] + boundary_distance_deg * math.cos(bearing_rad)
                        )
                        boundary_points.append(boundary_point)

                        references_info.append({
                            'bearing': ref['bearing'],
                            'distance': ref['distance'] if 'distance' in ref else 50.0,
                            'centroid': centroid,
                            'boundary_point': boundary_point,
                            'boundary_distance': boundary_distance_m
                        })
                    except Exception as e:
                        print(f"Error processing reference object{i}: {e}")

        result = {
            'location_id': point_data.get('id', f'Point_{point_index}'),
            'final_coordinate': (116.425, 39.950) if not centroids else centroids[0],
            'confidence': 0.8,
            'centroids': centroids,
            'boundary_points': boundary_points,
            'boundary_distances': [10.0] * len(boundary_points),
            'distribution_count': len(polygons_list),
            'references_info': references_info,
            'reference_polygons': polygons_list
        }

        if boundary_points:
            boundary_array = np.array(boundary_points)

            min_x, max_x = boundary_array[:, 0].min(), boundary_array[:, 0].max()
            min_y, max_y = boundary_array[:, 1].min(), boundary_array[:, 1].max()

            x_margin = (max_x - min_x) * 0.4
            y_margin = (max_y - min_y) * 0.4

            x_grid = np.linspace(min_x - x_margin, max_x + x_margin, 80)
            y_grid = np.linspace(min_y - y_margin, max_y + y_margin, 80)
            X, Y = np.meshgrid(x_grid, y_grid)

            Z = np.zeros_like(X)

            for i, centroid in enumerate(centroids):
                sigma_x = 0.0005  
                sigma_y = 0.0005

                Z += np.exp(-((X - centroid[0]) ** 2 / (2 * sigma_x ** 2) +
                              (Y - centroid[1]) ** 2 / (2 * sigma_y ** 2)))

            if Z.max() > 0:
                Z = Z / Z.max()

            result['fused_distribution'] = Z
            result['grid_info'] = (x_grid, y_grid, X, Y, 39.9)

            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2

            result['confidence_region'] = {
                'x_min': center_x - 0.0003,
                'x_max': center_x + 0.0003,
                'y_min': center_y - 0.0003,
                'y_max': center_y + 0.0003,
                'area_m2': (0.0006 * self.visualizer.longitude_to_meters) *
                           (0.0006 * self.visualizer.latitude_to_meters)
            }

        self.visualizer.create_sci_visualization(result, point_index)

        self.results[result['location_id']] = result

        return result

    def process_all_points(self, data=None, use_original_locator=True):
   
        if data is None:
            data = self.raw_data

        if data is None:
            print("Error: Data not loaded")
            return []

        all_results = []

        print("=" * 60)
        print("Start processing all waypoints")
        print("=" * 60)

        points_to_process = data['points']

        for i, point_data in enumerate(points_to_process):
            print(f"\n {i + 1}/{len(points_to_process)}: {point_data['id']}")

            if use_original_locator and self.has_original_locator:
                result = self.process_single_point_with_original(point_data, i)
            else:
                result = self.simulate_processing_for_visualization(point_data, i)

            if result:
                all_results.append(result)

        print(f"\nProcessing complete! Successfully processed {len(all_results)} locations")

    
        self.save_results(all_results)

        return all_results

    def save_results(self, results):
    
        if not results:
            return

        data_dir = os.path.join(self.output_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)

    
        json_filename = os.path.join(data_dir, 'enhanced_fuzzy_geolocation_results.json')

        output_data = []
        for result in results:
            final_coord = result.get('final_coordinate', (0, 0))

            output_result = {
                'location_id': result['location_id'],
                'final_coordinate': {
                    'longitude': float(final_coord[0]),
                    'latitude': float(final_coord[1])
                },
                'confidence': float(result['confidence']),
                'distribution_count': result['distribution_count']
            }

            conf_region = result.get('confidence_region')
            if conf_region:
                output_result['confidence_region'] = {
                    'x_min': float(conf_region.get('x_min', 0)),
                    'x_max': float(conf_region.get('x_max', 0)),
                    'y_min': float(conf_region.get('y_min', 0)),
                    'y_max': float(conf_region.get('y_max', 0)),
                    'area_m2': float(conf_region.get('area_m2', 0))
                }

            output_data.append(output_result)

        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\nJSON result has been saved to {json_filename}")


def main():

    output_directory = 'sci_fuzzy_geolocation_results7'
    enhancer = EnhancedFuzzyGeoLocator(
        fuzziness_levels={
            'high': {'delta_theta': 60.0, 'delta_distance': 1.0},
            'medium': {'delta_theta': 30.0, 'delta_distance': 0.5},
            'low': {'delta_theta': 15.0, 'delta_distance': 0.2}
        },
        grid_resolution=100,
        output_dir=output_directory
    )

  
    json_file = ''  
    data = enhancer.load_data_from_json(json_file)

    if data is None:
        print("Failed to load JSON data. Please check the file path and format.")
        return []

    use_original = enhancer.has_original_locator
    if not use_original:
        print("Note: Visualization will be performed using simulated processing")

    results = enhancer.process_all_points(data, use_original_locator=use_original)

    if results:
        print("\n" + "=" * 60)
        print("Summary of Processing")
        print("=" * 60)

        confidences = [r.get('confidence', 0) for r in results]
        areas = [r['confidence_region']['area_m2'] for r in results
                 if r.get('confidence_region')]

        print(f"Total number of processing points: {len(results)}")
        if confidences:
            print(f"Average confidence: {np.mean(confidences):.3f} "
                  f"(scope: {np.min(confidences):.3f} - {np.max(confidences):.3f})")

        if areas:
            print(f"Average confidence region area: {np.mean(areas):.1f} 平方米")

        print(f"The visualization image is saved in: {os.path.join(output_directory, 'images')}")

        img_dir = os.path.join(output_directory, 'images')
        if os.path.exists(img_dir):
            img_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
            print(f"Total number of images generated: {len(img_files)}")


    return results


if __name__ == "__main__":
    results = main()

    if results:
        print(f"\nInformation on the first point:")
        print(f"  ID: {results[0]['location_id']}")
        print(f"  最终坐标: ({results[0]['final_coordinate'][0]:.6f}, "
              f"{results[0]['final_coordinate'][1]:.6f})")
        print(f"  置信度: {results[0].get('confidence', 0):.3f}")
        print(f"  参照物数量: {results[0]['distribution_count']}")
