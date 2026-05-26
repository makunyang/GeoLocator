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

# 设置中文字体（保留但仅用于特殊情况）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False


class ImprovedVisualizer:
    """
    改进的可视化器：生成符合SCI论文风格的定位结果图
    """

    def __init__(self, output_dir='results'):
        """
        初始化可视化器

        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        self.img_dir = os.path.join(output_dir, 'images')

        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

        # SCI论文配色方案
        self.sci_colors = {
            'background': 'white',
            'text': 'black',
            'building_fill': '#F8F9FA',  # 非常浅的灰色填充
            'building_edge': '#495057',  # 深灰色边框
            'boundary_point': '#1E88E5',  # 蓝色
            'centroid': '#D32F2F',  # 红色
            'final_point': '#388E3C',  # 绿色
            'confidence_region': '#7B1FA2',  # 紫色
            'direction_arrow': '#FF9800',  # 橙色
            'fuzzy_high': '#C62828',  # 深红色（高隶属度）
            'fuzzy_low': '#1565C0',  # 深蓝色（低隶属度）
            'text_box': '#FAFAFA'  # 非常浅的灰色文本框
        }

        # 参照物颜色列表（用于区分不同参照物）
        self.ref_colors = ['#E53935', '#43A047', '#FB8C00', '#039BE5',
                           '#8E24AA', '#F4511E', '#00ACC1', '#5E35B1']

        # 在纬度39.9°（北京）处的转换系数
        self.longitude_to_meters = 85000
        self.latitude_to_meters = 111000

    def parse_wkt_to_polygons(self, wkt_string):
        """
        解析WKT字符串，提取所有多边形用于可视化

        Args:
            wkt_string: WKT格式字符串

        Returns:
            polygons: 多边形坐标列表
        """
        try:
            geom = wkt.loads(wkt_string)
            polygons = []

            if geom.geom_type == 'MultiPolygon':
                for polygon in geom.geoms:
                    # 提取外边界坐标
                    if hasattr(polygon, 'exterior'):
                        polygons.append(np.array(polygon.exterior.coords))
                    # 处理可能的多个环
                    elif hasattr(polygon, 'geoms'):
                        for poly in polygon.geoms:
                            if hasattr(poly, 'exterior'):
                                polygons.append(np.array(poly.exterior.coords))
            elif geom.geom_type == 'Polygon':
                polygons.append(np.array(geom.exterior.coords))

            return polygons
        except Exception as e:
            print(f"WKT解析错误: {e}")
            return []

    def create_sci_visualization(self, result, point_index=None):
        """
        创建SCI论文风格的可视化图 - 三个子图分别保存
        """
        if result is None:
            return

        # 创建2D融合结果图
        self._create_2d_plot(result, point_index)

        # 创建3D分布图
        self._create_3d_plot(result, point_index)

        # 创建统计信息图
        self._create_statistics_plot(result, point_index)

    def _create_2d_plot(self, result, point_index=None):
        """
        创建并保存2D融合结果图
        """
        if result is None:
            return

        # 创建图形
        fig = plt.figure(figsize=(10, 8), facecolor=self.sci_colors['background'])

        # 设置标题
        location_id = result['location_id']
        main_title = f'2D Fuzzy Distribution - {location_id}'
        if point_index is not None:
            main_title += f' (Point {point_index + 1})'

        # 创建单个子图
        ax = fig.add_subplot(111)
        ax.set_facecolor(self.sci_colors['background'])

        # 绘制2D融合结果
        self._plot_2d_fused_result(ax, result)

        # 设置主标题
        fig.suptitle(main_title, fontsize=14, fontweight='bold',
                     color=self.sci_colors['text'], y=0.95)
        plt.tight_layout(rect=[0, 0, 1, 0.93])

        # 保存图片
        img_filename = f"{location_id}_2d_plot.png"
        img_path = os.path.join(self.img_dir, img_filename)
        plt.savefig(img_path, dpi=300, bbox_inches='tight',
                    facecolor=self.sci_colors['background'])
        print(f"  2D可视化图已保存: {img_path}")
        plt.close(fig)

    def _create_3d_plot(self, result, point_index=None):
        """
        创建并保存3D分布图
        """
        if result is None:
            return

        # 创建图形
        fig = plt.figure(figsize=(10, 8), facecolor=self.sci_colors['background'])

        # 设置标题
        location_id = result['location_id']
        main_title = f'3D Fuzzy Distribution - {location_id}'
        if point_index is not None:
            main_title += f' (Point {point_index + 1})'

        # 创建3D子图
        ax = fig.add_subplot(111, projection='3d')

        # 绘制3D分布
        self._plot_3d_distribution(ax, result)

        # 设置主标题
        fig.suptitle(main_title, fontsize=14, fontweight='bold',
                     color=self.sci_colors['text'], y=0.95)
        plt.tight_layout(rect=[0, 0, 1, 0.93])

        # 保存图片
        img_filename = f"{location_id}_3d_plot.png"
        img_path = os.path.join(self.img_dir, img_filename)
        plt.savefig(img_path, dpi=300, bbox_inches='tight',
                    facecolor=self.sci_colors['background'])
        print(f"  3D可视化图已保存: {img_path}")
        plt.close(fig)

    def _create_statistics_plot(self, result, point_index=None):
        """
        创建并保存统计信息图
        """
        if result is None:
            return

        # 创建图形
        fig = plt.figure(figsize=(8, 10), facecolor=self.sci_colors['background'])

        # 设置标题
        location_id = result['location_id']
        main_title = f'Location Statistics - {location_id}'
        if point_index is not None:
            main_title += f' (Point {point_index + 1})'

        # 创建单个子图
        ax = fig.add_subplot(111)
        ax.set_facecolor(self.sci_colors['background'])

        # 绘制统计信息
        self._plot_statistics(ax, result)

        # 设置主标题
        fig.suptitle(main_title, fontsize=14, fontweight='bold',
                     color=self.sci_colors['text'], y=0.97)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # 保存图片
        img_filename = f"{location_id}_statistics.png"
        img_path = os.path.join(self.img_dir, img_filename)
        plt.savefig(img_path, dpi=300, bbox_inches='tight',
                    facecolor=self.sci_colors['background'])
        print(f"  统计信息图已保存: {img_path}")
        plt.close(fig)

    def _plot_2d_fused_result(self, ax, result):
        """
        绘制2D融合结果子图（包含参照物轮廓）
        """
        ax.set_title("2D Fuzzy Distribution with Reference Buildings",
                     fontsize=12, fontweight='bold', pad=10)

        # 获取网格数据
        x_grid, y_grid, X, Y, avg_latitude = result['grid_info']
        fused_dist = result['fused_distribution']

        # 首先绘制模糊分布
        if fused_dist is not None:
            # 绘制模糊分布
            im = ax.contourf(X, Y, fused_dist, levels=20,
                             cmap='viridis', alpha=0.7, zorder=1)

            # 添加颜色条，放在右侧
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.15)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label('Membership Degree', fontsize=9)
            cbar.ax.tick_params(labelsize=12)

        # 绘制参照物轮廓（从reference_polygons）
        if 'reference_polygons' in result and result['reference_polygons']:
            patches = []

            for i, polygons in enumerate(result['reference_polygons']):
                color_idx = i % len(self.ref_colors)
                ref_color = self.ref_colors[color_idx]

                for polygon_coords in polygons:
                    if len(polygon_coords) > 0:
                        # 创建多边形补丁
                        polygon_patch = MplPolygon(
                            polygon_coords,
                            closed=True,
                            facecolor=ref_color,
                            edgecolor='black',
                            linewidth=1.5,
                            alpha=0.4,  # 半透明填充
                            zorder=2,
                        )
                        ax.add_patch(polygon_patch)
                        patches.append(polygon_patch)

                        # 在多边形中心添加标签（但避免遮挡）
                        if len(polygon_coords) > 0:
                            center_x = np.mean(polygon_coords[:, 0])
                            center_y = np.mean(polygon_coords[:, 1])

                            # 检查标签位置是否合适
                            label_text = f'R{i + 1}'
                            ax.text(center_x, center_y, label_text,
                                    fontsize=9, fontweight='bold',
                                    ha='center', va='center',
                                    color='black',
                                    bbox=dict(boxstyle='round,pad=0.2',
                                              facecolor='white', alpha=0.7),
                                    zorder=3)

        # 标记最终位置
        final_coord = result['final_coordinate']
        if final_coord:
            ax.plot(final_coord[0], final_coord[1], marker='*',
                    markersize=15, color=self.sci_colors['final_point'],
                    markeredgecolor='black', markeredgewidth=1.5,
                    zorder=5)

        # 标记质心和边界点
        for i, (centroid, boundary) in enumerate(zip(result['centroids'],
                                                     result['boundary_points'])):
            color_idx = i % len(self.ref_colors)
            ref_color = self.ref_colors[color_idx]

            # 质心
            ax.plot(centroid[0], centroid[1], marker='o', markersize=7,
                    color=ref_color, alpha=0.9, markeredgecolor='black',
                    markeredgewidth=1, zorder=4)

            # 边界点
            ax.plot(boundary[0], boundary[1], marker='s', markersize=9,
                    color=self.sci_colors['boundary_point'], alpha=0.9,
                    markeredgecolor='black', markeredgewidth=1, zorder=4)

            # 绘制从质心到边界的连线（虚线）
            ax.plot([centroid[0], boundary[0]], [centroid[1], boundary[1]],
                    'k--', linewidth=1, alpha=0.5, zorder=3)

            # 绘制方向箭头（从边界点）
            if i < len(result['references_info']):
                bearing = result['references_info'][i]['bearing']
                distance = result['references_info'][i]['distance']

                # 将距离转换为度
                bearing_rad = math.radians(bearing)
                distance_degrees = distance / (self.longitude_to_meters *
                                               math.cos(math.radians(avg_latitude)))

                dx = distance_degrees * math.sin(bearing_rad)
                dy = distance_degrees * math.cos(bearing_rad)

                # 绘制箭头
                arrow = ax.arrow(boundary[0], boundary[1], dx, dy,
                                 head_width=distance_degrees * 0.08,
                                 head_length=distance_degrees * 0.12,
                                 fc=self.sci_colors['direction_arrow'],
                                 ec=self.sci_colors['direction_arrow'],
                                 alpha=0.7, width=distance_degrees * 0.01,
                                 length_includes_head=True, zorder=3)

                # 标注方位和距离 - 放在质心附近，避免遮挡箭头
                # 计算偏移量，确保标注在质心旁边
                offset_x = 0.00005  # 小偏移
                offset_y = 0.00005

                # 根据箭头方向选择标注位置
                if dx > 0:  # 箭头向右
                    text_x = centroid[0] - offset_x * 2
                else:  # 箭头向左
                    text_x = centroid[0] + offset_x * 2

                if dy > 0:  # 箭头向上
                    text_y = centroid[1] - offset_y * 2
                else:  # 箭头向下
                    text_y = centroid[1] + offset_y * 2

                # 在质心附近标注方位和距离
                ax.text(text_x, text_y,
                        f'{bearing}°\n{distance}m',
                        fontsize=7.5, ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.2',
                                  facecolor='white', alpha=0.8,
                                  edgecolor=ref_color, linewidth=1),
                        zorder=4)

        # 绘制置信区域
        conf_region = result['confidence_region']
        if conf_region:
            rect = plt.Rectangle((conf_region['x_min'], conf_region['y_min']),
                                 conf_region['x_max'] - conf_region['x_min'],
                                 conf_region['y_max'] - conf_region['y_min'],
                                 fill=False, edgecolor=self.sci_colors['confidence_region'],
                                 linewidth=2, linestyle='--', alpha=0.8,
                                 zorder=2)
            ax.add_patch(rect)

        # # 设置坐标轴
        # ax.set_xlabel('Longitude', fontsize=10)
        # ax.set_ylabel('Latitude', fontsize=10)
        ax.set_xticks([])  # 隐藏x轴刻度
        ax.set_yticks([])  # 隐藏y轴刻度
        ax.grid(True, alpha=0.2, linestyle='--', zorder=1)

        # 创建自定义图例
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

        # 添加参照物图例
        ref_count = min(len(result.get('reference_polygons', [])), 4)
        for i in range(ref_count):
            color_idx = i % len(self.ref_colors)
            legend_elements.append(
                plt.Rectangle((0, 0), 1, 1, facecolor=self.ref_colors[color_idx],
                              alpha=0.4, edgecolor='black', label=f'Building {i + 1}')
            )

        # # 将图例放在图形内部右上角
        # ax.legend(handles=legend_elements, fontsize=8, loc='upper right',
        #           framealpha=0.9, ncol=1, bbox_to_anchor=(1.4, 1.0))

        if fused_dist is not None:  # 确保颜色条存在
            # 修复：Bbox无centery属性，用center[1]获取垂直中心；调整y为颜色条顶部（y1）实现右上定位
            cax_pos = cax.get_position()
            fig = ax.get_figure()
            fig.legend(
                handles=legend_elements,
                fontsize=8,
                framealpha=0.9,
                ncol=1,
                loc='upper left',  # 图例以左上角对齐定位点
                # 定位点：颜色条右侧+0.01间距，颜色条顶部（y1）
                bbox_to_anchor=(cax_pos.x1 + 0.1, cax_pos.y1),
                frameon=True,
                edgecolor='gray'
            )
        else:
            # 没有颜色条时，保留原图例位置
            ax.legend(handles=legend_elements, fontsize=8, loc='upper right',
                      framealpha=0.9, ncol=1, bbox_to_anchor=(1.5, 1.0))


        # 设置等比例
        ax.set_aspect('equal', adjustable='box')

        # 自动调整坐标轴范围以显示所有内容
        all_points = []

        # 添加参照物多边形点
        if 'reference_polygons' in result:
            for polygons in result['reference_polygons']:
                for polygon in polygons:
                    if len(polygon) > 0:
                        all_points.extend(polygon)

        # 添加边界点
        if 'boundary_points' in result:
            all_points.extend(result['boundary_points'])

        # 添加最终位置
        if 'final_coordinate' in result:
            all_points.append(result['final_coordinate'])

        if all_points:
            all_points = np.array(all_points)
            x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
            y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()

            # 添加15%的边距
            x_margin = (x_max - x_min) * 0.15
            y_margin = (y_max - y_min) * 0.15

            ax.set_xlim(x_min - x_margin, x_max + x_margin)
            ax.set_ylim(y_min - y_margin, y_max + y_margin)

    def _plot_3d_distribution(self, ax, result):
        """
        绘制3D分布子图（曲面+等高线）
        """
        ax.set_title("3D Fuzzy Distribution Surface",
                     fontsize=12, fontweight='bold', pad=1)

        # 获取网格数据
        x_grid, y_grid, X, Y, avg_latitude = result['grid_info']
        Z = result['fused_distribution']

        # 检查Z的最大值
        max_z = np.max(Z)

        if max_z > 0:
            # 绘制3D曲面
            surf = ax.plot_surface(X, Y, Z, cmap='plasma', alpha=0.8,
                                   rstride=2, cstride=2, edgecolor='none',
                                   linewidth=0.5, antialiased=True, zorder=1)

            # 在底部平面绘制等高线投影
            offset = np.min(Z) - 0.1 * (np.max(Z) - np.min(Z))
            ax.contourf(X, Y, Z, zdir='z', offset=offset,
                        cmap='plasma', alpha=0.6, levels=10, zorder=2)

            # 标记最高点
            max_pos = np.unravel_index(np.argmax(Z), Z.shape)
            ax.scatter(X[max_pos], Y[max_pos], Z[max_pos] + 0.02,
                       c='red', s=100, marker='*', edgecolors='black',
                       label='Max Membership', zorder=3)

            # 标记最终位置
            final_coord = result['final_coordinate']
            if final_coord:
                # 找到最近网格点的隶属度值
                x_idx = np.argmin(np.abs(x_grid - final_coord[0]))
                y_idx = np.argmin(np.abs(y_grid - final_coord[1]))
                if 0 <= x_idx < len(x_grid) and 0 <= y_idx < len(y_grid):
                    z_value = Z[y_idx, x_idx]
                    ax.scatter([final_coord[0]], [final_coord[1]], [z_value],
                               c='green', s=150, marker='*', edgecolors='black',
                               label='Final Position', zorder=3)
        else:
            # 如果分布全为零，绘制平面
            ax.plot_surface(X, Y, np.zeros_like(Z), color='gray', alpha=0.5, zorder=1)

        # 设置坐标轴标签
        ax.set_xlabel('Longitude', fontsize=10, labelpad=10)
        ax.set_ylabel('Latitude', fontsize=10, labelpad=10)
        ax.set_zlabel('Membership', fontsize=10, labelpad=10)

        # 设置视角
        ax.view_init(elev=25, azim=45)

        # 添加简化的图例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='*', color='w', markerfacecolor='green',
                   markersize=10, markeredgecolor='black', label='Final Position'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
                   markersize=10, markeredgecolor='black', label='Max Membership'),
        ]
        ax.legend(handles=legend_elements, fontsize=8, loc='upper right')

    def _plot_statistics(self, ax, result):
        """
        绘制统计信息子图
        """
        ax.set_title("Location Statistics", fontsize=12, fontweight='bold', pad=10, y=1)
        ax.axis('off')

        final_coord = result['final_coordinate']
        conf_region = result['confidence_region']
        boundary_dists = result.get('boundary_distances', [])

        # 准备统计文本
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

        # 添加参照物信息
        if 'references_info' in result and result['references_info']:
            stats_text += f"\nReference Information:\n"
            for i, ref_info in enumerate(result['references_info'][:4]):  # 只显示前4个
                stats_text += f"  R{i + 1}: {ref_info['bearing']}°, "
                stats_text += f"{ref_info['distance']} m\n"

            if len(result['references_info']) > 4:
                stats_text += f"  ... and {len(result['references_info']) - 4} more\n"

        # 添加算法信息
        stats_text += f"""

Algorithm Parameters:

  Grid Resolution: 100×100
  Fusion Method: Product
  Defuzzification: Centroid
"""

        # 绘制文本框
        ax.text(0.05, 0.95, stats_text, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round',
                          facecolor=self.sci_colors['text_box'],
                          alpha=0.9, edgecolor='gray', linewidth=1),
                fontfamily='monospace')


class EnhancedFuzzyGeoLocator:
    """
    增强的模糊地理定位器：包含SCI论文风格的可视化
    """

    def __init__(self, fuzziness_levels=None, grid_resolution=100, output_dir='sci_results'):
        """
        初始化增强定位器

        Args:
            fuzziness_levels: 模糊度等级参数配置
            grid_resolution: 网格分辨率
            output_dir: 输出文件夹路径
        """
        self.output_dir = output_dir

        # 创建SCI可视化器
        self.visualizer = ImprovedVisualizer(output_dir)

        self.results = {}

        # 存储原始数据用于后续处理
        self.raw_data = None

        # 导入原始定位器（假设fuzzy_geolocator.py在同一目录）
        try:
            from fuzzy_geolocator import FuzzyGeoLocator

            # 创建原始定位器实例
            self.locator = FuzzyGeoLocator(fuzziness_levels, grid_resolution, output_dir)
            self.has_original_locator = True
        except ImportError:
            print("警告: 无法导入原始fuzzy_geolocator，仅支持可视化功能")
            self.has_original_locator = False

    def load_data_from_json(self, json_file):
        """
        从JSON文件加载数据

        Args:
            json_file: JSON文件路径

        Returns:
            处理后的数据
        """
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            print(f"已加载JSON文件，共{len(data['points'])}个定位点")

            # 保存原始数据
            self.raw_data = data

            # 如果有原始定位器，也加载数据到其中
            if self.has_original_locator:
                self.locator.data = data

            return data

        except Exception as e:
            print(f"加载JSON数据失败: {e}")
            return None

    def extract_polygons_from_references(self, references):
        """
        从参照物列表中提取多边形信息

        Args:
            references: 参照物列表

        Returns:
            多边形列表
        """
        polygons_list = []

        for ref in references:
            if 'wkt' in ref:
                polygons = self.visualizer.parse_wkt_to_polygons(ref['wkt'])
                polygons_list.append(polygons)

        return polygons_list

    def process_single_point_with_original(self, point_data, point_index=None):
        """
        使用原始定位器处理单个定位点

        Args:
            point_data: 点数据
            point_index: 点索引

        Returns:
            处理结果
        """
        if not self.has_original_locator:
            print("错误: 原始定位器不可用")
            return None

        # 使用原始定位器处理（不保存图片）
        result = self.locator.process_location_point(point_data, point_index, save_images=False)

        if result:
            # 添加WKT多边形信息用于可视化
            if 'references' in point_data:
                polygons_list = self.extract_polygons_from_references(point_data['references'])
                result['reference_polygons'] = polygons_list

            # 生成SCI风格可视化图（分别保存三个子图）
            self.visualizer.create_sci_visualization(result, point_index)

            # 保存结果
            self.results[result['location_id']] = result

        return result

    def simulate_processing_for_visualization(self, point_data, point_index=None):
        """
        模拟处理用于可视化（当原始定位器不可用时）

        Args:
            point_data: 点数据
            point_index: 点索引

        Returns:
            模拟结果
        """
        # 提取多边形信息
        polygons_list = []
        centroids = []
        boundary_points = []
        references_info = []

        if 'references' in point_data:
            polygons_list = self.extract_polygons_from_references(point_data['references'])

            # 为每个参照物计算质心
            for i, ref in enumerate(point_data['references']):
                if 'wkt' in ref and 'bearing' in ref:
                    try:
                        # 使用WKT解析几何体
                        geom = wkt.loads(ref['wkt'])
                        centroid_geom = geom.centroid
                        centroid = (centroid_geom.x, centroid_geom.y)
                        centroids.append(centroid)

                        # 计算边界点（简单模拟）
                        bearing_rad = math.radians(ref['bearing'])
                        boundary_distance_m = 10  # 假设10米
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
                        print(f"处理参照物{i}时出错: {e}")

        # 创建模拟结果
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

        # 创建模拟网格
        if boundary_points:
            boundary_array = np.array(boundary_points)

            # 创建简单网格
            min_x, max_x = boundary_array[:, 0].min(), boundary_array[:, 0].max()
            min_y, max_y = boundary_array[:, 1].min(), boundary_array[:, 1].max()

            # 扩展范围
            x_margin = (max_x - min_x) * 0.4
            y_margin = (max_y - min_y) * 0.4

            x_grid = np.linspace(min_x - x_margin, max_x + x_margin, 80)
            y_grid = np.linspace(min_y - y_margin, max_y + y_margin, 80)
            X, Y = np.meshgrid(x_grid, y_grid)

            # 创建基于参照物的高斯分布
            Z = np.zeros_like(X)

            for i, centroid in enumerate(centroids):
                sigma_x = 0.0005  # 固定sigma值
                sigma_y = 0.0005

                Z += np.exp(-((X - centroid[0]) ** 2 / (2 * sigma_x ** 2) +
                              (Y - centroid[1]) ** 2 / (2 * sigma_y ** 2)))

            # 归一化
            if Z.max() > 0:
                Z = Z / Z.max()

            result['fused_distribution'] = Z
            result['grid_info'] = (x_grid, y_grid, X, Y, 39.9)

            # 创建置信区域
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

        # 生成SCI风格可视化图（分别保存三个子图）
        self.visualizer.create_sci_visualization(result, point_index)

        # 保存结果
        self.results[result['location_id']] = result

        return result

    def process_all_points(self, data=None, use_original_locator=True):
        """
        处理所有定位点

        Args:
            data: 输入数据
            use_original_locator: 是否使用原始定位器

        Returns:
            所有处理结果
        """
        if data is None:
            data = self.raw_data

        if data is None:
            print("错误: 没有加载数据")
            return []

        all_results = []

        print("=" * 60)
        print("开始处理所有定位点（SCI风格可视化）")
        print("=" * 60)

        # 处理所有点
        points_to_process = data['points']

        for i, point_data in enumerate(points_to_process):
            print(f"\n处理点 {i + 1}/{len(points_to_process)}: {point_data['id']}")

            if use_original_locator and self.has_original_locator:
                result = self.process_single_point_with_original(point_data, i)
            else:
                result = self.simulate_processing_for_visualization(point_data, i)

            if result:
                all_results.append(result)

        print(f"\n处理完成! 成功处理 {len(all_results)} 个定位点")

        # 保存结果
        self.save_results(all_results)

        return all_results

    def save_results(self, results):
        """
        保存处理结果

        Args:
            results: 结果列表
        """
        if not results:
            return

        # 创建数据目录
        data_dir = os.path.join(self.output_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)

        # 保存JSON格式结果
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

        print(f"\nJSON结果已保存到 {json_filename}")


def main():
    """
    主函数：使用JSON数据和增强的定位器
    """
    # 创建增强定位器
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

    # 加载JSON数据
    json_file = './data/data-turth.json'  # 请确保文件路径正确
    data = enhancer.load_data_from_json(json_file)

    if data is None:
        print("无法加载JSON数据，请检查文件路径和格式")
        return []

    # 处理所有点（使用原始定位器）
    use_original = enhancer.has_original_locator
    if not use_original:
        print("注意: 将使用模拟处理进行可视化")

    # 处理全部的点
    results = enhancer.process_all_points(data, use_original_locator=use_original)

    # 打印总结
    if results:
        print("\n" + "=" * 60)
        print("SCI风格模糊地理定位系统 - 处理总结")
        print("=" * 60)

        confidences = [r.get('confidence', 0) for r in results]
        areas = [r['confidence_region']['area_m2'] for r in results
                 if r.get('confidence_region')]

        print(f"处理点位总数: {len(results)}")
        if confidences:
            print(f"平均置信度: {np.mean(confidences):.3f} "
                  f"(范围: {np.min(confidences):.3f} - {np.max(confidences):.3f})")

        if areas:
            print(f"平均置信区域面积: {np.mean(areas):.1f} 平方米")

        print(f"可视化图保存在: {os.path.join(output_directory, 'images')}")

        # 统计生成图片数量
        img_dir = os.path.join(output_directory, 'images')
        if os.path.exists(img_dir):
            img_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
            print(f"生成图片总数: {len(img_files)}")
            print(f"  2D图: {len([f for f in img_files if '2d_plot' in f])}")
            print(f"  3D图: {len([f for f in img_files if '3d_plot' in f])}")
            print(f"  统计图: {len([f for f in img_files if 'statistics' in f])}")

    return results


if __name__ == "__main__":
    # 执行主函数
    results = main()

    # 显示第一个点的可视化信息
    if results:
        print(f"\n第一个点的信息:")
        print(f"  ID: {results[0]['location_id']}")
        print(f"  最终坐标: ({results[0]['final_coordinate'][0]:.6f}, "
              f"{results[0]['final_coordinate'][1]:.6f})")
        print(f"  置信度: {results[0].get('confidence', 0):.3f}")
        print(f"  参照物数量: {results[0]['distribution_count']}")