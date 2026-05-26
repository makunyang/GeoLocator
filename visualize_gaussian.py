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
  Membership: Gaussian
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
    增强的模糊地理定位器：内置高斯衰减，无外部依赖
    """

    def __init__(self, fuzziness_levels=None, grid_resolution=100, output_dir='sci_results'):
        """
        初始化增强定位器（内置高斯衰减）
        """
        self.output_dir = output_dir
        self.visualizer = ImprovedVisualizer(output_dir)
        self.results = {}
        self.raw_data = None
        self.grid_resolution = grid_resolution

        # 模糊度等级参数（与原代码完全一致）
        if fuzziness_levels is None:
            self.fuzziness_levels = {
                'high': {'delta_theta': 60.0, 'delta_distance': 1.0},
                'medium': {'delta_theta': 30.0, 'delta_distance': 0.5},
                'low': {'delta_theta': 15.0, 'delta_distance': 0.2}
            }
        else:
            self.fuzziness_levels = fuzziness_levels

        # 经纬度转米系数
        self.longitude_to_meters = 85000
        self.latitude_to_meters = 111000

    def extract_polygons_from_references(self, references):
        """
        从参照物列表中提取多边形信息
        """
        polygons_list = []
        for ref in references:
            if 'wkt' in ref:
                polygons = self.visualizer.parse_wkt_to_polygons(ref['wkt'])
                polygons_list.append(polygons)
        return polygons_list

    # ===================== 核心：内置高斯衰减函数 =====================
    def _gaussian_membership(self, diff, delta):
        """高斯非线性隶属度函数"""
        if delta == 0:
            return 1.0 if diff == 0 else 0.0
        return math.exp(- (diff ** 2) / (2 * delta ** 2))

    def calculate_fuzzy_membership(self, point_lon, point_lat, boundary_lon, boundary_lat,
                                   bearing_deg, distance_meters, avg_latitude, delta_theta, delta_distance):
        """高斯衰减模糊隶属度计算（核心）"""
        dx = point_lon - boundary_lon
        dy = point_lat - boundary_lat
        dx_meters = dx * (self.longitude_to_meters * math.cos(math.radians(avg_latitude)))
        dy_meters = dy * self.latitude_to_meters

        actual_angle = math.degrees(math.atan2(dx_meters, dy_meters))
        actual_angle = actual_angle + 360 if actual_angle < 0 else actual_angle
        actual_distance = math.sqrt(dx_meters ** 2 + dy_meters ** 2)
        angle_diff = min(abs(actual_angle - bearing_deg), 360 - abs(actual_angle - bearing_deg))

        # 高斯衰减
        ma = self._gaussian_membership(angle_diff, delta_theta)
        md = 0.0
        if distance_meters > 0:
            rel_diff = abs(actual_distance - distance_meters) / distance_meters
            md = self._gaussian_membership(rel_diff, delta_distance)

        return max(ma * md, 0.0)

    def calculate_fuzzy_distribution(self, ref, boundary_point, avg_latitude, xg, yg, X, Y):
        """计算单参照物高斯模糊分布"""
        delta_theta = self.fuzziness_levels[self._assign_fuzziness_level(ref)]['delta_theta']
        delta_distance = self.fuzziness_levels[self._assign_fuzziness_level(ref)]['delta_distance']
        fuzzy_dist = np.zeros_like(X)
        bearing, distance = ref['bearing'], ref['distance']
        blon, blat = boundary_point

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                fuzzy_dist[i, j] = self.calculate_fuzzy_membership(
                    X[i,j], Y[i,j], blon, blat, bearing, distance, avg_latitude, delta_theta, delta_distance
                )
        return fuzzy_dist

    def fuse_distributions(self, distributions):
        """乘积融合模糊分布"""
        fused = np.ones_like(distributions[0])
        for d in distributions:
            fused *= d
        return fused

    def defuzzify(self, fuzzy_dist, xg, yg):
        """重心法去模糊化"""
        total = np.sum(fuzzy_dist)
        if total == 0:
            return tuple(np.mean(np.array([bp for bp in self.current_bps]), axis=0)), 0.1
        yi, xi = np.indices(fuzzy_dist.shape)
        cx, cy = np.sum(xi*fuzzy_dist)/total, np.sum(yi*fuzzy_dist)/total
        xi, yi = int(np.clip(cx, 0, len(xg)-2)), int(np.clip(cy, 0, len(yg)-2))
        dx, dy = cx-xi, cy-yi
        x = (1-dx)*(1-dy)*xg[xi] + dx*(1-dy)*xg[xi+1] + (1-dx)*dy*xg[xi] + dx*dy*xg[xi+1]
        y = (1-dx)*(1-dy)*yg[yi] + dx*(1-dy)*yg[yi+1] + (1-dx)*dy*yg[yi] + dx*dy*yg[yi+1]
        return (x,y), np.max(fuzzy_dist)

    def calculate_confidence_region(self, dist, xg, yg):
        """计算50%置信区域"""
        if dist is None or np.max(dist)==0:
            return None
        mask = dist >= np.max(dist)*0.5
        if not np.any(mask):
            return None
        ys, xs = np.where(mask)
        x1, x2 = xg[xs.min()], xg[xs.max()]
        y1, y2 = yg[ys.min()], yg[ys.max()]
        area = abs((x2-x1)*self.longitude_to_meters * (y2-y1)*self.latitude_to_meters)
        return {'x_min':x1,'x_max':x2,'y_min':y1,'y_max':y2,'area_m2':area}

    def _assign_fuzziness_level(self, ref):
        """自动分配模糊度等级"""
        d = ref['distance']
        if d < 20:
            return 'low'
        elif d < 50:
            return 'medium'
        else:
            return 'high'

    def calculate_boundary_reference_point(self, geom, centroid, bearing):
        """计算边界参考点（与原代码完全一致）"""
        try:
            centroid_point = Point(centroid[0], centroid[1])
            bearing_rad = math.radians(bearing)
            ray_length = 0.01
            dx = ray_length * math.sin(bearing_rad)
            dy = ray_length * math.cos(bearing_rad)
            ray_end = Point(centroid[0] + dx, centroid[1] + dy)
            ray_line = LineString([centroid_point, ray_end])
            intersection = geom.boundary.intersection(ray_line)

            if intersection.is_empty:
                return centroid, 0.0
            if intersection.geom_type == 'Point':
                boundary_point = (intersection.x, intersection.y)
            else:
                points = []
                if hasattr(intersection, 'geoms'):
                    for g in intersection.geoms:
                        if g.geom_type == 'Point':
                            points.append((g.x, g.y))
                if not points:
                    return centroid, 0.0
                distances = [math.hypot(p[0]-centroid[0], p[1]-centroid[1]) for p in points]
                boundary_point = points[np.argmin(distances)]

            dx_m = (boundary_point[0] - centroid[0]) * self.longitude_to_meters
            dy_m = (boundary_point[1] - centroid[1]) * self.latitude_to_meters
            boundary_distance = math.sqrt(dx_m ** 2 + dy_m ** 2)
            return boundary_point, boundary_distance
        except:
            return centroid, 0.0

    def parse_wkt_to_geometry(self, wkt_string, bearing):
        """解析WKT几何"""
        try:
            geom = wkt.loads(wkt_string)
            c = geom.centroid
            centroid = (c.x, c.y)
            bp, bd = self.calculate_boundary_reference_point(geom, centroid, bearing)
            return geom, centroid, bp, bd
        except:
            return None, None, None, 0.0

    def create_search_grid(self, boundary_points, max_distance_meters=100):
        """创建搜索网格"""
        arr = np.array(boundary_points)
        avg_lat = np.mean(arr[:,1])
        d_lon = max_distance_meters / (self.longitude_to_meters * math.cos(math.radians(avg_lat)))
        d_lat = max_distance_meters / self.latitude_to_meters
        min_x = arr[:,0].min() - d_lon
        max_x = arr[:,0].max() + d_lon
        min_y = arr[:,1].min() - d_lat
        max_y = arr[:,1].max() + d_lat
        xg = np.linspace(min_x, max_x, self.grid_resolution)
        yg = np.linspace(min_y, max_y, self.grid_resolution)
        X, Y = np.meshgrid(xg, yg)
        return xg, yg, X, Y, avg_lat

    # ===================== 处理逻辑（高斯衰减） =====================
    def process_single_point(self, point_data, point_index=None):
        """内置高斯衰减处理单个点（替换原外部调用）"""
        loc_id = point_data.get('id', f'Point_{point_index}')
        refs = point_data.get('references', [])
        centroids, bps, bdists, ref_infos = [], [], [], []

        for r in refs:
            geom, c, bp, bd = self.parse_wkt_to_geometry(r['wkt'], r['bearing'])
            if c and bp:
                centroids.append(c)
                bps.append(bp)
                bdists.append(bd)
                ref_infos.append({'bearing':r['bearing'],'distance':r['distance'],'centroid':c,'boundary_point':bp,'boundary_distance':bd})

        if not bps:
            return None
        self.current_bps = bps
        max_d = max([r['distance'] for r in refs]) if refs else 100
        xg, yg, X, Y, avg_lat = self.create_search_grid(bps, max_d*3)

        # 高斯模糊分布计算 + 融合
        dists = []
        for i, r in enumerate(refs):
            geom, c, bp, bd = self.parse_wkt_to_geometry(r['wkt'], r['bearing'])
            if not bp:
                continue
            fd = self.calculate_fuzzy_distribution(r, bp, avg_lat, xg, yg, X, Y)
            dists.append(fd)
        fused = self.fuse_distributions(dists) if dists else None
        coord, conf = self.defuzzify(fused, xg, yg)
        region = self.calculate_confidence_region(fused, xg, yg)
        # ✅ 修复：调用自身的方法，而非可视化器
        polygons_list = self.extract_polygons_from_references(refs)

        # 构造结果（完全兼容原可视化）
        result = {
            'location_id': loc_id, 'final_coordinate': coord, 'confidence': conf,
            'confidence_region': region, 'centroids': centroids, 'boundary_points': bps,
            'boundary_distances': bdists, 'fused_distribution': fused,
            'grid_info': (xg, yg, X, Y, avg_lat), 'distribution_count': len(refs),
            'references_info': ref_infos, 'reference_polygons': polygons_list
        }
        self.results[loc_id] = result
        self.visualizer.create_sci_visualization(result, point_index)
        return result

    # ===================== 原有功能完全保留 =====================
    def load_data_from_json(self, json_file):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"已加载JSON文件，共{len(data['points'])}个定位点")
            self.raw_data = data
            return data
        except Exception as e:
            print(f"加载JSON数据失败: {e}")
            return None

    def process_all_points(self, data=None):
        if data is None:
            data = self.raw_data
        if data is None:
            return []

        all_results = []
        print("=" * 60)
        print("开始处理所有定位点（内置高斯衰减）")
        print("=" * 60)

        for i, point_data in enumerate(data['points']):
            print(f"\n处理点 {i + 1}/{len(data['points'])}: {point_data['id']}")
            result = self.process_single_point(point_data, i)
            if result:
                all_results.append(result)

        print(f"\n处理完成! 成功处理 {len(all_results)} 个定位点")
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
                'final_coordinate': {'longitude': float(final_coord[0]), 'latitude': float(final_coord[1])},
                'confidence': float(result['confidence']),
                'distribution_count': result['distribution_count']
            }
            conf_region = result.get('confidence_region')
            if conf_region:
                output_result['confidence_region'] = {
                    'x_min': float(conf_region.get('x_min', 0)), 'x_max': float(conf_region.get('x_max', 0)),
                    'y_min': float(conf_region.get('y_min', 0)), 'y_max': float(conf_region.get('y_max', 0)),
                    'area_m2': float(conf_region.get('area_m2', 0))
                }
            output_data.append(output_result)

        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nJSON结果已保存到 {json_filename}")


def main():
    """
    主函数：使用JSON数据 + 内置高斯衰减
    """
    output_directory = 'sci_fuzzy_geolocation_results_gaussian2'
    enhancer = EnhancedFuzzyGeoLocator(
        fuzziness_levels={
            'high': {'delta_theta': 60.0, 'delta_distance': 1.0},
            'medium': {'delta_theta': 30.0, 'delta_distance': 0.5},
            'low': {'delta_theta': 15.0, 'delta_distance': 0.2}
        },
        grid_resolution=100,
        output_dir=output_directory
    )

    # 加载JSON数据（与原代码完全一致）
    json_file = './data/data3-deepseek.json'
    data = enhancer.load_data_from_json(json_file)

    if data is None:
        print("无法加载JSON数据，请检查文件路径和格式")
        return []

    # 处理所有点（内置高斯衰减）
    results = enhancer.process_all_points(data)

    # 打印总结（完全不变）
    if results:
        print("\n" + "=" * 60)
        print("SCI风格模糊地理定位系统 - 高斯衰减版")
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

        img_dir = os.path.join(output_directory, 'images')
        if os.path.exists(img_dir):
            img_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
            print(f"生成图片总数: {len(img_files)}")
            print(f"  2D图: {len([f for f in img_files if '2d_plot' in f])}")
            print(f"  3D图: {len([f for f in img_files if '3d_plot' in f])}")
            print(f"  统计图: {len([f for f in img_files if 'statistics' in f])}")

    return results


if __name__ == "__main__":
    results = main()
    if results:
        print(f"\n第一个点的信息:")
        print(f"  ID: {results[0]['location_id']}")
        print(f"  最终坐标: ({results[0]['final_coordinate'][0]:.6f}, "
              f"{results[0]['final_coordinate'][1]:.6f})")
        print(f"  置信度: {results[0].get('confidence', 0):.3f}")
        print(f"  参照物数量: {results[0]['distribution_count']}")