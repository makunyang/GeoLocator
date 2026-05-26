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

warnings.filterwarnings('ignore')


plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class FuzzyGeoLocator:


    def __init__(self, fuzziness_levels=None, grid_resolution=100, output_dir='results', primary_model='gaussian'):

        if fuzziness_levels is None:
            self.fuzziness_levels = {
                'high': {'delta_theta': 60.0, 'delta_distance': 1.0},
                'medium': {'delta_theta': 30.0, 'delta_distance': 0.5},
                'low': {'delta_theta': 15.0, 'delta_distance': 0.2}
            }
        else:
            self.fuzziness_levels = fuzziness_levels

        self.grid_resolution = grid_resolution
        self.results = {}
        self.data = None
        self.primary_model = primary_model 

 
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.img_dir = os.path.join(output_dir, 'images')
        self.data_dir = os.path.join(output_dir, 'data')
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

 
        self.longitude_to_meters = 85000
        self.latitude_to_meters = 111000


    def _linear_membership(self, diff, delta):

        if diff <= delta:
            return 1.0 - (diff / delta)
        return 0.0

    def _gaussian_membership(self, diff, delta):

        if delta == 0:
            return 1.0 if diff == 0 else 0.0
        return math.exp(- (diff ** 2) / (2 * delta ** 2))

    def _exponential_membership(self, diff, delta):

        if delta == 0:
            return 1.0 if diff == 0 else 0.0
        return math.exp(- diff / delta)

    
    def calculate_fuzzy_membership(self, point_lon, point_lat, boundary_lon, boundary_lat,
                                   bearing_deg, distance_meters, avg_latitude,
                                   delta_theta, delta_distance, model_type='linear'):
    
        dx = point_lon - boundary_lon
        dy = point_lat - boundary_lat

        dx_meters = dx * (self.longitude_to_meters * math.cos(math.radians(avg_latitude)))
        dy_meters = dy * self.latitude_to_meters

        actual_angle = math.degrees(math.atan2(dx_meters, dy_meters))
        if actual_angle < 0:
            actual_angle += 360

        actual_distance = math.sqrt(dx_meters ** 2 + dy_meters ** 2)

        angle_diff = min(abs(actual_angle - bearing_deg), 360 - abs(actual_angle - bearing_deg))

     
        if model_type == 'gaussian':
            ma = self._gaussian_membership(angle_diff, delta_theta)
        elif model_type == 'exponential':
            ma = self._exponential_membership(angle_diff, delta_theta)
        else:
            ma = self._linear_membership(angle_diff, delta_theta)

  
        md = 0.0
        if distance_meters > 0 and actual_distance >= 0:
            rel_diff = abs(actual_distance - distance_meters) / distance_meters
            if model_type == 'gaussian':
                md = self._gaussian_membership(rel_diff, delta_distance)
            elif model_type == 'exponential':
                md = self._exponential_membership(rel_diff, delta_distance)
            else:
                md = self._linear_membership(rel_diff, delta_distance)

        return max(ma * md, 0.0)

    def calculate_fuzzy_distribution_for_point(self, reference_data, fuzziness_level,
                                               x_grid, y_grid, X, Y, boundary_point, avg_latitude, model_type='linear'):
        delta_theta = self.fuzziness_levels[fuzziness_level]['delta_theta']
        delta_distance = self.fuzziness_levels[fuzziness_level]['delta_distance']
        fuzzy_dist = np.zeros_like(X)
        ref_distance = reference_data['distance']
        bearing = reference_data['bearing']
        boundary_lon, boundary_lat = boundary_point

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                fuzzy_dist[i, j] = self.calculate_fuzzy_membership(
                    X[i, j], Y[i, j], boundary_lon, boundary_lat,
                    bearing, ref_distance, avg_latitude,
                    delta_theta, delta_distance, model_type
                )
        return fuzzy_dist


    def load_excel_data(self, filepath):
        try:
            df = pd.read_excel(filepath)
            print(f"已加载Excel文件，共{len(df)}行数据")
            df['点ID'] = df['点ID'].ffill().astype(int)
            points = []
            for point_id, group in df.groupby('点ID'):
                references = []
                for _, row in group.iterrows():
                    reference = {
                        'type': 'polygon',
                        'wkt': row['WKT'],
                        'bearing': float(row['角度']),
                        'distance': float(row['距离'])
                    }
                    references.append(reference)
                points.append({'id': f"Point{int(point_id)}", 'references': references})
            self.data = {'points': points}
            print(f"处理后得到 {len(points)} 个定位点")
            return self.data
        except Exception as e:
            print(f"加载Excel数据失败: {e}")
            return None

    def calculate_boundary_reference_point(self, geom, centroid, bearing):
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
        try:
            geom = wkt.loads(wkt_string)
            c = geom.centroid
            centroid = (c.x, c.y)
            bp, bd = self.calculate_boundary_reference_point(geom, centroid, bearing)
            return geom, centroid, bp, bd
        except:
            return None, None, None, 0.0

    def create_search_grid(self, boundary_points, max_distance_meters=100):
        if not boundary_points:
            return None, None, None, None, None
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

    def fuse_distributions(self, distributions, fusion_method='product'):
        if not distributions:
            return None
        if fusion_method == 'product':
            fused = np.ones_like(distributions[0][0])
            for d,_ in distributions: fused *= d
        elif fusion_method == 'min':
            fused = distributions[0][0].copy()
            for d,_ in distributions[1:]: fused = np.minimum(fused, d)
        elif fusion_method == 'average':
            fused = np.zeros_like(distributions[0][0])
            for d,_ in distributions: fused += d
            fused /= len(distributions)
        else:
            fused = np.ones_like(distributions[0][0])
            for d,_ in distributions: fused *= d
        return fused

    def defuzzify(self, fuzzy_dist, xg, yg, method='centroid'):
        if fuzzy_dist is None or np.max(fuzzy_dist) == 0:
            return None, 0
        if method == 'centroid':
            total = np.sum(fuzzy_dist)
            if total == 0: return None, 0
            yi, xi = np.indices(fuzzy_dist.shape)
            cx = np.sum(xi * fuzzy_dist) / total
            cy = np.sum(yi * fuzzy_dist) / total
            xi = int(np.clip(cx, 0, len(xg)-2))
            yi = int(np.clip(cy, 0, len(yg)-2))
            dx, dy = cx-xi, cy-yi
            x = (1-dx)*(1-dy)*xg[xi] + dx*(1-dy)*xg[xi+1] + (1-dx)*dy*xg[xi] + dx*dy*xg[xi+1]
            y = (1-dx)*(1-dy)*yg[yi] + dx*(1-dy)*yg[yi+1] + (1-dx)*dy*yg[yi] + dx*dy*yg[yi+1]
            return (x,y), np.max(fuzzy_dist)
        else:
            yp, xp = np.unravel_index(np.argmax(fuzzy_dist), fuzzy_dist.shape)
            return (xg[xp], yg[yp]), fuzzy_dist[yp,xp]

    def calculate_confidence_region(self, dist, xg, yg, level=0.5):
        if dist is None or np.max(dist)==0: return None
        mask = dist >= np.max(dist)*level
        if not np.any(mask): return None
        ys, xs = np.where(mask)
        x1, x2 = xg[xs.min()], xg[xs.max()]
        y1, y2 = yg[ys.min()], yg[ys.max()]
        area = abs((x2-x1)*self.longitude_to_meters * (y2-y1)*self.latitude_to_meters)
        return {'x_min':x1,'x_max':x2,'y_min':y1,'y_max':y2,'area_m2':area}

    def _assign_fuzziness_level(self, ref, idx, total):
        d = ref['distance']
        if d<20: return 'low'
        elif d<50: return 'medium'
        else: return 'high'

    def process_location_point(self, point_data, point_index=None, save_images=True):
        loc_id = point_data.get('id', f'Point_{point_index}')
        refs = point_data.get('references', [])
        print(f"\n{'='*60}\n处理位置: {loc_id}")

        centroids = []
        bps = []
        bdists = []
        ref_infos = []

        for r in refs:
            geom, c, bp, bd = self.parse_wkt_to_geometry(r['wkt'], r['bearing'])
            if c and bp:
                centroids.append(c)
                bps.append(bp)
                bdists.append(bd)
                ref_infos.append({'bearing':r['bearing'],'distance':r['distance'],'centroid':c,'boundary_point':bp,'boundary_distance':bd})

        if not bps:
            print(f"无有效边界点")
            return None

        max_d = max([r['distance'] for r in refs]) if refs else 100
        xg, yg, X, Y, avg_lat = self.create_search_grid(bps, max_d*3)
        if xg is None: return None

        models = ['linear', 'gaussian', 'exponential']
        model_results = {}

        for m in models:
            dists = []
            for i, r in enumerate(refs):
                lev = self._assign_fuzziness_level(r, i, len(refs))
                geom, c, bp, bd = self.parse_wkt_to_geometry(r['wkt'], r['bearing'])
                if not bp: continue
                fd = self.calculate_fuzzy_distribution_for_point(r, lev, xg, yg, X, Y, bp, avg_lat, m)
                dists.append((fd, {}))
            fused = self.fuse_distributions(dists) if dists else None
            if fused is None or np.max(fused)==0:
                fused = self.fuse_distributions(dists, 'average') if dists else None
            coord, conf = self.defuzzify(fused, xg, yg)
            if coord is None:
                coord = tuple(np.mean(bps, axis=0))
                conf = 0.1
            region = self.calculate_confidence_region(fused, xg, yg)
            model_results[m] = {
                'coord': coord, 'conf': conf, 'region': region, 'fused': fused
            }

        final = model_results[self.primary_model]
        result = {
            'location_id': loc_id,
            'final_coordinate': final['coord'],
            'confidence': final['conf'],
            'confidence_region': final['region'],
            'centroids': centroids,
            'boundary_points': bps,
            'boundary_distances': bdists,
            'fused_distribution': final['fused'],
            'grid_info': (xg, yg, X, Y, avg_lat),
            'distribution_count': len(refs),
            'references_info': ref_infos,
            'model_results': model_results,
            'primary_model': self.primary_model
        }
        self.results[loc_id] = result
        if save_images:
            self.visualize_results(result, point_index)
            self.visualize_3d_results(result, point_index)
        return result

    def visualize_results(self, result, point_index=None):
        if result is None: return
        fig, axes = plt.subplots(2,2,figsize=(14,12))
        tid = result['location_id']
        if point_index is not None: tid += f' (点{point_index+1})'
        fig.suptitle(f'模糊地理定位结果 - {tid} (主模型:{result["primary_model"]})', fontsize=16, fontweight='bold')
        xg, yg, X, Y, avg_lat = result['grid_info']

        ax = axes[0,0]
        if result['fused_distribution'] is not None:
            ax.contourf(X,Y,result['fused_distribution'],20,cmap='YlOrRd',alpha=0.8)
            fx,fy = result['final_coordinate']
            ax.plot(fx,fy,'g*',markersize=15,label='最终位置',markeredgecolor='black')
            for i,c in enumerate(result['centroids']):
                ax.plot(c[0],c[1],'ro',markersize=6,alpha=0.7,label='质心'if i==0 else "")
                if i<len(result['boundary_points']):
                    bp = result['boundary_points'][i]
                    ax.plot(bp[0],bp[1],'bs',markersize=8,alpha=0.7,label='边界点'if i==0 else "")
                    ax.plot([c[0],bp[0]],[c[1],bp[1]],'k--',lw=1,alpha=0.5)
                    ax.text(c[0],c[1],f'C{i+1}',fontsize=8)
                    ax.text(bp[0],bp[1],f'B{i+1}',fontsize=8)
        ax.set_title('融合后模糊分布');ax.set_xlabel('经度');ax.set_ylabel('纬度');ax.grid(alpha=0.3);ax.legend()

        ax=axes[0,1]
        if result['fused_distribution'] is not None:
            ax.imshow(result['fused_distribution'],extent=[xg.min(),xg.max(),yg.min(),yg.max()],origin='lower',cmap='hot',alpha=0.7)
            fx,fy=result['final_coordinate']
            ax.plot(fx,fy,'g*',markersize=15,label='最终位置',markeredgecolor='black')
            cr=result['confidence_region']
            if cr:
                r=plt.Rectangle((cr['x_min'],cr['y_min']),cr['x_max']-cr['x_min'],cr['y_max']-cr['y_min'],fill=False,ec='blue',lw=2,ls='--',label='50%置信区域')
                ax.add_patch(r)
        ax.set_title('去模糊化结果');ax.set_xlabel('经度');ax.set_ylabel('纬度');ax.grid(alpha=0.3);ax.legend()

        ax=axes[1,0]
        for i,info in enumerate(result['references_info']):
            bp=info['boundary_point']
            c=info['centroid']
            if bp and c:
                ax.plot(bp[0],bp[1],'bs',markersize=10,mfc='white',mec='blue',mew=2,label='边界点'if i==0 else "")
                ax.plot(c[0],c[1],'ro',markersize=8,alpha=0.7,label='质心'if i==0 else "")
                ax.text(bp[0],bp[1],f'B{i+1}',fontsize=10,ha='center',va='bottom',bbox=dict(boxstyle='round,pad=0.3',fc='lightblue',alpha=0.7))
        if result['boundary_points']:
            arr=np.array(result['boundary_points'])
            pad=max(arr[:,0].max()-arr[:,0].min(),arr[:,1].max()-arr[:,1].min())*0.3
            ax.set_xlim(arr[:,0].min()-pad,arr[:,0].max()+pad)
            ax.set_ylim(arr[:,1].min()-pad,arr[:,1].max()+pad)
        for info in result['references_info']:
            bp=info['boundary_point']
            if not bp:continue
            b=info['bearing']
            d=info['distance']
            rad=math.radians(b)
            dd=d/(self.longitude_to_meters*math.cos(math.radians(avg_lat)))
            dx,dy=dd*math.sin(rad),dd*math.cos(rad)
            ax.arrow(bp[0],bp[1],dx,dy,head_width=dd*0.1,head_length=dd*0.15,fc='red',ec='darkred',alpha=0.8,lw=1,length_includes_head=True)
        ax.set_title('边界点与方向');ax.set_xlabel('经度');ax.set_ylabel('纬度');ax.grid(alpha=0.3);ax.set_aspect('equal',adjustable='box');ax.legend()

        ax=axes[1,1];ax.axis('off')
        fx,fy=result['final_coordinate']
        cr=result['confidence_region']
        txt=f"""
        定位统计信息:
        位置ID: {result['location_id']}
        主模型: {result['primary_model']}
        参照物数量: {result['distribution_count']}
        最终坐标: ({fx:.6f},{fy:.6f})
        置信度: {result['confidence']:.3f}
        """
        if cr:
            txt+=f"""
        置信区域: {cr['area_m2']:.1f}㎡
        """
        if result['boundary_distances']:
            txt+=f"""
        平均边界距离: {np.mean(result['boundary_distances']):.2f}m
        """
        mr = result.get('model_results', {})
        txt += f"""
        --- 三模型对比 ---
        线性结果:    ({mr['linear']['coord'][0]:.6f},{mr['linear']['coord'][1]:.6f}) 置信度:{mr['linear']['conf']:.3f}
        高斯结果:    ({mr['gaussian']['coord'][0]:.6f},{mr['gaussian']['coord'][1]:.6f}) 置信度:{mr['gaussian']['conf']:.3f}
        指数结果:    ({mr['exponential']['coord'][0]:.6f},{mr['exponential']['coord'][1]:.6f}) 置信度:{mr['exponential']['conf']:.3f}
        """
        ax.text(0.05,0.95,txt,fontsize=8,va='top',bbox=dict(boxstyle='round',fc='wheat',alpha=0.8))
        plt.tight_layout()
        fp=os.path.join(self.img_dir,f"{result['location_id']}_2d.png")
        plt.savefig(fp,dpi=150,bbox_inches='tight')
        plt.close()

    def visualize_3d_results(self, result, point_index=None):
        if result is None or result['fused_distribution'] is None: return
        fig = plt.figure(figsize=(12,9))
        xg,yg,X,Y,_ = result['grid_info']
        Z = result['fused_distribution']
        maxz = np.max(Z) if Z is not None else 0

        ax1=fig.add_subplot(221,projection='3d')
        if maxz>0: ax1.plot_surface(X,Y,Z,cmap='viridis',alpha=0.8,rstride=2,cstride=2,ec='none')
        ax1.set_title(f'3D分布曲面 ({result["primary_model"]})')
        ax1.set_xlabel('经度');ax1.set_ylabel('纬度');ax1.set_zlabel('隶属度')
        ax1.view_init(30,45)

        ax2=fig.add_subplot(222,projection='3d')
        if maxz>0: ax2.contour(X,Y,Z,np.linspace(0,maxz,10),cmap='hot',lw=2)
        ax2.set_title('等高线')
        ax2.view_init(90,45)

        ax3=fig.add_subplot(223,projection='3d')
        if maxz>0: ax3.plot_wireframe(X,Y,Z,rstride=5,cstride=5,color='blue',alpha=0.6,lw=0.5)
        fx,fy=result['final_coordinate']
        xi=np.argmin(np.abs(xg-fx))
        yi=np.argmin(np.abs(yg-fy))
        zv=Z[yi,xi] if (0<=xi<len(xg)and 0<=yi<len(yg)) else 0
        ax3.scatter([fx],[fy],[zv],c='green',s=200,marker='*',label='最终位置')
        ax3.set_title('线框图');ax3.legend();ax3.view_init(20,60)

        ax4=fig.add_subplot(224,projection='3d')
        if maxz>0:
            xs=np.linspace(0,X.shape[1]-1,10,dtype=int)
            ys=np.linspace(0,X.shape[0]-1,10,dtype=int)
            for i in xs:
                for j in ys:
                    ax4.bar3d(X[j,i],Y[j,i],0,(xg[1]-xg[0])/3,(yg[1]-yg[0])/3,Z[j,i],alpha=0.7,color=plt.cm.viridis(Z[j,i]/maxz))
        ax4.set_title('柱状图');ax4.view_init(25,45)

        plt.suptitle(f'3D结果 - {result["location_id"]}',fontsize=16)
        plt.tight_layout()
        fp=os.path.join(self.img_dir,f"{result['location_id']}_3d.png")
        plt.savefig(fp,dpi=150,bbox_inches='tight')
        plt.close()

    def process_all_points(self, data=None, save_results=True):
        if data is None:
            if self.data is None:
                print("无数据")
                return []
            data = self.data
        ress=[]
        for i,p in enumerate(data['points']):
            res=self.process_location_point(p,i,True)
            if res: ress.append(res)
        if save_results and ress:
            self.save_results(ress)
        return ress

    def save_results(self, results):
        jp=os.path.join(self.data_dir,'results.json')
        cp=os.path.join(self.data_dir,'results.csv')
        xp=os.path.join(self.data_dir,'detailed.xlsx')
        cmp=os.path.join(self.data_dir,'model_comparison.csv')

        out=[]
        csv_rows=[]
        comp_rows=[]
        detailed=[]

        for r in results:
            fx,fy=r['final_coordinate']
            mr=r.get('model_results',{})
            cr=r.get('confidence_region',{})

            out.append({
                'id':r['location_id'],
                'primary_model': r['primary_model'],
                'lon':fx,'lat':fy,'conf':r['confidence'],
                'models':{k:{'lon':v['coord'][0],'lat':v['coord'][1],'conf':v['conf']} for k,v in mr.items()}
            })

            csv_rows.append({
                'id':r['location_id'],'primary_model': r['primary_model'],
                'lon':fx,'lat':fy,'conf':r['confidence'],
                'area':cr.get('area_m2',0) if cr else 0
            })

            for m in ['linear','gaussian','exponential']:
                d=mr[m]
                comp_rows.append({
                    'id':r['location_id'],'model':m,
                    'lon':d['coord'][0],'lat':d['coord'][1],'conf':d['conf']
                })

            for i,info in enumerate(r['references_info']):
                detailed.append({
                    'id':r['location_id'],'ref':i+1,
                    'bearing':info['bearing'],'distance':info['distance'],
                    'cen_lon':info['centroid'][0] if info['centroid'] else None,
                    'cen_lat':info['centroid'][1] if info['centroid'] else None,
                    'bp_lon':info['boundary_point'][0] if info['boundary_point'] else None,
                    'bp_lat':info['boundary_point'][1] if info['boundary_point'] else None,
                    'bp_dist':info['boundary_distance']
                })

        with open(jp,'w',encoding='utf-8')as f:
            json.dump(out,f,indent=2,ensure_ascii=False)

        pd.DataFrame(csv_rows).to_csv(cp,index=False,encoding='utf-8-sig')
        pd.DataFrame(detailed).to_excel(xp,index=False)
        pd.DataFrame(comp_rows).to_csv(cmp,index=False,encoding='utf-8-sig')

    def generate_summary_report(self, results):
        if not results:
            print("无有效结果，无法生成报告")
            return

        total_points = len(results) 
        ref_counts = [r['distribution_count'] for r in results]  
        avg_ref_count = round(np.mean(ref_counts), 1)  

        primary_confs = [r['confidence'] for r in results]
        avg_primary_conf = round(np.mean(primary_confs), 3)  
        conf_min = round(np.min(primary_confs), 3)  
        conf_max = round(np.max(primary_confs), 3) 
        area_list = []
        for r in results:
            cr = r['confidence_region']
            area = cr['area_m2'] if (cr and cr.get('area_m2') is not None) else 0
            area_list.append(area)
        avg_area = round(np.mean(area_list), 1)  
        bd_list = []
        for r in results:
            bds = r['boundary_distances']
            avg_bd = np.mean(bds) if bds else 0
            bd_list.append(avg_bd)
        avg_bd = round(np.mean(bd_list), 2)  

        models = ['linear', 'gaussian', 'exponential']
        model_summary = {}
        for m in models:
            m_confs = [r['model_results'][m]['conf'] for r in results]
            m_avg_conf = round(np.mean(m_confs), 3)
            m_area_list = []
            for r in results:
                m_cr = r['model_results'][m]['region']
                m_area = m_cr['area_m2'] if (m_cr and m_cr.get('area_m2') is not None) else 0
                m_area_list.append(m_area)
            m_avg_area = round(np.mean(m_area_list), 1)
            model_summary[m] = {
                'avg_conf': m_avg_conf,
                'avg_area': m_avg_area,
                'conf_range': f"{round(np.min(m_confs), 3)} - {round(np.max(m_confs), 3)}"
            }

        print("\n" + "="*80)
        print(" 模糊地理定位系统 - 三模型对比总结报告")
        print("="*80)
        print(f"主模型: {self.primary_model}")
        print(f"处理点位总数: {total_points}")
        print(f"平均置信度: {avg_primary_conf}")
        print(f"置信度范围: {conf_min} - {conf_max}")
        print(f"平均置信区域面积: {avg_area} 平方米")
        print(f"平均参照物数量: {avg_ref_count}")
        print(f"平均质心到边界距离: {avg_bd} 米")
        print("-"*80)
        print("分模型整体性能汇总:")
        for m in models:
            print(f"  [{m:10s}] 平均置信度: {model_summary[m]['avg_conf']:4f} | 平均置信区域面积: {model_summary[m]['avg_area']:6.1f}㎡ | 置信度范围: {model_summary[m]['conf_range']}")
        print("-"*80)
        print(f"\n点位详细对比:")
        print(f"{'点位ID':<8s} {'模型':<10s} {'经度':<12s} {'纬度':<12s} {'置信度':<8s}")
        print("-"*60)
        for r in results:
            for m in models:
                d = r['model_results'][m]
                print(f"{r['location_id']:<8s} {m:<10s} {d['coord'][0]:.6f} {d['coord'][1]:.6f} {d['conf']:.3f}")

        rp=os.path.join(self.output_dir,'summary_report.txt')
        with open(rp,'w',encoding='utf-8')as f:
            f.write("模糊地理定位系统 - 三模型对比总结报告\n")
            f.write("="*80 + "\n")
            f.write(f"主模型: {self.primary_model}\n")
            f.write(f"处理点位总数: {total_points}\n")
            f.write(f"平均置信度: {avg_primary_conf}\n")
            f.write(f"置信度范围: {conf_min} - {conf_max}\n")
            f.write(f"平均置信区域面积: {avg_area} 平方米\n")
            f.write(f"平均参照物数量: {avg_ref_count}\n")
            f.write(f"平均质心到边界距离: {avg_bd} 米\n")
            f.write("-"*80 + "\n")
            f.write("分模型整体性能汇总:\n")
            for m in models:
                f.write(f"  [{m:10s}] 平均置信度: {model_summary[m]['avg_conf']:4f} | 平均置信区域面积: {model_summary[m]['avg_area']:6.1f}㎡ | 置信度范围: {model_summary[m]['conf_range']}\n")
            f.write("-"*80 + "\n")
            f.write("\n各点位详细信息:\n")
            f.write(f"{'点位ID':<8s} {'模型':<10s} {'经度':<12s} {'纬度':<12s} {'置信度':<8s}\n")
            f.write("-"*60 + "\n")
            for r in results:
                for m in models:
                    d = r['model_results'][m]
                    f.write(f"{r['location_id']:<8s} {m:<10s} {d['coord'][0]:.6f} {d['coord'][1]:.6f} {d['conf']:.3f}\n")

        print(f"\n✅ 完整报告已保存至: {rp}")
        print(f"✅ 模型对比表已保存至: {os.path.join(self.data_dir,'model_comparison.csv')}")


# ====================================================
def main():
    # ==========================================
    output_dir = r''
    excel_file = r''
    primary_model = 'gaussian'  # 'linear' / 'gaussian' / 'exponential'
    # ========================================================

    locator = FuzzyGeoLocator(grid_resolution=80, output_dir=output_dir, primary_model=primary_model)
    data = locator.load_excel_data(excel_file)
    if not data: return []
    results = locator.process_all_points(data)
    if results:
        locator.generate_summary_report(results)
    return results


if __name__ == "__main__":
    main()
