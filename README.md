# Fuzzy Geographic Locator with Multi-Model Fusion

This repository implements a **fuzzy logic‑based geographic localization system** that fuses multiple spatial references (polygons with direction and distance) to estimate a target location. The system supports three membership models (linear, Gaussian, exponential), various fusion strategies, and provides both quantitative evaluation and SCI‑style visualizations.

## ✨ Key Features

- **Fuzzy membership functions** – Linear, Gaussian, and exponential decay for directional and distance uncertainty.
- **Multi‑reference fusion** – Product, min, and average fusion of fuzzy distributions.
- **Defuzzification** – Centroid method to obtain final coordinates.
- **Ablation studies** – Disable fuzzy modeling or multi‑reference fusion to evaluate contributions.
- **Geometric alternative** – Least‑squares localization using ideal points computed from boundaries, bearings, and distances.
- **Robust estimation** – Huber loss and Fermat point methods to handle outlier references.
- **High‑quality visualization** – 2D contour plots, 3D surfaces, and statistical summary figures following SCI publication standards.
- **Evaluation metrics** – Average localization error (meters) and recall within a user‑defined threshold (e.g., 50 m).

## 📂 Repository Structure

| File | Description |
|------|-------------|
| `fuzzy_geolocator.py` | Core fuzzy geolocator with linear membership (original version). |
| `gaussian.py` | Enhanced version with Gaussian/exponential models and primary model selection. |
| `evaluate_fuzzy_geolocator2.py` | Evaluation script: runs original model and ablations, computes errors against ground truth. |
| `LLM+jihejisuan.py` | Geometric least‑squares localization using boundary points + bearing/distance (no fuzzy logic). |
| `main3.py` | Robust localization: removes outliers (Z‑score), then applies Huber and Fermat point estimators. |
| `visualization2.py` | SCI‑style visualizer that reads results and generates separate 2D, 3D, and statistics plots. |
| `visualize_gaussian.py` | Standalone version of the visualizer that **embeds Gaussian fuzzy processing** – no external dependency on `fuzzy_geolocator`. |

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/fuzzy-geolocator.git
cd fuzzy-geolocator
```

### 2. Install dependencies

All scripts require Python 3.8+ and the following packages:

```bash
pip install numpy pandas matplotlib shapely scipy openpyxl
```

### 3. Prepare input data

The system expects an Excel file with the following columns (names adjustable in code):

- `点ID` – Point identifier (grouping rows for the same location)
- `WKT` – Well‑Known Text of a polygon (reference building/region)
- `角度` – Bearing from the boundary point toward the target (degrees)
- `距离` – Distance from the boundary point to the target (meters)

A typical row group for one point:

| 点ID | WKT | 角度 | 距离 |
|------|-----|------|------|
| 1 | POLYGON((116.40 39.90, ...)) | 45 | 30 |
| 1 | POLYGON((116.41 39.91, ...)) | 120 | 55 |

Ground truth coordinates (for evaluation) should be in a separate Excel file, e.g., `实验记录-0.xlsx`, with a column containing `Point (lon lat)` strings.

### 4. Run the fuzzy geolocator (Gaussian version)

```bash
python gaussian.py
```

Adjust paths and the `primary_model` parameter inside `main()`:

```python
primary_model = 'gaussian'   # 'linear', 'gaussian', or 'exponential'
```

Outputs:
- `results/` – Images (2D, 3D, statistics) and data (JSON, CSV, Excel).
- Console summary of model comparisons.

### 5. Run evaluation with ground truth

```bash
python evaluate_fuzzy_geolocator2.py
```

This will compute:
- Original model
- Ablation: no fuzzy modeling (Δθ = Δd = 0)
- Ablation: no multi‑reference fusion (average of individual defuzzifications)

Metrics: average error (m) and recall @ 50 m.

### 6. Run geometric least‑squares localization

```bash
python LLM+jihejisuan.py
```

Outputs an Excel file with `POINT(lon lat)` WKT for each point.

### 7. Run robust estimation (Huber + Fermat)

```bash
python main3.py
```

Produces three result files: least squares, Huber robust, and Fermat point.

## 📊 Example Outputs

### 2D Fuzzy Distribution

![](example/Point1_2d_plot.png)

*Contour plot of fused membership with reference polygons, boundary points, centroids, direction arrows, final position (green star), and 50 % confidence region (dashed rectangle).*

### 3D Distribution Surface

![](example/Point1_3d_plot.png)

*Surface plot of the fuzzy distribution, with contour projection on the bottom plane.*

### Evaluation Metrics (Console)

```
原模型 - 平均定位误差: 12.34 米, 误差<50米召回率: 92.00%
无模糊建模 - 平均定位误差: 28.76 米, 误差<50米召回率: 68.00%
无多约束融合 - 平均定位误差: 19.45 米, 误差<50米召回率: 84.00%
```

## 🧪 Customization

- **Grid resolution** – Change `grid_resolution` in the locator initializer (default 80–100).
- **Fuzziness levels** – Modify `delta_theta` (degrees) and `delta_distance` (relative) for high/medium/low uncertainty.
- **Fusion method** – Switch between `'product'`, `'min'`, or `'average'` in `fuse_distributions()`.
- **Confidence region** – Change the threshold level (e.g., 0.5 → 0.8) inside `calculate_confidence_region()`.

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@software{fuzzy_geolocator,
  author = {Your Name},
  title = {Fuzzy Geographic Locator with Multi-Model Fusion},
  year = {2025},
  url = {https://github.com/yourusername/fuzzy-geolocator}
}
```

## 📬 Contact

For questions or suggestions, please open an issue or contact the maintainer.

---

**License:** MIT
