import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter
from scipy.spatial import cKDTree
from qgis.core import (QgsProject, QgsVectorLayer, QgsFeature, QgsGeometry, 
                       QgsPointXY, QgsField, QgsRasterDataProvider)
from PyQt5.QtCore import QVariant

# --- 1. Read User Inputs ---
# (Assume you have pulled these from your UI line edits)
base_thresh = 2.5 
p2p_thresh = 5.0
max_dipole_dist_px = 10.0
complex_radius_px = 20.0

# --- 2. Load Raster Data ---
layer = self.dlg.mMapLayerComboBox.currentLayer() # From QGIS combobox
provider = layer.dataProvider()
extent = provider.extent()
cols = provider.xSize()
rows = provider.ySize()
pixel_size_x = layer.rasterUnitsPerPixelX() # Should be 0.25
pixel_size_y = layer.rasterUnitsPerPixelY()

# Read raster band into numpy array
block = provider.block(1, extent, cols, rows)
grid = np.zeros((rows, cols))
for i in range(rows):
    for j in range(cols):
        grid[i, j] = block.value(i, j)

# Handle NoData values (replace with 0 or mean)
nodata = provider.sourceNoDataValue(1)
grid[grid == nodata] = 0 

# --- 3. Detect Extrema ---
# Use a 3x3 window to find local peaks and troughs
local_max = maximum_filter(grid, size=3) == grid
local_min = minimum_filter(grid, size=3) == grid

# Apply base amplitude threshold
pos_peaks_mask = (local_max) & (grid > base_thresh)
neg_peaks_mask = (local_min) & (grid < -base_thresh)

# Get pixel coordinates of peaks (Y=row, X=col)
pos_coords = np.argwhere(pos_peaks_mask)
neg_coords = np.argwhere(neg_peaks_mask)
all_peaks = np.vstack((pos_coords, neg_coords))

# --- 4. Spatial Clustering & Classification ---
# Use KDTree for fast spatial searching
tree = cKDTree(all_peaks)
targets = [] # List to hold target dictionaries
processed_indices = set()

for i, peak in enumerate(all_peaks):
    if i in processed_indices:
        continue
        
    # Find all peaks within the Complex Cluster Radius (20 wavelengths)
    neighbors = tree.query_ball_point(peak, r=complex_radius_px)
    
    # COMPLEX: More than 2 peaks clustered
    if len(neighbors) > 2:
        group_coords = all_peaks[neighbors]
        # Center of mass for the complex cluster
        center_y, center_x = np.mean(group_coords, axis=0)
        
        # Find absolute max and min in this cluster
        vals = [grid[c[0], c[1]] for c in group_coords]
        max_val = max(vals)
        min_val = min(vals)
        p2p = abs(max_val - min_val)
        
        targets.append({
            'desc': 'Complex', 'x_px': center_x, 'y_px': center_y,
            'pos_val': max_val, 'neg_val': min_val, 'p2p': p2p,
            'dist': 0.0, 'wl': 20.0
        })
        processed_indices.update(neighbors)
        
    # MONOPOLE or DIPOLE (1 or 2 peaks)
    else:
        is_pos = grid[peak[0], peak[1]] > 0
        search_radius = max_dipole_dist_px
        
        # Look for the opposite pole
        opposite_tree = cKDTree(neg_coords) if is_pos else cKDTree(pos_coords)
        nearest_opp_dist, nearest_opp_idx = opposite_tree.query(peak, k=1)
        
        if nearest_opp_dist <= search_radius:
            opp_peak = neg_coords[nearest_opp_idx] if is_pos else pos_coords[nearest_opp_idx]
            val1 = grid[peak[0], peak[1]]
            val2 = grid[opp_peak[0], opp_peak[1]]
            p2p = abs(val1 - val2)
            
            # DIPOLE: Passes 5nT threshold
            if p2p >= p2p_thresh:
                center_y = (peak[0] + opp_peak[0]) / 2.0
                center_x = (peak[1] + opp_peak[1]) / 2.0
                
                targets.append({
                    'desc': 'Dipole', 'x_px': center_x, 'y_px': center_y,
                    'pos_val': val1 if is_pos else val2, 
                    'neg_val': val2 if is_pos else val1,
                    'p2p': p2p, 'dist': nearest_opp_dist, 'wl': nearest_opp_dist
                })
                # Add opposite peak to processed indices
                # (Requires index mapping logic in full script to prevent double-counting)
        
        else:
            # MONOPOLE
            val = grid[peak[0], peak[1]]
            targets.append({
                'desc': 'Monopole', 'x_px': peak[1], 'y_px': peak[0],
                'pos_val': val if is_pos else 0, 
                'neg_val': 0 if is_pos else val,
                'p2p': abs(val), 'dist': 0.0, 'wl': 1.0
            })
        
        processed_indices.add(i)

# --- 5. Generate Shapefile Output ---
vector_layer = QgsVectorLayer("Point?crs=" + layer.crs().authid(), "Magnetic_Targets", "memory")
pr = vector_layer.dataProvider()

# Define Attributes
pr.addAttributes([
    QgsField("Target_ID", QVariant.Int),
    QgsField("X_Coord", QVariant.Double),
    QgsField("Y_Coord", QVariant.Double),
    QgsField("Peak_Pos", QVariant.Double),
    QgsField("Peak_Neg", QVariant.Double),
    QgsField("P2P_Amp", QVariant.Double),
    QgsField("Desc", QVariant.String),
    QgsField("P2P_Dist", QVariant.Double),
    QgsField("Wavelength", QVariant.Double)
])
vector_layer.updateFields()

# Write Features
features = []
for idx, t in enumerate(targets):
    feat = QgsFeature()
    
    # Convert pixel coords back to real-world coordinates
    real_x = extent.xMinimum() + (t['x_px'] * pixel_size_x) + (pixel_size_x / 2)
    real_y = extent.yMaximum() - (t['y_px'] * pixel_size_y) - (pixel_size_y / 2)
    
    feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(real_x, real_y)))
    feat.setAttributes([
        idx + 1, real_x, real_y, t['pos_val'], t['neg_val'], 
        t['p2p'], t['desc'], t['dist'], t['wl']
    ])
    features.append(feat)

pr.addFeatures(features)
vector_layer.updateExtents()
QgsProject.instance().addMapLayer(vector_layer)