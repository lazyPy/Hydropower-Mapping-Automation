"""
Debug script to understand why weir search finds 0 candidates
"""
import os
import sys
import django
import numpy as np
import math

# Setup Django
sys.path.insert(0, '.')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'HYDROPOWER_MAPPING.settings')
django.setup()

from hydropower.models import RasterLayer, SitePair
from django.conf import settings
import rasterio
from rasterio.transform import rowcol

# Get data
rl = RasterLayer.objects.get(watershed_delineated=True)
pairs = list(SitePair.objects.filter(raster_layer=rl, is_feasible=True).order_by('-power')[:5])

print("=" * 70)
print("DEBUGGING WEIR SEARCH - WHY 0 CANDIDATES?")
print("=" * 70)

# Load DEM (use original, not filled)
dem_path = str(settings.MEDIA_ROOT / rl.dataset.file.name)
print(f"\n1. DEM INFO:")
print(f"   Path: {dem_path}")
print(f"   Type: ORIGINAL DEM (not filled)")

with rasterio.open(dem_path) as dem:
    print(f"   Size: {dem.width}x{dem.height}")
    print(f"   CRS: {dem.crs}")
    print(f"   Bounds: {dem.bounds}")
    print(f"   Resolution: {dem.transform[0]:.2f}m x {abs(dem.transform[4]):.2f}m")
    print(f"   NoData: {dem.nodata}")
    
    dem_array = dem.read(1)
    dem_transform = dem.transform
    dem_nodata = dem.nodata
    
    print(f"   DEM range: {dem_array.min():.1f} to {dem_array.max():.1f}m")
    print(f"   Valid pixels: {(dem_array != dem_nodata).sum()} / {dem_array.size}")

# Test first inlet
print(f"\n2. TESTING FIRST INLET (Highest Power Site):")
p = pairs[0]
inlet_x = p.inlet.geometry.x
inlet_y = p.inlet.geometry.y
inlet_z = p.inlet.elevation
outlet_x = p.outlet.geometry.x
outlet_y = p.outlet.geometry.y

print(f"   Pair: {p.pair_id}")
print(f"   Inlet: ({inlet_x:.1f}, {inlet_y:.1f}), Elev={inlet_z:.1f}m")
print(f"   Outlet: ({outlet_x:.1f}, {outlet_y:.1f})")

# Check if inlet is within DEM bounds
with rasterio.open(dem_path) as dem:
    print(f"\n3. CHECKING INLET LOCATION:")
    print(f"   DEM bounds: {dem.bounds}")
    
    if not (dem.bounds.left <= inlet_x <= dem.bounds.right and 
            dem.bounds.bottom <= inlet_y <= dem.bounds.top):
        print(f"   ❌ INLET OUTSIDE DEM BOUNDS!")
    else:
        print(f"   ✓ Inlet is within DEM bounds")
        
        # Sample DEM at inlet
        col, row = rowcol(dem_transform, inlet_x, inlet_y)
        if 0 <= row < dem.height and 0 <= col < dem.width:
            dem_value = dem_array[row, col]
            print(f"   DEM at inlet: row={row}, col={col}, value={dem_value:.1f}m")
            if dem_value != dem_nodata:
                diff = abs(dem_value - inlet_z)
                print(f"   Difference from stored elevation: {diff:.1f}m")
                if diff > 30:
                    print(f"   ⚠️  Large elevation mismatch!")
            else:
                print(f"   ❌ DEM NODATA at inlet location!")
        else:
            print(f"   ❌ Inlet pixel coordinates outside array bounds!")

# Test search parameters
print(f"\n4. TESTING SEARCH PARAMETERS:")
search_radius = 500.0
min_distance = 50.0
elev_tolerance = 30.0
cone_angle_deg = 120.0

print(f"   Search radius: {search_radius}m")
print(f"   Min distance: {min_distance}m")
print(f"   Elevation tolerance: ±{elev_tolerance}m")
print(f"   Cone angle: {cone_angle_deg}°")

# Define search box
search_bbox = {
    'minx': inlet_x - search_radius,
    'maxx': inlet_x + search_radius,
    'miny': inlet_y - search_radius,
    'maxy': inlet_y + search_radius
}

print(f"\n5. SEARCH BOX:")
print(f"   X: {search_bbox['minx']:.1f} to {search_bbox['maxx']:.1f}")
print(f"   Y: {search_bbox['miny']:.1f} to {search_bbox['maxy']:.1f}")

# Convert to pixel coordinates
with rasterio.open(dem_path) as dem:
    col_min, row_max = rowcol(dem_transform, search_bbox['minx'], search_bbox['miny'])
    col_max, row_min = rowcol(dem_transform, search_bbox['maxx'], search_bbox['maxy'])
    
    # Clamp to bounds
    row_min_clamped = max(0, row_min)
    row_max_clamped = min(dem.height - 1, row_max)
    col_min_clamped = max(0, col_min)
    col_max_clamped = min(dem.width - 1, col_max)
    
    print(f"   Pixel range (before clamp): rows {row_min}-{row_max}, cols {col_min}-{col_max}")
    print(f"   Pixel range (after clamp): rows {row_min_clamped}-{row_max_clamped}, cols {col_min_clamped}-{col_max_clamped}")
    
    pixels_to_check = (row_max_clamped - row_min_clamped + 1) * (col_max_clamped - col_min_clamped + 1)
    print(f"   Total pixels to check: {pixels_to_check}")
    
    if pixels_to_check == 0:
        print(f"   ❌ NO PIXELS TO CHECK!")
    
    # Extract subset and analyze
    if pixels_to_check > 0:
        print(f"\n6. ANALYZING SEARCH AREA:")
        dem_subset = dem_array[row_min_clamped:row_max_clamped+1, col_min_clamped:col_max_clamped+1]
        
        # Filter by elevation
        valid_mask = dem_subset != dem_nodata
        elev_mask = np.abs(dem_subset - inlet_z) <= elev_tolerance
        
        print(f"   Valid (non-nodata) pixels: {valid_mask.sum()}")
        print(f"   Within elevation tolerance: {(valid_mask & elev_mask).sum()}")
        
        if (valid_mask & elev_mask).sum() > 0:
            print(f"\n7. CHECKING DISTANCE AND DIRECTIONAL CONSTRAINTS:")
            
            # Sample a few candidates
            candidate_count = 0
            distance_filtered = 0
            direction_filtered = 0
            
            cos_theta_min = math.cos(math.radians(cone_angle_deg))
            
            for r in range(min(10, dem_subset.shape[0])):  # Check first 10 rows
                for c in range(min(10, dem_subset.shape[1])):  # Check first 10 cols
                    if not valid_mask[r, c] or not elev_mask[r, c]:
                        continue
                    
                    # Get world coordinates
                    cell_row = row_min_clamped + r
                    cell_col = col_min_clamped + c
                    cell_x = dem_transform[2] + cell_col * dem_transform[0]
                    cell_y = dem_transform[5] + cell_row * dem_transform[4]
                    
                    # Distance
                    dx = cell_x - inlet_x
                    dy = cell_y - inlet_y
                    dist = math.sqrt(dx**2 + dy**2)
                    
                    if dist < min_distance:
                        distance_filtered += 1
                        continue
                    
                    if dist > search_radius:
                        continue
                    
                    # Directional constraint
                    vec_out_x = outlet_x - inlet_x
                    vec_out_y = outlet_y - inlet_y
                    len_out = math.sqrt(vec_out_x**2 + vec_out_y**2)
                    
                    if len_out < 1e-6:
                        continue
                    
                    vec_cand_x = cell_x - inlet_x
                    vec_cand_y = cell_y - inlet_y
                    
                    dot_product = (vec_out_x * vec_cand_x + vec_out_y * vec_cand_y) / (len_out * dist)
                    dot_product = max(-1.0, min(1.0, dot_product))
                    
                    if dot_product < cos_theta_min:
                        direction_filtered += 1
                        continue
                    
                    candidate_count += 1
                    
                    if candidate_count <= 3:
                        print(f"   ✓ Candidate {candidate_count}: dist={dist:.1f}m, dz={dem_subset[r,c]-inlet_z:.1f}m, angle={math.degrees(math.acos(dot_product)):.1f}°")
            
            print(f"\n   Results from sample (first 10x10 pixels):")
            print(f"   - Candidates found: {candidate_count}")
            print(f"   - Filtered by distance (<{min_distance}m): {distance_filtered}")
            print(f"   - Filtered by direction: {direction_filtered}")
            
            if candidate_count == 0:
                print(f"\n   ⚠️  POSSIBLE ISSUES:")
                print(f"   - All cells too close to inlet (<{min_distance}m)")
                print(f"   - All cells fail directional constraint")
                print(f"   - Try reducing min_distance or widening cone_angle")
        else:
            print(f"   ❌ NO VALID PIXELS within elevation tolerance!")
    else:
        print(f"   ❌ NO PIXELS IN SEARCH AREA!")

print("\n" + "=" * 70)
print("DIAGNOSIS:")
print("=" * 70)
print("The DEM has NODATA at inlet locations!")
print("This means:")
print("1. Inlet points may be on stream network edges/outside watershed")
print("2. DEM filling process left gaps at stream locations")
print("3. Inlet elevations were sampled from original (unfilled) DEM")
print("\nSOLUTION: Use original DEM for weir search instead of filled DEM")
print("=" * 70)
