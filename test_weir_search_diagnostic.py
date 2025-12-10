"""
Diagnostic Test: Weir Search with Relaxed Constraints

This tests the weir search with more permissive constraints to find candidates.
"""

import os
os.environ['GDAL_LIBRARY_PATH'] = r'D:\Desktop\Hydro HEC-HMS\env\Lib\site-packages\osgeo\gdal.dll'
os.environ['DJANGO_SETTINGS_MODULE'] = 'HYDROPOWER_MAPPING.settings'

import django
django.setup()

from hydropower.main_channel_weir_search import run_main_channel_weir_search, WeirSearchConfig
from hydropower.models import RasterLayer, SitePair, WeirCandidate

print("=" * 80)
print("DIAGNOSTIC WEIR SEARCH TEST (Relaxed Constraints)")
print("=" * 80)

# Get raster layer
raster_layer = RasterLayer.objects.first()
print(f"RasterLayer ID: {raster_layer.id}")

# DEM path
dem_path = f"media/preprocessed/dem_{raster_layer.id}/filled_dem.tif"
print(f"DEM Path: {dem_path}")
print()

# Relaxed configuration
config = WeirSearchConfig(
    search_radius_m=1000.0,      # Increased from 500
    min_distance_m=50.0,         # Decreased from 100
    elevation_tolerance_m=50.0,  # Increased from 20
    cone_angle_deg=180.0,        # Increased from 90 (full hemisphere)
    max_candidates_per_inlet=20  # Increased from 5
)

print("Relaxed Configuration:")
print(f"  Search Radius: {config.search_radius_m}m")
print(f"  Min Distance: {config.min_distance_m}m")
print(f"  Elevation Tolerance: ±{config.elevation_tolerance_m}m")
print(f"  Cone Angle: {config.cone_angle_deg}°")
print(f"  Max Candidates: {config.max_candidates_per_inlet}")
print()

# Check one inlet manually
pairs = SitePair.objects.filter(
    raster_layer=raster_layer,
    is_feasible=True,
    pair_id__startswith='HP_'
).order_by('rank')[:5]

print("Top 5 Pairs:")
for pair in pairs:
    print(f"  {pair.pair_id}: Inlet={pair.inlet.site_id} at ({pair.inlet.geometry.x:.1f}, {pair.inlet.geometry.y:.1f}, z={pair.inlet.elevation:.1f}m)")
print()

# Run weir search
print("Running weir search with relaxed constraints...")
print("-" * 80)

results = run_main_channel_weir_search(
    raster_layer_id=raster_layer.id,
    dem_path=dem_path,
    top_n_pairs=5,  # Just 5 pairs for quick test
    config=config,
    generate_infrastructure=False
)

print()
print("=" * 80)
print("DIAGNOSTIC RESULTS")
print("=" * 80)
print(f"Total Weir Candidates: {results['total_candidates']}")
print(f"Inlets Processed: {results['inlets_processed']}")
print(f"Best Weirs Identified: {len(results['best_weirs'])}")
print()

if results['best_weirs']:
    print("Best Weir Candidates:")
    print("-" * 80)
    for i, weir in enumerate(results['best_weirs'], 1):
        print(f"{i}. Inlet: {weir['inlet_site_id']:20s}")
        print(f"   Score: {weir['suitability_score']:5.1f}")
        print(f"   Distance: {weir['distance_from_inlet']:6.1f}m")
        print(f"   Elev Diff: {weir['elevation_difference']:+6.1f}m")
        print(f"   Position: ({weir['x']:.1f}, {weir['y']:.1f})")
        print()
else:
    print("⚠️  Still no candidates found!")
    print()
    print("Possible reasons:")
    print("1. DEM nodata values at search locations")
    print("2. All pixels fail elevation constraint")
    print("3. Search area outside DEM bounds")
    print()
    
    # Check DEM at first inlet
    import rasterio
    from rasterio.transform import rowcol
    
    first_pair = pairs[0]
    inlet_x = first_pair.inlet.geometry.x
    inlet_y = first_pair.inlet.geometry.y
    inlet_z = first_pair.inlet.elevation
    
    print(f"Checking DEM at inlet {first_pair.inlet.site_id}:")
    print(f"  Coordinates: ({inlet_x:.1f}, {inlet_y:.1f})")
    print(f"  Inlet elevation: {inlet_z:.1f}m")
    
    with rasterio.open(dem_path) as dem:
        row, col = rowcol(dem.transform, inlet_x, inlet_y)
        print(f"  DEM pixel: row={row}, col={col}")
        print(f"  DEM bounds: {dem.bounds}")
        
        if 0 <= row < dem.height and 0 <= col < dem.width:
            dem_value = float(dem.read(1)[row, col])
            print(f"  DEM value at inlet: {dem_value:.1f}m")
            print(f"  Match: {'✓' if abs(dem_value - inlet_z) < 5 else '✗'}")
            
            # Sample nearby area
            print(f"\n  Sampling 3x3 area around inlet:")
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    r, c = row + dr, col + dc
                    if 0 <= r < dem.height and 0 <= c < dem.width:
                        val = float(dem.read(1)[r, c])
                        print(f"    ({r},{c}): {val:.1f}m", end="")
                        if val > -9999:
                            print(f" [valid]")
                        else:
                            print(f" [nodata]")
        else:
            print(f"  ✗ Inlet is OUTSIDE DEM bounds!")

print()
print("=" * 80)
