"""
Quick Test: Weir Search with Top 10 Pairs (Small Scale Test)

This tests the weir search functionality with a small subset to verify it works.
"""

import os
os.environ['GDAL_LIBRARY_PATH'] = r'D:\Desktop\Hydro HEC-HMS\env\Lib\site-packages\osgeo\gdal.dll'
os.environ['DJANGO_SETTINGS_MODULE'] = 'HYDROPOWER_MAPPING.settings'

import django
django.setup()

from hydropower.main_channel_weir_search import run_main_channel_weir_search, WeirSearchConfig
from hydropower.models import RasterLayer, SitePair, WeirCandidate

print("=" * 80)
print("WEIR SEARCH TEST (Top 10 Pairs)")
print("=" * 80)

# Get raster layer
raster_layer = RasterLayer.objects.first()
print(f"RasterLayer ID: {raster_layer.id}")
print(f"Dataset: {raster_layer.dataset.name}")

# DEM path
dem_path = f"media/preprocessed/dem_{raster_layer.id}/filled_dem.tif"
print(f"DEM Path: {dem_path}")
print(f"DEM Exists: {os.path.exists(dem_path)}")
print()

# Check main channel pairs
pair_count = SitePair.objects.filter(
    raster_layer=raster_layer,
    is_feasible=True,
    pair_id__startswith='HP_'
).count()
print(f"Main Channel Pairs Available: {pair_count}")
print()

# Run weir search with top 10 pairs (small test)
print("Running weir search with top 10 pairs...")
print("-" * 80)

config = WeirSearchConfig(
    search_radius_m=500.0,
    min_distance_m=100.0,
    elevation_tolerance_m=20.0,
    cone_angle_deg=90.0,
    max_candidates_per_inlet=5  # Reduced for test
)

results = run_main_channel_weir_search(
    raster_layer_id=raster_layer.id,
    dem_path=dem_path,
    top_n_pairs=10,  # Small test with top 10
    config=config,
    generate_infrastructure=False  # Skip infrastructure for speed
)

print()
print("=" * 80)
print("TEST RESULTS")
print("=" * 80)
print(f"Total Weir Candidates: {results['total_candidates']}")
print(f"Inlets Processed: {results['inlets_processed']}")
print(f"Best Weirs Identified: {len(results['best_weirs'])}")
print()

if results['best_weirs']:
    print("Best Weir Candidates:")
    print("-" * 80)
    for i, weir in enumerate(results['best_weirs'][:5], 1):
        print(f"{i}. Inlet: {weir['inlet_site_id']:15s} | "
              f"Score: {weir['suitability_score']:5.1f} | "
              f"Distance: {weir['distance_from_inlet']:6.1f}m | "
              f"Elev Diff: {weir['elevation_difference']:+6.1f}m")

print()
print("Database Check:")
print("-" * 80)
total_weirs = WeirCandidate.objects.filter(raster_layer=raster_layer).count()
best_weirs_db = WeirCandidate.objects.filter(raster_layer=raster_layer, rank_within_inlet=1).count()
print(f"Total Weir Candidates in DB: {total_weirs}")
print(f"Best Weirs (Rank 1): {best_weirs_db}")
print()
print("=" * 80)
print("✓ TEST COMPLETED SUCCESSFULLY")
print("=" * 80)
