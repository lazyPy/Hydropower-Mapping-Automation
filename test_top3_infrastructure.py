"""
Test: Top 3 Best Weir Infrastructure Generation

This tests the new top 3 weir infrastructure generation feature.
"""

import os
os.environ['GDAL_LIBRARY_PATH'] = r'D:\Desktop\Hydro HEC-HMS\env\Lib\site-packages\osgeo\gdal.dll'
os.environ['DJANGO_SETTINGS_MODULE'] = 'HYDROPOWER_MAPPING.settings'

import django
django.setup()

from hydropower.main_channel_weir_search import run_main_channel_weir_search
from hydropower.models import RasterLayer, SitePair, WeirCandidate
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

print("=" * 80)
print("TOP 3 WEIR INFRASTRUCTURE GENERATION TEST")
print("=" * 80)

# Get raster layer
raster_layer = RasterLayer.objects.first()
print(f"RasterLayer ID: {raster_layer.id}")
print(f"Dataset: {raster_layer.dataset.name}")
print()

# DEM path
dem_path = f"media/preprocessed/dem_{raster_layer.id}/filled_dem.tif"

# Check main channel pairs
pair_count = SitePair.objects.filter(
    raster_layer=raster_layer,
    is_feasible=True,
    pair_id__startswith='HP_'
).count()
print(f"Main Channel Pairs Available: {pair_count}")
print()

print("Running weir search on top 50 pairs with TOP 3 infrastructure generation...")
print("-" * 80)

# Run weir search with infrastructure for top 3
results = run_main_channel_weir_search(
    raster_layer_id=raster_layer.id,
    dem_path=dem_path,
    top_n_pairs=50,
    generate_infrastructure=True  # Enable infrastructure generation
)

print()
print("=" * 80)
print("RESULTS")
print("=" * 80)
print(f"Total Weir Candidates: {results['total_candidates']}")
print(f"Inlets Processed: {results['inlets_processed']}")
print(f"Best Weirs Identified: {len(results['best_weirs'])}")
print(f"Infrastructure Layouts Generated: {results['infrastructure_generated']}")
print()

if results['best_weirs']:
    # Sort by score to show top 3
    sorted_weirs = sorted(
        results['best_weirs'],
        key=lambda w: (w.get('suitability_score', 0), -w.get('distance_from_inlet', 999999)),
        reverse=True
    )
    
    print("TOP 3 Best Weir Locations (with Infrastructure):")
    print("-" * 80)
    for i, weir in enumerate(sorted_weirs[:3], 1):
        print(f"🏆 RANK {i}: {weir['inlet_site_id']}")
        print(f"   Score: {weir['suitability_score']:.2f}")
        print(f"   Distance from Inlet: {weir['distance_from_inlet']:.1f}m")
        print(f"   Elevation Difference: {weir['elevation_difference']:+.2f}m")
        print(f"   Coordinates: ({weir['x']:.2f}, {weir['y']:.2f})")
        
        # Check infrastructure in database
        site_pair = SitePair.objects.filter(
            raster_layer=raster_layer,
            inlet_id=weir['inlet_id'],
            pair_id__startswith='HP_'
        ).order_by('rank').first()
        
        if site_pair and site_pair.intake_basin_geom:
            print(f"   ✅ Infrastructure Generated: {site_pair.pair_id}")
            print(f"      - Intake Basin: {site_pair.intake_basin_geom}")
            print(f"      - Channel Length: {site_pair.channel_length:.0f}m" if site_pair.channel_length else "      - Channel: N/A")
            print(f"      - Penstock Length: {site_pair.penstock_length:.0f}m" if site_pair.penstock_length else "      - Penstock: N/A")
        else:
            print(f"   ⚠️  Infrastructure Not Found")
        print()
    
    if len(sorted_weirs) > 3:
        print(f"Other Best Weirs (Rank 4-{len(sorted_weirs)}):")
        print("-" * 80)
        for i, weir in enumerate(sorted_weirs[3:], 4):
            print(f"  {i}. {weir['inlet_site_id']} - "
                  f"Score: {weir['suitability_score']:.1f}, "
                  f"Distance: {weir['distance_from_inlet']:.0f}m "
                  f"(No infrastructure generated)")

print()
print("=" * 80)
print("Database Verification:")
print("-" * 80)

# Count infrastructure
infra_count = SitePair.objects.filter(
    raster_layer=raster_layer,
    pair_id__startswith='HP_',
    intake_basin_geom__isnull=False
).count()

print(f"Site Pairs with Infrastructure: {infra_count}")
print()

# Show infrastructure details
if infra_count > 0:
    print("Infrastructure Components:")
    infra_pairs = SitePair.objects.filter(
        raster_layer=raster_layer,
        pair_id__startswith='HP_',
        intake_basin_geom__isnull=False
    ).order_by('rank')[:5]
    
    for pair in infra_pairs:
        print(f"  • {pair.pair_id} (Rank {pair.rank})")
        print(f"    Intake: {pair.intake_basin_geom}")
        if pair.channel_length:
            print(f"    Channel: {pair.channel_length:.0f}m")
        if pair.penstock_length:
            print(f"    Penstock: {pair.penstock_length:.0f}m")

print()
print("=" * 80)
print("✓ TEST COMPLETED SUCCESSFULLY")
print("=" * 80)
