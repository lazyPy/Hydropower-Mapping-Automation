"""
Test script for weir candidate search
"""
import os
import sys
import django

# Setup Django
sys.path.insert(0, '.')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'HYDROPOWER_MAPPING.settings')
django.setup()

from hydropower.models import RasterLayer, SitePair, WeirCandidate
from hydropower.weir_search import WeirSearch, WeirSearchConfig
from django.conf import settings

# Get raster layer
rl = RasterLayer.objects.get(watershed_delineated=True)
print(f'Testing weir search on RasterLayer {rl.id}')

# Get top 50 pairs
pairs = list(SitePair.objects.filter(raster_layer=rl, is_feasible=True).order_by('-power')[:50])
print(f'Found {len(pairs)} top pairs\n')

# Show top 5 pairs
print('Top 5 pairs:')
for p in pairs[:5]:
    print(f'  {p.pair_id}:')
    print(f'    Inlet: ({p.inlet.geometry.x:.1f}, {p.inlet.geometry.y:.1f}), Elev={p.inlet.elevation:.1f}m')
    print(f'    Outlet: ({p.outlet.geometry.x:.1f}, {p.outlet.geometry.y:.1f}), Elev={p.outlet.elevation:.1f}m')
    print(f'    Head={p.head:.1f}m, Power={p.power:.1f}kW')

# Convert to pairs list
pairs_list = [{
    'pair_id': p.pair_id,
    'inlet_node_id': p.inlet.site_id,
    'outlet_node_id': p.outlet.site_id,
    'inlet_x': p.inlet.geometry.x,
    'inlet_y': p.inlet.geometry.y,
    'inlet_elevation': p.inlet.elevation,
    'outlet_x': p.outlet.geometry.x,
    'outlet_y': p.outlet.geometry.y,
    'power_kw': p.power
} for p in pairs]

# Test weir search with relaxed constraints
print('\nTesting with relaxed constraints...')
config = WeirSearchConfig(
    search_radius_m=500.0,
    min_distance_m=50.0,  # Reduced from 100m
    elevation_tolerance_m=30.0,  # Increased from 20m
    max_candidates_per_inlet=10,
    cone_angle_deg=120.0,  # Increased from 90° (wider cone)
    top_n_pairs=50
)

ws = WeirSearch(config)
ws.load_dem(str(settings.MEDIA_ROOT / rl.filled_dem_path))

inlets = ws.extract_top_inlets(pairs_list)
print(f'Extracted {len(inlets)} unique inlets')

candidates = ws.search_weir_candidates(inlets)
print(f'\nFound {len(candidates)} weir candidates')

if len(candidates) > 0:
    print('\nTop 5 candidates:')
    for c in candidates[:5]:
        print(f'  {c["candidate_id"]}:')
        print(f'    Weir: ({c["weir_x"]:.1f}, {c["weir_y"]:.1f}), Elev={c["weir_z"]:.1f}m')
        print(f'    Distance={c["distance_from_inlet"]:.1f}m, dZ={c["elevation_difference"]:.1f}m')
        print(f'    Rank={c["rank_within_inlet"]}, Score={c.get("suitability_score", 0):.1f}')
    
    # Save to database
    print('\nSaving to database...')
    ws.save_to_postgis(inlets, candidates, rl.id)
    
    # Verify
    saved_count = WeirCandidate.objects.filter(raster_layer=rl).count()
    print(f'Verified: {saved_count} weir candidates in database')
    print('SUCCESS!')
else:
    print('No candidates found - may need to adjust search parameters or check DEM data')
