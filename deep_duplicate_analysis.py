import os
import sys
import django
import rasterio
from pathlib import Path
import numpy as np

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'HYDROPOWER_MAPPING.settings')
django.setup()

from hydropower.models import SitePair
from django.conf import settings

print("\n" + "=" * 100)
print("DEEP HYDROLOGICAL ANALYSIS OF DUPLICATE DISCHARGE VALUES")
print("=" * 100)

pairs = SitePair.objects.all().select_related('inlet', 'outlet', 'raster_layer').order_by('discharge')

# Get raster paths
raster = pairs.first().raster_layer
flow_accum_path = Path(settings.MEDIA_ROOT) / raster.flow_accumulation_path
flow_dir_path = Path(settings.MEDIA_ROOT) / raster.flow_direction_path
filled_dem_path = Path(settings.MEDIA_ROOT) / raster.filled_dem_path

print(f"\nAnalyzing {pairs.count()} site pairs")
print(f"Flow Accumulation: {flow_accum_path.name}")
print(f"Q_outlet: {raster.discharge_q_outlet} m³/s")

# Open all rasters
with rasterio.open(flow_accum_path) as flow_accum_src, \
     rasterio.open(flow_dir_path) as flow_dir_src, \
     rasterio.open(filled_dem_path) as dem_src:
    
    # Get cell size
    cell_width = abs(flow_accum_src.transform[0])
    cell_height = abs(flow_accum_src.transform[4])
    cell_area_m2 = cell_width * cell_height
    
    print(f"Cell size: {cell_width}m × {cell_height}m = {cell_area_m2:.2f} m²")
    
    # Group by discharge value
    discharge_groups = {}
    
    for pair in pairs:
        # Get coordinates
        inlet_x, inlet_y = pair.inlet.geometry.x, pair.inlet.geometry.y
        outlet_x, outlet_y = pair.outlet.geometry.x, pair.outlet.geometry.y
        
        # Get raster values at inlet
        inlet_row, inlet_col = flow_accum_src.index(inlet_x, inlet_y)
        inlet_flow_accum = flow_accum_src.read(1, window=((inlet_row, inlet_row+1), (inlet_col, inlet_col+1)))[0, 0]
        inlet_flow_dir = flow_dir_src.read(1, window=((inlet_row, inlet_row+1), (inlet_col, inlet_col+1)))[0, 0]
        inlet_elev = dem_src.read(1, window=((inlet_row, inlet_row+1), (inlet_col, inlet_col+1)))[0, 0]
        
        # Get raster values at outlet
        outlet_row, outlet_col = flow_accum_src.index(outlet_x, outlet_y)
        outlet_flow_accum = flow_accum_src.read(1, window=((outlet_row, outlet_row+1), (outlet_col, outlet_col+1)))[0, 0]
        outlet_flow_dir = flow_dir_src.read(1, window=((outlet_row, outlet_row+1), (outlet_col, outlet_col+1)))[0, 0]
        outlet_elev = dem_src.read(1, window=((outlet_row, outlet_row+1), (outlet_col, outlet_col+1)))[0, 0]
        
        # Calculate distances
        distance_cells = np.sqrt((outlet_row - inlet_row)**2 + (outlet_col - inlet_col)**2)
        distance_m = np.sqrt((outlet_x - inlet_x)**2 + (outlet_y - inlet_y)**2)
        
        # Calculate expected discharge from formula
        max_flow_accum = flow_accum_src.read(1).max()
        max_drainage_area_m2 = max_flow_accum * cell_area_m2
        outlet_drainage_area_m2 = outlet_flow_accum * cell_area_m2
        
        q_calculated = raster.discharge_q_outlet * (outlet_drainage_area_m2 / max_drainage_area_m2)
        
        discharge_key = round(pair.discharge, 3)
        
        if discharge_key not in discharge_groups:
            discharge_groups[discharge_key] = []
        
        discharge_groups[discharge_key].append({
            'pair_id': pair.pair_id,
            'inlet_id': pair.inlet.site_id,
            'outlet_id': pair.outlet.site_id,
            'inlet_elev': inlet_elev,
            'outlet_elev': outlet_elev,
            'head': pair.head,
            'inlet_flow_accum': inlet_flow_accum,
            'outlet_flow_accum': outlet_flow_accum,
            'inlet_flow_dir': inlet_flow_dir,
            'outlet_flow_dir': outlet_flow_dir,
            'distance_m': distance_m,
            'distance_cells': distance_cells,
            'outlet_drainage_km2': outlet_drainage_area_m2 / 1e6,
            'q_calculated': q_calculated,
            'q_stored': pair.discharge,
            'q_match': abs(q_calculated - pair.discharge) < 0.001,
            'power_kw': pair.power,
            'inlet_row': inlet_row,
            'inlet_col': inlet_col,
            'outlet_row': outlet_row,
            'outlet_col': outlet_col
        })

print("\n" + "=" * 100)
print("DETAILED ANALYSIS OF DUPLICATE GROUPS")
print("=" * 100)

duplicate_count = 0
for discharge, sites in sorted(discharge_groups.items()):
    if len(sites) > 1:
        duplicate_count += 1
        print(f"\n{'=' * 100}")
        print(f"GROUP {duplicate_count}: Discharge = {discharge} m³/s ({len(sites)} site pairs)")
        print(f"{'=' * 100}")
        
        # Check if all have same outlet flow accumulation
        outlet_accums = [s['outlet_flow_accum'] for s in sites]
        all_same_accum = len(set(outlet_accums)) == 1
        
        if all_same_accum:
            print(f"✓ EXPECTED: All outlets have SAME flow accumulation ({outlet_accums[0]:.0f} cells)")
            print(f"  → Same drainage area = same discharge = HYDROLOGICALLY CORRECT")
        else:
            print(f"⚠ SUSPICIOUS: Outlets have DIFFERENT flow accumulation: {set(outlet_accums)}")
            print(f"  → Different drainage areas should NOT have same discharge!")
        
        print(f"\nDetailed breakdown:")
        print(f"{'Site Pair':<25} {'Outlet Accum':<15} {'Drainage (km²)':<15} {'Calc Q':<12} {'Match':<8} {'Elev Drop':<12} {'Distance'}")
        print("-" * 100)
        
        for site in sites:
            match_symbol = "✓" if site['q_match'] else "✗"
            print(f"{site['pair_id']:<25} {site['outlet_flow_accum']:>8.0f} cells   "
                  f"{site['outlet_drainage_km2']:>10.6f} km²  "
                  f"{site['q_calculated']:>8.3f} m³/s  {match_symbol:<8} "
                  f"{site['head']:>8.2f} m     {site['distance_m']:>7.1f} m")
        
        # Check spatial clustering
        print(f"\nSpatial Analysis:")
        print(f"{'Site Pair':<25} {'Inlet (row,col)':<20} {'Outlet (row,col)':<20} {'Flow Dir (I→O)'}")
        print("-" * 100)
        
        for site in sites:
            print(f"{site['pair_id']:<25} ({site['inlet_row']:>4},{site['inlet_col']:>4})          "
                  f"({site['outlet_row']:>4},{site['outlet_col']:>4})           "
                  f"I:{site['inlet_flow_dir']:>3.0f}° O:{site['outlet_flow_dir']:>3.0f}°")
        
        # Check if outlets are at same location
        outlet_positions = [(s['outlet_row'], s['outlet_col']) for s in sites]
        unique_positions = set(outlet_positions)
        
        if len(unique_positions) == 1:
            print(f"\n⚠ CRITICAL ISSUE: All outlets are at SAME RASTER CELL!")
            print(f"   This suggests potential site pairing error - multiple inlets paired to same outlet")
        elif len(unique_positions) < len(sites):
            print(f"\n⚠ ISSUE: Some outlets share raster cells ({len(unique_positions)} unique positions for {len(sites)} outlets)")
        else:
            print(f"\n✓ Outlets are at different raster cells")

print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)

total_duplicates = sum(1 for sites in discharge_groups.values() if len(sites) > 1)
total_in_duplicate_groups = sum(len(sites) for sites in discharge_groups.values() if len(sites) > 1)

print(f"Total discharge values with duplicates: {total_duplicates}")
print(f"Total site pairs in duplicate groups: {total_in_duplicate_groups}")

# Check for calculation errors
calculation_mismatches = []
for discharge, sites in discharge_groups.items():
    for site in sites:
        if not site['q_match']:
            calculation_mismatches.append(site)

if calculation_mismatches:
    print(f"\n⚠ WARNING: {len(calculation_mismatches)} site pairs have discharge calculation mismatches!")
    print("  This suggests the stored discharge doesn't match the current flow accumulation raster.")
else:
    print(f"\n✓ All stored discharge values match calculated values from flow accumulation")

# Check for spatial issues
same_outlet_groups = 0
for discharge, sites in discharge_groups.items():
    if len(sites) > 1:
        outlet_positions = [(s['outlet_row'], s['outlet_col']) for s in sites]
        if len(set(outlet_positions)) < len(sites):
            same_outlet_groups += 1

if same_outlet_groups > 0:
    print(f"\n⚠ CRITICAL: {same_outlet_groups} duplicate groups have outlets sharing raster cells!")
    print("  → This is likely a site pairing algorithm issue")
    print("  → Multiple inlets should NOT be paired to the same outlet location")
else:
    print(f"\n✓ No duplicate groups have outlets at the same raster cell")
    print("  → Duplicates are due to different outlets having same drainage area")
    print("  → This is HYDROLOGICALLY NORMAL for watersheds with similar sub-catchments")

print("=" * 100)
