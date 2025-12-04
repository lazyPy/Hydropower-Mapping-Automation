#!/usr/bin/env python
"""
Check for duplicate discharge values in site points.
"""
import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'HYDROPOWER_MAPPING.settings')
django.setup()

from hydropower.models import SitePoint, SitePair
from django.db.models import Count
import rasterio
from pathlib import Path
from django.conf import settings

print("=" * 80)
print("DISCHARGE DUPLICATE ANALYSIS")
print("=" * 80)

# SitePoint doesn't have discharge - it's in SitePair
print("\n[1] SITE POINTS - Basic Information")
print("-" * 80)

sites = SitePoint.objects.all().order_by('site_id')
print(f"Total site points: {sites.count()}")
print(f"  - Inlet sites: {sites.filter(site_type='INLET').count()}")
print(f"  - Outlet sites: {sites.filter(site_type='OUTLET').count()}\n")

# Check SitePair discharge values
print("\n[2] SITE PAIRS - Discharge Analysis (Detailed)")
print("-" * 80)

pairs = SitePair.objects.all().select_related('inlet', 'outlet', 'raster_layer').order_by('discharge')
print(f"Total site pairs: {pairs.count()}\n")

if pairs.count() > 0:
    # Get flow accumulation raster path
    raster = pairs.first().raster_layer
    flow_accum_path = Path(settings.MEDIA_ROOT) / raster.flow_accumulation_path if raster and raster.flow_accumulation_path else None
    
    print(f"{'Pair ID':<25} {'Discharge':>10} {'Head':>8} {'Power':>10} {'Inlet Elev':>11} {'Outlet Elev':>12}")
    print("-" * 100)
    
    # Store discharge values with their site info
    discharge_map = {}
    
    for pair in pairs:
        print(f"{pair.pair_id:<25} {pair.discharge:>8.3f} m³/s {pair.head:>6.2f} m {pair.power:>8.2f} kW "
              f"{pair.inlet.elevation:>9.2f} m {pair.outlet.elevation:>10.2f} m")
        
        # Group by discharge value
        if pair.discharge not in discharge_map:
            discharge_map[pair.discharge] = []
        discharge_map[pair.discharge].append(pair)
    
    # Now get flow accumulation for each site
    if flow_accum_path and flow_accum_path.exists():
        print("\n[3] FLOW ACCUMULATION ANALYSIS")
        print("-" * 100)
        
        with rasterio.open(flow_accum_path) as src:
            print(f"\nExtracting flow accumulation values from raster...")
            
            for pair in pairs:
                # Get inlet flow accumulation
                inlet_coords = [(pair.inlet.geometry.x, pair.inlet.geometry.y)]
                inlet_row_col = [src.index(x, y) for x, y in inlet_coords]
                inlet_flow_accum = src.read(1, window=((inlet_row_col[0][0], inlet_row_col[0][0]+1), 
                                                       (inlet_row_col[0][1], inlet_row_col[0][1]+1)))[0, 0]
                
                # Get outlet flow accumulation
                outlet_coords = [(pair.outlet.geometry.x, pair.outlet.geometry.y)]
                outlet_row_col = [src.index(x, y) for x, y in outlet_coords]
                outlet_flow_accum = src.read(1, window=((outlet_row_col[0][0], outlet_row_col[0][0]+1), 
                                                        (outlet_row_col[0][1], outlet_row_col[0][1]+1)))[0, 0]
                
                print(f"\n{pair.pair_id}:")
                print(f"  Discharge: {pair.discharge:.3f} m³/s")
                print(f"  Inlet  - FlowAccum: {inlet_flow_accum:>12.0f} cells")
                print(f"  Outlet - FlowAccum: {outlet_flow_accum:>12.0f} cells")
    
    # Find duplicates in site pairs
    print("\n\n[4] DUPLICATE DISCHARGE VALUES IN SITE PAIRS")
    print("-" * 100)
    
    pair_duplicates = SitePair.objects.values('discharge').annotate(
        count=Count('discharge')
    ).filter(count__gt=1).order_by('-count')
    
    if pair_duplicates.exists():
        print(f"\n!! Found {pair_duplicates.count()} discharge values that appear multiple times:\n")
        
        for dup in pair_duplicates:
            discharge_val = dup['discharge']
            count = dup['count']
            print(f"\n{'='*80}")
            print(f"Discharge {discharge_val:.3f} m³/s appears in {count} site pairs:")
            print(f"{'='*80}")
            
            pairs_with_val = SitePair.objects.filter(discharge=discharge_val).select_related('inlet', 'outlet')
            
            for i, p in enumerate(pairs_with_val, 1):
                print(f"\n  [{i}] {p.pair_id}")
                print(f"      Head: {p.head:>8.2f} m")
                print(f"      Power: {p.power:>10.2f} kW")
                print(f"      Inlet:  {p.inlet.site_id:<15} Elev: {p.inlet.elevation:>8.2f} m")
                print(f"      Outlet: {p.outlet.site_id:<15} Elev: {p.outlet.elevation:>8.2f} m")
    else:
        print("\n✓ No duplicate discharge values found in SitePair!")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
