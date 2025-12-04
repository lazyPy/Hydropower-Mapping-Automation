#!/usr/bin/env python
"""
Debug discharge calculation to understand why many sites have same value.
"""

import os
import sys
from pathlib import Path

# Set up Django environment
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'HYDROPOWER_MAPPING.settings')

import django
django.setup()

import numpy as np
from hydropower.models import RasterLayer, SitePair
from hydropower.spatial_discharge import SpatialDischargeCalculator

# Get raster layer
raster = RasterLayer.objects.get(id=87)
print(f"Raster: {raster.id}")
print(f"Flow accumulation path: {raster.flow_accumulation_path}")
print()

# Initialize calculator
calculator = SpatialDischargeCalculator(raster)
print(f"Max flow accumulation: {calculator.max_flow_acc:.0f} cells")
print(f"Cell size: {calculator.cell_size:.1f}m")
print(f"Basin area: {calculator.basin_area_km2:.2f} km²")
print()

# Check flow accumulation at each inlet
print("Flow accumulation at each inlet point:")
print("-" * 80)
print(f"{'Site ID':<25} {'Inlet X':<12} {'Inlet Y':<12} {'Flow Acc':>12} {'Area Ratio':>12} {'Discharge':>12}")
print("-" * 80)

site_pairs = SitePair.objects.filter(raster_layer=raster).select_related('inlet')
flow_acc_values = []

for sp in site_pairs:
    inlet_x = sp.inlet.geometry.x
    inlet_y = sp.inlet.geometry.y
    
    flow_acc = calculator.get_flow_accumulation_at_point(inlet_x, inlet_y)
    area_ratio = flow_acc / calculator.max_flow_acc if flow_acc else 0
    
    flow_acc_values.append((sp.pair_id, flow_acc, area_ratio, sp.discharge))
    
    print(f"{sp.pair_id:<25} {inlet_x:<12.2f} {inlet_y:<12.2f} {flow_acc or 0:>12.0f} {area_ratio:>12.4f} {sp.discharge:>12.4f}")

print("-" * 80)
print()

# Show unique flow accumulation values
unique_fa = set(fa for _, fa, _, _ in flow_acc_values if fa)
print(f"Total inlets: {len(flow_acc_values)}")
print(f"Unique flow accumulation values: {len(unique_fa)}")
print()

# Check if the issue is flow accumulation granularity
print("Unique flow accumulation values (sorted):")
for fa in sorted(unique_fa):
    count = sum(1 for _, f, _, _ in flow_acc_values if f == fa)
    ratio = fa / calculator.max_flow_acc
    print(f"  {fa:>10.0f} cells ({ratio*100:>6.2f}% of basin) - {count} sites")
