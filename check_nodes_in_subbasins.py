#!/usr/bin/env python
"""Check how many HP nodes are inside vs outside subbasins"""

import django
import os

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'HYDROPOWER_MAPPING.settings')
django.setup()

from hydropower.models import HPNode, WatershedPolygon, RasterLayer

# Get raster layer
rl = RasterLayer.objects.filter(is_preprocessed=True).order_by('-preprocessing_date').first()
print(f"RasterLayer ID: {rl.id}")

# Get all HP nodes and watersheds
nodes = HPNode.objects.filter(raster_layer=rl)
watersheds = WatershedPolygon.objects.filter(raster_layer=rl)

print(f"Total HP nodes: {nodes.count()}")
print(f"Total watersheds: {watersheds.count()}")

# Check spatial containment for all nodes
print(f"\nChecking spatial containment...")
nodes_inside = 0
nodes_outside = 0

for node in nodes:
    is_inside = False
    for ws in watersheds:
        if ws.geometry.contains(node.geometry):
            is_inside = True
            break
    
    if is_inside:
        nodes_inside += 1
    else:
        nodes_outside += 1

print(f"\nResults:")
print(f"  HP nodes INSIDE subbasins: {nodes_inside} ({nodes_inside/nodes.count()*100:.1f}%)")
print(f"  HP nodes OUTSIDE subbasins: {nodes_outside} ({nodes_outside/nodes.count()*100:.1f}%)")

# Show some examples
print(f"\nExamples of nodes outside subbasins:")
count = 0
for node in nodes:
    is_inside = any(ws.geometry.contains(node.geometry) for ws in watersheds)
    if not is_inside:
        print(f"  {node.node_id}: ({node.geometry.x:.1f}, {node.geometry.y:.1f}), Elev {node.elevation:.1f}m")
        count += 1
        if count >= 5:
            break
