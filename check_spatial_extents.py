#!/usr/bin/env python
"""Check spatial extents of HP nodes vs watersheds"""

import django
import os

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'HYDROPOWER_MAPPING.settings')
django.setup()

from hydropower.models import HPNode, WatershedPolygon, RasterLayer
from django.contrib.gis.geos import MultiPoint

# Get raster layer
rl = RasterLayer.objects.filter(is_preprocessed=True).order_by('-preprocessing_date').first()
print(f"RasterLayer ID: {rl.id}\n")

# Get HP nodes and watersheds
nodes = HPNode.objects.filter(raster_layer=rl)
watersheds = WatershedPolygon.objects.filter(raster_layer=rl)

print(f"HP Nodes: {nodes.count()}")
print(f"Watersheds: {watersheds.count()}\n")

# Get bounding boxes
if nodes.exists():
    node_geoms = [node.geometry for node in nodes]
    nodes_multipoint = MultiPoint(node_geoms)
    nodes_extent = nodes_multipoint.extent
    print(f"HP Nodes Extent (minx, miny, maxx, maxy):")
    print(f"  {nodes_extent}")
    print(f"  Width: {nodes_extent[2] - nodes_extent[0]:.1f}m")
    print(f"  Height: {nodes_extent[3] - nodes_extent[1]:.1f}m\n")

if watersheds.exists():
    # Get combined extent of all watersheds
    ws_list = list(watersheds)
    all_minx = min(ws.geometry.extent[0] for ws in ws_list)
    all_miny = min(ws.geometry.extent[1] for ws in ws_list)
    all_maxx = max(ws.geometry.extent[2] for ws in ws_list)
    all_maxy = max(ws.geometry.extent[3] for ws in ws_list)
    
    print(f"Watersheds Combined Extent:")
    print(f"  ({all_minx:.1f}, {all_miny:.1f}, {all_maxx:.1f}, {all_maxy:.1f})")
    print(f"  Width: {all_maxx - all_minx:.1f}m")
    print(f"  Height: {all_maxy - all_miny:.1f}m\n")
    
    # Calculate distance between extents
    node_center_x = (nodes_extent[0] + nodes_extent[2]) / 2
    node_center_y = (nodes_extent[1] + nodes_extent[3]) / 2
    ws_center_x = (all_minx + all_maxx) / 2
    ws_center_y = (all_miny + all_maxy) / 2
    
    distance = ((node_center_x - ws_center_x)**2 + (node_center_y - ws_center_y)**2)**0.5
    
    print(f"Distance between centers: {distance:.1f}m ({distance/1000:.1f}km)\n")
    
    # Check overlap
    overlap_x = not (nodes_extent[2] < all_minx or nodes_extent[0] > all_maxx)
    overlap_y = not (nodes_extent[3] < all_miny or nodes_extent[1] > all_maxy)
    overlap = overlap_x and overlap_y
    
    print(f"Extents overlap: {overlap}")
    if not overlap:
        print(f"  ❌ HP nodes and watersheds are in DIFFERENT locations!")
        print(f"  This explains why no nodes are inside watersheds.")
    else:
        print(f"  ✓ Extents overlap, but nodes may still be outside polygon boundaries")

# Show some sample coordinates
print(f"\nSample HP node locations:")
for node in nodes[:3]:
    print(f"  {node.node_id}: ({node.geometry.x:.1f}, {node.geometry.y:.1f})")

print(f"\nSample watershed locations:")
for ws in watersheds.order_by('-area_km2')[:3]:
    extent = ws.geometry.extent
    center_x = (extent[0] + extent[2]) / 2
    center_y = (extent[1] + extent[3]) / 2
    print(f"  Watershed {ws.watershed_id}: center ({center_x:.1f}, {center_y:.1f}), area {ws.area_km2:.4f} km²")
