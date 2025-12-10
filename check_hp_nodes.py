#!/usr/bin/env python
"""Check HP nodes and subbasins in the database"""

import django
import os

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'HYDROPOWER_MAPPING.settings')
django.setup()

from hydropower.models import HPNode, WatershedPolygon, StreamNetwork

print(f"HP Nodes: {HPNode.objects.count()}")
print(f"Watersheds: {WatershedPolygon.objects.count()}")
print(f"Stream Network segments: {StreamNetwork.objects.count()}")

print("\nWatershed details:")
for ws in WatershedPolygon.objects.all()[:5]:
    print(f"  Watershed {ws.watershed_id}: {ws.area_km2:.2f} km², {ws.stream_count} streams")

print("\nHP Node details (first 10):")
for node in HPNode.objects.all()[:10]:
    print(f"  {node.node_id}: Elev {node.elevation:.1f}m, Chainage {node.chainage:.2f}km")

# Check if HP nodes have watershed association
print("\nChecking HP node - Watershed relationships:")
node_model_fields = [f.name for f in HPNode._meta.get_fields()]
print(f"  HPNode fields: {node_model_fields}")
if 'watershed' in node_model_fields or 'watershed_polygon' in node_model_fields:
    print("  HPNode has watershed field")
else:
    print("  HPNode DOES NOT have watershed field (needs to be added!)")

# Check stream segments in watersheds
print("\nStream segments per watershed:")
for ws in WatershedPolygon.objects.all()[:3]:
    streams = StreamNetwork.objects.filter(
        raster_layer=ws.raster_layer,
        geometry__intersects=ws.geometry
    )
    print(f"  Watershed {ws.watershed_id}: {streams.count()} stream segments")
