#!/usr/bin/env python
"""Analyze StreamNetwork for potential HP node generation"""

import django
import os

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'HYDROPOWER_MAPPING.settings')
django.setup()

from hydropower.models import StreamNetwork, RasterLayer, WatershedPolygon

rl = RasterLayer.objects.latest('preprocessing_date')
streams = StreamNetwork.objects.filter(raster_layer=rl)

print(f"RasterLayer ID: {rl.id}")
print(f"Total stream segments: {streams.count()}")

total_length = sum(s.length_m for s in streams)
print(f"Total stream length: {total_length:.1f}m ({total_length/1000:.1f}km)")
print(f"Average segment length: {total_length/streams.count():.1f}m")

print(f"\nStream order distribution:")
for order in range(1, 6):
    order_streams = streams.filter(stream_order=order)
    if order_streams.exists():
        count = order_streams.count()
        length = sum(s.length_m for s in order_streams)
        print(f"  Order {order}: {count} segments, {length:.1f}m ({length/1000:.1f}km)")

print(f"\nLongest stream segments:")
for s in streams.order_by('-length_m')[:5]:
    print(f"  ID {s.id}: {s.length_m:.1f}m, Order {s.stream_order}")

print(f"\nShortest stream segments:")
for s in streams.order_by('length_m')[:5]:
    print(f"  ID {s.id}: {s.length_m:.1f}m, Order {s.stream_order}")

# Check how many streams are inside watersheds
watersheds = WatershedPolygon.objects.filter(raster_layer=rl)
print(f"\nWatersheds: {watersheds.count()}")

# Sample check
sample_ws = watersheds.order_by('-area_km2').first()
if sample_ws:
    streams_in_ws = streams.filter(geometry__intersects=sample_ws.geometry)
    print(f"\nLargest watershed (ID {sample_ws.watershed_id}, {sample_ws.area_km2:.4f} km²):")
    print(f"  Streams intersecting: {streams_in_ws.count()}")
