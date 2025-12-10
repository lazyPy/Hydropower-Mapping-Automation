#!/usr/bin/env python
"""Debug watershed-stream spatial relationships"""

import django
import os

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'HYDROPOWER_MAPPING.settings')
django.setup()

from hydropower.models import RasterLayer, WatershedPolygon, StreamNetwork
from django.contrib.gis.geos import GEOSGeometry

# Get raster layer
rl = RasterLayer.objects.filter(
    is_preprocessed=True,
    watershed_delineated=True
).order_by('-preprocessing_date').first()

print(f"RasterLayer ID: {rl.id}")
print(f"Watersheds: {WatershedPolygon.objects.filter(raster_layer=rl).count()}")
print(f"Streams: {StreamNetwork.objects.filter(raster_layer=rl).count()}")

# Get a few watersheds and streams
watersheds = WatershedPolygon.objects.filter(raster_layer=rl).order_by('-area_km2')[:3]
streams = StreamNetwork.objects.filter(raster_layer=rl).order_by('-length_m')[:5]

print(f"\n=== Top 3 Watersheds ===")
for ws in watersheds:
    print(f"Watershed {ws.watershed_id}:")
    print(f"  Area: {ws.area_km2:.4f} km²")
    print(f"  Bounds: {ws.geometry.extent}")
    print(f"  Type: {ws.geometry.geom_type}")

print(f"\n=== Top 5 Streams ===")
for stream in streams:
    print(f"Stream {stream.id}:")
    print(f"  Length: {stream.length_m:.1f} m")
    print(f"  Order: {stream.stream_order}")
    print(f"  Bounds: {stream.geometry.extent}")
    print(f"  Type: {stream.geometry.geom_type}")

# Check spatial intersection
print(f"\n=== Spatial Intersection Test ===")
ws1 = watersheds[0]
print(f"Testing Watershed {ws1.watershed_id} against all streams...")

# Method 1: Django ORM intersects
intersecting_streams = StreamNetwork.objects.filter(
    raster_layer=rl,
    geometry__intersects=ws1.geometry
)
print(f"  Method 1 (ORM intersects): {intersecting_streams.count()} streams")

# Method 2: Check within expanded bounds
from django.contrib.gis.geos import Polygon
minx, miny, maxx, maxy = ws1.geometry.extent
# Expand bounds by 100m
buffer = 100.0
bounds_geom = Polygon.from_bbox((minx-buffer, miny-buffer, maxx+buffer, maxy+buffer))
bounds_geom.srid = 32651

streams_in_bounds = StreamNetwork.objects.filter(
    raster_layer=rl,
    geometry__intersects=bounds_geom
)
print(f"  Method 2 (expanded bounds): {streams_in_bounds.count()} streams")

# Method 3: Check all streams manually
print(f"\n  Checking first 10 streams manually...")
for stream in streams[:10]:
    intersects = stream.geometry.intersects(ws1.geometry)
    distance = stream.geometry.distance(ws1.geometry)
    print(f"    Stream {stream.id}: intersects={intersects}, distance={distance:.1f}m")
