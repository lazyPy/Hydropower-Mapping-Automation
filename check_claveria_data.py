#!/usr/bin/env python
"""
Check the Claveria Subbasin shapefile and compare with current data
"""

import geopandas as gpd
import os

# File paths
subbasin_shp = r"d:\Desktop\Hydro HEC-HMS\INPUT DATA - Claveria\WATERSHED DATA\Bridge & River\Claveria Subbasin.shp"
main_channel_gpkg = r"d:\Desktop\Hydro HEC-HMS\INPUT DATA - Claveria\Nominal-Channel.gpkg"

print("="*60)
print("CLAVERIA SUBBASIN SHAPEFILE")
print("="*60)

# Read subbasin
subbasin = gpd.read_file(subbasin_shp)
print(f"\nFeatures: {len(subbasin)}")
print(f"CRS: {subbasin.crs}")
print(f"Bounds: {subbasin.total_bounds}")
print(f"Total area: {subbasin.geometry.area.sum() / 1_000_000:.2f} km²")

print(f"\nColumns: {list(subbasin.columns)}")
print(f"\nFirst few rows:")
print(subbasin.head())

print(f"\n{'='*60}")
print("NOMINAL CHANNEL GPKG")
print("="*60)

# Read main channel
channel = gpd.read_file(main_channel_gpkg)
print(f"\nFeatures: {len(channel)}")
print(f"CRS: {channel.crs}")
print(f"Bounds: {channel.total_bounds}")
print(f"Total length: {channel.geometry.length.sum() / 1000:.2f} km")

print(f"\nColumns: {list(channel.columns)}")

print(f"\n{'='*60}")
print("SPATIAL COMPARISON")
print("="*60)

# Check if they overlap
if subbasin.crs != channel.crs:
    print(f"\n⚠️  CRS MISMATCH!")
    print(f"  Subbasin CRS: {subbasin.crs}")
    print(f"  Channel CRS: {channel.crs}")
    print(f"  Reprojecting subbasin to match channel...")
    subbasin = subbasin.to_crs(channel.crs)

# Check intersection
subbasin_union = subbasin.geometry.unary_union
channel_union = channel.geometry.unary_union

intersects = subbasin_union.intersects(channel_union)
print(f"\nGeometries intersect: {intersects}")

if intersects:
    intersection = subbasin_union.intersection(channel_union)
    intersection_length = intersection.length if hasattr(intersection, 'length') else sum(g.length for g in intersection.geoms)
    total_channel_length = channel_union.length
    
    print(f"\nChannel length INSIDE subbasins: {intersection_length / 1000:.2f} km")
    print(f"Total channel length: {total_channel_length / 1000:.2f} km")
    print(f"Percentage inside: {intersection_length / total_channel_length * 100:.1f}%")
else:
    print(f"\n❌ Channel and subbasins DO NOT INTERSECT!")
    print(f"   These are from different areas.")

# Compare bounds
sb_bounds = subbasin.total_bounds
ch_bounds = channel.total_bounds

print(f"\nSubbasin bounds: ({sb_bounds[0]:.0f}, {sb_bounds[1]:.0f}) to ({sb_bounds[2]:.0f}, {sb_bounds[3]:.0f})")
print(f"Channel bounds:  ({ch_bounds[0]:.0f}, {ch_bounds[1]:.0f}) to ({ch_bounds[2]:.0f}, {ch_bounds[3]:.0f})")

# Calculate distance between centers
sb_center = [(sb_bounds[0] + sb_bounds[2])/2, (sb_bounds[1] + sb_bounds[3])/2]
ch_center = [(ch_bounds[0] + ch_bounds[2])/2, (ch_bounds[1] + ch_bounds[3])/2]
distance = ((sb_center[0] - ch_center[0])**2 + (sb_center[1] - ch_center[1])**2)**0.5

print(f"\nDistance between centers: {distance / 1000:.1f} km")
