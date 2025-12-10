"""
Debug script to analyze why systematic pairing produced no results
"""
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'HYDROPOWER_MAPPING.settings')
django.setup()

from hydropower.models import StreamNetwork, RasterLayer
import numpy as np

# Get the latest raster layer
raster = RasterLayer.objects.latest('id')
streams = StreamNetwork.objects.filter(raster_layer=raster)

print(f'=== STREAM NETWORK ANALYSIS ===\n')
print(f'Total streams: {streams.count()}')
print(f'\nStream lengths (m):')
lengths = [s.length_m for s in streams if s.length_m]
if lengths:
    print(f'  Min: {min(lengths):.1f}m')
    print(f'  Max: {max(lengths):.1f}m')
    print(f'  Mean: {np.mean(lengths):.1f}m')
    print(f'  Median: {np.median(lengths):.1f}m')
    print(f'  Streams < 100m: {sum(1 for l in lengths if l < 100)} ({sum(1 for l in lengths if l < 100)/len(lengths)*100:.1f}%)')

print(f'\n(Elevation analysis skipped - no from_elevation/to_elevation attributes)')

print(f'\n=== DENSIFIED NODES SIMULATION ===\n')

# Simulate node densification
node_spacing = 100.0  # Default from config
total_nodes_created = 0
multi_node_segments = 0

for stream in streams:
    if stream.length_m and stream.length_m > 0:
        num_nodes = max(2, int(np.ceil(stream.length_m / node_spacing)))
        total_nodes_created += num_nodes
        if num_nodes >= 2:
            multi_node_segments += 1

print(f'Node spacing: {node_spacing}m')
print(f'Total nodes that would be created: {total_nodes_created}')
print(f'Segments with 2+ nodes: {multi_node_segments}/{streams.count()} ({multi_node_segments/streams.count()*100:.1f}%)')

# Check if it would trigger global pairing mode
threshold = 0.1  # 10% threshold from code
if multi_node_segments < streams.count() * threshold:
    print(f'\n⚠️  GLOBAL PAIRING MODE would be triggered (< 10% have multiple nodes)')
    print(f'   This mode has a max_outlets_to_check limit of 50, which may be too restrictive')
else:
    print(f'\n✓ WITHIN-SEGMENT PAIRING MODE would be used')

print(f'\n=== POTENTIAL ISSUES ===\n')

# Check for very short segments
very_short = sum(1 for l in lengths if l < 50)
if very_short > len(lengths) * 0.5:
    print(f'⚠️  {very_short} segments ({very_short/len(lengths)*100:.1f}%) are < 50m long')
    print(f'   These won\'t create valid pairs with min_river_distance=50m')

# Elevation-based checks skipped

# Check if segments are mostly single-node after densification
if total_nodes_created < streams.count() * 1.5:
    print(f'⚠️  Average nodes per segment: {total_nodes_created/streams.count():.1f}')
    print(f'   Most segments will have only 1-2 nodes, limiting pairing options')

print(f'\n=== RECOMMENDATIONS ===\n')
print('1. Lower min_river_distance from 50m to 20m for short streams')
print('2. Lower min_head from 10m to 5m to capture micro-hydro potential')
print('3. Reduce node_spacing from 100m to 50m to create more nodes')
print('4. Remove max_outlets_to_check=50 limit in global pairing mode')
print('5. Consider using original pairing method for networks with many short segments')
