"""
Deep dive debug script to understand why systematic pairing fails
even with adaptive constraints
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

print('=== SYSTEMATIC PAIRING FAILURE ANALYSIS ===\n')

# Simulate densified node creation with 50m spacing
print('Simulating node densification (50m spacing):\n')

segments_with_pairs = 0
total_potential_pairs = 0

for stream in streams[:10]:  # Sample first 10
    if stream.length_m and stream.length_m > 0:
        num_nodes = max(2, int(np.ceil(stream.length_m / 50.0)))
        
        # Simulate node spacing
        if num_nodes >= 2:
            # Distance between consecutive nodes
            node_spacing_actual = stream.length_m / (num_nodes - 1)
            
            # Check how many valid pairs this segment could generate
            pairs_in_segment = 0
            for i in range(num_nodes - 1):
                for j in range(i + 1, num_nodes):
                    river_dist = (j - i) * node_spacing_actual
                    
                    # Check constraints
                    if river_dist >= 10.0 and river_dist <= 8000.0:
                        # Would also need to check head (elevation difference)
                        # But we can't do that without DEM values at each node
                        pairs_in_segment += 1
            
            if pairs_in_segment > 0:
                segments_with_pairs += 1
                total_potential_pairs += pairs_in_segment
            
            print(f'Stream {stream.id}: {stream.length_m:.1f}m, {num_nodes} nodes, '
                  f'spacing={node_spacing_actual:.1f}m, potential_pairs={pairs_in_segment}')

print(f'\nSegments that could produce pairs: {segments_with_pairs}/10')
print(f'Total potential pairs (distance check only): {total_potential_pairs}')

print('\n=== KEY INSIGHT ===')
print('Even with 10m min_river_distance, segments are too short!')
print('Average segment length: 12m')
print('With 50m node spacing, most segments get only 2 nodes')
print('Distance between those 2 nodes: ~12m')
print('But the constraint check also needs:')
print('  1. Positive head (inlet elevation > outlet elevation)')
print('  2. Head >= 5m minimum')
print('\nFor such short segments, the elevation drop might be < 5m')
print('even if there is some elevation difference.')

print('\n=== SOLUTION ===')
print('1. Further reduce min_head from 5m to 2m for micro-hydro')
print('2. Reduce node_spacing from 50m to 10m (match segment length)')
print('3. OR: Disable systematic pairing for networks with median length < 50m')
print('   (rely on original pairing method which pairs across entire network)')
