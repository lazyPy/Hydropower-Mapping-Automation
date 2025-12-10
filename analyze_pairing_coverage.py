"""
Analyze site pairing coverage along the river channel.
Compare current pairing with expected dense coverage.
"""

import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'HYDROPOWER_MAPPING.settings')
django.setup()

from hydropower.models import HPNode, SitePair
from collections import defaultdict

# Get all HP nodes
all_nodes = HPNode.objects.order_by('distance_along_channel')
print(f"Total HP nodes: {all_nodes.count()}")

# Get all site pairs
all_pairs = SitePair.objects.all()
print(f"Total site pairs: {all_pairs.count()}")

# Find nodes that are NOT used in any pair (as inlet or outlet)
inlet_ids = set(SitePair.objects.values_list('inlet_id', flat=True))
outlet_ids = set(SitePair.objects.values_list('outlet_id', flat=True))
used_node_ids = inlet_ids | outlet_ids

print(f"\nNodes used as inlets: {len(inlet_ids)}")
print(f"Nodes used as outlets: {len(outlet_ids)}")
print(f"Total nodes used in pairs: {len(used_node_ids)}")
print(f"Nodes NOT used: {all_nodes.count() - len(used_node_ids)}")

# Analyze pairing density along river
# Group nodes into 1km segments
segment_size = 1000  # meters
segments = defaultdict(lambda: {'total_nodes': 0, 'used_nodes': 0, 'pairs': 0})

for node in all_nodes:
    segment_idx = int(node.distance_along_channel / segment_size)
    segments[segment_idx]['total_nodes'] += 1
    if node.id in used_node_ids:
        segments[segment_idx]['used_nodes'] += 1

# Count pairs per segment (based on inlet location)
# Note: SitePoint doesn't have distance_along_channel, need to lookup HP node
inlet_hp_nodes = {node.id: node for node in all_nodes}
for pair in all_pairs:
    if pair.inlet_id and pair.inlet_id in inlet_hp_nodes:
        inlet_node = inlet_hp_nodes[pair.inlet_id]
        segment_idx = int(inlet_node.distance_along_channel / segment_size)
        segments[segment_idx]['pairs'] += 1

print(f"\n{'='*80}")
print(f"PAIRING DENSITY BY RIVER SEGMENT (1km segments)")
print(f"{'='*80}")
print(f"{'Segment':<10} {'Distance (km)':<15} {'Total Nodes':<15} {'Used Nodes':<15} {'Pairs':<10} {'Coverage %':<12}")
print(f"{'-'*80}")

total_segments = 0
segments_with_pairs = 0

for segment_idx in sorted(segments.keys()):
    seg = segments[segment_idx]
    distance_km = (segment_idx * segment_size) / 1000.0
    coverage = (seg['used_nodes'] / seg['total_nodes'] * 100) if seg['total_nodes'] > 0 else 0
    
    total_segments += 1
    if seg['pairs'] > 0:
        segments_with_pairs += 1
    
    print(f"{segment_idx:<10} {distance_km:<15.2f} {seg['total_nodes']:<15} {seg['used_nodes']:<15} {seg['pairs']:<10} {coverage:<12.1f}")

print(f"{'-'*80}")
print(f"\nSegments with pairs: {segments_with_pairs}/{total_segments} ({segments_with_pairs/total_segments*100:.1f}%)")

# Analyze pair statistics
print(f"\n{'='*80}")
print(f"SITE PAIR STATISTICS")
print(f"{'='*80}")

if all_pairs.exists():
    heads = [pair.head for pair in all_pairs if pair.head]
    distances = [pair.river_distance for pair in all_pairs if pair.river_distance]
    powers = [pair.power for pair in all_pairs if pair.power]
    
    if heads:
        print(f"Head range: {min(heads):.2f} - {max(heads):.2f} m (avg: {sum(heads)/len(heads):.2f} m)")
    if distances:
        print(f"Distance range: {min(distances):.2f} - {max(distances):.2f} m (avg: {sum(distances)/len(distances):.2f} m)")
    if powers:
        print(f"Power range: {min(powers):.2f} - {max(powers):.2f} kW (avg: {sum(powers)/len(powers):.2f} kW)")

print("\nDone.")
