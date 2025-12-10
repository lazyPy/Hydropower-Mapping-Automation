"""
Debug script to understand why only 2 site pairs were generated.
"""
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'HYDROPOWER_MAPPING.settings')
django.setup()

from hydropower.models import HPNode, SitePair
import statistics

def analyze_pairing_constraints():
    """Analyze HP nodes and pairing constraints to understand the low pair count."""
    
    print("\n" + "="*80)
    print("HP NODE ANALYSIS")
    print("="*80)
    
    nodes = list(HPNode.objects.order_by('distance_along_channel'))
    elevations = [n.elevation for n in nodes]
    
    print(f"\nTotal HP Nodes: {len(nodes)}")
    print(f"Elevation Range: {min(elevations):.1f}m - {max(elevations):.1f}m")
    print(f"Elevation Mean: {statistics.mean(elevations):.1f}m")
    print(f"Elevation StdDev: {statistics.stdev(elevations):.1f}m")
    print(f"Channel Length: {nodes[-1].distance_along_channel:.1f}m")
    
    # Check elevation profile
    print(f"\nELEVATION PROFILE (every 5th node):")
    for i in range(0, len(nodes), 5):
        n = nodes[i]
        print(f"  {n.node_id}: {n.elevation:.1f}m @ {n.distance_along_channel:.0f}m")
    
    # Analyze head differences
    print(f"\n" + "="*80)
    print("HEAD DIFFERENCE ANALYSIS")
    print("="*80)
    
    head_differences = []
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            inlet = nodes[i]
            outlet = nodes[j]
            head = inlet.elevation - outlet.elevation
            distance = outlet.distance_along_channel - inlet.distance_along_channel
            head_differences.append({
                'inlet': inlet.node_id,
                'outlet': outlet.node_id,
                'head': head,
                'distance': distance
            })
    
    # Filter by constraints
    min_head = 10.0
    max_head = 500.0
    min_distance = 200.0
    max_distance = 5000.0
    
    print(f"\nConstraints:")
    print(f"  Head: {min_head}m - {max_head}m")
    print(f"  Distance: {min_distance}m - {max_distance}m")
    
    total_combinations = len(head_differences)
    
    # Count failures by constraint
    head_too_low = sum(1 for hd in head_differences if hd['head'] < min_head)
    head_too_high = sum(1 for hd in head_differences if hd['head'] > max_head)
    distance_too_short = sum(1 for hd in head_differences if hd['distance'] < min_distance)
    distance_too_long = sum(1 for hd in head_differences if hd['distance'] > max_distance)
    
    print(f"\nTotal Possible Combinations: {total_combinations}")
    print(f"\nConstraint Violations:")
    print(f"  Head < {min_head}m: {head_too_low} ({head_too_low/total_combinations*100:.1f}%)")
    print(f"  Head > {max_head}m: {head_too_high} ({head_too_high/total_combinations*100:.1f}%)")
    print(f"  Distance < {min_distance}m: {distance_too_short} ({distance_too_short/total_combinations*100:.1f}%)")
    print(f"  Distance > {max_distance}m: {distance_too_long} ({distance_too_long/total_combinations*100:.1f}%)")
    
    # Find feasible pairs
    feasible = [
        hd for hd in head_differences
        if min_head <= hd['head'] <= max_head
        and min_distance <= hd['distance'] <= max_distance
    ]
    
    print(f"\nFeasible Pairs: {len(feasible)} ({len(feasible)/total_combinations*100:.1f}%)")
    
    if feasible:
        print(f"\nTop 10 Feasible Pairs (by head):")
        feasible_sorted = sorted(feasible, key=lambda x: x['head'], reverse=True)[:10]
        for hd in feasible_sorted:
            print(f"  {hd['inlet']} → {hd['outlet']}: Head={hd['head']:.1f}m, Distance={hd['distance']:.0f}m")
    
    # Check actual saved pairs
    print(f"\n" + "="*80)
    print("SAVED SITE PAIRS")
    print("="*80)
    
    saved_pairs = SitePair.objects.filter(
        source='MAIN_CHANNEL',
        is_validated=True
    ).order_by('-rank')
    
    print(f"\nTotal Saved Pairs: {saved_pairs.count()}")
    
    for pair in saved_pairs:
        print(f"\nPair ID {pair.id} (Rank {pair.rank}):")
        print(f"  Inlet: {pair.inlet.site_point_id} @ {pair.inlet.elevation:.1f}m")
        print(f"  Outlet: {pair.outlet.site_point_id} @ {pair.outlet.elevation:.1f}m")
        print(f"  Head: {pair.head:.1f}m")
        print(f"  Distance: {pair.euclidean_distance:.1f}m")
        print(f"  Power: {pair.power:.1f}kW")
        print(f"  Score: {pair.score:.1f}")
    
    # Recommendations
    print(f"\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if len(feasible) < 10:
        print("\n⚠️  Very few feasible pairs! Consider:")
        print("  1. Reduce min_head from 10m to 5m")
        print("  2. Increase max_distance from 5000m to 10000m")
        print("  3. Reduce sampling interval from 500m to 250m (more nodes)")
        
        # Calculate what constraints would yield more pairs
        if head_too_low > total_combinations * 0.5:
            print("\n  🔍 Most failures due to insufficient head (<10m)")
            print("     Try: min_head = 5m")
        
        if distance_too_long > total_combinations * 0.5:
            print("\n  🔍 Most failures due to excessive distance (>5000m)")
            print("     Try: max_distance = 10000m")

if __name__ == '__main__':
    analyze_pairing_constraints()
