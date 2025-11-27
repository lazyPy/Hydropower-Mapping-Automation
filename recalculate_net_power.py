"""
Recalculate Power with Net Head (After Friction Losses)
========================================================
This applies head loss calculations for more accurate power output.
"""

import os
os.environ['GDAL_LIBRARY_PATH'] = r'D:\Desktop\Hydro HEC-HMS\env\Lib\site-packages\osgeo\gdal.dll'
os.environ['DJANGO_SETTINGS_MODULE'] = 'HYDROPOWER_MAPPING.settings'

import django
django.setup()

import numpy as np
from hydropower.models import SitePair

# Constants
RHO = 1000.0  # kg/m³
G = 9.81      # m/s²
ETA = 0.7     # 70% efficiency

# Penstock parameters (realistic values)
VELOCITY = 4.0  # m/s design velocity
C_HAZEN = 140   # Hazen-Williams coefficient for steel
ENTRANCE_LOSS = 0.5
EXIT_LOSS = 1.0
BEND_LOSS = 0.25  # Per 90° bend
NUM_BENDS = 2


def calculate_head_losses(gross_head, penstock_length, discharge):
    """Calculate friction and minor losses in penstock."""
    if discharge <= 0 or penstock_length <= 0:
        return 0, gross_head
    
    # Calculate diameter from velocity and discharge
    diameter = np.sqrt((4 * discharge) / (np.pi * VELOCITY))
    
    # Friction loss (Hazen-Williams)
    friction_loss = 10.67 * penstock_length * (discharge ** 1.852) / ((C_HAZEN ** 1.852) * (diameter ** 4.87))
    
    # Minor losses
    velocity_head = (VELOCITY ** 2) / (2 * G)
    minor_loss = (ENTRANCE_LOSS + EXIT_LOSS + NUM_BENDS * BEND_LOSS) * velocity_head
    
    total_loss = friction_loss + minor_loss
    net_head = max(0, gross_head - total_loss)
    
    return total_loss, net_head


def main():
    print("=" * 70)
    print("  RECALCULATING POWER WITH NET HEAD (FRICTION LOSSES)")
    print("=" * 70 + "\n")
    
    pairs = SitePair.objects.all()
    print(f"Processing {pairs.count()} site pairs...\n")
    
    results = []
    
    for pair in pairs:
        # Estimate penstock length (30% of river distance as typical)
        penstock_length = pair.euclidean_distance * 0.8  # Penstock follows terrain
        
        # Calculate head losses
        total_loss, net_head = calculate_head_losses(
            pair.head, 
            penstock_length, 
            pair.discharge
        )
        
        # Recalculate power with net head
        gross_power = RHO * G * pair.discharge * pair.head * ETA / 1000
        net_power = RHO * G * pair.discharge * net_head * ETA / 1000
        
        # Update database
        pair.power = net_power
        pair.save()
        
        efficiency_factor = (net_head / pair.head * 100) if pair.head > 0 else 0
        
        results.append({
            'pair_id': pair.pair_id,
            'gross_head': pair.head,
            'penstock_length': penstock_length,
            'head_loss': total_loss,
            'net_head': net_head,
            'efficiency': efficiency_factor,
            'gross_power': gross_power,
            'net_power': net_power,
            'rank': pair.rank
        })
    
    # Sort by rank
    results.sort(key=lambda x: x['rank'] if x['rank'] else 999)
    
    # Print results
    print("-" * 100)
    print(f"{'Rank':<6}{'Pair ID':<20}{'Gross H':<10}{'Penstock':<10}{'Loss':<8}{'Net H':<10}{'Eff %':<8}{'Gross P':<12}{'Net P':<12}")
    print("-" * 100)
    
    total_gross = 0
    total_net = 0
    
    for r in results[:15]:
        print(f"{r['rank'] or '-':<6}"
              f"{r['pair_id'][:19]:<20}"
              f"{r['gross_head']:<10.1f}"
              f"{r['penstock_length']:<10.1f}"
              f"{r['head_loss']:<8.2f}"
              f"{r['net_head']:<10.1f}"
              f"{r['efficiency']:<8.1f}"
              f"{r['gross_power']:<12.1f}"
              f"{r['net_power']:<12.1f}")
        total_gross += r['gross_power']
        total_net += r['net_power']
    
    if len(results) > 15:
        for r in results[15:]:
            total_gross += r['gross_power']
            total_net += r['net_power']
        print(f"... and {len(results) - 15} more")
    
    print("\n" + "=" * 70)
    print("SUMMARY (with head losses)")
    print("=" * 70)
    
    net_powers = [r['net_power'] for r in results]
    efficiencies = [r['efficiency'] for r in results]
    
    print(f"Total Gross Power: {sum(r['gross_power'] for r in results):.2f} kW ({sum(r['gross_power'] for r in results)/1000:.2f} MW)")
    print(f"Total Net Power:   {sum(net_powers):.2f} kW ({sum(net_powers)/1000:.2f} MW)")
    print(f"Average Head Loss: {100 - np.mean(efficiencies):.1f}%")
    print(f"Power Reduction:   {(1 - sum(net_powers)/sum(r['gross_power'] for r in results))*100:.1f}%")
    
    print("\nNet Power Range: {:.2f} - {:.2f} kW".format(min(net_powers), max(net_powers)))
    
    # Classification
    micro = sum(1 for p in net_powers if p < 100)
    mini = sum(1 for p in net_powers if 100 <= p < 1000)
    small = sum(1 for p in net_powers if p >= 1000)
    print(f"\nClassification (Net Power):")
    print(f"  Micro (<100 kW): {micro} sites")
    print(f"  Mini (100-1000 kW): {mini} sites")
    print(f"  Small (>1000 kW): {small} sites")


if __name__ == '__main__':
    main()
