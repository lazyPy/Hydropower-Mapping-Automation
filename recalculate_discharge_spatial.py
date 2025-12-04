#!/usr/bin/env python
r"""
Recalculate Discharge with Spatially-Varying Values

This script updates all site pairs with spatially-varying discharge values
based on drainage area (flow accumulation). Instead of assigning the same
discharge to all sites, each site now gets a unique discharge proportional
to its upstream contributing area.

Formula: Q_site = Q_outlet × (A_site / A_outlet)

Where:
- Q_outlet = Peak discharge at basin outlet (from HEC-HMS)
- A_site = Contributing area at site (from flow accumulation)
- A_outlet = Total basin area (max flow accumulation)

Usage:
    .\env\Scripts\Activate.ps1; python recalculate_discharge_spatial.py

Options:
    --q-outlet VALUE    Override outlet discharge (m³/s), default: from HMS
    --raster-layer ID   Process specific raster layer ID
    --dry-run           Show what would be done without making changes
"""

import os
import sys
import argparse
from pathlib import Path

# Set up Django environment
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'HYDROPOWER_MAPPING.settings')

import django
django.setup()

from django.conf import settings
from django.db.models import Max
from django.utils import timezone
from hydropower.models import RasterLayer, SitePair, HMSRun, TimeSeries
from hydropower.spatial_discharge import update_site_pair_discharge_spatial, SpatialDischargeCalculator


def main():
    parser = argparse.ArgumentParser(description='Recalculate discharge with spatially-varying values')
    parser.add_argument('--q-outlet', type=float, help='Override outlet discharge (m³/s)')
    parser.add_argument('--raster-layer', type=int, help='Process specific raster layer ID')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without changes')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Spatially-Varying Discharge Calculation")
    print("=" * 70)
    print()
    
    # Get raster layer(s) to process
    if args.raster_layer:
        raster_layers = RasterLayer.objects.filter(id=args.raster_layer)
    else:
        raster_layers = RasterLayer.objects.filter(
            site_pairing_completed=True,
            flow_accumulation_path__isnull=False
        ).exclude(flow_accumulation_path='')
    
    if not raster_layers.exists():
        print("ERROR: No eligible raster layers found.")
        print("Requirements:")
        print("  - Flow accumulation must be computed (run DEM preprocessing)")
        print("  - Site pairing must be completed")
        return 1
    
    total_updated = 0
    
    for raster in raster_layers:
        print(f"\n[Processing: {raster.dataset.name if raster.dataset else 'Raster #' + str(raster.id)}]")
        print("-" * 50)
        
        # Check flow accumulation
        if not raster.flow_accumulation_path:
            print("  SKIP: Flow accumulation not available")
            continue
        
        # Get site pairs count
        site_count = SitePair.objects.filter(raster_layer=raster).count()
        if site_count == 0:
            print("  SKIP: No site pairs found")
            continue
        
        print(f"  Site pairs: {site_count}")
        
        # Determine Q_outlet
        if args.q_outlet:
            q_outlet = args.q_outlet
            print(f"  Q_outlet (override): {q_outlet:.2f} m³/s")
        else:
            # Try to get from HMS data
            hms_run = HMSRun.objects.first()
            if hms_run:
                peak_data = TimeSeries.objects.filter(
                    dataset=hms_run.dataset,
                    data_type='DISCHARGE',
                    value__gt=0
                ).aggregate(peak=Max('value'))
                q_outlet = peak_data.get('peak') or 10.0
                print(f"  Q_outlet (from HMS): {q_outlet:.2f} m³/s")
            else:
                q_outlet = 10.0
                print(f"  Q_outlet (default): {q_outlet:.2f} m³/s")
        
        if args.dry_run:
            print("\n  DRY RUN - No changes will be made")
            
            # Still show what would happen
            try:
                calculator = SpatialDischargeCalculator(raster)
                stats = calculator.get_discharge_summary_stats()
                
                print(f"\n  Flow Accumulation Statistics:")
                print(f"    Max cells (outlet): {stats.get('max_cells', 'N/A'):.0f}")
                print(f"    Basin area: {stats.get('basin_area_km2', 'N/A'):.2f} km²")
                print(f"    Cell size: {stats.get('cell_size_m', 'N/A'):.1f} m")
                
                # Sample discharge calculation
                sample_ratios = [0.01, 0.1, 0.5, 1.0]
                print(f"\n  Sample discharge values (Q_outlet = {q_outlet:.2f} m³/s):")
                for ratio in sample_ratios:
                    q = q_outlet * ratio
                    print(f"    {ratio*100:.0f}% of outlet area: {q:.3f} m³/s")
                
            except Exception as e:
                print(f"  Error: {e}")
            
            continue
        
        # Run the spatial discharge calculation
        print("\n  Updating site pairs with spatially-varying discharge...")
        
        try:
            stats = update_site_pair_discharge_spatial(raster, q_outlet)
            
            if 'error' in stats:
                print(f"  ERROR: {stats['error']}")
                continue
            
            total_updated += stats.get('updated', 0)
            
            print(f"\n  Results:")
            print(f"    Updated: {stats.get('updated', 0)} / {stats.get('total', 0)} site pairs")
            
            if stats.get('discharge_min') is not None:
                print(f"\n  Discharge Distribution:")
                print(f"    Q_min:  {stats['discharge_min']:.4f} m³/s")
                print(f"    Q_max:  {stats['discharge_max']:.2f} m³/s")
                print(f"    Q_mean: {stats['discharge_mean']:.3f} m³/s")
                print(f"    Q_std:  {stats.get('discharge_std', 0):.3f} m³/s")
            
            if stats.get('power_min') is not None:
                print(f"\n  Power Output:")
                print(f"    P_min:  {stats['power_min']:.2f} kW")
                print(f"    P_max:  {stats['power_max']:.2f} kW")
                print(f"    P_mean: {stats['power_mean']:.2f} kW")
                print(f"    Total:  {stats['power_total']:.2f} kW ({stats['power_total']/1000:.3f} MW)")
            
            # Mark discharge as computed on this raster layer
            raster.discharge_computed = True
            raster.discharge_computation_date = timezone.now()
            raster.discharge_q_outlet = q_outlet
            raster.save(update_fields=['discharge_computed', 'discharge_computation_date', 'discharge_q_outlet'])
            print(f"\n  ✓ Marked discharge_computed = True")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print(f"Total site pairs updated: {total_updated}")
    print("=" * 70)
    
    if total_updated > 0:
        print("\nDischarge values are now spatially-varying based on drainage area!")
        print("Each site has a unique discharge proportional to its upstream area.")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
