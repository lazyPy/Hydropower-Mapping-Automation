"""
Fix Discharge Association for Site Pairs
========================================
This script:
1. Cleans invalid discharge values from TimeSeries
2. Associates HMS discharge data with Site Pairs
3. Calculates power output (P = ρ × g × Q × H × η)

Run with: python fix_discharge_association.py
"""

import os
import sys

# Set environment before Django imports
os.environ['GDAL_LIBRARY_PATH'] = r'D:\Desktop\Hydro HEC-HMS\env\Lib\site-packages\osgeo\gdal.dll'
os.environ['DJANGO_SETTINGS_MODULE'] = 'HYDROPOWER_MAPPING.settings'

import django
django.setup()

from hydropower.models import TimeSeries, SitePair, HMSRun
from django.db.models import Min, Max, Avg

# Constants for power calculation
RHO = 1000.0  # Water density kg/m³
G = 9.81      # Gravity m/s²
ETA = 0.7     # Efficiency (70%)


def clean_invalid_discharge():
    """Remove invalid nodata values from TimeSeries."""
    print("=" * 60)
    print("STEP 1: CLEANING INVALID DISCHARGE VALUES")
    print("=" * 60)
    
    # Find and delete invalid records (IEEE float nodata)
    invalid_ts = TimeSeries.objects.filter(value__lt=-1e30)
    invalid_count = invalid_ts.count()
    
    if invalid_count > 0:
        print(f"Found {invalid_count} invalid discharge records (nodata values)")
        invalid_ts.delete()
        print(f"Deleted {invalid_count} invalid records")
    else:
        print("No invalid records found")
    
    # Verify remaining data
    ts = TimeSeries.objects.filter(data_type='DISCHARGE')
    if ts.exists():
        d = ts.aggregate(min_v=Min('value'), max_v=Max('value'), avg_v=Avg('value'))
        print(f"Remaining discharge records: {ts.count()}")
        print(f"Discharge range: {d['min_v']:.4f} - {d['max_v']:.4f} m³/s (avg: {d['avg_v']:.4f})")
    else:
        print("WARNING: No discharge records remaining!")
    
    return ts.exists()


def get_representative_discharge(hms_run_id):
    """Get peak discharge from HMS run for site association."""
    print("\n" + "=" * 60)
    print("STEP 2: EXTRACTING REPRESENTATIVE DISCHARGE")
    print("=" * 60)
    
    try:
        hms_run = HMSRun.objects.get(id=hms_run_id)
        print(f"HMS Run: {hms_run.event_name}")
        print(f"Return Period: {hms_run.return_period or 'Not specified'}")
        print(f"Peak Discharge: {hms_run.peak_discharge:.4f} m³/s")
        
        # Get discharge by station
        ts = TimeSeries.objects.filter(
            dataset=hms_run.dataset,
            data_type='DISCHARGE',
            value__gte=0  # Only positive values
        )
        
        # Get peak discharge per station
        from django.db.models import Max as DBMax
        stations = ts.values('station_id').annotate(
            peak_q=DBMax('value')
        ).order_by('-peak_q')
        
        print(f"\nStation Peak Discharges:")
        discharge_data = {}
        for s in stations:
            print(f"  {s['station_id']}: {s['peak_q']:.4f} m³/s")
            discharge_data[s['station_id']] = s['peak_q']
        
        # Use outlet/sink peak discharge as representative
        # Typically use the highest discharge (basin outlet)
        representative_q = max(discharge_data.values()) if discharge_data else hms_run.peak_discharge
        print(f"\nRepresentative Q (peak): {representative_q:.4f} m³/s")
        
        return representative_q, hms_run
        
    except HMSRun.DoesNotExist:
        print(f"ERROR: HMS Run with ID {hms_run_id} not found")
        return None, None


def assign_discharge_to_sites(representative_q, hms_run):
    """Assign discharge and calculate power for all site pairs."""
    print("\n" + "=" * 60)
    print("STEP 3: ASSIGNING DISCHARGE & CALCULATING POWER")
    print("=" * 60)
    
    site_pairs = SitePair.objects.all()
    total = site_pairs.count()
    
    print(f"Total site pairs to process: {total}")
    print(f"Using efficiency η = {ETA} ({ETA*100:.0f}%)")
    print(f"Formula: P = ρ × g × Q × H × η / 1000 (kW)")
    
    updated_count = 0
    results = []
    
    for pair in site_pairs:
        # For run-of-river hydropower, discharge typically scales with drainage area
        # For simplicity, we use the representative Q (can be enhanced with spatial matching)
        
        # Assign discharge
        pair.discharge = representative_q
        pair.hms_run = hms_run
        pair.return_period = hms_run.return_period if hms_run else ""
        
        # Calculate power: P = ρ × g × Q × H × η (Watts) / 1000 (kW)
        power_watts = RHO * G * pair.discharge * pair.head * ETA
        pair.power = power_watts / 1000.0  # Convert to kW
        
        # Save
        pair.save()
        updated_count += 1
        
        results.append({
            'pair_id': pair.pair_id,
            'head': pair.head,
            'discharge': pair.discharge,
            'power': pair.power,
            'rank': pair.rank
        })
    
    print(f"\nUpdated {updated_count} site pairs with discharge and power")
    
    # Show results sorted by rank
    results.sort(key=lambda x: x['rank'] if x['rank'] else 999)
    
    print("\n" + "-" * 70)
    print(f"{'Rank':<6}{'Pair ID':<25}{'Head (m)':<12}{'Q (m³/s)':<12}{'Power (kW)':<12}")
    print("-" * 70)
    
    for r in results[:15]:  # Show top 15
        print(f"{r['rank'] or '-':<6}{r['pair_id'][:24]:<25}{r['head']:<12.2f}{r['discharge']:<12.4f}{r['power']:<12.2f}")
    
    if len(results) > 15:
        print(f"... and {len(results) - 15} more site pairs")
    
    # Summary statistics
    powers = [r['power'] for r in results if r['power']]
    if powers:
        print("\n" + "=" * 60)
        print("POWER OUTPUT SUMMARY")
        print("=" * 60)
        print(f"Total sites: {len(powers)}")
        print(f"Power range: {min(powers):.2f} - {max(powers):.2f} kW")
        print(f"Average power: {sum(powers)/len(powers):.2f} kW")
        print(f"Total potential: {sum(powers):.2f} kW ({sum(powers)/1000:.2f} MW)")
        
        # Classify by power output
        micro = sum(1 for p in powers if p < 100)
        mini = sum(1 for p in powers if 100 <= p < 1000)
        small = sum(1 for p in powers if p >= 1000)
        print(f"\nClassification:")
        print(f"  Micro (<100 kW): {micro} sites")
        print(f"  Mini (100-1000 kW): {mini} sites")
        print(f"  Small (>1000 kW): {small} sites")


def main():
    print("\n" + "=" * 70)
    print("  HYDROPOWER SITE DISCHARGE ASSOCIATION & POWER CALCULATION")
    print("=" * 70 + "\n")
    
    # Step 1: Clean invalid values
    has_data = clean_invalid_discharge()
    
    if not has_data:
        print("\nERROR: No valid discharge data found. Cannot proceed.")
        return
    
    # Step 2: Get representative discharge
    # Get the first HMS Run ID dynamically
    first_hms = HMSRun.objects.first()
    if not first_hms:
        print("\nERROR: No HMS Run found in database.")
        return
    
    representative_q, hms_run = get_representative_discharge(hms_run_id=first_hms.id)
    
    if representative_q is None:
        print("\nERROR: Could not extract representative discharge. Cannot proceed.")
        return
    
    # Step 3: Assign discharge and calculate power
    assign_discharge_to_sites(representative_q, hms_run)
    
    print("\n" + "=" * 70)
    print("  COMPLETED - Site pairs now have discharge and power values!")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
