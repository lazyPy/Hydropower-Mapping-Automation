r"""
Re-run Site Pairing with Updated Configuration

This script re-runs the inlet-outlet pairing algorithm with the updated
configuration (lower thresholds, drainage-area weighted discharge) to
discover more hydropower site candidates.

Run with: python manage.py shell < rerun_site_pairing.py
Or:       .\env\Scripts\Activate.ps1; python manage.py runscript rerun_site_pairing
"""

import os
import sys
import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'HYDROPOWER_MAPPING.settings')
django.setup()

from hydropower.models import RasterLayer, SitePair, SitePoint, HMSRun, StreamNetwork
from hydropower.site_pairing import InletOutletPairing, PairingConfig, calculate_head_losses
from hydropower.discharge_association import (
    DischargeAssociator, DischargeConfig,
    associate_discharge_with_drainage_area,
    get_flow_accumulation_at_point,
    get_max_flow_accumulation
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Re-run site pairing with updated configuration."""
    
    print("=" * 70)
    print("Re-running Site Pairing with Updated Configuration")
    print("=" * 70)
    
    # Check current state
    raster_layers = RasterLayer.objects.filter(
        flow_accumulation_path__isnull=False
    ).order_by('-preprocessing_date')
    
    if not raster_layers.exists():
        print("ERROR: No processed raster layers found!")
        print("Run DEM preprocessing first.")
        return
    
    # Use the most recent raster layer
    raster_layer = raster_layers.first()
    dataset_name = raster_layer.dataset.name if raster_layer.dataset else "Unknown"
    print(f"\nUsing RasterLayer: {dataset_name} (ID: {raster_layer.id})")
    
    # Check for HMS data
    hms_runs = HMSRun.objects.all()
    hms_run = hms_runs.first() if hms_runs.exists() else None
    
    if hms_run:
        print(f"Using HMSRun: {hms_run.event_name} (ID: {hms_run.id})")
        print(f"  - Peak discharge: {hms_run.peak_discharge:.2f} m³/s" if hms_run.peak_discharge else "")
    else:
        print("WARNING: No HMSRun found. Sites will not have discharge/power data.")
    
    # Show current site statistics
    current_sites = SitePair.objects.filter(raster_layer=raster_layer)
    print(f"\nCurrent sites: {current_sites.count()}")
    
    # Check stream network
    streams = StreamNetwork.objects.filter(raster_layer=raster_layer)
    print(f"Stream segments: {streams.count()}")
    
    # Create new configuration with lower thresholds
    config = PairingConfig()
    print(f"\nNew Configuration:")
    print(f"  - min_head: {config.min_head}m (was 10m)")
    print(f"  - min_stream_order: {config.min_stream_order} (was 2)")
    print(f"  - min_river_distance: {config.min_river_distance}m (was 100m)")
    print(f"  - max_river_distance: {config.max_river_distance}m (was 5000m)")
    print(f"  - spacing_buffer: {config.spacing_buffer}m (was 200m)")
    print(f"  - validate_infrastructure_slope: {config.validate_infrastructure_slope}")
    
    # Ask for confirmation
    response = input("\nProceed with re-pairing? This will ADD new sites. [y/N]: ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Load DEM
    dem_path = raster_layer.dataset.file.path if raster_layer.dataset and raster_layer.dataset.file else None
    if not dem_path:
        print("ERROR: DEM file not found!")
        return
    
    print(f"\nLoading DEM: {dem_path}")
    
    # Create pairing instance with raster_layer_id for persisting main river flags
    pairing = InletOutletPairing(config=config, raster_layer_id=raster_layer.id)
    
    # Load DEM data
    pairing.load_dem(dem_path)
    
    # Load stream network from PostGIS - convert to GeoDataFrame
    print("Loading stream network from PostGIS...")
    streams_qs = StreamNetwork.objects.filter(raster_layer=raster_layer)
    if streams_qs.exists():
        import geopandas as gpd
        from shapely.geometry import shape
        from shapely import wkt
        
        streams_data = []
        for stream in streams_qs:
            # Convert Django GEOS geometry to Shapely
            geom = wkt.loads(stream.geometry.wkt)
            streams_data.append({
                'geometry': geom,
                'stream_order': stream.stream_order,
                'length_m': stream.length_m,
                'from_node': stream.from_node,
                'to_node': stream.to_node
            })
        
        pairing.stream_network = gpd.GeoDataFrame(streams_data, crs="EPSG:32651")
        pairing._build_stream_graph()
        pairing._build_spatial_index()
        print(f"Loaded {len(pairing.stream_network)} stream segments from PostGIS")
    else:
        print("ERROR: No stream network found in database!")
        return
    
    # Load watershed polygons for boundary validation (CRITICAL FIX)
    print("Loading watershed boundaries for validation...")
    pairing.load_watershed_polygons(raster_layer.id)
    if pairing.watershed_gdf is not None:
        print(f"Loaded {len(pairing.watershed_gdf)} watershed polygons")
        print("[OK] Sites will be constrained to watershed boundaries")
    else:
        print("WARNING: No watershed boundaries found - sites may fall outside basin")
    
    # Run pairing algorithm
    print("\nRunning pairing algorithm...")
    pairs = pairing.run_pairing()
    
    print(f"\nFound {len(pairs)} candidate site pairs")
    
    if len(pairs) == 0:
        print("No new sites found. Check stream network and DEM data.")
        return
    
    # Show head distribution
    heads = [p['head'] for p in pairs]
    print(f"\nHead distribution:")
    print(f"  - Min: {min(heads):.1f}m")
    print(f"  - Max: {max(heads):.1f}m")
    print(f"  - Mean: {sum(heads)/len(heads):.1f}m")
    
    # Count by head range
    micro = sum(1 for h in heads if h < 10)
    small = sum(1 for h in heads if 10 <= h < 50)
    medium = sum(1 for h in heads if 50 <= h < 100)
    large = sum(1 for h in heads if h >= 100)
    print(f"\nBy head category:")
    print(f"  - Micro (<10m): {micro}")
    print(f"  - Small (10-50m): {small}")
    print(f"  - Medium (50-100m): {medium}")
    print(f"  - Large (>100m): {large}")
    
    # Save to database
    response = input("\nSave new sites to database? [y/N]: ")
    if response.lower() != 'y':
        print("Not saved. Exiting.")
        return
    
    print("\nSaving sites...")
    hms_run_id = hms_run.id if hms_run else None
    site_point_ids, site_pair_ids = pairing.save_to_postgis(
        pairs, raster_layer.id, hms_run_id
    )
    
    print(f"Created {len(site_point_ids)} site points and {len(site_pair_ids)} site pairs")
    
    # Associate discharge with drainage-area weighted method
    if hms_run:
        print("\nAssociating discharge using drainage-area weighted method...")
        
        new_sites = SitePair.objects.filter(id__in=site_pair_ids)
        
        stats = associate_discharge_with_drainage_area(
            new_sites,
            hms_run.id,
            raster_layer,
            use_flow_duration=False,  # Use peak discharge
            exceedance_percentile=30.0
        )
        
        print(f"  - Updated: {stats['updated']}")
        print(f"  - Failed: {stats['failed']}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    all_sites = SitePair.objects.filter(raster_layer=raster_layer)
    sites_with_power = all_sites.filter(power__isnull=False, power__gt=0)
    
    print(f"Total sites: {all_sites.count()}")
    print(f"Sites with power: {sites_with_power.count()}")
    
    if sites_with_power.exists():
        from django.db.models import Sum, Avg, Min, Max
        
        stats = sites_with_power.aggregate(
            total_power=Sum('power'),
            avg_power=Avg('power'),
            min_power=Min('power'),
            max_power=Max('power'),
            total_discharge=Sum('discharge'),
            avg_head=Avg('head')
        )
        
        print(f"\nPower Statistics:")
        print(f"  - Total: {stats['total_power']/1000:.2f} MW")
        print(f"  - Average: {stats['avg_power']:.1f} kW")
        print(f"  - Range: {stats['min_power']:.1f} - {stats['max_power']:.1f} kW")
        print(f"  - Total Discharge: {stats['total_discharge']:.1f} m³/s")
        print(f"  - Average Head: {stats['avg_head']:.1f} m")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
