#!/usr/bin/env python
"""
Test script for main river focus in site pairing algorithm.

This script:
1. Loads the existing stream network from database
2. Identifies main river segments using longest path algorithm
3. Reruns site pairing with main_river_only=True
4. Compares results with full network approach

Usage:
    .\env\Scripts\Activate.ps1; python test_main_river_focus.py
"""

import os
import sys
import django
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'HYDROPOWER_MAPPING.settings')
django.setup()

from django.conf import settings
from hydropower.models import RasterLayer, StreamNetwork, SitePair, SitePoint
from hydropower.site_pairing import InletOutletPairing, PairingConfig
import geopandas as gpd
from shapely import wkt
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_main_river_identification():
    """Test main river identification logic"""
    print("\n" + "="*80)
    print("TEST 1: Main River Identification")
    print("="*80)
    
    # Get the latest RasterLayer
    raster = RasterLayer.objects.order_by('-dataset__uploaded_at').first()
    if not raster:
        print("ERROR: No RasterLayer found in database")
        return
    
    print(f"\nUsing RasterLayer ID: {raster.id}")
    
    # Load stream network
    streams = StreamNetwork.objects.filter(raster_layer=raster)
    print(f"Total stream segments: {streams.count()}")
    
    # Convert to GeoDataFrame
    stream_data = []
    for stream in streams:
        stream_data.append({
            'geometry': wkt.loads(stream.geometry.wkt),
            'stream_order': stream.stream_order,
            'length_m': stream.length_m,
            'is_main_river': stream.is_main_river,
        })
    
    streams_gdf = gpd.GeoDataFrame(stream_data, crs='EPSG:32651')
    
    # Export to temporary file
    with tempfile.NamedTemporaryFile(suffix='.gpkg', delete=False) as tmp:
        temp_gpkg = tmp.name
    streams_gdf.to_file(temp_gpkg, driver='GPKG')
    
    # Test main river identification with main_river_only=True
    config = PairingConfig(
        main_river_only=True,
        min_main_river_order=3
    )
    pairing = InletOutletPairing(config=config)
    
    # Load DEM and streams
    dem_path = str(settings.MEDIA_ROOT / raster.filled_dem_path)
    pairing.load_dem(dem_path)
    pairing.load_stream_network(temp_gpkg)
    
    # Check results
    main_river_count = pairing.stream_network['is_main_river'].sum()
    main_river_length = pairing.stream_network[pairing.stream_network['is_main_river']]['length_m'].sum()
    total_length = pairing.stream_network['length_m'].sum()
    
    print(f"\nMain River Statistics:")
    print(f"  Main river segments: {main_river_count} / {len(pairing.stream_network)} ({main_river_count/len(pairing.stream_network)*100:.1f}%)")
    print(f"  Main river length: {main_river_length/1000:.2f} km / {total_length/1000:.2f} km ({main_river_length/total_length*100:.1f}%)")
    
    if 'stream_order' in pairing.stream_network.columns:
        order_stats = pairing.stream_network.groupby('stream_order')['is_main_river'].agg(['count', 'sum'])
        print(f"\nMain River by Stream Order:")
        print(order_stats)
    
    # Clean up
    import os
    try:
        os.remove(temp_gpkg)
    except:
        pass
    
    return main_river_count > 0


def test_site_pairing_comparison():
    """Compare site pairing with and without main river focus"""
    print("\n" + "="*80)
    print("TEST 2: Site Pairing Comparison (Full Network vs Main River)")
    print("="*80)
    
    # Get the latest RasterLayer
    raster = RasterLayer.objects.order_by('-dataset__uploaded_at').first()
    if not raster:
        print("ERROR: No RasterLayer found in database")
        return
    
    # Load stream network
    streams = StreamNetwork.objects.filter(raster_layer=raster)
    stream_data = []
    for stream in streams:
        stream_data.append({
            'geometry': wkt.loads(stream.geometry.wkt),
            'stream_order': stream.stream_order,
            'length_m': stream.length_m,
            'is_main_river': stream.is_main_river,
        })
    
    streams_gdf = gpd.GeoDataFrame(stream_data, crs='EPSG:32651')
    
    # Export to temporary file
    with tempfile.NamedTemporaryFile(suffix='.gpkg', delete=False) as tmp:
        temp_gpkg = tmp.name
    streams_gdf.to_file(temp_gpkg, driver='GPKG')
    
    dem_path = str(settings.MEDIA_ROOT / raster.filled_dem_path)
    
    # Test 1: Full network (main_river_only=False)
    print("\n--- Test 1: Full Network (all streams) ---")
    config_full = PairingConfig(
        min_head=5.0,
        max_head=200.0,
        min_river_distance=50.0,
        max_river_distance=2000.0,
        min_stream_order=1,
        max_candidates_per_type=50,
        max_outlets_per_inlet=3,
        main_river_only=False  # Include all streams
    )
    
    pairing_full = InletOutletPairing(config=config_full)
    pairing_full.load_dem(dem_path)
    pairing_full.load_stream_network(temp_gpkg)
    
    inlets_full = pairing_full.identify_inlet_candidates()
    outlets_full = pairing_full.identify_outlet_candidates()
    
    print(f"Full Network:")
    print(f"  Inlet candidates: {len(inlets_full)}")
    print(f"  Outlet candidates: {len(outlets_full)}")
    
    # Test 2: Main river only (main_river_only=True)
    print("\n--- Test 2: Main River Only (practical sites) ---")
    config_main = PairingConfig(
        min_head=5.0,
        max_head=200.0,
        min_river_distance=50.0,
        max_river_distance=2000.0,
        min_stream_order=1,
        max_candidates_per_type=50,
        max_outlets_per_inlet=3,
        main_river_only=True,  # Main river only
        min_main_river_order=3
    )
    
    pairing_main = InletOutletPairing(config=config_main)
    pairing_main.load_dem(dem_path)
    pairing_main.load_stream_network(temp_gpkg)
    
    inlets_main = pairing_main.identify_inlet_candidates()
    outlets_main = pairing_main.identify_outlet_candidates()
    
    print(f"Main River Only:")
    print(f"  Inlet candidates: {len(inlets_main)}")
    print(f"  Outlet candidates: {len(outlets_main)}")
    
    # Comparison
    print(f"\n--- Comparison ---")
    print(f"Reduction in inlet candidates: {len(inlets_full) - len(inlets_main)} ({(1 - len(inlets_main)/len(inlets_full))*100:.1f}%)")
    print(f"Reduction in outlet candidates: {len(outlets_full) - len(outlets_main)} ({(1 - len(outlets_main)/len(outlets_full))*100:.1f}%)")
    print(f"\nThis focuses the analysis on {len(inlets_main)} practical inlet locations")
    print(f"on the main river stem where real hydropower plants would be built.")
    
    # Clean up
    import os
    try:
        os.remove(temp_gpkg)
    except:
        pass


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("MAIN RIVER FOCUS - TEST SUITE")
    print("="*80)
    
    try:
        # Test 1: Main river identification
        success = test_main_river_identification()
        if not success:
            print("\nERROR: Main river identification failed")
            return
        
        # Test 2: Site pairing comparison
        test_site_pairing_comparison()
        
        print("\n" + "="*80)
        print("ALL TESTS COMPLETED")
        print("="*80)
        print("\nRecommendation:")
        print("  ✓ Use main_river_only=True in production (practical sites)")
        print("  ✓ Use main_river_only=False for theoretical potential mapping")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
