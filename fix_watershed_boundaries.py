"""
Fix Watershed Boundary Violations for Existing Site Pairs

This script identifies and removes site pairs where the inlet or outlet
points fall outside the watershed boundary. This fixes the Claveria issue
where HPP sites were placed outside the river basin.

Run with: .\\env\\Scripts\\Activate.ps1; python fix_watershed_boundaries.py
"""

import os
import sys
import django
from pathlib import Path

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'HYDROPOWER_MAPPING.settings')
django.setup()

from django.conf import settings
from hydropower.models import RasterLayer, SitePair, SitePoint, WatershedPolygon
from shapely import wkt
from shapely.geometry import Point
from shapely.ops import unary_union
import geopandas as gpd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_point_in_watershed(point_geom, watershed_union):
    """
    Check if a point is within the watershed boundary.
    
    Args:
        point_geom: Django GEOS Point
        watershed_union: Shapely Polygon (unified watershed boundary)
    
    Returns:
        True if point is within watershed, False otherwise
    """
    # Convert GEOS point to Shapely
    shapely_point = wkt.loads(point_geom.wkt)
    
    # Check intersection with watershed
    return watershed_union.contains(shapely_point) or watershed_union.intersects(shapely_point)


def load_watershed_boundary(raster_layer):
    """
    Load watershed boundary, prioritizing subbasin shapefile over micro-watersheds.
    
    Returns:
        Shapely Polygon (unified watershed boundary), or None if not found
    """
    # First, try to load subbasin shapefile (preferred - actual basin boundary)
    vector_cache = Path(settings.MEDIA_ROOT) / 'vector_cache'
    if vector_cache.exists():
        # Look for subbasin geojson files
        subbasin_files = list(vector_cache.glob('subbasin_*.geojson'))
        if subbasin_files:
            # Use the first subbasin file found
            subbasin_path = subbasin_files[0]
            logger.info(f"Loading subbasin shapefile: {subbasin_path}")
            try:
                gdf = gpd.read_file(subbasin_path)
                if not gdf.empty:
                    watershed_union = unary_union(gdf.geometry.tolist())
                    area_km2 = watershed_union.area / 1_000_000
                    logger.info(f"✓ Using subbasin shapefile ({area_km2:.2f} km²)")
                    return watershed_union
            except Exception as e:
                logger.warning(f"Failed to load subbasin shapefile: {e}")
    
    # Fall back to database WatershedPolygon
    watersheds = WatershedPolygon.objects.filter(raster_layer=raster_layer)
    if watersheds.exists():
        logger.info(f"Loading {watersheds.count()} micro-watershed polygons from database")
        polygons = [wkt.loads(w.geometry.wkt) for w in watersheds]
        watershed_union = unary_union(polygons)
        area_km2 = watershed_union.area / 1_000_000
        logger.info(f"✓ Using micro-watersheds ({area_km2:.2f} km²)")
        return watershed_union
    
    return None


def main():
    """Identify and fix site pairs outside watershed boundaries."""
    
    print("=" * 70)
    print("Fix Watershed Boundary Violations")
    print("=" * 70)
    
    # Get all raster layers with site pairs
    raster_layers = RasterLayer.objects.filter(site_pairs__isnull=False).distinct()
    
    if not raster_layers.exists():
        print("No raster layers with site pairs found.")
        return
    
    for raster_layer in raster_layers:
        dataset_name = raster_layer.dataset.name if raster_layer.dataset else "Unknown"
        print(f"\nProcessing: {dataset_name} (ID: {raster_layer.id})")
        
        # Load watershed boundary (prefers subbasin shapefile)
        watershed_union = load_watershed_boundary(raster_layer)
        
        if watershed_union is None:
            print(f"  WARNING: No watershed boundary found, skipping")
            continue
        
        area_km2 = watershed_union.area / 1_000_000
        print(f"  Watershed area: {area_km2:.2f} km²")
        
        # Get all site pairs
        site_pairs = SitePair.objects.filter(raster_layer=raster_layer)
        print(f"  Site pairs: {site_pairs.count()}")
        
        # Check each site pair
        invalid_pairs = []
        
        for pair in site_pairs:
            inlet_valid = check_point_in_watershed(pair.inlet.geometry, watershed_union)
            outlet_valid = check_point_in_watershed(pair.outlet.geometry, watershed_union)
            
            if not (inlet_valid and outlet_valid):
                invalid_pairs.append({
                    'pair': pair,
                    'inlet_valid': inlet_valid,
                    'outlet_valid': outlet_valid
                })
        
        if not invalid_pairs:
            print(f"  ✓ All site pairs are within watershed boundaries")
            continue
        
        print(f"  ✗ Found {len(invalid_pairs)} invalid site pairs:")
        
        for item in invalid_pairs[:10]:  # Show first 10
            pair = item['pair']
            inlet_status = "✓" if item['inlet_valid'] else "✗"
            outlet_status = "✓" if item['outlet_valid'] else "✗"
            print(f"    - {pair.pair_id}: Inlet {inlet_status}, Outlet {outlet_status}")
        
        if len(invalid_pairs) > 10:
            print(f"    ... and {len(invalid_pairs) - 10} more")
        
        # Ask for confirmation
        response = input(f"\n  Remove {len(invalid_pairs)} invalid site pairs? [y/N]: ")
        if response.lower() != 'y':
            print("  Skipped.")
            continue
        
        # Update meets_watershed_constraint flag
        for item in invalid_pairs:
            pair = item['pair']
            pair.meets_watershed_constraint = False
            pair.is_feasible = False
            pair.save()
        
        print(f"  ✓ Marked {len(invalid_pairs)} site pairs as infeasible (meets_watershed_constraint=False)")
        
        # Optionally delete them
        response = input(f"  Delete these site pairs from database? [y/N]: ")
        if response.lower() == 'y':
            # Get site point IDs to delete
            inlet_ids = [item['pair'].inlet.id for item in invalid_pairs]
            outlet_ids = [item['pair'].outlet.id for item in invalid_pairs]
            
            # Delete site pairs (cascades to site points)
            for item in invalid_pairs:
                item['pair'].delete()
            
            print(f"  ✓ Deleted {len(invalid_pairs)} site pairs and their points")
        else:
            print(f"  Site pairs marked as infeasible but not deleted")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    # Final statistics
    for raster_layer in raster_layers:
        dataset_name = raster_layer.dataset.name if raster_layer.dataset else "Unknown"
        all_pairs = SitePair.objects.filter(raster_layer=raster_layer)
        feasible_pairs = all_pairs.filter(is_feasible=True)
        infeasible_pairs = all_pairs.filter(is_feasible=False)
        outside_watershed = all_pairs.filter(meets_watershed_constraint=False)
        
        print(f"\n{dataset_name}:")
        print(f"  Total site pairs: {all_pairs.count()}")
        print(f"  Feasible: {feasible_pairs.count()}")
        print(f"  Infeasible: {infeasible_pairs.count()}")
        print(f"  Outside watershed: {outside_watershed.count()}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
