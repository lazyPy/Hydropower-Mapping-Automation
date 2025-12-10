#!/usr/bin/env python
"""
Import Claveria Subbasins into database and associate HP nodes with them
"""

import django
import os
import sys

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'HYDROPOWER_MAPPING.settings')
django.setup()

from hydropower.models import WatershedPolygon, HPNode, RasterLayer
from django.contrib.gis.geos import GEOSGeometry
from django.db import transaction
import geopandas as gpd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Get raster layer
    rl = RasterLayer.objects.filter(is_preprocessed=True).order_by('-preprocessing_date').first()
    logger.info(f"RasterLayer ID: {rl.id}")
    
    # Load Claveria Subbasin shapefile
    subbasin_shp = r"d:\Desktop\Hydro HEC-HMS\INPUT DATA - Claveria\WATERSHED DATA\Bridge & River\Claveria Subbasin.shp"
    logger.info(f"\nLoading Claveria Subbasin shapefile...")
    subbasin_gdf = gpd.read_file(subbasin_shp)
    
    # Ensure CRS is EPSG:32651
    if subbasin_gdf.crs != 'EPSG:32651':
        logger.info(f"Reprojecting from {subbasin_gdf.crs} to EPSG:32651")
        subbasin_gdf = subbasin_gdf.to_crs('EPSG:32651')
    
    logger.info(f"Loaded {len(subbasin_gdf)} Claveria Subbasins")
    
    # Delete old DEM-delineated watersheds
    old_count = WatershedPolygon.objects.filter(raster_layer=rl).count()
    logger.info(f"\nDeleting {old_count} old DEM-delineated watersheds...")
    WatershedPolygon.objects.filter(raster_layer=rl).delete()
    
    # Import Claveria Subbasins
    logger.info(f"\nImporting Claveria Subbasins...")
    with transaction.atomic():
        for idx, row in subbasin_gdf.iterrows():
            # Convert geometry to GEOS
            # Convert MultiPolygon to Polygon if needed (take largest polygon)
            geom = row.geometry
            if geom.geom_type == 'MultiPolygon':
                # Take the largest polygon
                geom = max(geom.geoms, key=lambda p: p.area)
            
            geom_wkt = geom.wkt
            geos_geom = GEOSGeometry(geom_wkt, srid=32651)
            
            # Calculate area
            area_m2 = row.geometry.area
            area_km2 = area_m2 / 1_000_000
            perimeter_m = row.geometry.length
            
            # Create WatershedPolygon
            ws = WatershedPolygon.objects.create(
                raster_layer=rl,
                watershed_id=idx + 1,
                geometry=geos_geom,
                area_m2=area_m2,
                area_km2=area_km2,
                perimeter_m=perimeter_m,
                stream_threshold=1000,  # Default value
            )
            
            logger.info(f"  Created Watershed {ws.watershed_id}: {row['name']}, {area_km2:.2f} km²")
    
    # Update watershed count in raster layer
    rl.watershed_count = len(subbasin_gdf)
    rl.save()
    
    # Associate HP nodes with watersheds
    logger.info(f"\nAssociating HP nodes with Claveria Subbasins...")
    nodes = HPNode.objects.filter(raster_layer=rl)
    watersheds = WatershedPolygon.objects.filter(raster_layer=rl)
    
    nodes_associated = 0
    with transaction.atomic():
        for node in nodes:
            for ws in watersheds:
                if ws.geometry.contains(node.geometry):
                    node.watershed = ws
                    node.save(update_fields=['watershed'])
                    nodes_associated += 1
                    break
    
    logger.info(f"  Associated {nodes_associated}/{nodes.count()} HP nodes with watersheds")
    
    # Verify
    logger.info(f"\n{'='*60}")
    logger.info(f"✓ SUCCESS: Imported Claveria Subbasins")
    logger.info(f"  Total watersheds in database: {WatershedPolygon.objects.filter(raster_layer=rl).count()}")
    logger.info(f"  HP nodes with watershed: {HPNode.objects.filter(raster_layer=rl, watershed__isnull=False).count()}")
    logger.info(f"{'='*60}")

if __name__ == '__main__':
    main()
