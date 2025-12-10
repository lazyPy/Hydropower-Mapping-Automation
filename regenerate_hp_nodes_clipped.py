#!/usr/bin/env python
"""
Generate HP nodes from Nominal-Channel clipped to Claveria Subbasins

This ensures HP nodes are only created where the main channel intersects
with the study area subbasins, solving the "nodes outside subbasins" issue.
"""

import django
import os
import sys

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'HYDROPOWER_MAPPING.settings')
django.setup()

from hydropower.hp_node_generation import HPNodeGenerator
from hydropower.models import RasterLayer, HPNode, SitePair, SitePoint, VectorLayer
import geopandas as gpd
from shapely.ops import linemerge
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Generate HP nodes from channel clipped to subbasins"""
    
    # Get raster layer
    rl = RasterLayer.objects.filter(is_preprocessed=True).order_by('-preprocessing_date').first()
    if not rl:
        logger.error("No raster layer found")
        sys.exit(1)
    
    logger.info(f"Using RasterLayer ID: {rl.id}")
    
    # File paths
    subbasin_shp = r"d:\Desktop\Hydro HEC-HMS\INPUT DATA - Claveria\WATERSHED DATA\Bridge & River\Claveria Subbasin.shp"
    main_channel_gpkg = r"d:\Desktop\Hydro HEC-HMS\INPUT DATA - Claveria\Nominal-Channel.gpkg"
    
    # Get DEM path
    from django.conf import settings
    dem_path = None
    if rl.filled_dem_path:
        relative_path = str(rl.filled_dem_path)
        dem_path = os.path.join(settings.MEDIA_ROOT, relative_path) if not os.path.isabs(relative_path) else relative_path
    
    if not dem_path or not os.path.exists(dem_path):
        logger.error(f"DEM not found: {dem_path}")
        sys.exit(1)
    
    logger.info(f"DEM path: {dem_path}")
    
    # Load subbasins and channel
    logger.info(f"\nLoading Claveria Subbasin shapefile...")
    subbasin = gpd.read_file(subbasin_shp)
    logger.info(f"  {len(subbasin)} subbasins, Total area: {subbasin.geometry.area.sum() / 1_000_000:.2f} km²")
    
    logger.info(f"\nLoading Nominal-Channel...")
    channel = gpd.read_file(main_channel_gpkg)
    logger.info(f"  {len(channel)} segments, Total length: {channel.geometry.length.sum() / 1000:.2f} km")
    
    # Ensure same CRS
    if subbasin.crs != channel.crs:
        logger.info(f"  Reprojecting subbasin from {subbasin.crs} to {channel.crs}")
        subbasin = subbasin.to_crs(channel.crs)
    
    # Clip channel to subbasins
    logger.info(f"\nClipping channel to subbasin boundaries...")
    subbasin_union = subbasin.union_all() if hasattr(subbasin, 'union_all') else subbasin.geometry.unary_union
    
    clipped_channel = channel[channel.intersects(subbasin_union)].copy()
    clipped_channel['geometry'] = clipped_channel.intersection(subbasin_union)
    
    # Remove empty geometries
    clipped_channel = clipped_channel[~clipped_channel.geometry.is_empty]
    
    clipped_length = clipped_channel.geometry.length.sum() / 1000
    logger.info(f"  Clipped channel length: {clipped_length:.2f} km")
    logger.info(f"  Percentage retained: {clipped_length / (channel.geometry.length.sum() / 1000) * 100:.1f}%")
    
    # Save clipped channel temporarily
    import tempfile
    temp_gpkg = os.path.join(tempfile.gettempdir(), 'clipped_channel.gpkg')
    clipped_channel.to_file(temp_gpkg, driver='GPKG')
    logger.info(f"  Saved clipped channel to: {temp_gpkg}")
    
    # Delete existing HP nodes and site pairs
    logger.info(f"\nDeleting existing HP nodes and site pairs...")
    HPNode.objects.filter(raster_layer=rl).delete()
    SitePair.objects.filter(raster_layer=rl).delete()
    SitePoint.objects.filter(raster_layer=rl).delete()
    logger.info(f"  Deleted old data")
    
    # Generate HP nodes from clipped channel
    logger.info(f"\n{'='*60}")
    logger.info(f"Generating HP nodes from clipped channel...")
    logger.info(f"{'='*60}\n")
    
    generator = HPNodeGenerator(sampling_interval_m=100.0)  # 100m intervals (matches QGIS NODE_SPACING_M)
    generator.load_dem(dem_path)
    
    # Load clipped channel
    clipped_gdf = gpd.read_file(temp_gpkg)
    
    # Merge into line(s)
    merged_geom = generator.merge_channel_lines(clipped_gdf)
    
    # Sample points
    all_sampled_points = []
    cumulative_distance = 0.0
    
    if merged_geom.geom_type == 'LineString':
        all_sampled_points = generator.sample_points_along_line(merged_geom)
    elif merged_geom.geom_type == 'MultiLineString':
        logger.info(f"Sampling {len(merged_geom.geoms)} channel segments")
        for seg_idx, segment in enumerate(merged_geom.geoms):
            seg_points = generator.sample_points_along_line(segment)
            adjusted_points = [(pt, dist + cumulative_distance) for pt, dist in seg_points]
            all_sampled_points.extend(adjusted_points)
            cumulative_distance += segment.length
    
    # Extract elevations and create HP nodes
    hp_nodes = []
    skipped_count = 0
    
    for idx, (point, distance) in enumerate(all_sampled_points):
        elevation = generator.extract_elevation(point.x, point.y)
        
        if elevation is None:
            skipped_count += 1
            continue
        
        node_id = f"HP_{idx+1:03d}"
        chainage_km = distance / 1000.0
        
        hp_node = {
            'node_id': node_id,
            'geometry': point,
            'elevation': elevation,
            'distance_along_channel': distance,
            'chainage': chainage_km,
            'raster_layer_id': rl.id,
            'vector_layer_id': None,
            'source_vector_name': 'Nominal-Channel (clipped to Claveria Subbasins)',
            'sampling_interval': 100.0,
        }
        
        hp_nodes.append(hp_node)
    
    logger.info(f"\nGenerated {len(hp_nodes)} HP nodes (skipped {skipped_count})")
    
    # Save to database
    generator.save_to_database(hp_nodes)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"✓ SUCCESS: Generated {len(hp_nodes)} HP nodes")
    logger.info(f"  Coverage: {clipped_length:.2f} km of channel")
    logger.info(f"  All nodes are within Claveria Subbasin boundaries")
    logger.info(f"{'='*60}")
    
    logger.info(f"\nNext step: Run 'python regenerate_site_pairing.py' to create site pairs")
    
    # Clean up temp file
    os.remove(temp_gpkg)

if __name__ == '__main__':
    main()
