#!/usr/bin/env python
"""
Run watershed-based HP node generation

This script generates HP nodes along ALL river segments within each watershed,
providing comprehensive coverage for hydropower site assessment.
"""

import django
import os
import sys

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'HYDROPOWER_MAPPING.settings')
django.setup()

from hydropower.watershed_hp_generation import generate_watershed_hp_nodes
from hydropower.models import RasterLayer
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Run watershed-based HP node generation"""
    
    # Get the most recent raster layer
    raster_layer = RasterLayer.objects.filter(
        is_preprocessed=True,
        watershed_delineated=True
    ).order_by('-preprocessing_date').first()
    
    if not raster_layer:
        logger.error("No preprocessed raster layer found. Run DEM preprocessing first.")
        sys.exit(1)
    
    logger.info(f"Using RasterLayer ID: {raster_layer.id}")
    logger.info(f"  Dataset: {raster_layer.dataset.name if raster_layer.dataset else 'N/A'}")
    
    # Get DEM path (use filled DEM or smoothed DEM)
    from django.conf import settings
    import os
    
    dem_path = None
    if raster_layer.filled_dem_path:
        relative_path = str(raster_layer.filled_dem_path)
        dem_path = os.path.join(settings.MEDIA_ROOT, relative_path) if not os.path.isabs(relative_path) else relative_path
    elif raster_layer.smoothed_dem_path:
        relative_path = str(raster_layer.smoothed_dem_path)
        dem_path = os.path.join(settings.MEDIA_ROOT, relative_path) if not os.path.isabs(relative_path) else relative_path
    else:
        logger.error("No preprocessed DEM found")
        sys.exit(1)
    
    if not os.path.exists(dem_path):
        logger.error(f"DEM file not found: {dem_path}")
        sys.exit(1)
    
    logger.info(f"  DEM path: {dem_path}")
    
    # Configuration
    sampling_interval = 200.0  # 200m intervals for dense coverage
    min_area = 0.005  # Only watersheds >= 0.005 km² (5000 m²) - very small for this dataset
    max_watersheds = 10  # Process top 10 watersheds for testing
    
    logger.info(f"\nConfiguration:")
    logger.info(f"  Sampling interval: {sampling_interval}m")
    logger.info(f"  Min watershed area: {min_area} km²")
    logger.info(f"  Max watersheds: {max_watersheds if max_watersheds else 'ALL'}")
    
    # Run generation
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting watershed-based HP node generation...")
    logger.info(f"{'='*60}\n")
    
    try:
        total_nodes = generate_watershed_hp_nodes(
            dem_path=dem_path,
            raster_layer_id=raster_layer.id,
            sampling_interval_m=sampling_interval,
            min_watershed_area_km2=min_area,
            max_watersheds=max_watersheds
        )
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✓ SUCCESS: Generated {total_nodes} HP nodes across watersheds")
        logger.info(f"{'='*60}\n")
        
    except Exception as e:
        logger.error(f"\n{'='*60}")
        logger.error(f"✗ ERROR: {e}")
        logger.error(f"{'='*60}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
