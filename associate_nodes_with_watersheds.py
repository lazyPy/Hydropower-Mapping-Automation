#!/usr/bin/env python
"""
Generate HP nodes from main channel, but only in areas that intersect watersheds

This creates HP nodes along the main channel, but filters them to only include
nodes that fall within or near (100m buffer) watershed boundaries.
"""

import django
import os
import sys

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'HYDROPOWER_MAPPING.settings')
django.setup()

from hydropower.models import HPNode, WatershedPolygon, RasterLayer
from django.contrib.gis.geos import MultiPolygon
from django.db import transaction
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Associate HP nodes with watersheds using spatial buffer"""
    
    # Get raster layer
    rl = RasterLayer.objects.filter(is_preprocessed=True).order_by('-preprocessing_date').first()
    if not rl:
        logger.error("No raster layer found")
        sys.exit(1)
    
    logger.info(f"Using RasterLayer ID: {rl.id}")
    
    # Get all HP nodes and watersheds
    all_nodes = list(HPNode.objects.filter(raster_layer=rl))
    watersheds = list(WatershedPolygon.objects.filter(raster_layer=rl))
    
    logger.info(f"Total HP nodes: {len(all_nodes)}")
    logger.info(f"Total watersheds: {len(watersheds)}")
    
    # Create a buffered union of all watersheds (100m buffer to catch nearby rivers)
    logger.info(f"\nCreating buffered watershed union (100m buffer)...")
    buffered_watersheds = []
    for ws in watersheds:
        buffered = ws.geometry.buffer(100.0)  # 100m buffer
        buffered_watersheds.append(buffered)
    
    # Merge all buffered watersheds
    from django.contrib.gis.geos import GEOSGeometry
    combined_buffer = buffered_watersheds[0]
    for buffered in buffered_watersheds[1:]:
        combined_buffer = combined_buffer.union(buffered)
    
    logger.info(f"Created combined watershed buffer")
    
    # Check which nodes fall within the buffered area
    logger.info(f"\nChecking HP nodes against buffered watersheds...")
    nodes_inside = []
    nodes_outside = []
    
    for idx, node in enumerate(all_nodes):
        if (idx + 1) % 50 == 0:
            logger.info(f"  Processed {idx + 1}/{len(all_nodes)} nodes...")
        
        if combined_buffer.contains(node.geometry):
            # Find the closest watershed
            min_distance = float('inf')
            closest_ws = None
            for ws in watersheds:
                dist = ws.geometry.distance(node.geometry)
                if dist < min_distance:
                    min_distance = dist
                    closest_ws = ws
            
            nodes_inside.append((node, closest_ws))
        else:
            nodes_outside.append(node)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Results:")
    logger.info(f"  HP nodes INSIDE buffered watersheds: {len(nodes_inside)}")
    logger.info(f"  HP nodes OUTSIDE buffered watersheds: {len(nodes_outside)}")
    logger.info(f"{'='*60}")
    
    if len(nodes_outside) == 0:
        logger.info("✓ All HP nodes are within watershed buffer!")
        return
    
    # Update nodes with watershed associations
    logger.info(f"\nUpdating watershed associations for {len(nodes_inside)} nodes...")
    with transaction.atomic():
        for node, watershed in nodes_inside:
            node.watershed = watershed
            node.save(update_fields=['watershed'])
    
    logger.info(f"✓ Updated watershed associations")
    
    # Report nodes that will remain outside
    logger.info(f"\n{len(nodes_outside)} nodes are outside the watershed buffer zone.")
    logger.info(f"These represent river sections outside the study area watersheds.")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"✓ SUCCESS: Associated HP nodes with watersheds")
    logger.info(f"{'='*60}")

if __name__ == '__main__':
    main()
