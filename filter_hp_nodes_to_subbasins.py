#!/usr/bin/env python
"""
Filter HP nodes to keep only those INSIDE subbasins

This script removes HP nodes that fall outside subbasin boundaries,
ensuring the visualization only shows nodes within the study area.
"""

import django
import os
import sys

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'HYDROPOWER_MAPPING.settings')
django.setup()

from hydropower.models import HPNode, WatershedPolygon, RasterLayer, SitePair, SitePoint
from django.db import transaction
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Filter HP nodes to keep only those inside subbasins"""
    
    # Get raster layer
    rl = RasterLayer.objects.filter(is_preprocessed=True).order_by('-preprocessing_date').first()
    if not rl:
        logger.error("No raster layer found")
        sys.exit(1)
    
    logger.info(f"Using RasterLayer ID: {rl.id}")
    
    # Get all HP nodes and watersheds
    all_nodes = HPNode.objects.filter(raster_layer=rl)
    watersheds = WatershedPolygon.objects.filter(raster_layer=rl)
    
    logger.info(f"Total HP nodes: {all_nodes.count()}")
    logger.info(f"Total watersheds: {watersheds.count()}")
    
    # Find nodes inside subbasins
    logger.info(f"\nChecking spatial containment...")
    nodes_to_keep = []
    nodes_to_delete = []
    
    for idx, node in enumerate(all_nodes):
        if (idx + 1) % 50 == 0:
            logger.info(f"  Processed {idx + 1}/{all_nodes.count()} nodes...")
        
        is_inside = False
        for ws in watersheds:
            if ws.geometry.contains(node.geometry):
                is_inside = True
                # Also associate the node with the watershed
                if node.watershed_id != ws.id:
                    node.watershed = ws
                    nodes_to_keep.append(node)
                break
        
        if not is_inside:
            nodes_to_delete.append(node.id)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Results:")
    logger.info(f"  HP nodes INSIDE subbasins: {len(nodes_to_keep)}")
    logger.info(f"  HP nodes OUTSIDE subbasins: {len(nodes_to_delete)}")
    logger.info(f"{'='*60}")
    
    if len(nodes_to_delete) == 0:
        logger.info("✓ All HP nodes are already inside subbasins!")
        return
    
    # Ask for confirmation
    logger.warning(f"\nThis will DELETE {len(nodes_to_delete)} HP nodes that are outside subbasins.")
    logger.warning(f"Associated site pairs will also be deleted.")
    
    response = input("\nProceed with deletion? (yes/no): ")
    if response.lower() != 'yes':
        logger.info("Operation cancelled.")
        return
    
    # Delete nodes outside subbasins
    logger.info(f"\nDeleting {len(nodes_to_delete)} HP nodes outside subbasins...")
    
    with transaction.atomic():
        # First, delete associated site pairs
        site_pairs = SitePair.objects.filter(raster_layer=rl)
        pairs_deleted = 0
        for pair in site_pairs:
            inlet_node_id = pair.inlet.site_id.replace('INLET_', '').replace('HP_', '')
            outlet_node_id = pair.outlet.site_id.replace('OUTLET_', '').replace('HP_', '')
            
            # Check if either inlet or outlet is being deleted
            if pair.inlet.id in nodes_to_delete or pair.outlet.id in nodes_to_delete:
                pair.delete()
                pairs_deleted += 1
        
        logger.info(f"  Deleted {pairs_deleted} site pairs")
        
        # Delete site points
        site_points = SitePoint.objects.filter(raster_layer=rl)
        points_deleted = site_points.delete()[0]
        logger.info(f"  Deleted {points_deleted} site points")
        
        # Delete HP nodes
        HPNode.objects.filter(id__in=nodes_to_delete).delete()
        logger.info(f"  Deleted {len(nodes_to_delete)} HP nodes")
        
        # Update watershed associations for remaining nodes
        if nodes_to_keep:
            for node in nodes_to_keep:
                node.save()
            logger.info(f"  Updated watershed associations for {len(nodes_to_keep)} nodes")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"✓ SUCCESS: Filtered HP nodes to subbasins only")
    logger.info(f"  Remaining HP nodes: {HPNode.objects.filter(raster_layer=rl).count()}")
    logger.info(f"{'='*60}")
    
    logger.info(f"\nNOTE: You need to regenerate site pairs using regenerate_site_pairing.py")

if __name__ == '__main__':
    main()
