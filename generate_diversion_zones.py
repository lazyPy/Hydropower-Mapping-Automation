#!/usr/bin/env python
r"""
Generate Diversion Zones for Existing Site Pairs

This script implements Step 2 of the hydropower site identification workflow:
- For each inlet point in existing site pairs, search the DEM
- Find cells with elevation similar to the inlet's elevation
- Generate polygon zones representing potential water diversion/storage areas

Usage:
    cd "d:\Desktop\Hydro HEC-HMS"
    .\env\Scripts\Activate.ps1
    python generate_diversion_zones.py [--search-radius 500] [--elevation-tolerance 2.0] [--limit 100]
"""

import os
import sys
import django
import argparse
import logging

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'HYDROPOWER_MAPPING.settings')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
django.setup()

from hydropower.diversion_zone import generate_diversion_zones, DiversionZoneConfig, DiversionZoneAnalyzer
from hydropower.models import RasterLayer, SitePair, DiversionZone

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Generate diversion zones for hydropower site pairs')
    parser.add_argument('--raster-layer', type=int, help='RasterLayer ID to process (default: latest)')
    parser.add_argument('--search-radius', type=float, default=500.0,
                        help='Search radius around inlet points (meters, default: 500)')
    parser.add_argument('--elevation-tolerance', type=float, default=2.0,
                        help='Elevation tolerance for similar elevation search (meters, default: 2.0)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Maximum number of zones to generate')
    parser.add_argument('--min-area', type=float, default=100.0,
                        help='Minimum zone area to keep (m², default: 100)')
    parser.add_argument('--apply-slope-filter', action='store_true',
                        help='Filter out areas with slope > 30 degrees')
    parser.add_argument('--clear-existing', action='store_true',
                        help='Clear existing diversion zones before generating new ones')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run analysis without saving to database')
    
    args = parser.parse_args()
    
    # Get raster layer
    if args.raster_layer:
        raster_layer = RasterLayer.objects.get(id=args.raster_layer)
    else:
        raster_layer = RasterLayer.objects.order_by('-id').first()
    
    if not raster_layer:
        logger.error("No RasterLayer found in database")
        sys.exit(1)
    
    logger.info(f"Processing RasterLayer: {raster_layer.id} - {raster_layer.dataset.name}")
    
    # Check for existing site pairs
    site_pair_count = SitePair.objects.filter(raster_layer=raster_layer, is_feasible=True).count()
    logger.info(f"Found {site_pair_count} feasible site pairs")
    
    if site_pair_count == 0:
        logger.warning("No site pairs found. Run site pairing first.")
        sys.exit(1)
    
    # Clear existing zones if requested
    if args.clear_existing:
        deleted_count, _ = DiversionZone.objects.filter(raster_layer=raster_layer).delete()
        logger.info(f"Cleared {deleted_count} existing diversion zones")
    
    # Configure and run analysis
    logger.info(f"Generating diversion zones with:")
    logger.info(f"  - Search radius: {args.search_radius}m")
    logger.info(f"  - Elevation tolerance: ±{args.elevation_tolerance}m")
    logger.info(f"  - Minimum area: {args.min_area} m²")
    logger.info(f"  - Slope filter: {args.apply_slope_filter}")
    
    if args.dry_run:
        logger.info("DRY RUN - Results will not be saved to database")
        
        # Run analysis without saving
        config = DiversionZoneConfig(
            search_radius=args.search_radius,
            elevation_tolerance=args.elevation_tolerance,
            min_area_m2=args.min_area,
            apply_slope_filter=args.apply_slope_filter,
            max_zones_per_run=args.limit or 1000
        )
        
        analyzer = DiversionZoneAnalyzer(config)
        zones = analyzer.analyze_site_pairs(raster_layer.id, limit=args.limit)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"DRY RUN RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Zones generated: {len(zones)}")
        
        if zones:
            total_area = sum(z['area_m2'] for z in zones)
            avg_area = total_area / len(zones)
            logger.info(f"Total area: {total_area/10000:.2f} hectares")
            logger.info(f"Average area: {avg_area:.2f} m² ({avg_area/10000:.4f} ha)")
            
            # Show top 5 zones
            logger.info(f"\nTop 5 zones by area:")
            for i, zone in enumerate(sorted(zones, key=lambda z: -z['area_m2'])[:5]):
                logger.info(f"  {i+1}. {zone['site_id']}: {zone['area_hectares']:.2f} ha @ {zone['reference_elevation']:.1f}m")
    else:
        # Run full analysis with database save
        result = generate_diversion_zones(
            raster_layer_id=raster_layer.id,
            search_radius=args.search_radius,
            elevation_tolerance=args.elevation_tolerance,
            limit=args.limit
        )
        
        logger.info(f"\n{'='*60}")
        logger.info(f"DIVERSION ZONE GENERATION COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Zones generated: {result['zones_generated']}")
        logger.info(f"Zones saved: {result['zones_saved']}")
        logger.info(f"Total area: {result['total_area_hectares']:.2f} hectares")
        logger.info(f"Average area: {result['average_area_hectares']:.4f} hectares")
        logger.info(f"Processing time: {result['processing_time_seconds']:.2f} seconds")
    
    logger.info("\nDone!")


if __name__ == '__main__':
    main()
