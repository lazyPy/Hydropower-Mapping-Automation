"""
Run Complete Weir Search and Infrastructure Workflow

This script demonstrates the complete workflow from main channel site pairing
to weir search and infrastructure generation.

Workflow Steps:
1. Filter top 50 optimal main channel site pairs
2. Extract unique inlet points
3. Search for weir candidates around each inlet
4. Highlight the best weir location (rank 1) for each inlet
5. Generate infrastructure layout for best weir candidates

Usage:
    # Activate virtual environment first
    .\env\Scripts\Activate.ps1
    
    # Run complete workflow
    python run_weir_workflow.py
    
    # Run with custom parameters
    python run_weir_workflow.py --top_n=30 --search_radius=1000
"""

import os
os.environ['GDAL_LIBRARY_PATH'] = r'D:\Desktop\Hydro HEC-HMS\env\Lib\site-packages\osgeo\gdal.dll'
os.environ['DJANGO_SETTINGS_MODULE'] = 'HYDROPOWER_MAPPING.settings'

import django
django.setup()

from hydropower.models import RasterLayer, SitePair, WeirCandidate
from hydropower.main_channel_weir_search import (
    run_main_channel_weir_search,
    WeirSearchConfig
)
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Run complete weir search and infrastructure workflow')
    parser.add_argument('--raster_layer', type=int, default=None, help='RasterLayer ID (default: latest)')
    parser.add_argument('--dem_path', type=str, default=None, help='DEM path (default: auto-detect from raster layer)')
    parser.add_argument('--top_n', type=int, default=200, help='Top N site pairs to consider (default: 200, from reference)')
    parser.add_argument('--search_radius', type=float, default=1500.0, help='Weir search radius (m, from reference)')
    parser.add_argument('--min_distance', type=float, default=500.0, help='Minimum distance from inlet (m, from reference)')
    parser.add_argument('--elevation_tolerance', type=float, default=20.0, help='Elevation tolerance (m)')
    parser.add_argument('--cone_angle', type=float, default=50.0, help='Directional cone angle (degrees, from reference)')
    parser.add_argument('--max_candidates', type=int, default=3, help='Max candidates per inlet (from reference)')
    parser.add_argument('--no_infrastructure', action='store_true', help='Skip infrastructure generation')
    
    args = parser.parse_args()
    
    # Get raster layer
    if args.raster_layer:
        raster_layer = RasterLayer.objects.get(id=args.raster_layer)
    else:
        raster_layer = RasterLayer.objects.order_by('-dataset__uploaded_at').first()
        if not raster_layer:
            logger.error("No RasterLayer found in database")
            return
    
    # Auto-detect DEM path if not provided
    if not args.dem_path:
        args.dem_path = f"media/preprocessed/dem_{raster_layer.id}/filled_dem.tif"
        if not os.path.exists(args.dem_path):
            logger.error(f"DEM file not found at auto-detected path: {args.dem_path}")
            logger.info("Please specify --dem_path manually")
            return
    
    logger.info("=" * 80)
    logger.info("WEIR SEARCH AND INFRASTRUCTURE WORKFLOW")
    logger.info("=" * 80)
    logger.info(f"RasterLayer: {raster_layer.id} - {raster_layer.dataset.name}")
    logger.info(f"DEM Path: {args.dem_path}")
    logger.info(f"Top N Pairs: {args.top_n}")
    logger.info("")
    
    # Check if main channel pairs exist
    main_channel_pairs = SitePair.objects.filter(
        raster_layer=raster_layer,
        is_feasible=True,
        pair_id__startswith='HP_'
    ).order_by('rank')
    
    pair_count = main_channel_pairs.count()
    logger.info(f"Main Channel Site Pairs Available: {pair_count}")
    
    if pair_count == 0:
        logger.error("No main channel site pairs found. Run main channel workflow first.")
        return
    
    # Display top 10 site pairs
    logger.info("")
    logger.info("Top 10 Main Channel Site Pairs:")
    logger.info("-" * 80)
    for i, pair in enumerate(main_channel_pairs[:10], 1):
        logger.info(f"{i:2d}. {pair.pair_id:20s} | "
                   f"Rank: {pair.rank:3d} | "
                   f"Head: {pair.head:6.1f}m | "
                   f"Q: {pair.discharge:6.2f}m³/s | "
                   f"P: {pair.power:8.1f}kW")
    logger.info("")
    
    # Configure weir search
    config = WeirSearchConfig(
        search_radius_m=args.search_radius,
        min_distance_m=args.min_distance,
        elevation_tolerance_m=args.elevation_tolerance,
        cone_angle_deg=args.cone_angle,
        max_candidates_per_inlet=args.max_candidates
    )
    
    logger.info("Weir Search Configuration:")
    logger.info(f"  Search Radius: {config.search_radius_m}m")
    logger.info(f"  Min Distance: {config.min_distance_m}m")
    logger.info(f"  Elevation Tolerance: ±{config.elevation_tolerance_m}m")
    logger.info(f"  Cone Angle: {config.cone_angle_deg}°")
    logger.info(f"  Max Candidates per Inlet: {config.max_candidates_per_inlet}")
    logger.info(f"  Generate Infrastructure: {not args.no_infrastructure}")
    logger.info("")
    
    # Run weir search
    logger.info("-" * 80)
    logger.info("Starting Weir Search...")
    logger.info("-" * 80)
    
    results = run_main_channel_weir_search(
        raster_layer_id=raster_layer.id,
        dem_path=args.dem_path,
        top_n_pairs=args.top_n,
        config=config,
        generate_infrastructure=not args.no_infrastructure
    )
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("WEIR SEARCH RESULTS")
    logger.info("=" * 80)
    logger.info(f"Total Weir Candidates: {results['total_candidates']}")
    logger.info(f"Inlets Processed: {results['inlets_processed']}")
    logger.info(f"Best Weirs Identified: {len(results['best_weirs'])}")
    
    if not args.no_infrastructure:
        logger.info(f"Infrastructure Layouts Generated: {results['infrastructure_generated']}")
    
    logger.info("")
    
    # Display best weirs
    if results['best_weirs']:
        logger.info("Best Weir Candidates (Top 20):")
        logger.info("-" * 80)
        logger.info(f"{'#':<4}{'Inlet':<20}{'Score':<10}{'Distance':<12}{'Elev Diff':<12}{'Position'}")
        logger.info("-" * 80)
        
        for i, weir in enumerate(results['best_weirs'][:20], 1):
            logger.info(f"{i:<4}"
                       f"{weir['inlet_site_id']:<20}"
                       f"{weir['suitability_score']:<10.1f}"
                       f"{weir['distance_from_inlet']:<12.1f}"
                       f"{weir['elevation_difference']:<12.1f}"
                       f"({weir['x']:.1f}, {weir['y']:.1f})")
    
    logger.info("")
    
    # Database summary
    logger.info("=" * 80)
    logger.info("DATABASE SUMMARY")
    logger.info("=" * 80)
    
    total_weirs = WeirCandidate.objects.filter(raster_layer=raster_layer).count()
    best_weirs_db = WeirCandidate.objects.filter(raster_layer=raster_layer, rank_within_inlet=1).count()
    pairs_with_infrastructure = SitePair.objects.filter(
        raster_layer=raster_layer,
        intake_basin_geom__isnull=False
    ).count()
    
    logger.info(f"Total Weir Candidates in DB: {total_weirs}")
    logger.info(f"Best Weirs (Rank 1): {best_weirs_db}")
    logger.info(f"Site Pairs with Infrastructure: {pairs_with_infrastructure}")
    logger.info("")
    
    # Display sample infrastructure details
    if not args.no_infrastructure and pairs_with_infrastructure > 0:
        logger.info("Sample Infrastructure Details (Top 5):")
        logger.info("-" * 80)
        
        pairs_with_infra = SitePair.objects.filter(
            raster_layer=raster_layer,
            intake_basin_geom__isnull=False
        ).order_by('rank')[:5]
        
        for pair in pairs_with_infra:
            logger.info(f"\n{pair.pair_id} (Rank {pair.rank}):")
            logger.info(f"  Intake Basin: {pair.intake_basin_geom}")
            logger.info(f"  Settling Basin: {pair.settling_basin_geom}")
            logger.info(f"  Channel Length: {pair.channel_length:.1f}m")
            logger.info(f"  Forebay Tank: {pair.forebay_tank_geom}")
            logger.info(f"  Penstock Length: {pair.penstock_length:.1f}m")
            logger.info(f"  Penstock Diameter: {pair.penstock_diameter:.2f}m")
            logger.info(f"  Powerhouse: {pair.powerhouse_geom}")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("WORKFLOW COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Next Steps:")
    logger.info("1. View weir candidates on map: http://localhost:8000/")
    logger.info("2. Query best weirs: api/geojson/weir-candidates/?best_only=true")
    logger.info("3. View infrastructure: api/geojson/site-pairs/?top_n=5")
    logger.info("4. Export results for engineering design")
    logger.info("")


if __name__ == '__main__':
    main()
