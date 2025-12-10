"""
Django Management Command: Run Weir Search and Infrastructure Generation

This command implements the complete weir search workflow:
1. Load top 50 optimal main channel site pairs
2. Extract unique inlet points
3. Search for candidate weir locations around each inlet
4. Highlight the best weir for each inlet
5. Generate infrastructure layout for best weir candidates

Usage:
    python manage.py run_weir_search --raster_layer=<id> --dem_path=<path>
    python manage.py run_weir_search --raster_layer=1 --dem_path="media/preprocessed/DEM_filled.tif" --top_n=50
    python manage.py run_weir_search --raster_layer=1 --no-infrastructure  # Skip infrastructure generation
"""

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from hydropower.models import RasterLayer, WeirCandidate, SitePair
from hydropower.main_channel_weir_search import (
    run_main_channel_weir_search,
    WeirSearchConfig
)
import logging
import os

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Run weir search for top main channel site pairs and generate infrastructure'

    def add_arguments(self, parser):
        parser.add_argument(
            '--raster_layer',
            type=int,
            required=True,
            help='RasterLayer ID for the DEM to process'
        )
        parser.add_argument(
            '--dem_path',
            type=str,
            help='Path to DEM GeoTIFF (default: media/preprocessed/DEM_filled.tif)'
        )
        parser.add_argument(
            '--top_n',
            type=int,
            default=50,
            help='Number of top site pairs to consider (default: 50)'
        )
        parser.add_argument(
            '--search_radius',
            type=float,
            default=500.0,
            help='Search radius around inlet in meters (default: 500m)'
        )
        parser.add_argument(
            '--min_distance',
            type=float,
            default=100.0,
            help='Minimum distance from inlet in meters (default: 100m)'
        )
        parser.add_argument(
            '--elevation_tolerance',
            type=float,
            default=20.0,
            help='Elevation tolerance in meters (default: 20m)'
        )
        parser.add_argument(
            '--cone_angle',
            type=float,
            default=90.0,
            help='Directional cone half-angle in degrees (default: 90°)'
        )
        parser.add_argument(
            '--max_candidates',
            type=int,
            default=10,
            help='Maximum candidates per inlet (default: 10)'
        )
        parser.add_argument(
            '--no-infrastructure',
            action='store_true',
            help='Skip infrastructure generation (only find weir candidates)'
        )

    def handle(self, *args, **options):
        raster_layer_id = options['raster_layer']
        top_n = options['top_n']
        generate_infrastructure = not options['no_infrastructure']
        
        # Get DEM path
        dem_path = options.get('dem_path')
        if not dem_path:
            dem_path = os.path.join('media', 'preprocessed', 'DEM_filled.tif')
        
        # Validate raster layer exists
        try:
            raster_layer = RasterLayer.objects.get(id=raster_layer_id)
        except RasterLayer.DoesNotExist:
            raise CommandError(f'RasterLayer with ID {raster_layer_id} does not exist')
        
        # Validate DEM file exists
        if not os.path.exists(dem_path):
            raise CommandError(f'DEM file not found: {dem_path}')
        
        # Check if site pairs exist
        pair_count = SitePair.objects.filter(
            raster_layer=raster_layer,
            is_feasible=True,
            pair_id__startswith='HP_'
        ).count()
        
        if pair_count == 0:
            raise CommandError(
                f'No main channel site pairs found for RasterLayer {raster_layer_id}. '
                'Run main channel workflow first.'
            )
        
        self.stdout.write(self.style.SUCCESS('=== Weir Search and Infrastructure Generation ==='))
        self.stdout.write(f'RasterLayer ID: {raster_layer_id}')
        self.stdout.write(f'DEM Path: {dem_path}')
        self.stdout.write(f'Top N Pairs: {top_n}')
        self.stdout.write(f'Available Main Channel Pairs: {pair_count}')
        self.stdout.write(f'Generate Infrastructure: {generate_infrastructure}')
        self.stdout.write('')
        
        # Configure weir search
        config = WeirSearchConfig(
            search_radius_m=options['search_radius'],
            min_distance_m=options['min_distance'],
            elevation_tolerance_m=options['elevation_tolerance'],
            cone_angle_deg=options['cone_angle'],
            max_candidates_per_inlet=options['max_candidates']
        )
        
        self.stdout.write(self.style.NOTICE('Configuration:'))
        self.stdout.write(f'  Search Radius: {config.search_radius_m}m')
        self.stdout.write(f'  Min Distance: {config.min_distance_m}m')
        self.stdout.write(f'  Elevation Tolerance: ±{config.elevation_tolerance_m}m')
        self.stdout.write(f'  Cone Angle: {config.cone_angle_deg}°')
        self.stdout.write(f'  Max Candidates per Inlet: {config.max_candidates_per_inlet}')
        self.stdout.write('')
        
        # Run weir search
        self.stdout.write(self.style.NOTICE('Step 1: Running weir search...'))
        
        try:
            results = run_main_channel_weir_search(
                raster_layer_id=raster_layer_id,
                dem_path=dem_path,
                top_n_pairs=top_n,
                config=config,
                generate_infrastructure=generate_infrastructure
            )
        except Exception as e:
            raise CommandError(f'Weir search failed: {e}')
        
        # Display results
        self.stdout.write('')
        self.stdout.write(self.style.SUCCESS('=== Weir Search Results ==='))
        self.stdout.write(f'Total Weir Candidates: {results["total_candidates"]}')
        self.stdout.write(f'Inlets Processed: {results["inlets_processed"]}')
        self.stdout.write(f'Best Weirs Identified: {len(results["best_weirs"])}')
        
        if generate_infrastructure:
            self.stdout.write(f'Infrastructure Layouts Generated: {results["infrastructure_generated"]}')
        
        self.stdout.write('')
        
        # Display best weir details
        if results['best_weirs']:
            self.stdout.write(self.style.NOTICE('Best Weir Candidates (Top 10):'))
            self.stdout.write('')
            
            for i, weir in enumerate(results['best_weirs'][:10], 1):
                self.stdout.write(
                    f'{i:2d}. Inlet: {weir["inlet_site_id"]:15s} | '
                    f'Score: {weir["suitability_score"]:5.1f} | '
                    f'Distance: {weir["distance_from_inlet"]:6.1f}m | '
                    f'Elev Diff: {weir["elevation_difference"]:+6.1f}m'
                )
        
        self.stdout.write('')
        
        # Summary statistics
        self.stdout.write(self.style.SUCCESS('=== Database Summary ==='))
        
        total_weir_candidates = WeirCandidate.objects.filter(
            raster_layer=raster_layer
        ).count()
        
        best_weirs_db = WeirCandidate.objects.filter(
            raster_layer=raster_layer,
            rank_within_inlet=1
        ).count()
        
        pairs_with_infrastructure = SitePair.objects.filter(
            raster_layer=raster_layer,
            intake_basin_geom__isnull=False
        ).count()
        
        self.stdout.write(f'Total Weir Candidates in DB: {total_weir_candidates}')
        self.stdout.write(f'Best Weirs (Rank 1): {best_weirs_db}')
        self.stdout.write(f'Site Pairs with Infrastructure: {pairs_with_infrastructure}')
        self.stdout.write('')
        
        # Next steps
        self.stdout.write(self.style.SUCCESS('=== Next Steps ==='))
        self.stdout.write('1. View weir candidates on the map')
        self.stdout.write('2. Inspect infrastructure layouts')
        self.stdout.write('3. Export results to GeoJSON/CSV')
        self.stdout.write('4. Run engineering design calculations')
        self.stdout.write('')
        
        self.stdout.write(self.style.SUCCESS('✓ Weir search completed successfully!'))
