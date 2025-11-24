"""
Management command to generate infrastructure layout for existing top-ranked site pairs.
Usage: python manage.py generate_infrastructure
"""

from django.core.management.base import BaseCommand
from django.db import transaction
from hydropower.models import SitePair
from hydropower.site_pairing import InletOutletPairing
import logging

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Generate infrastructure layout for existing top-ranked site pairs'

    def add_arguments(self, parser):
        parser.add_argument(
            '--top-n',
            type=int,
            default=5,
            help='Number of top-ranked sites to generate infrastructure for (default: 5)'
        )
        parser.add_argument(
            '--force',
            action='store_true',
            help='Regenerate infrastructure even if it already exists'
        )

    def handle(self, *args, **options):
        top_n = options['top_n']
        force = options['force']
        
        self.stdout.write(self.style.WARNING(f'Generating infrastructure for top {top_n} ranked sites...'))
        
        # Get top N ranked site pairs
        queryset = SitePair.objects.filter(
            rank__isnull=False,
            rank__lte=top_n,
            is_feasible=True
        ).order_by('rank')
        
        total = queryset.count()
        
        if total == 0:
            self.stdout.write(self.style.ERROR('No ranked site pairs found. Please run site pairing first.'))
            return
        
        self.stdout.write(f'Found {total} site pairs to process')
        
        updated_count = 0
        skipped_count = 0
        error_count = 0
        
        for site_pair in queryset:
            # Skip if infrastructure already exists (unless --force)
            if not force and site_pair.intake_basin_geom and site_pair.powerhouse_geom:
                self.stdout.write(self.style.WARNING(f'  Skipping {site_pair.pair_id} (rank {site_pair.rank}) - infrastructure already exists'))
                skipped_count += 1
                continue
            
            try:
                # Calculate infrastructure layout
                self.stdout.write(f'  Processing {site_pair.pair_id} (rank {site_pair.rank})...')
                
                infrastructure = InletOutletPairing.calculate_infrastructure_layout(site_pair)
                
                # Update site pair with infrastructure geometries
                with transaction.atomic():
                    site_pair.intake_basin_geom = infrastructure['intake_basin_geom']
                    site_pair.settling_basin_geom = infrastructure['settling_basin_geom']
                    site_pair.channel_geom = infrastructure['channel_geom']
                    site_pair.channel_length = infrastructure['channel_length']
                    site_pair.forebay_tank_geom = infrastructure['forebay_tank_geom']
                    site_pair.penstock_geom = infrastructure['penstock_geom']
                    site_pair.penstock_length = infrastructure['penstock_length']
                    site_pair.penstock_diameter = infrastructure['penstock_diameter']
                    site_pair.powerhouse_geom = infrastructure['powerhouse_geom']
                    site_pair.save()
                
                self.stdout.write(self.style.SUCCESS(f'    ✓ Generated infrastructure for {site_pair.pair_id}'))
                updated_count += 1
                
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'    ✗ Error processing {site_pair.pair_id}: {str(e)}'))
                logger.error(f"Error generating infrastructure for {site_pair.pair_id}: {str(e)}", exc_info=True)
                error_count += 1
        
        # Summary
        self.stdout.write('')
        self.stdout.write(self.style.SUCCESS('=== Summary ==='))
        self.stdout.write(f'Total site pairs: {total}')
        self.stdout.write(self.style.SUCCESS(f'Updated: {updated_count}'))
        if skipped_count > 0:
            self.stdout.write(self.style.WARNING(f'Skipped: {skipped_count}'))
        if error_count > 0:
            self.stdout.write(self.style.ERROR(f'Errors: {error_count}'))
        
        self.stdout.write('')
        self.stdout.write(self.style.SUCCESS('Infrastructure generation complete!'))
        self.stdout.write('Refresh your map view to see the infrastructure components.')
