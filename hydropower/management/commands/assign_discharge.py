"""
Django management command to assign discharge to existing site pairs.

Usage:
    python manage.py assign_discharge
"""

from django.core.management.base import BaseCommand
from django.db.models import Min, Max, Avg
from hydropower.models import SitePair, HMSRun, RasterLayer
from hydropower.discharge_association import DischargeAssociator, DischargeConfig
import logging

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Assign discharge and power to site pairs that are missing this data'

    def add_arguments(self, parser):
        parser.add_argument(
            '--hms-run-id',
            type=int,
            help='HMS Run ID to use for discharge data (default: use most recent)',
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=100,
            help='Number of pairs to process per batch (default: 100)',
        )
        parser.add_argument(
            '--extraction-method',
            type=str,
            default='peak',
            choices=['peak', 'average', 'median'],
            help='Discharge extraction method (default: peak)',
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('\n' + '='*60))
        self.stdout.write(self.style.SUCCESS('DISCHARGE ASSIGNMENT TO EXISTING SITE PAIRS'))
        self.stdout.write(self.style.SUCCESS('='*60 + '\n'))

        # Get all site pairs without discharge OR with discharge=0 (failed previous run)
        pairs_without_discharge = SitePair.objects.filter(
            discharge__isnull=True
        ) | SitePair.objects.filter(discharge=0)
        total_count = pairs_without_discharge.count()

        if total_count == 0:
            self.stdout.write(self.style.SUCCESS("All site pairs already have discharge data!"))
            return

        self.stdout.write(f"Found {total_count} site pairs without discharge data")

        # Get HMS run
        hms_run_id = options.get('hms_run_id')
        if hms_run_id:
            try:
                hms_run = HMSRun.objects.get(id=hms_run_id)
            except HMSRun.DoesNotExist:
                self.stdout.write(self.style.ERROR(f"HMS Run ID {hms_run_id} not found"))
                return
        else:
            hms_runs = HMSRun.objects.all()
            if not hms_runs.exists():
                self.stdout.write(self.style.ERROR("No HMS runs available. Cannot assign discharge."))
                return
            hms_run = hms_runs.order_by('-created_at').first()

        self.stdout.write(f"Using HMS run: {hms_run.dataset.name} (ID: {hms_run.id})")

        # Create discharge associator
        extraction_method = options.get('extraction_method', 'peak')
        config = DischargeConfig(
            extraction_method=extraction_method,
            assignment_strategy='scaled'
        )
        associator = DischargeAssociator(config=config)

        # Create discharge summary
        self.stdout.write("Creating discharge summary from HMS time series data...")
        discharge_summary = associator.create_discharge_summary(hms_run.id)

        if not discharge_summary:
            self.stdout.write(self.style.ERROR("Failed to create discharge summary"))
            return

        self.stdout.write(f"Discharge summary: {len(discharge_summary)} stations")
        self.stdout.write(f"Range: {min(discharge_summary.values()):.2f} - {max(discharge_summary.values()):.2f} m³/s\n")

        # Process pairs in batches
        batch_size = options.get('batch_size', 100)
        updated_count = 0
        failed_count = 0

        for i in range(0, total_count, batch_size):
            batch = pairs_without_discharge[i:i+batch_size]
            progress = f"[{i+1:4d}-{min(i+batch_size, total_count):4d} of {total_count}]"
            self.stdout.write(f"{progress} Processing batch {i//batch_size + 1}...", ending='')

            batch_updated = 0
            for pair in batch:
                try:
                    # Always assign HMS run first (even if pair doesn't have one yet)
                    if pair.hms_run is None:
                        pair.hms_run = hms_run
                    
                    # Calculate scaled discharge based on head
                    discharge = associator.calculate_scaled_discharge(pair.head, discharge_summary)

                    if discharge is not None:
                        pair.discharge = discharge
                        pair.calculate_power()  # Calculate power from discharge and head
                        pair.save()
                        updated_count += 1
                        batch_updated += 1
                    else:
                        logger.warning(f"Discharge calculation returned None for pair {pair.pair_id} (head={pair.head:.2f}m)")
                        failed_count += 1

                except Exception as e:
                    logger.error(f"Error processing pair {pair.pair_id}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    failed_count += 1

            self.stdout.write(self.style.SUCCESS(f" [OK] {batch_updated} updated"))

        # Final statistics
        self.stdout.write(self.style.SUCCESS('\n' + '='*60))
        self.stdout.write(self.style.SUCCESS('SUMMARY'))
        self.stdout.write(self.style.SUCCESS('='*60))
        self.stdout.write(f"Total processed:       {total_count}")
        self.stdout.write(self.style.SUCCESS(f"Successfully updated:  {updated_count}"))
        if failed_count > 0:
            self.stdout.write(self.style.WARNING(f"Failed:                {failed_count}"))

        # Database statistics
        all_pairs = SitePair.objects.all()
        pairs_with_discharge = SitePair.objects.filter(discharge__isnull=False)
        coverage = pairs_with_discharge.count() / all_pairs.count() * 100 if all_pairs.count() > 0 else 0

        self.stdout.write(f"\nDatabase statistics:")
        self.stdout.write(f"  Total site pairs:      {all_pairs.count()}")
        self.stdout.write(f"  Pairs with discharge:  {pairs_with_discharge.count()}")
        self.stdout.write(self.style.SUCCESS(f"  Coverage:              {coverage:.1f}%"))

        if pairs_with_discharge.exists():
            stats = pairs_with_discharge.aggregate(
                min_q=Min('discharge'),
                max_q=Max('discharge'),
                avg_q=Avg('discharge'),
                min_p=Min('power'),
                max_p=Max('power'),
                avg_p=Avg('power')
            )

            self.stdout.write(f"\n  Discharge: {stats['min_q']:.2f} - {stats['max_q']:.2f} m³/s (avg: {stats['avg_q']:.2f})")
            self.stdout.write(f"  Power:     {stats['min_p']:.1f} - {stats['max_p']:.1f} kW (avg: {stats['avg_p']:.1f})")

            total_power_mw = stats['avg_p'] * pairs_with_discharge.count() / 1000
            self.stdout.write(self.style.SUCCESS(f"  Total potential: {total_power_mw:.2f} MW"))

        self.stdout.write(self.style.SUCCESS('\n' + '='*60))
        self.stdout.write(self.style.SUCCESS('[OK] Discharge assignment complete!'))
        self.stdout.write(self.style.SUCCESS('='*60 + '\n'))
