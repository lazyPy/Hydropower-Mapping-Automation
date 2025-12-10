"""
Main Channel Hydropower Workflow Script

This script orchestrates the complete main channel hydropower site assessment workflow:
1. Generate HP nodes along main channel from vector layer
2. Perform systematic site pairing on HP nodes (identify top 50 optimal pairs)
3. Search for weir/intake locations near inlet points
4. Generate infrastructure layouts

Usage:
    python run_main_channel_workflow.py

Configuration:
    - Modify paths and parameters in the script below
    - Ensure database is properly configured
    - Make sure DEM and main channel vector are available

Author: Hydropower Mapping System
Date: December 2025
"""

import os
import sys
import django
import logging
from pathlib import Path

# Setup Django environment
sys.path.append(str(Path(__file__).parent))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'HYDROPOWER_MAPPING.settings')
django.setup()

from hydropower.hp_node_generation import generate_hp_nodes_from_main_channel
from hydropower.main_channel_pairing import run_main_channel_site_pairing, PairingConstraints
from hydropower.main_channel_weir_search import run_main_channel_weir_search, WeirSearchConfig
from hydropower.models import RasterLayer, VectorLayer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
MAIN_CHANNEL_PATH = r"INPUT DATA - Claveria\Nominal-Channel.gpkg"
MAIN_CHANNEL_LAYER_NAME = "main_channel"  # Layer name in GeoPackage

# Get DEM from database (use latest or specify ID)
RASTER_LAYER_ID = None  # Set to specific ID or None for latest

# HP Node Generation Parameters - ALIGNED WITH REFERENCE (reference_here.py)
SAMPLING_INTERVAL_M = 100.0  # Sample every 100m along channel (NODE_SPACING_M from reference)

# Site Pairing Parameters - ALIGNED WITH REFERENCE (reference_here.py)
PAIRING_DISCHARGE_M3S = 7.0  # Design discharge (m³/s) - CONSTANT_Q_CMS from reference
PAIRING_CONSTRAINTS = PairingConstraints(
    min_head_m=20.0,          # MIN_HEAD_PAIR_M from reference
    max_head_m=500.0,
    min_distance_m=20.0,
    max_distance_m=2000.0,    # PAIR_SEARCH_RADIUS_M from reference
    min_power_kw=5.0,
    efficiency=0.8,           # EFFICIENCY from reference
    top_n_pairs=200           # TOP_N_PAIRS from reference
)

# Weir Search Parameters - ALIGNED WITH REFERENCE (reference_here.py)
WEIR_CONFIG = WeirSearchConfig(
    search_radius_m=1500.0,         # WEIR_SEARCH_RADIUS_M from reference
    min_distance_m=500.0,           # WEIR_MIN_DIST_M from reference
    elevation_tolerance_m=20.0,     # WEIR_ELEV_TOLERANCE_M from reference
    max_candidates_per_inlet=3,     # WEIR_MAX_CANDIDATES_PER_IN from reference
    cone_angle_deg=50.0,            # WEIR_ANGLE_LIMIT_DEG from reference
    pixel_sampling_factor=4
)


# =============================================================================
# WORKFLOW FUNCTIONS
# =============================================================================

def get_raster_layer():
    """Get raster layer from database."""
    if RASTER_LAYER_ID:
        raster = RasterLayer.objects.get(id=RASTER_LAYER_ID)
        logger.info(f"Using specified RasterLayer ID {RASTER_LAYER_ID}: {raster.dataset.name}")
    else:
        raster = RasterLayer.objects.order_by('-id').first()
        if not raster:
            raise ValueError("No RasterLayer found in database. Upload a DEM first.")
        logger.info(f"Using latest RasterLayer ID {raster.id}: {raster.dataset.name}")
    
    return raster


def step1_generate_hp_nodes(raster_layer):
    """
    Step 1: Generate HP nodes along main channel.
    
    Samples points at regular intervals along the main river channel.
    """
    logger.info("=" * 80)
    logger.info("STEP 1: GENERATE HP NODES")
    logger.info("=" * 80)
    
    # Check if main channel file exists
    if not Path(MAIN_CHANNEL_PATH).exists():
        raise FileNotFoundError(f"Main channel vector not found: {MAIN_CHANNEL_PATH}")
    
    # Get DEM path
    dem_path = raster_layer.dataset.file.path
    
    # Generate HP nodes
    node_count = generate_hp_nodes_from_main_channel(
        main_channel_path=MAIN_CHANNEL_PATH,
        dem_path=dem_path,
        raster_layer_id=raster_layer.id,
        vector_layer_id=None,
        layer_name=MAIN_CHANNEL_LAYER_NAME,
        sampling_interval_m=SAMPLING_INTERVAL_M
    )
    
    logger.info(f"✓ Generated {node_count} HP nodes")
    return node_count


def step2_site_pairing(raster_layer):
    """
    Step 2: Systematic site pairing on HP nodes.
    
    Pairs HP nodes to identify optimal inlet-outlet combinations.
    Returns top N pairs ranked by multi-criteria scoring.
    """
    logger.info("=" * 80)
    logger.info("STEP 2: MAIN CHANNEL SITE PAIRING")
    logger.info("=" * 80)
    
    pair_count = run_main_channel_site_pairing(
        raster_layer_id=raster_layer.id,
        discharge_m3s=PAIRING_DISCHARGE_M3S,
        constraints=PAIRING_CONSTRAINTS
    )
    
    logger.info(f"✓ Generated {pair_count} site pairs (top {PAIRING_CONSTRAINTS.top_n_pairs})")
    return pair_count


def step3_weir_search(raster_layer):
    """
    Step 3: Search for weir/intake locations.
    
    For each inlet point from top-ranked pairs, searches DEM for
    candidate weir/diversion locations with hydraulic constraints.
    """
    logger.info("=" * 80)
    logger.info("STEP 3: WEIR SEARCH")
    logger.info("=" * 80)
    
    # Get DEM path
    dem_path = raster_layer.dataset.file.path
    
    candidate_count = run_main_channel_weir_search(
        raster_layer_id=raster_layer.id,
        dem_path=dem_path,
        top_n_pairs=PAIRING_CONSTRAINTS.top_n_pairs,
        config=WEIR_CONFIG
    )
    
    logger.info(f"✓ Generated {candidate_count} weir candidates")
    return candidate_count


def step4_infrastructure_generation(raster_layer):
    """
    Step 4: Generate infrastructure layouts.
    
    For top-ranked site pairs, generates run-of-river hydropower
    infrastructure components (intake, channel, penstock, powerhouse).
    """
    logger.info("=" * 80)
    logger.info("STEP 4: INFRASTRUCTURE GENERATION")
    logger.info("=" * 80)
    
    # This step is already implemented in the existing infrastructure generation
    # which runs during site pairing if needed, or can be run separately
    
    from hydropower.models import SitePair
    
    # Count site pairs with infrastructure
    pairs_with_infra = SitePair.objects.filter(
        raster_layer=raster_layer,
        pair_id__startswith='HP_',
        powerhouse_geom__isnull=False
    ).count()
    
    logger.info(f"✓ Infrastructure layouts available for {pairs_with_infra} site pairs")
    logger.info("  (Infrastructure is auto-generated during visualization)")
    
    return pairs_with_infra


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def run_complete_workflow():
    """
    Run the complete main channel hydropower workflow.
    
    Steps:
    1. Generate HP nodes from main channel vector
    2. Perform systematic site pairing (top 50 optimal pairs)
    3. Search for weir/intake locations near inlets
    4. Generate infrastructure layouts
    """
    logger.info("\n")
    logger.info("╔" + "═" * 78 + "╗")
    logger.info("║" + " " * 20 + "MAIN CHANNEL HYDROPOWER WORKFLOW" + " " * 26 + "║")
    logger.info("╚" + "═" * 78 + "╝")
    logger.info("\n")
    
    try:
        # Get raster layer
        raster_layer = get_raster_layer()
        logger.info(f"\nUsing DEM: {raster_layer.dataset.name}")
        logger.info(f"Raster ID: {raster_layer.id}")
        logger.info(f"Dimensions: {raster_layer.width} x {raster_layer.height}")
        logger.info(f"Resolution: {raster_layer.pixel_size_x:.2f}m x {raster_layer.pixel_size_y:.2f}m")
        logger.info("")
        
        # Run workflow steps
        results = {}
        
        # Step 1: Generate HP nodes
        results['hp_nodes'] = step1_generate_hp_nodes(raster_layer)
        
        # Step 2: Site pairing
        results['site_pairs'] = step2_site_pairing(raster_layer)
        
        # Step 3: Weir search
        results['weir_candidates'] = step3_weir_search(raster_layer)
        
        # Step 4: Infrastructure generation
        results['infrastructure'] = step4_infrastructure_generation(raster_layer)
        
        # Summary
        logger.info("\n")
        logger.info("╔" + "═" * 78 + "╗")
        logger.info("║" + " " * 28 + "WORKFLOW COMPLETE" + " " * 33 + "║")
        logger.info("╚" + "═" * 78 + "╝")
        logger.info("\n")
        logger.info("RESULTS SUMMARY:")
        logger.info(f"  • HP Nodes Generated:      {results['hp_nodes']}")
        logger.info(f"  • Site Pairs Identified:   {results['site_pairs']}")
        logger.info(f"  • Weir Candidates Found:   {results['weir_candidates']}")
        logger.info(f"  • Infrastructure Layouts:  {results['infrastructure']}")
        logger.info("\n")
        logger.info("VIEW RESULTS:")
        logger.info("  1. Open map interface: http://localhost:8000/")
        logger.info("  2. Use Processing Layers sidebar to toggle:")
        logger.info("     - HP Node Generation (Step 1)")
        logger.info("     - Main Channel Site Pairing (Step 2)")
        logger.info("     - Weir Search (Step 3)")
        logger.info("\n")
        logger.info("NEXT STEPS:")
        logger.info("  • Review top-ranked site pairs on map")
        logger.info("  • Examine weir candidate locations (rank #1 = best)")
        logger.info("  • Export results for detailed analysis")
        logger.info("\n")
        
        return results
        
    except Exception as e:
        logger.error(f"\n❌ WORKFLOW FAILED: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    # Verify Django is properly configured
    try:
        from django.conf import settings
        logger.info(f"Django configured: {settings.DATABASES['default']['NAME']}")
    except Exception as e:
        logger.error(f"Django configuration error: {e}")
        sys.exit(1)
    
    # Run workflow
    try:
        results = run_complete_workflow()
        sys.exit(0)
    except KeyboardInterrupt:
        logger.warning("\nWorkflow interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nWorkflow failed: {e}")
        sys.exit(1)
