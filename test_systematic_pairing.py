"""
Test script for enhanced systematic IO pairing algorithm

This script demonstrates the new features:
1. Node densification at regular intervals along channels
2. Pair-specific discharge calculation using flow accumulation
3. Systematic nested-loop pairing (inlet i, outlet j where j > i)
4. Chainage-based distance calculation
5. Deduplication and power-based ranking
"""

import os
import sys
import django
from pathlib import Path

# Setup Django
sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'HYDROPOWER_MAPPING.settings')
django.setup()

from hydropower.site_pairing import InletOutletPairing, PairingConfig
from hydropower.models import RasterLayer, VectorLayer, SitePoint, SitePair, StreamNetwork
from shapely.geometry import Point
import geopandas as gpd
from django.conf import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_systematic_pairing():
    """Test the systematic IO pairing with node densification"""
    
    # Configuration
    config = PairingConfig(
        # Enable node densification and systematic pairing
        use_node_densification=True,
        node_spacing=10.0,  # 10m node spacing (adjusted for short segments)
        
        # Enable flow accumulation-based discharge
        use_flow_accumulation=True,
        q_outlet_design=7.0,  # Design Q at outlet (m³/s)
        constant_q_fallback=7.749,  # Fallback if FA fails
        
        # Head and distance constraints (tuned for dataset)
        min_head=5.0,  # Minimum 5m head (lowered for short segments)
        max_head=500.0,
        min_river_distance=10.0,  # 10m minimum (segments are 10-14m)
        max_river_distance=500.0,  # 500m max search radius
        
        # Efficiency
        efficiency=0.7,  # 70% efficiency (conservative for micro-hydro)
        
        # Performance limits
        max_stream_segments=5000,
        max_candidates_per_type=2000
    )
    
    # Initialize pairing algorithm
    pairing = InletOutletPairing(config)
    
    # Load most recent raster layer
    try:
        raster_layer = RasterLayer.objects.filter(
            watershed_delineated=True
        ).order_by('-id').first()
        
        if not raster_layer:
            logger.error("No RasterLayer found with completed watershed delineation")
            return
        
        logger.info(f"Using RasterLayer ID: {raster_layer.id}")
        logger.info(f"  DEM: {raster_layer.filled_dem_path or raster_layer.dataset.file.path}")
        logger.info(f"  Flow Accum: {raster_layer.flow_accumulation_path}")
        
    except Exception as e:
        logger.error(f"Error loading RasterLayer: {e}")
        return
    
    # Check for required files
    from django.conf import settings
    dem_path = raster_layer.filled_dem_path or raster_layer.dataset.file.path
    if not os.path.isabs(dem_path):
        dem_path = os.path.join(settings.MEDIA_ROOT, dem_path)
    if not os.path.exists(dem_path):
        logger.error(f"DEM not found: {dem_path}")
        return
    
    flowacc_path = raster_layer.flow_accumulation_path
    if flowacc_path and not os.path.isabs(flowacc_path):
        flowacc_path = os.path.join(settings.MEDIA_ROOT, flowacc_path)
    
    if not raster_layer.flow_accumulation_path or not os.path.exists(flowacc_path):
        logger.error(f"Flow accumulation raster not found: {flowacc_path}")
        logger.info("Will use constant discharge fallback")
        config.use_flow_accumulation = False
    
    # Export stream network to temporary shapefile
    try:
        streams = StreamNetwork.objects.filter(raster_layer=raster_layer)
        
        if not streams.exists():
            logger.error("No stream network found in database")
            return
        
        logger.info(f"Found {streams.count()} stream segments in database")
        
        # Export to GeoDataFrame
        from shapely import wkt
        stream_data = []
        for stream in streams:
            # Convert GEOS geometry to Shapely
            geom_wkt = stream.geometry.wkt
            shapely_geom = wkt.loads(geom_wkt)
            stream_data.append({
                'geometry': shapely_geom,
                'stream_order': stream.stream_order,
                'length_m': stream.length_m,
                'from_node': stream.from_node,
                'to_node': stream.to_node
            })
        
        if not stream_data:
            logger.error("No stream segments found")
            return
        
        # Create temporary shapefile
        import tempfile
        temp_dir = tempfile.mkdtemp()
        stream_path = os.path.join(temp_dir, 'streams.shp')
        
        stream_gdf = gpd.GeoDataFrame(stream_data, crs='EPSG:32651')
        stream_gdf.to_file(stream_path)
        
        logger.info(f"Exported stream network to: {stream_path}")
        logger.info(f"Stream orders range: {stream_gdf['stream_order'].min()} - {stream_gdf['stream_order'].max()}")
            
    except Exception as e:
        logger.error(f"Error loading stream network: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load DEM
    logger.info("Loading DEM...")
    pairing.load_dem(dem_path)
    
    # Load flow accumulation with outlet calibration
    if config.use_flow_accumulation and raster_layer.flow_accumulation_path:
        logger.info("Loading flow accumulation raster...")
        
        # Try to find outlet point for calibration
        outlet_point = None
        # Note: Outlet vector detection would need a proper field to identify outlet layers
        # For now, skip outlet calibration or manually specify coordinates
        
        # Manual outlet coordinates (optional - comment out if not available)
        # outlet_point = Point(123456.0, 987654.0)  # Replace with actual coordinates
        
        if False:  # Disabled - no layer_type field in VectorLayer
            try:
                outlet_shp_path = outlet_vector.shapefile_path
                if not os.path.isabs(outlet_shp_path):
                    outlet_shp_path = os.path.join(settings.MEDIA_ROOT, outlet_shp_path)
                outlet_gdf = gpd.read_file(outlet_shp_path)
                if len(outlet_gdf) > 0:
                    geom = outlet_gdf.iloc[0].geometry
                    outlet_point = Point(geom.x, geom.y)
                    logger.info(f"Found outlet point: ({outlet_point.x:.1f}, {outlet_point.y:.1f})")
            except Exception as e:
                logger.warning(f"Error loading outlet: {e}")
        
        pairing.load_flow_accumulation(
            flowacc_path,
            outlet_point=outlet_point
        )
    
    # Load stream network
    logger.info("Loading stream network...")
    pairing.load_stream_network(stream_path)
    
    # Load watershed boundaries
    logger.info("Loading watershed boundaries...")
    pairing.load_watershed_polygons(raster_layer.id)
    
    # Run pairing algorithm
    logger.info("="*60)
    logger.info("STARTING SYSTEMATIC IO PAIRING")
    logger.info("="*60)
    
    pairs = pairing.run_pairing()
    
    # Print results
    logger.info("="*60)
    logger.info("PAIRING RESULTS")
    logger.info("="*60)
    logger.info(f"Total site pairs found: {len(pairs)}")
    
    if len(pairs) > 0:
        logger.info("\nTop 10 site pairs by power:")
        logger.info(f"{'Rank':<6} {'Head(m)':<10} {'Dist(m)':<10} {'Q(m³/s)':<10} {'Power(kW)':<12} {'Score':<10}")
        logger.info("-"*60)
        
        for i, pair in enumerate(pairs[:10]):
            logger.info(
                f"{pair['rank']:<6} "
                f"{pair['head']:<10.1f} "
                f"{pair['river_distance']:<10.1f} "
                f"{pair.get('discharge', 0):<10.3f} "
                f"{pair.get('power_kw', 0):<12.1f} "
                f"{pair['score']:<10.2f}"
            )
        
        # Save to database
        logger.info("\n" + "="*60)
        logger.info("SAVING TO DATABASE")
        logger.info("="*60)
        
        try:
            # Note: Not deleting existing pairs to avoid foreign key issues
            # If you need to re-run, manually clear the database first
            logger.info("Saving new site pairs (not deleting existing ones)...")
            
            # Save new pairs
            site_point_ids, site_pair_ids = pairing.save_to_postgis(
                pairs,
                raster_layer_id=raster_layer.id
            )
            
            logger.info(f"Successfully saved {len(site_pair_ids)} site pairs to database")
            logger.info(f"  SitePoint IDs: {len(site_point_ids)}")
            logger.info(f"  SitePair IDs: {len(site_pair_ids)}")
            
            # Update raster layer
            raster_layer.site_pair_count = len(site_pair_ids)
            raster_layer.save()
            
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        logger.warning("No site pairs found!")
    
    logger.info("\n" + "="*60)
    logger.info("TEST COMPLETE")
    logger.info("="*60)


if __name__ == "__main__":
    test_systematic_pairing()
