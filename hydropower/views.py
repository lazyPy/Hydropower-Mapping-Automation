"""
Simplified views for map-only application.
All data processing is done via console script (INPUT DATA/process_data.py).
This module contains only:
- Map view (main page)
- GeoJSON API endpoints for map layers
- MVT tile endpoints for large datasets
"""

from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.cache import cache_page
from django.db import models as django_models
from django.conf import settings
import logging

from .models import RasterLayer, VectorLayer

logger = logging.getLogger(__name__)


def map_view(request):
    """
    Display interactive map with site pairs, watersheds, and stream networks.
    
    Features:
    - Leaflet.js map with multiple base layers (OSM, Satellite, Topo)
    - Center on Matiao River Basin, Pantukan, Davao de Oro (7.178256°N, 126.002943°E)
    - CRS: Data in EPSG:32651 (UTM Zone 51N), display in EPSG:3857 (Web Mercator)
    - Filter sidebar for site pairs (head, discharge, power)
    - Layer toggles for watersheds, streams, subbasins, bridges
    - Marker clustering for performance with large datasets
    """
    from hydropower.models import RasterLayer
    
    # Get raster layer ID from query string, or use the most recent one
    raster_layer_id = request.GET.get('raster_layer', None)
    if not raster_layer_id:
        # Default to the most recent raster layer
        latest_raster = RasterLayer.objects.order_by('-dataset__uploaded_at').first()
        if latest_raster:
            raster_layer_id = latest_raster.id
    
    context = {
        'raster_layer_id': raster_layer_id
    }
    return render(request, 'map_view.html', context)


# ============================================================================
# GeoJSON API Endpoints for Map Layers
# ============================================================================

# Removed @cache_page decorator - infrastructure data changes frequently during development
def geojson_site_pairs(request):
    """
    GeoJSON endpoint for site pairs (inlet-outlet lines and points).
    
    Returns FeatureCollection with:
    - LineString geometry (inlet-outlet connection)
    - Inlet/outlet point coordinates in properties
    - Head, discharge, power attributes
    - Infrastructure components (intake, channel, penstock, powerhouse) for top-ranked sites
    - CRS transformation: EPSG:32651 → EPSG:4326 (for Leaflet)
    
    Query parameters:
    - min_head: Minimum head (m)
    - min_discharge: Minimum discharge (m³/s)
    - min_power: Minimum power (kW)
    - scenario: Filter by return period/scenario
    - raster_layer: Filter by source DEM ID
    - bbox: Bounding box filter (minLng,minLat,maxLng,maxLat)
    - top_n: Show only top N ranked sites (e.g., top_n=5 for best 5 sites)
    """
    from django.contrib.gis.geos import GEOSGeometry
    from pyproj import Transformer
    import json
    
    try:
        # Get query parameters
        min_head = request.GET.get('min_head', 0)
        min_discharge = request.GET.get('min_discharge', 0)
        min_power = request.GET.get('min_power', 0)
        scenario = request.GET.get('scenario')
        raster_layer_id = request.GET.get('raster_layer')
        bbox = request.GET.get('bbox')  # Format: "minLng,minLat,maxLng,maxLat"
        top_n = request.GET.get('top_n')  # Show only top N ranked sites
        
        # Build query
        from .models import SitePair
        queryset = SitePair.objects.select_related('inlet', 'outlet', 'hms_run').all()
        
        # Apply filters
        if min_head:
            queryset = queryset.filter(head__gte=float(min_head))
        if min_discharge:
            queryset = queryset.filter(discharge__gte=float(min_discharge))
        if min_power:
            queryset = queryset.filter(power__gte=float(min_power))
        if scenario:
            queryset = queryset.filter(return_period=scenario)
        if raster_layer_id:
            queryset = queryset.filter(raster_layer_id=int(raster_layer_id))
        
        # Filter by top N ranked sites (e.g., top_n=5 for best 5 sites)
        if top_n:
            queryset = queryset.filter(rank__isnull=False, rank__lte=int(top_n))
        
        # Apply spatial bounding box filter
        if bbox:
            try:
                min_lng, min_lat, max_lng, max_lat = map(float, bbox.split(','))
                # Transform bbox from EPSG:4326 to EPSG:32651 for PostGIS query
                from pyproj import Transformer
                transformer_to_utm = Transformer.from_crs('EPSG:4326', 'EPSG:32651', always_xy=True)
                min_x, min_y = transformer_to_utm.transform(min_lng, min_lat)
                max_x, max_y = transformer_to_utm.transform(max_lng, max_lat)
                
                # Create bounding box polygon for PostGIS
                from django.contrib.gis.geos import Polygon
                bbox_polygon = Polygon.from_bbox((min_x, min_y, max_x, max_y))
                bbox_polygon.srid = 32651
                
                # Filter sites where inlet or outlet is within bbox
                queryset = queryset.filter(
                    django_models.Q(inlet__geometry__within=bbox_polygon) |
                    django_models.Q(outlet__geometry__within=bbox_polygon)
                )
            except Exception as e:
                logger.warning(f"Invalid bbox parameter: {bbox}, error: {str(e)}")
        
        # Limit results for performance (unless top_n is specified)
        if not top_n:
            max_results = 5000
            queryset = queryset[:max_results]
        
        # CRS transformer: EPSG:32651 (UTM Zone 51N) → EPSG:4326 (WGS84 lat/lng)
        transformer = Transformer.from_crs('EPSG:32651', 'EPSG:4326', always_xy=True)
        
        # Build GeoJSON FeatureCollection
        features = []
        infrastructure_features = []  # Separate features for infrastructure components
        for pair in queryset:
            # Transform LineString geometry
            line_coords = list(pair.geometry.coords)
            transformed_coords = [transformer.transform(x, y) for x, y in line_coords]
            
            # Transform inlet/outlet points
            inlet_x, inlet_y = pair.inlet.geometry.x, pair.inlet.geometry.y
            outlet_x, outlet_y = pair.outlet.geometry.x, pair.outlet.geometry.y
            inlet_lng, inlet_lat = transformer.transform(inlet_x, inlet_y)
            outlet_lng, outlet_lat = transformer.transform(outlet_x, outlet_y)
            
            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'LineString',
                    'coordinates': transformed_coords
                },
                'properties': {
                    'id': pair.id,
                    'pair_id': pair.pair_id,
                    'head': round(pair.head, 2),
                    'discharge': round(pair.discharge, 2) if pair.discharge else None,
                    'power': round(pair.power, 2) if pair.power else None,
                    'efficiency': pair.efficiency,
                    'river_distance': round(pair.river_distance, 2),
                    'inlet_id': pair.inlet.site_id,
                    'inlet_lat': round(inlet_lat, 6),
                    'inlet_lng': round(inlet_lng, 6),
                    'inlet_elevation': round(pair.inlet.elevation, 2),
                    'outlet_id': pair.outlet.site_id,
                    'outlet_lat': round(outlet_lat, 6),
                    'outlet_lng': round(outlet_lng, 6),
                    'outlet_elevation': round(pair.outlet.elevation, 2),
                    'hms_element': pair.hms_element_id,
                    'return_period': pair.return_period,
                    'score': round(pair.score, 2) if pair.score else None,
                    'rank': pair.rank,
                }
            }
            features.append(feature)
            
            # Add infrastructure components for top-ranked sites (rank <= 5)
            if pair.rank and pair.rank <= 5:
                logger.info(f"Processing infrastructure for {pair.pair_id} (rank {pair.rank})")
                
                # Intake Basin
                if pair.intake_basin_geom:
                    x, y = pair.intake_basin_geom.x, pair.intake_basin_geom.y
                    lng, lat = transformer.transform(x, y)
                    infrastructure_features.append({
                        'type': 'Feature',
                        'geometry': {'type': 'Point', 'coordinates': [lng, lat]},
                        'properties': {'type': 'intake_basin', 'pair_id': pair.pair_id, 'rank': pair.rank}
                    })
                    logger.debug(f"  Added intake basin at {lng}, {lat}")
                
                # Settling Basin
                if pair.settling_basin_geom:
                    x, y = pair.settling_basin_geom.x, pair.settling_basin_geom.y
                    lng, lat = transformer.transform(x, y)
                    # Extract actual elevation from DEM at settling basin location
                    try:
                        from rasterio import open as rio_open
                        from rasterio.transform import rowcol
                        if pair.raster_layer and pair.raster_layer.dataset and pair.raster_layer.dataset.file:
                            with rio_open(pair.raster_layer.dataset.file.path) as dem:
                                row, col = rowcol(dem.transform, x, y)
                                if 0 <= row < dem.height and 0 <= col < dem.width:
                                    settling_elevation = float(dem.read(1)[row, col])
                                    if settling_elevation < -9999:  # Nodata check
                                        settling_elevation = pair.inlet.elevation  # Fallback
                                else:
                                    settling_elevation = pair.inlet.elevation  # Fallback
                        else:
                            settling_elevation = pair.inlet.elevation  # Fallback if no DEM
                    except Exception as e:
                        logger.warning(f"Could not extract settling basin elevation: {e}")
                        settling_elevation = pair.inlet.elevation  # Fallback
                    infrastructure_features.append({
                        'type': 'Feature',
                        'geometry': {'type': 'Point', 'coordinates': [lng, lat]},
                        'properties': {'type': 'settling_basin', 'pair_id': pair.pair_id, 'rank': pair.rank, 'elevation_m': round(settling_elevation, 2) if settling_elevation else None}
                    })

                # Connector: Intake -> Settling (short link to visually join intake and settling)
                if pair.intake_basin_geom and pair.settling_basin_geom:
                    intake_xy = (pair.intake_basin_geom.x, pair.intake_basin_geom.y)
                    settling_xy = (pair.settling_basin_geom.x, pair.settling_basin_geom.y)
                    intake_trans = transformer.transform(intake_xy[0], intake_xy[1])
                    settling_trans = transformer.transform(settling_xy[0], settling_xy[1])
                    infrastructure_features.append({
                        'type': 'Feature',
                        'geometry': {'type': 'LineString', 'coordinates': [intake_trans, settling_trans]},
                        'properties': {'type': 'intake_to_settling', 'pair_id': pair.pair_id, 'rank': pair.rank}
                    })
                
                # Channel
                if pair.channel_geom:
                    channel_coords = list(pair.channel_geom.coords)
                    channel_transformed = [transformer.transform(x, y) for x, y in channel_coords]
                    infrastructure_features.append({
                        'type': 'Feature',
                        'geometry': {'type': 'LineString', 'coordinates': channel_transformed},
                        'properties': {'type': 'channel', 'pair_id': pair.pair_id, 'rank': pair.rank, 'length_m': round(pair.channel_length, 2) if pair.channel_length else None}
                    })
                
                # Forebay Tank
                if pair.forebay_tank_geom:
                    x, y = pair.forebay_tank_geom.x, pair.forebay_tank_geom.y
                    lng, lat = transformer.transform(x, y)
                    # Extract actual elevation from DEM at forebay tank location
                    try:
                        from rasterio import open as rio_open
                        from rasterio.transform import rowcol
                        if pair.raster_layer and pair.raster_layer.dataset and pair.raster_layer.dataset.file:
                            with rio_open(pair.raster_layer.dataset.file.path) as dem:
                                row, col = rowcol(dem.transform, x, y)
                                if 0 <= row < dem.height and 0 <= col < dem.width:
                                    forebay_elevation = float(dem.read(1)[row, col])
                                    if forebay_elevation < -9999:  # Nodata check
                                        forebay_elevation = pair.inlet.elevation - (pair.head * 0.1)  # Fallback: 10% head loss
                                else:
                                    forebay_elevation = pair.inlet.elevation - (pair.head * 0.1)  # Fallback
                        else:
                            forebay_elevation = pair.inlet.elevation - (pair.head * 0.1)  # Fallback if no DEM
                    except Exception as e:
                        logger.warning(f"Could not extract forebay tank elevation: {e}")
                        forebay_elevation = pair.inlet.elevation - (pair.head * 0.1)  # Fallback
                    infrastructure_features.append({
                        'type': 'Feature',
                        'geometry': {'type': 'Point', 'coordinates': [lng, lat]},
                        'properties': {'type': 'forebay_tank', 'pair_id': pair.pair_id, 'rank': pair.rank, 'elevation_m': round(forebay_elevation, 2) if forebay_elevation else None}
                    })
                
                # Penstock
                if pair.penstock_geom:
                    penstock_coords = list(pair.penstock_geom.coords)
                    penstock_transformed = [transformer.transform(x, y) for x, y in penstock_coords]
                    infrastructure_features.append({
                        'type': 'Feature',
                        'geometry': {'type': 'LineString', 'coordinates': penstock_transformed},
                        'properties': {'type': 'penstock', 'pair_id': pair.pair_id, 'rank': pair.rank, 'length_m': round(pair.penstock_length, 2) if pair.penstock_length else None, 'diameter_m': round(pair.penstock_diameter, 2) if pair.penstock_diameter else None}
                    })
                
                # Powerhouse
                if pair.powerhouse_geom:
                    x, y = pair.powerhouse_geom.x, pair.powerhouse_geom.y
                    lng, lat = transformer.transform(x, y)
                    # Extract actual elevation from DEM at powerhouse location
                    try:
                        from rasterio import open as rio_open
                        from rasterio.transform import rowcol
                        if pair.raster_layer and pair.raster_layer.dataset and pair.raster_layer.dataset.file:
                            with rio_open(pair.raster_layer.dataset.file.path) as dem:
                                row, col = rowcol(dem.transform, x, y)
                                if 0 <= row < dem.height and 0 <= col < dem.width:
                                    powerhouse_elevation = float(dem.read(1)[row, col])
                                    if powerhouse_elevation < -9999:  # Nodata check
                                        powerhouse_elevation = pair.outlet.elevation  # Fallback
                                else:
                                    powerhouse_elevation = pair.outlet.elevation  # Fallback
                        else:
                            powerhouse_elevation = pair.outlet.elevation  # Fallback if no DEM
                    except Exception as e:
                        logger.warning(f"Could not extract powerhouse elevation: {e}")
                        powerhouse_elevation = pair.outlet.elevation  # Fallback
                    infrastructure_features.append({
                        'type': 'Feature',
                        'geometry': {'type': 'Point', 'coordinates': [lng, lat]},
                        'properties': {'type': 'powerhouse', 'pair_id': pair.pair_id, 'rank': pair.rank, 'power_kw': round(pair.power, 2) if pair.power is not None else None, 'elevation_m': round(powerhouse_elevation, 2) if powerhouse_elevation else None}
                    })
        
        # Combine site pairs and infrastructure into single FeatureCollection
        all_features = features + infrastructure_features
        
        geojson = {
            'type': 'FeatureCollection',
            'crs': {
                'type': 'name',
                'properties': {'name': 'EPSG:4326'}
            },
            'features': all_features,
            'metadata': {
                'site_pairs_count': len(features),
                'infrastructure_count': len(infrastructure_features),
                'filtered_by_top_n': int(top_n) if top_n else None
            }
        }
        
        return JsonResponse(geojson, safe=False)
        
    except Exception as e:
        logger.error(f"Error generating site pairs GeoJSON: {str(e)}", exc_info=True)
        return JsonResponse({'error': str(e)}, status=500)


@cache_page(60 * 5)  # Cache for 5 minutes
def geojson_watersheds(request):
    """
    GeoJSON endpoint for watershed boundaries.
    
    Returns FeatureCollection with:
    - Polygon geometry (watershed boundaries)
    - Watershed statistics (area, perimeter, etc.)
    - CRS transformation: EPSG:32651 → EPSG:4326
    
    Query parameters:
    - raster_layer: Filter by source DEM ID
    """
    from pyproj import Transformer
    
    try:
        raster_layer_id = request.GET.get('raster_layer')
        
        from .models import WatershedPolygon
        queryset = WatershedPolygon.objects.all()
        
        if raster_layer_id:
            queryset = queryset.filter(raster_layer_id=int(raster_layer_id))
        
        # Limit results
        queryset = queryset[:100]
        
        # CRS transformer
        transformer = Transformer.from_crs('EPSG:32651', 'EPSG:4326', always_xy=True)
        
        features = []
        for watershed in queryset:
            # Get geometry and transform coordinates
            from shapely import wkb
            from shapely.ops import transform
            
            # Convert GeoDjango geometry to Shapely
            geom_shapely = wkb.loads(bytes(watershed.geometry.wkb))
            
            # Transform coordinates using pyproj transformer
            geom_transformed = transform(transformer.transform, geom_shapely)
            
            # Convert to GeoJSON dict
            import json
            geom_dict = json.loads(json.dumps(geom_transformed.__geo_interface__))
            
            feature = {
                'type': 'Feature',
                'geometry': geom_dict,
                'properties': {
                    'id': watershed.id,
                    'watershed_id': watershed.watershed_id,
                    'area_m2': round(watershed.area_m2, 2),
                    'area_km2': round(watershed.area_m2 / 1e6, 2),
                    'perimeter_m': round(watershed.perimeter_m, 2),
                    'stream_length_m': round(watershed.stream_length_m, 2) if watershed.stream_length_m else None,
                }
            }
            features.append(feature)
        
        geojson = {
            'type': 'FeatureCollection',
            'crs': {
                'type': 'name',
                'properties': {'name': 'EPSG:4326'}
            },
            'features': features
        }
        
        return JsonResponse(geojson, safe=False)
        
    except Exception as e:
        logger.error(f"Error generating watersheds GeoJSON: {str(e)}", exc_info=True)
        return JsonResponse({'error': str(e)}, status=500)


@cache_page(60 * 10)  # Cache for 10 minutes (streams rarely change)
def geojson_streams(request):
    """
    GeoJSON endpoint for stream network.
    
    Returns FeatureCollection with:
    - LineString geometry (stream segments)
    - Stream order, length, topology attributes
    - CRS transformation: EPSG:32651 → EPSG:4326
    
    Query parameters:
    - raster_layer: Filter by source DEM ID
    - min_order: Minimum stream order
    """
    from pyproj import Transformer
    
    try:
        raster_layer_id = request.GET.get('raster_layer')
        min_order = request.GET.get('min_order', 1)
        
        from .models import StreamNetwork
        queryset = StreamNetwork.objects.all()
        
        if raster_layer_id:
            queryset = queryset.filter(raster_layer_id=int(raster_layer_id))
        
        queryset = queryset.filter(stream_order__gte=int(min_order))
        
        # Limit results
        queryset = queryset[:5000]
        
        # CRS transformer
        transformer = Transformer.from_crs('EPSG:32651', 'EPSG:4326', always_xy=True)
        
        features = []
        for stream in queryset:
            # Transform LineString coordinates
            line_coords = list(stream.geometry.coords)
            transformed_coords = [transformer.transform(x, y) for x, y in line_coords]
            
            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'LineString',
                    'coordinates': transformed_coords
                },
                'properties': {
                    'id': stream.id,
                    'stream_order': stream.stream_order,
                    'length_m': round(stream.length_m, 2),
                    'from_node': stream.from_node,
                    'to_node': stream.to_node,
                    'is_outlet': stream.is_outlet,
                    'is_confluence': stream.is_confluence,
                }
            }
            features.append(feature)
        
        geojson = {
            'type': 'FeatureCollection',
            'crs': {
                'type': 'name',
                'properties': {'name': 'EPSG:4326'}
            },
            'features': features
        }
        
        return JsonResponse(geojson, safe=False)
        
    except Exception as e:
        logger.error(f"Error generating streams GeoJSON: {str(e)}", exc_info=True)
        return JsonResponse({'error': str(e)}, status=500)


@cache_page(60 * 10)  # Cache for 10 minutes
def geojson_subbasins(request):
    """
    GeoJSON endpoint for subbasins (from cached shapefile GeoJSON).
    
    Returns FeatureCollection with:
    - Polygon geometry (subbasin boundaries)
    - Subbasin attributes from shapefile
    - CRS transformation: EPSG:32651 → EPSG:4326
    """
    from pyproj import Transformer
    from pathlib import Path
    import json
    
    try:
        features = []
        
        # Read from cached GeoJSON files (subbasin_*.geojson)
        cache_dir = Path(settings.MEDIA_ROOT) / 'vector_cache'
        
        if cache_dir.exists():
            for cache_file in cache_dir.glob('subbasin_*.geojson'):
                try:
                    with open(cache_file, 'r') as f:
                        cached_data = json.load(f)
                    
                    # Transform coordinates from EPSG:32651 to EPSG:4326
                    transformer = Transformer.from_crs('EPSG:32651', 'EPSG:4326', always_xy=True)
                    
                    for feature in cached_data.get('features', []):
                        # Transform geometry coordinates
                        geom = feature['geometry']
                        if geom['type'] == 'Polygon':
                            new_coords = []
                            for ring in geom['coordinates']:
                                new_ring = [list(transformer.transform(x, y)) for x, y in ring]
                                new_coords.append(new_ring)
                            geom['coordinates'] = new_coords
                        elif geom['type'] == 'MultiPolygon':
                            new_coords = []
                            for polygon in geom['coordinates']:
                                new_polygon = []
                                for ring in polygon:
                                    new_ring = [list(transformer.transform(x, y)) for x, y in ring]
                                    new_polygon.append(new_ring)
                                new_coords.append(new_polygon)
                            geom['coordinates'] = new_coords
                        
                        # Add source info to properties
                        feature['properties']['source_file'] = cache_file.stem
                        features.append(feature)
                        
                except Exception as e:
                    logger.error(f"Error reading cache file {cache_file}: {str(e)}")
                    continue
        
        geojson = {
            'type': 'FeatureCollection',
            'crs': {
                'type': 'name',
                'properties': {'name': 'EPSG:4326'}
            },
            'features': features
        }
        
        return JsonResponse(geojson, safe=False)
        
    except Exception as e:
        logger.error(f"Error generating subbasins GeoJSON: {str(e)}", exc_info=True)
        return JsonResponse({'error': str(e)}, status=500)


@cache_page(60 * 10)  # Cache for 10 minutes
def geojson_bridges(request):
    """
    GeoJSON endpoint for bridges/outlets (from cached shapefile GeoJSON).
    
    Returns FeatureCollection with:
    - Point geometry (bridge/outlet locations)
    - Bridge attributes from shapefile
    - CRS transformation: EPSG:32651 → EPSG:4326
    """
    from pyproj import Transformer
    from pathlib import Path
    import json
    
    try:
        features = []
        
        # Read from cached GeoJSON files (bridge_*.geojson or outlet_*.geojson)
        cache_dir = Path(settings.MEDIA_ROOT) / 'vector_cache'
        
        if cache_dir.exists():
            for cache_file in list(cache_dir.glob('bridge_*.geojson')) + list(cache_dir.glob('outlet_*.geojson')):
                try:
                    with open(cache_file, 'r') as f:
                        cached_data = json.load(f)
                    
                    # Transform coordinates from EPSG:32651 to EPSG:4326
                    transformer = Transformer.from_crs('EPSG:32651', 'EPSG:4326', always_xy=True)
                    
                    for feature in cached_data.get('features', []):
                        # Transform geometry coordinates
                        geom = feature['geometry']
                        if geom['type'] == 'Point':
                            x, y = geom['coordinates'][:2]
                            lon, lat = transformer.transform(x, y)
                            geom['coordinates'] = [lon, lat]
                        elif geom['type'] == 'MultiPoint':
                            new_coords = []
                            for point in geom['coordinates']:
                                x, y = point[:2]
                                lon, lat = transformer.transform(x, y)
                                new_coords.append([lon, lat])
                            geom['coordinates'] = new_coords
                        
                        # Add source info to properties
                        feature['properties']['source_file'] = cache_file.stem
                        features.append(feature)
                        
                except Exception as e:
                    logger.error(f"Error reading cache file {cache_file}: {str(e)}")
                    continue
        
        geojson = {
            'type': 'FeatureCollection',
            'crs': {
                'type': 'name',
                'properties': {'name': 'EPSG:4326'}
            },
            'features': features
        }
        
        return JsonResponse(geojson, safe=False)
        
    except Exception as e:
        logger.error(f"Error generating bridges GeoJSON: {str(e)}", exc_info=True)
        return JsonResponse({'error': str(e)}, status=500)


def geojson_weir_candidates(request):
    """
    GeoJSON endpoint for weir/diversion candidates (Objective 3).
    
    These points represent potential water intake/weir locations near inlet nodes,
    identified by searching the DEM with directional constraints toward outlets.
    
    Returns FeatureCollection with:
    - Point geometry (weir candidate locations)
    - Candidate metadata (elevation, distance, ranking)
    - Best weir highlighted (rank_within_inlet=1) with special styling
    - Connection lines from weir to inlet for visualization
    - CRS transformation: EPSG:32651 → EPSG:4326
    
    Query parameters:
    - raster_layer: Filter by source DEM ID
    - inlet_node_id: Filter to candidates for a specific inlet
    - min_rank: Minimum suitability rank (1=best)
    - max_candidates: Maximum number of candidates to return (default: 500)
    - best_only: Show only best weir (rank=1) for each inlet (default: false)
    """
    from pyproj import Transformer
    
    try:
        raster_layer_id = request.GET.get('raster_layer')
        inlet_node_id = request.GET.get('inlet_node_id')
        min_rank = request.GET.get('min_rank', 1)
        max_candidates = request.GET.get('max_candidates', 500)
        best_only = request.GET.get('best_only', 'false').lower() == 'true'
        
        # Import model
        from .models import WeirCandidate
        
        queryset = WeirCandidate.objects.select_related('inlet_point').all()
        
        # Apply filters
        if raster_layer_id:
            queryset = queryset.filter(raster_layer_id=int(raster_layer_id))
        
        if inlet_node_id:
            queryset = queryset.filter(inlet_node_id=inlet_node_id)
        
        if best_only:
            # Show only rank 1 candidates
            queryset = queryset.filter(rank_within_inlet=1)
        elif int(min_rank) > 1:
            queryset = queryset.filter(rank_within_inlet__lte=int(min_rank))
        
        # Order by rank and limit
        queryset = queryset.order_by('inlet_point', 'rank_within_inlet')[:int(max_candidates)]
        
        # CRS transformer
        transformer = Transformer.from_crs('EPSG:32651', 'EPSG:4326', always_xy=True)
        
        features = []
        connection_features = []  # Lines connecting weir to inlet
        best_weirs_count = 0
        
        for candidate in queryset:
            # Transform geometry
            weir_lng, weir_lat = transformer.transform(candidate.geometry.x, candidate.geometry.y)
            inlet_lng, inlet_lat = transformer.transform(candidate.inlet_point.geometry.x, candidate.inlet_point.geometry.y)
            
            # Determine if this is the best weir for this inlet
            is_best = (candidate.rank_within_inlet == 1)
            if is_best:
                best_weirs_count += 1
            
            # Weir point feature
            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [weir_lng, weir_lat]
                },
                'properties': {
                    'id': candidate.id,
                    'candidate_id': candidate.candidate_id,
                    'inlet_node_id': candidate.inlet_node_id,
                    'inlet_point_id': candidate.inlet_point.site_id if candidate.inlet_point else None,
                    'elevation': round(candidate.elevation, 2),
                    'inlet_elevation': round(candidate.inlet_elevation, 2),
                    'elevation_difference': round(candidate.elevation_difference, 2),
                    'distance_from_inlet': round(candidate.distance_from_inlet, 2),
                    'angle_to_outlet_deg': round(candidate.angle_to_outlet_deg, 2) if candidate.angle_to_outlet_deg else None,
                    'is_toward_outlet': candidate.is_toward_outlet,
                    'outlet_count': candidate.outlet_count,
                    'outlet_node_ids': candidate.outlet_node_ids,
                    'pair_count': candidate.pair_count,
                    'pair_ids_list': candidate.pair_ids_list,
                    'suitability_score': round(candidate.suitability_score, 2) if candidate.suitability_score else None,
                    'rank_within_inlet': candidate.rank_within_inlet,
                    'is_best': is_best,  # Highlight best weir
                    'search_radius': candidate.search_radius,
                    'elevation_tolerance': candidate.elevation_tolerance,
                    'min_distance': candidate.min_distance,
                    'cone_angle_deg': candidate.cone_angle_deg,
                    # Inlet coordinates for drawing links
                    'inlet_lng': round(inlet_lng, 6),
                    'inlet_lat': round(inlet_lat, 6),
                    # Weir coordinates
                    'weir_lng': round(weir_lng, 6),
                    'weir_lat': round(weir_lat, 6),
                }
            }
            features.append(feature)
            
            # Add connection line from weir to inlet (for best weirs only)
            if is_best:
                connection_feature = {
                    'type': 'Feature',
                    'geometry': {
                        'type': 'LineString',
                        'coordinates': [[weir_lng, weir_lat], [inlet_lng, inlet_lat]]
                    },
                    'properties': {
                        'type': 'weir_to_inlet_connection',
                        'candidate_id': candidate.candidate_id,
                        'inlet_node_id': candidate.inlet_node_id,
                        'is_best': True,
                        'distance_m': round(candidate.distance_from_inlet, 2)
                    }
                }
                connection_features.append(connection_feature)
        
        # Combine weir points and connection lines
        all_features = features + connection_features
        
        geojson = {
            'type': 'FeatureCollection',
            'crs': {
                'type': 'name',
                'properties': {'name': 'EPSG:4326'}
            },
            'features': all_features,
            'metadata': {
                'total_candidates': len(features),
                'best_weirs': best_weirs_count,
                'connection_lines': len(connection_features),
                'unique_inlets': len(set(f['properties']['inlet_node_id'] for f in features)),
                'best_only_filter': best_only
            }
        }
        
        return JsonResponse(geojson, safe=False)
        
    except Exception as e:
        logger.error(f"Error generating weir candidates GeoJSON: {str(e)}", exc_info=True)
        return JsonResponse({'error': str(e)}, status=500)


@cache_page(60 * 5)  # Cache for 5 minutes
def geojson_hp_nodes(request):
    """
    GeoJSON endpoint for HP nodes (hydropower nodes along main channel).
    
    These are candidate inlet/outlet points sampled systematically along
    the main river channel for site assessment.
    
    Returns FeatureCollection with:
    - Point geometry (HP node locations)
    - Node attributes (elevation, chainage, distance along channel)
    - CRS transformation: EPSG:32651 → EPSG:4326
    
    Query parameters:
    - raster_layer: Filter by source DEM ID
    - max_nodes: Maximum number of nodes to return (default: 1000)
    """
    from pyproj import Transformer
    
    try:
        raster_layer_id = request.GET.get('raster_layer')
        max_nodes = request.GET.get('max_nodes', 1000)
        
        # Import model
        from .models import HPNode
        
        queryset = HPNode.objects.select_related('raster_layer').all()
        
        # Apply filters
        if raster_layer_id:
            queryset = queryset.filter(raster_layer_id=int(raster_layer_id))
        
        # Order by distance along channel and limit
        queryset = queryset.order_by('distance_along_channel')[:int(max_nodes)]
        
        # CRS transformer
        transformer = Transformer.from_crs('EPSG:32651', 'EPSG:4326', always_xy=True)
        
        features = []
        for node in queryset:
            # Transform geometry
            node_lng, node_lat = transformer.transform(node.geometry.x, node.geometry.y)
            
            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [node_lng, node_lat]
                },
                'properties': {
                    'id': node.id,
                    'node_id': node.node_id,
                    'elevation': round(node.elevation, 2),
                    'distance_along_channel': round(node.distance_along_channel, 2),
                    'chainage_km': round(node.chainage, 3),
                    'sampling_interval': node.sampling_interval,
                    'can_be_inlet': node.can_be_inlet,
                    'can_be_outlet': node.can_be_outlet,
                    'source_vector_name': node.source_vector_name,
                    'created_at': node.created_at.isoformat(),
                }
            }
            features.append(feature)
        
        geojson = {
            'type': 'FeatureCollection',
            'crs': {
                'type': 'name',
                'properties': {'name': 'EPSG:4326'}
            },
            'features': features,
            'metadata': {
                'total_nodes': len(features),
                'channel_length_m': features[-1]['properties']['distance_along_channel'] if features else 0,
                'channel_length_km': round(features[-1]['properties']['distance_along_channel'] / 1000.0, 2) if features else 0,
            }
        }
        
        return JsonResponse(geojson, safe=False)
        
    except Exception as e:
        logger.error(f"Error generating HP nodes GeoJSON: {str(e)}", exc_info=True)
        return JsonResponse({'error': str(e)}, status=500)


# ============================================================================
# Processing Layers API - For HEC-HMS-style Layer Visualization
# ============================================================================

def api_processing_layers(request):
    """
    API endpoint to list all available processing layers for map visualization.
    
    Returns JSON list of layers with their metadata, enabling HEC-HMS-style
    layer toggling in the map interface.
    
    Query parameters:
    - raster_layer: Filter by source DEM ID
    - visible_only: Only return visible layers (default: false)
    """
    from .models import ProcessingLayer, RasterLayer
    
    try:
        raster_layer_id = request.GET.get('raster_layer')
        visible_only = request.GET.get('visible_only', '').lower() == 'true'
        
        queryset = ProcessingLayer.objects.all().order_by('processing_step', 'name')
        
        if raster_layer_id:
            queryset = queryset.filter(raster_layer_id=int(raster_layer_id))
        
        if visible_only:
            queryset = queryset.filter(is_visible=True)
        
        layers = []
        for layer in queryset:
            layers.append({
                'id': layer.id,
                'name': layer.name,
                'layer_type': layer.layer_type,
                'layer_type_display': layer.get_layer_type_display(),
                'data_format': layer.data_format,
                'file_path': layer.file_path,
                'style': layer.style,
                'description': layer.description,
                'processing_step': layer.processing_step,
                'is_visible': layer.is_visible,
                'is_base_layer': layer.is_base_layer,
                'statistics': layer.statistics,
                'bounds': {
                    'minx': layer.bounds_minx,
                    'miny': layer.bounds_miny,
                    'maxx': layer.bounds_maxx,
                    'maxy': layer.bounds_maxy
                } if layer.bounds_minx else None,
                'raster_layer_id': layer.raster_layer_id,
                'created_at': layer.created_at.isoformat()
            })
        
        return JsonResponse({
            'layers': layers,
            'count': len(layers)
        })
        
    except Exception as e:
        logger.error(f"Error listing processing layers: {str(e)}", exc_info=True)
        return JsonResponse({'error': str(e)}, status=500)


def api_processing_status(request):
    """
    API endpoint to get current processing status and available intermediate outputs.
    
    Returns status of each processing step with layer availability,
    similar to HEC-HMS processing status display.
    """
    from .models import RasterLayer, StreamNetwork, WatershedPolygon, SitePair, ProcessingLayer
    
    try:
        # Get latest raster layer (or specific one if provided)
        raster_layer_id = request.GET.get('raster_layer')
        
        if raster_layer_id:
            raster = RasterLayer.objects.get(id=int(raster_layer_id))
        else:
            raster = RasterLayer.objects.order_by('-id').first()
        
        if not raster:
            return JsonResponse({
                'status': 'no_data',
                'message': 'No DEM data loaded. Upload a DEM to begin processing.'
            })
        
        # Build processing status
        processing_steps = [
            {
                'step': 1,
                'name': 'Flow Direction',
                'status': 'completed' if raster.flow_direction_path else 'not_started',
                'has_layer': bool(raster.flow_direction_path),
                'layer_type': 'FLOW_DIRECTION',
                'description': 'D8 flow direction raster'
            },
            {
                'step': 2,
                'name': 'Flow Accumulation',
                'status': 'completed' if raster.flow_accumulation_path else 'not_started',
                'has_layer': bool(raster.flow_accumulation_path),
                'layer_type': 'FLOW_ACCUMULATION',
                'description': 'Upstream contributing area (cells)'
            },
            {
                'step': 3,
                'name': 'Stream Network',
                'status': 'completed' if raster.stream_count > 0 else 'not_started',
                'has_layer': raster.stream_count > 0,
                'layer_type': 'STREAM_VECTOR',
                'count': raster.stream_count,
                'description': f'{raster.stream_count} stream segments extracted'
            },
            {
                'step': 4,
                'name': 'Watershed Delineation',
                'status': 'completed' if raster.watershed_delineated else 'not_started',
                'has_layer': raster.watershed_count > 0,
                'layer_type': 'WATERSHED_VECTOR',
                'count': raster.watershed_count,
                'description': f'{raster.watershed_count} watersheds delineated'
            },
            {
                'step': 5,
                'name': 'Subbasins',
                'status': 'completed',
                'has_layer': True,
                'layer_type': 'SUBBASIN',
                'description': 'HEC-HMS subbasin boundaries'
            },
            {
                'step': 6,
                'name': 'Bridges/POIs',
                'status': 'completed',
                'has_layer': True,
                'layer_type': 'BRIDGE_POI',
                'description': 'Bridge locations and points of interest'
            },
            {
                'step': 7,
                'name': 'Discharge Calculation',
                'status': 'completed' if raster.discharge_computed else 'not_started',
                'has_layer': bool(raster.discharge_raster_path),
                'layer_type': 'DISCHARGE_RASTER',
                'description': f'Spatially-varying discharge (Q_outlet={raster.discharge_q_outlet or "N/A"} m³/s)'
            },
        ]
        
        # Add main channel workflow steps (new systematic workflow)
        from .models import HPNode, WeirCandidate
        
        hp_node_count = raster.hp_nodes.count()
        main_channel_pairs_count = SitePair.objects.filter(
            raster_layer=raster,
            pair_id__startswith='HP_'
        ).count()
        weir_candidate_count = WeirCandidate.objects.filter(raster_layer=raster).count()
        
        processing_steps.extend([
            {
                'step': 8,
                'name': 'HP Node Generation',
                'status': 'completed' if hp_node_count > 0 else 'not_started',
                'has_layer': hp_node_count > 0,
                'layer_type': 'HP_NODES',
                'count': hp_node_count,
                'description': f'{hp_node_count} HP nodes along main channel'
            },
            {
                'step': 9,
                'name': 'Site Pairing',
                'status': 'completed' if main_channel_pairs_count > 0 else 'not_started',
                'has_layer': main_channel_pairs_count > 0,
                'layer_type': 'MAIN_CHANNEL_PAIRS',
                'count': main_channel_pairs_count,
                'description': f'{main_channel_pairs_count} optimal site pairs (top 50)'
            },
            {
                'step': 10,
                'name': 'Weir Search',
                'status': 'completed' if weir_candidate_count > 0 else 'not_started',
                'has_layer': weir_candidate_count > 0,
                'layer_type': 'WEIR_CANDIDATES',
                'count': weir_candidate_count,
                'description': f'{weir_candidate_count} weir candidates identified'
            },
            {
                'step': 11,
                'name': 'Infrastructure',
                'status': 'completed' if raster.site_pair_count > 0 else 'not_started',
                'has_layer': raster.site_pair_count > 0,
                'layer_type': 'INFRASTRUCTURE',
                'description': 'Run-of-river hydropower infrastructure layout'
            },
        ])
        
        # Get available processing layers
        available_layers = list(ProcessingLayer.objects.filter(
            raster_layer=raster
        ).values('id', 'name', 'layer_type', 'is_visible', 'processing_step'))
        
        return JsonResponse({
            'status': 'ok',
            'raster_layer_id': raster.id,
            'raster_name': raster.dataset.name if raster.dataset else 'Unknown',
            'processing_steps': processing_steps,
            'available_layers': available_layers,
            'overall_progress': sum(1 for s in processing_steps if s['status'] == 'completed') / len(processing_steps) * 100
        })
        
    except RasterLayer.DoesNotExist:
        return JsonResponse({'error': 'Raster layer not found'}, status=404)
    except Exception as e:
        logger.error(f"Error getting processing status: {str(e)}", exc_info=True)
        return JsonResponse({'error': str(e)}, status=500)


def api_raster_tile(request, layer_type, z, x, y):
    """
    Serve raster tiles for intermediate processing layers.
    
    This enables visualization of processing steps like HEC-HMS does:
    - filled_dem: Depression-filled DEM (terrain visualization)
    - flow_direction: D8 flow direction (arrow colors)
    - flow_accumulation: Upstream contributing area (blue gradient)
    
    URL format: /api/tiles/{layer_type}/{z}/{x}/{y}.png
    
    Uses XYZ tile convention (same as OpenStreetMap/Leaflet)
    """
    import numpy as np
    from PIL import Image
    import io
    from pathlib import Path
    import math
    
    try:
        # Get raster layer
        raster_layer_id = request.GET.get('raster_layer')
        
        if raster_layer_id:
            raster = RasterLayer.objects.get(id=int(raster_layer_id))
        else:
            raster = RasterLayer.objects.order_by('-id').first()
        
        if not raster:
            return HttpResponse(status=404)
        
        # Get raster path based on layer type
        layer_paths = {
            'flow_accumulation': raster.flow_accumulation_path,
            'filled_dem': raster.filled_dem_path,
            'flow_direction': raster.flow_direction_path,
        }
        
        raster_path = layer_paths.get(layer_type)
        
        if not raster_path:
            return HttpResponse(status=404)
        
        full_path = Path(settings.MEDIA_ROOT) / raster_path
        
        if not full_path.exists():
            return HttpResponse(status=404)
        
        # Import rasterio for raster reading
        try:
            import rasterio
            from rasterio.windows import from_bounds
            from rasterio.enums import Resampling
            from pyproj import Transformer
        except ImportError:
            logger.error("rasterio not installed")
            return HttpResponse(status=500)
        
        # XYZ tile to bounds (Web Mercator EPSG:3857)
        def tile_bounds(z, x, y):
            """Convert XYZ tile coordinates to Web Mercator bounds."""
            n = 2.0 ** z
            # Web Mercator bounds
            world_size = 20037508.342789244 * 2
            tile_size = world_size / n
            
            min_x = -20037508.342789244 + x * tile_size
            max_x = min_x + tile_size
            max_y = 20037508.342789244 - y * tile_size
            min_y = max_y - tile_size
            
            return min_x, min_y, max_x, max_y
        
        # Get tile bounds in Web Mercator
        bounds_3857 = tile_bounds(z, x, y)
        
        with rasterio.open(full_path) as src:
            # Transform bounds to raster CRS
            transformer = Transformer.from_crs("EPSG:3857", src.crs, always_xy=True)
            
            min_x, min_y = transformer.transform(bounds_3857[0], bounds_3857[1])
            max_x, max_y = transformer.transform(bounds_3857[2], bounds_3857[3])
            
            # Check if tile intersects raster
            raster_bounds = src.bounds
            if (max_x < raster_bounds.left or min_x > raster_bounds.right or
                max_y < raster_bounds.bottom or min_y > raster_bounds.top):
                # Return transparent tile if no intersection
                tile = Image.new('RGBA', (256, 256), (0, 0, 0, 0))
                buffer = io.BytesIO()
                tile.save(buffer, format='PNG')
                buffer.seek(0)
                return HttpResponse(buffer.getvalue(), content_type='image/png')
            
            # Clamp bounds to raster extent
            min_x = max(min_x, raster_bounds.left)
            max_x = min(max_x, raster_bounds.right)
            min_y = max(min_y, raster_bounds.bottom)
            max_y = min(max_y, raster_bounds.top)
            
            # Read window
            try:
                window = from_bounds(min_x, min_y, max_x, max_y, src.transform)
                
                # Read data with resampling to 256x256
                data = src.read(
                    1,
                    window=window,
                    out_shape=(256, 256),
                    resampling=Resampling.bilinear
                )
            except Exception as e:
                logger.warning(f"Error reading window: {e}")
                # Return transparent tile on error
                tile = Image.new('RGBA', (256, 256), (0, 0, 0, 0))
                buffer = io.BytesIO()
                tile.save(buffer, format='PNG')
                buffer.seek(0)
                return HttpResponse(buffer.getvalue(), content_type='image/png')
            
            # Get nodata value
            nodata = src.nodata
            
            # Create RGBA image based on layer type
            rgba = np.zeros((256, 256, 4), dtype=np.uint8)
            
            # Create mask for valid data
            if nodata is not None:
                valid_mask = ~np.isnan(data) & (data != nodata)
            else:
                valid_mask = ~np.isnan(data)
            
            if not np.any(valid_mask):
                # Return transparent tile if no valid data
                tile = Image.new('RGBA', (256, 256), (0, 0, 0, 0))
                buffer = io.BytesIO()
                tile.save(buffer, format='PNG')
                buffer.seek(0)
                return HttpResponse(buffer.getvalue(), content_type='image/png')
            
            # Apply color ramp based on layer type
            if layer_type == 'filled_dem':
                # Terrain color ramp (green-yellow-brown-white)
                vmin = np.nanpercentile(data[valid_mask], 2)
                vmax = np.nanpercentile(data[valid_mask], 98)
                normalized = np.clip((data - vmin) / (vmax - vmin + 1e-10), 0, 1)
                
                # Terrain colors: green (low) -> yellow -> brown -> white (high)
                rgba[valid_mask, 0] = (normalized[valid_mask] * 200 + 55).astype(np.uint8)  # R
                rgba[valid_mask, 1] = ((1 - normalized[valid_mask] * 0.5) * 200).astype(np.uint8)  # G
                rgba[valid_mask, 2] = (normalized[valid_mask] * 100).astype(np.uint8)  # B
                rgba[valid_mask, 3] = 180  # Alpha
                
            elif layer_type == 'flow_direction':
                # Flow direction: 8 distinct colors for 8 directions
                # D8 encoding: 1=E, 2=SE, 4=S, 8=SW, 16=W, 32=NW, 64=N, 128=NE
                direction_colors = {
                    1: (255, 0, 0),      # E - Red
                    2: (255, 127, 0),    # SE - Orange
                    4: (255, 255, 0),    # S - Yellow
                    8: (127, 255, 0),    # SW - Yellow-Green
                    16: (0, 255, 0),     # W - Green
                    32: (0, 255, 255),   # NW - Cyan
                    64: (0, 127, 255),   # N - Light Blue
                    128: (127, 0, 255),  # NE - Purple
                }
                
                for direction, color in direction_colors.items():
                    mask = valid_mask & (data == direction)
                    rgba[mask, 0] = color[0]
                    rgba[mask, 1] = color[1]
                    rgba[mask, 2] = color[2]
                    rgba[mask, 3] = 180
                
            elif layer_type == 'flow_accumulation':
                # Flow accumulation: logarithmic blue scale (streams are brighter)
                # Use log scale since values range from 1 to millions
                log_data = np.log10(data + 1)
                vmin = 0
                vmax = np.nanpercentile(log_data[valid_mask], 99)
                normalized = np.clip((log_data - vmin) / (vmax - vmin + 1e-10), 0, 1)
                
                # Blue gradient - brighter for higher accumulation (streams)
                rgba[valid_mask, 0] = (normalized[valid_mask] * 100).astype(np.uint8)  # R
                rgba[valid_mask, 1] = (normalized[valid_mask] * 150).astype(np.uint8)  # G
                rgba[valid_mask, 2] = (100 + normalized[valid_mask] * 155).astype(np.uint8)  # B
                rgba[valid_mask, 3] = (50 + normalized[valid_mask] * 180).astype(np.uint8)  # Alpha varies
            
            # Create PIL Image
            tile = Image.fromarray(rgba, 'RGBA')
        
        # Save to bytes
        buffer = io.BytesIO()
        tile.save(buffer, format='PNG', optimize=True)
        buffer.seek(0)
        
        response = HttpResponse(buffer.getvalue(), content_type='image/png')
        response['Cache-Control'] = 'public, max-age=3600'  # Cache for 1 hour
        return response
        
    except RasterLayer.DoesNotExist:
        return HttpResponse(status=404)
    except Exception as e:
        logger.error(f"Error generating raster tile: {str(e)}", exc_info=True)
        return HttpResponse(status=500)


def api_discharge_stats(request):
    """
    API endpoint to get discharge statistics for the current dataset.
    
    Returns discharge distribution across all site pairs, enabling
    visualization of spatially-varying discharge.
    """
    from .models import SitePair
    from django.db.models import Min, Max, Avg, StdDev, Count
    
    try:
        raster_layer_id = request.GET.get('raster_layer')
        
        queryset = SitePair.objects.all()
        
        if raster_layer_id:
            queryset = queryset.filter(raster_layer_id=int(raster_layer_id))
        
        # Filter out null discharge values
        queryset = queryset.filter(discharge__isnull=False, discharge__gt=0)
        
        stats = queryset.aggregate(
            count=Count('id'),
            min_discharge=Min('discharge'),
            max_discharge=Max('discharge'),
            avg_discharge=Avg('discharge'),
            std_discharge=StdDev('discharge'),
            min_power=Min('power'),
            max_power=Max('power'),
            avg_power=Avg('power'),
        )
        
        # Get discharge distribution (histogram bins)
        discharges = list(queryset.values_list('discharge', flat=True))
        
        if discharges:
            import numpy as np
            hist, bin_edges = np.histogram(discharges, bins=10)
            distribution = [
                {'range': f'{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}', 'count': int(hist[i])}
                for i in range(len(hist))
            ]
        else:
            distribution = []
        
        return JsonResponse({
            'statistics': {
                'count': stats['count'] or 0,
                'min_discharge_m3s': round(stats['min_discharge'] or 0, 3),
                'max_discharge_m3s': round(stats['max_discharge'] or 0, 2),
                'avg_discharge_m3s': round(stats['avg_discharge'] or 0, 3),
                'std_discharge_m3s': round(stats['std_discharge'] or 0, 3),
                'min_power_kw': round(stats['min_power'] or 0, 2),
                'max_power_kw': round(stats['max_power'] or 0, 2),
                'avg_power_kw': round(stats['avg_power'] or 0, 2),
            },
            'distribution': distribution,
            'is_spatially_varying': stats['std_discharge'] is not None and stats['std_discharge'] > 0.01
        })
        
    except Exception as e:
        logger.error(f"Error getting discharge stats: {str(e)}", exc_info=True)
        return JsonResponse({'error': str(e)}, status=500)


@cache_page(60 * 15)  # Cache MVT tiles for 15 minutes
def mvt_site_pairs(request, z, x, y):
    """
    Mapbox Vector Tile (MVT) endpoint for site pairs.
    Optimized for large datasets (>5000 features).
    
    URL format: /api/mvt/site-pairs/{z}/{x}/{y}.pbf
    
    Returns binary Protobuf-encoded vector tile.
    Use with Leaflet.VectorGrid or Mapbox GL JS.
    
    Query parameters: Same as geojson_site_pairs
    """
    try:
        from django.db import connection
        import struct
        
        # Parse tile coordinates
        zoom = int(z)
        tile_x = int(x)
        tile_y = int(y)
        
        # Get query parameters for filtering
        min_head = request.GET.get('min_head', 0)
        min_discharge = request.GET.get('min_discharge', 0)
        min_power = request.GET.get('min_power', 0)
        scenario = request.GET.get('scenario')
        
        # Build SQL query for MVT generation
        # PostGIS ST_AsMVT generates MVT directly
        sql = """
        WITH filtered_sites AS (
            SELECT 
                sp.id,
                sp.pair_id,
                sp.head,
                sp.discharge,
                sp.power,
                sp.efficiency,
                sp.river_distance,
                sp.return_period,
                -- Transform geometry to Web Mercator (EPSG:3857) for MVT
                ST_Transform(sp.geometry, 3857) AS geom
            FROM hydropower_sitepair sp
            WHERE 1=1
        """
        
        params = []
        
        if min_head:
            sql += " AND sp.head >= %s"
            params.append(float(min_head))
        if min_discharge:
            sql += " AND sp.discharge >= %s"
            params.append(float(min_discharge))
        if min_power:
            sql += " AND sp.power >= %s"
            params.append(float(min_power))
        if scenario:
            sql += " AND sp.return_period = %s"
            params.append(scenario)
        
        sql += """
        ),
        mvt_data AS (
            SELECT 
                id,
                pair_id,
                head,
                discharge,
                power,
                efficiency,
                river_distance,
                return_period,
                -- Clip geometry to tile bounds
                ST_AsMVTGeom(
                    geom,
                    ST_TileEnvelope(%s, %s, %s),  -- z, x, y
                    4096,  -- extent (tile size)
                    256,   -- buffer (to avoid clipping at edges)
                    true   -- clip geometry
                ) AS geom
            FROM filtered_sites
            WHERE ST_Intersects(
                geom,
                ST_Transform(ST_TileEnvelope(%s, %s, %s), 3857)
            )
        )
        SELECT ST_AsMVT(mvt_data.*, 'site_pairs', 4096, 'geom') AS mvt
        FROM mvt_data;
        """
        
        # Add tile coordinates to params (z, x, y repeated twice)
        params.extend([zoom, tile_x, tile_y, zoom, tile_x, tile_y])
        
        # Execute query
        with connection.cursor() as cursor:
            cursor.execute(sql, params)
            result = cursor.fetchone()
            
            if result and result[0]:
                mvt_data = bytes(result[0])
                return HttpResponse(mvt_data, content_type='application/x-protobuf')
            else:
                # Return empty tile
                return HttpResponse(b'', content_type='application/x-protobuf')
    
    except Exception as e:
        logger.error(f"Error generating MVT tile {z}/{x}/{y}: {str(e)}", exc_info=True)
        return HttpResponse(b'', content_type='application/x-protobuf', status=500)
