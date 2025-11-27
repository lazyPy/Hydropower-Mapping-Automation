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
    return render(request, 'map_view.html')


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
                        'properties': {'type': 'powerhouse', 'pair_id': pair.pair_id, 'rank': pair.rank, 'power_kw': round(pair.power, 2) if pair.power else None, 'elevation_m': round(powerhouse_elevation, 2) if powerhouse_elevation else None}
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
