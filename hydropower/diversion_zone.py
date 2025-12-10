"""
Diversion Zone Analysis for Hydropower Site Selection

This module implements Step 2 of the hydropower site identification workflow:
- Search DEM around each inlet (upper) point
- Find DEM cells with elevation similar to the inlet's elevation
- Generate polygons representing potential water diversion/storage areas

These zones represent areas where water could be diverted or stored at constant head,
useful for identifying:
- Potential intake/weir locations
- Natural reservoir/storage areas
- Water diversion corridors

Based on the research methodology workflow:
1. Input: River point layer with coordinates and elevation
2. Step 1: Find inlet-outlet pairs with ≥10m head (implemented in site_pairing.py)
3. Step 2: Search DEM around upper point to find similar elevation zones (THIS MODULE)
"""

import rasterio
from rasterio.transform import rowcol, xy
from rasterio.features import shapes
from rasterio.mask import mask as rasterio_mask
import numpy as np
from shapely.geometry import Point, Polygon, MultiPolygon, shape, mapping
from shapely.ops import unary_union
from shapely.validation import make_valid
import geopandas as gpd
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
import logging
from django.contrib.gis.geos import Point as GEOSPoint, MultiPolygon as GEOSMultiPolygon, Polygon as GEOSPolygon
from scipy import ndimage
import time

logger = logging.getLogger(__name__)


@dataclass
class DiversionZoneConfig:
    """Configuration parameters for diversion zone analysis"""
    
    # Search parameters
    search_radius: float = 500.0  # meters - radius around inlet point to search
    elevation_tolerance: float = 2.0  # meters - allowed elevation difference from inlet
    
    # Minimum thresholds
    min_area_m2: float = 100.0  # minimum zone area to keep (m²)
    min_pixel_count: int = 10  # minimum number of pixels to form a valid zone
    
    # Contiguity settings
    require_contiguous: bool = False  # whether to require single contiguous zone
    fill_holes: bool = True  # fill small holes in the zone polygon
    max_hole_area_m2: float = 50.0  # maximum hole area to fill
    
    # Simplification
    simplify_tolerance: float = 5.0  # polygon simplification tolerance (meters)
    
    # Slope constraints (optional)
    max_slope_degrees: float = 30.0  # maximum slope for suitable areas
    apply_slope_filter: bool = False  # whether to filter by slope
    
    # Performance
    max_zones_per_run: int = 1000  # limit number of zones to generate


class DiversionZoneAnalyzer:
    """
    Analyzer for finding similar elevation zones around inlet points.
    
    This class implements the DEM-based search algorithm to find areas
    around each inlet point that have similar elevation, representing
    potential water diversion or storage zones.
    """
    
    def __init__(self, config: Optional[DiversionZoneConfig] = None):
        """
        Initialize the analyzer.
        
        Args:
            config: DiversionZoneConfig object with analysis parameters
        """
        self.config = config or DiversionZoneConfig()
        self.dem_path = None
        self.dem_array = None
        self.dem_transform = None
        self.dem_crs = None
        self.dem_nodata = None
        self.pixel_size = None
        self.slope_array = None  # Cached slope array
        
    def load_dem(self, dem_path: str):
        """
        Load DEM for elevation analysis.
        
        Args:
            dem_path: Path to DEM GeoTIFF file
        """
        self.dem_path = dem_path
        with rasterio.open(dem_path) as src:
            self.dem_array = src.read(1)
            self.dem_transform = src.transform
            self.dem_crs = src.crs
            self.dem_nodata = src.nodata
            self.dem_bounds = src.bounds
            # Get pixel size (assume square pixels)
            self.pixel_size = abs(src.transform[0])
            logger.info(f"Loaded DEM: {src.width}x{src.height}, pixel size: {self.pixel_size:.2f}m, CRS={src.crs}")
    
    def _calculate_slope(self) -> np.ndarray:
        """
        Calculate slope from DEM if not already cached.
        
        Returns:
            Slope array in degrees
        """
        if self.slope_array is not None:
            return self.slope_array
        
        logger.info("Calculating slope from DEM...")
        
        # Calculate gradient
        dy, dx = np.gradient(self.dem_array, self.pixel_size)
        
        # Calculate slope in degrees
        slope_radians = np.arctan(np.sqrt(dx**2 + dy**2))
        self.slope_array = np.degrees(slope_radians)
        
        # Handle nodata
        if self.dem_nodata is not None:
            nodata_mask = self.dem_array == self.dem_nodata
            self.slope_array[nodata_mask] = np.nan
        
        logger.info(f"Slope calculated: min={np.nanmin(self.slope_array):.1f}°, max={np.nanmax(self.slope_array):.1f}°")
        return self.slope_array
    
    def extract_elevation(self, x: float, y: float) -> Optional[float]:
        """
        Extract elevation from DEM at given coordinates.
        
        Args:
            x: X coordinate (easting)
            y: Y coordinate (northing)
        
        Returns:
            Elevation in meters, or None if outside bounds or nodata
        """
        if self.dem_array is None:
            raise ValueError("DEM not loaded. Call load_dem() first.")
        
        try:
            row, col = rowcol(self.dem_transform, x, y)
            
            if 0 <= row < self.dem_array.shape[0] and 0 <= col < self.dem_array.shape[1]:
                elevation = float(self.dem_array[row, col])
                if self.dem_nodata is not None and elevation == self.dem_nodata:
                    return None
                if elevation < -9999:
                    return None
                return elevation
            return None
        except Exception as e:
            logger.warning(f"Error extracting elevation at ({x}, {y}): {e}")
            return None
    
    def find_similar_elevation_zone(
        self,
        inlet_point: Point,
        inlet_elevation: Optional[float] = None,
        search_radius: Optional[float] = None,
        elevation_tolerance: Optional[float] = None
    ) -> Optional[Dict]:
        """
        Find DEM cells with similar elevation around an inlet point.
        
        This is the core algorithm implementing Step 2 of the workflow:
        1. Create a circular search window around the inlet point
        2. Find all DEM cells within the window that have elevation within tolerance
        3. Convert the matching cells to a polygon
        4. Calculate statistics and return results
        
        Args:
            inlet_point: Shapely Point geometry of the inlet location
            inlet_elevation: Elevation of inlet (extracted from DEM if not provided)
            search_radius: Override default search radius (meters)
            elevation_tolerance: Override default elevation tolerance (meters)
        
        Returns:
            Dictionary with zone geometry and statistics, or None if no valid zone found
        """
        start_time = time.time()
        
        # Use config values if not overridden
        search_radius = search_radius or self.config.search_radius
        elevation_tolerance = elevation_tolerance or self.config.elevation_tolerance
        
        # Get inlet coordinates
        inlet_x, inlet_y = inlet_point.x, inlet_point.y
        
        # Get inlet elevation if not provided
        if inlet_elevation is None:
            inlet_elevation = self.extract_elevation(inlet_x, inlet_y)
            if inlet_elevation is None:
                logger.warning(f"Could not extract elevation at inlet ({inlet_x}, {inlet_y})")
                return None
        
        # Calculate the bounding box for the search window
        min_x = inlet_x - search_radius
        max_x = inlet_x + search_radius
        min_y = inlet_y - search_radius
        max_y = inlet_y + search_radius
        
        # Convert to row/col indices
        # Note: rasterio's rowcol returns (row, col) where row increases downward
        try:
            # Get center pixel
            center_row, center_col = rowcol(self.dem_transform, inlet_x, inlet_y)
            
            # Calculate pixel radius
            pixel_radius = int(np.ceil(search_radius / self.pixel_size))
            
            # Define window bounds using pixel radius from center
            min_row = center_row - pixel_radius
            max_row = center_row + pixel_radius + 1  # +1 to include the edge
            min_col = center_col - pixel_radius
            max_col = center_col + pixel_radius + 1
            
            # Clamp to array bounds
            min_row = max(0, min_row)
            max_row = min(self.dem_array.shape[0], max_row)
            min_col = max(0, min_col)
            max_col = min(self.dem_array.shape[1], max_col)
            
            # Check if window has valid size
            if max_row - min_row < 3 or max_col - min_col < 3:
                logger.warning(f"Window too small for inlet at ({inlet_x:.1f}, {inlet_y:.1f}): {max_row-min_row}x{max_col-min_col}")
                return None
            
            # Check if center point is within bounds
            if not (min_row <= center_row < max_row and min_col <= center_col < max_col):
                logger.warning(f"Inlet at ({inlet_x:.1f}, {inlet_y:.1f}) is outside DEM bounds")
                return None
                
        except Exception as e:
            logger.warning(f"Error calculating window bounds: {e}")
            return None
        
        # Extract the DEM window
        dem_window = self.dem_array[min_row:max_row, min_col:max_col].copy()
        
        # Create distance mask (circular search window)
        # Create coordinate grids for the window
        window_rows = np.arange(min_row, max_row)
        window_cols = np.arange(min_col, max_col)
        col_grid, row_grid = np.meshgrid(window_cols, window_rows)
        
        # Calculate distance from center in pixels, then convert to meters
        row_dist = (row_grid - center_row) * self.pixel_size
        col_dist = (col_grid - center_col) * self.pixel_size
        distance_from_center = np.sqrt(row_dist**2 + col_dist**2)
        
        # Create circular mask
        circular_mask = distance_from_center <= search_radius
        
        # Create elevation mask (cells within tolerance of inlet elevation)
        elevation_diff = np.abs(dem_window - inlet_elevation)
        elevation_mask = elevation_diff <= elevation_tolerance
        
        # Handle nodata
        if self.dem_nodata is not None:
            nodata_mask = dem_window != self.dem_nodata
        else:
            nodata_mask = dem_window > -9999
        
        # Combine masks
        combined_mask = circular_mask & elevation_mask & nodata_mask
        
        # Apply slope filter if enabled
        if self.config.apply_slope_filter:
            slope_array = self._calculate_slope()
            slope_window = slope_array[min_row:max_row, min_col:max_col]
            slope_mask = slope_window <= self.config.max_slope_degrees
            combined_mask = combined_mask & slope_mask
        
        # Count matching pixels
        pixel_count = np.sum(combined_mask)
        
        if pixel_count < self.config.min_pixel_count:
            logger.debug(f"Zone at ({inlet_x}, {inlet_y}) has only {pixel_count} pixels, below threshold")
            return None
        
        # Convert mask to polygon(s)
        # Create a binary raster for vectorization
        binary_mask = combined_mask.astype(np.uint8)
        
        # Fill small holes if configured
        if self.config.fill_holes:
            binary_mask = ndimage.binary_fill_holes(binary_mask).astype(np.uint8)
        
        # Vectorize the mask to polygons
        # Create a transform for the window based on pixel coordinates
        window_transform = rasterio.transform.Affine(
            self.dem_transform.a,  # pixel width
            self.dem_transform.b,  # rotation (usually 0)
            self.dem_transform.c + min_col * self.dem_transform.a,  # x origin offset
            self.dem_transform.d,  # rotation (usually 0)
            self.dem_transform.e,  # pixel height (negative)
            self.dem_transform.f + min_row * self.dem_transform.e   # y origin offset
        )
        
        # Extract shapes from binary mask
        polygons = []
        try:
            for geom, value in shapes(binary_mask, mask=binary_mask == 1, transform=window_transform):
                if value == 1:
                    poly = shape(geom)
                    if poly.is_valid and poly.area >= self.config.min_area_m2:
                        polygons.append(poly)
        except Exception as e:
            logger.warning(f"Error vectorizing zone: {e}")
            return None
        
        if not polygons:
            logger.debug(f"No valid polygons generated for inlet at ({inlet_x}, {inlet_y})")
            return None
        
        # Merge polygons and simplify
        if len(polygons) == 1:
            zone_geom = polygons[0]
        else:
            zone_geom = unary_union(polygons)
        
        # Make valid and simplify
        zone_geom = make_valid(zone_geom)
        if self.config.simplify_tolerance > 0:
            zone_geom = zone_geom.simplify(self.config.simplify_tolerance, preserve_topology=True)
        
        # Ensure MultiPolygon type
        if isinstance(zone_geom, Polygon):
            zone_geom = MultiPolygon([zone_geom])
        elif not isinstance(zone_geom, MultiPolygon):
            # Handle GeometryCollection or other types
            polys = [g for g in zone_geom.geoms if isinstance(g, Polygon)]
            if not polys:
                return None
            zone_geom = MultiPolygon(polys)
        
        # Calculate statistics
        area_m2 = zone_geom.area
        if area_m2 < self.config.min_area_m2:
            logger.debug(f"Zone area {area_m2:.1f} m² below threshold")
            return None
        
        # Get elevations within zone for statistics
        zone_elevations = dem_window[combined_mask]
        
        # Get slope statistics if available
        mean_slope = None
        max_slope = None
        if self.slope_array is not None:
            slope_window = self.slope_array[min_row:max_row, min_col:max_col]
            zone_slopes = slope_window[combined_mask]
            mean_slope = float(np.nanmean(zone_slopes))
            max_slope = float(np.nanmax(zone_slopes))
        
        # Calculate fragment statistics
        fragment_count = len(zone_geom.geoms)
        largest_fragment_area = max(g.area for g in zone_geom.geoms)
        
        processing_time = time.time() - start_time
        
        result = {
            'geometry': zone_geom,
            'centroid': zone_geom.centroid,
            'reference_elevation': inlet_elevation,
            'elevation_tolerance': elevation_tolerance,
            'search_radius': search_radius,
            'min_elevation': float(np.min(zone_elevations)),
            'max_elevation': float(np.max(zone_elevations)),
            'mean_elevation': float(np.mean(zone_elevations)),
            'area_m2': area_m2,
            'area_hectares': area_m2 / 10000.0,
            'perimeter_m': zone_geom.length,
            'pixel_count': int(pixel_count),
            'mean_slope': mean_slope,
            'max_slope': max_slope,
            'is_contiguous': fragment_count == 1,
            'fragment_count': fragment_count,
            'largest_fragment_area_m2': largest_fragment_area,
            'processing_time_seconds': processing_time
        }
        
        return result
    
    def analyze_site_pairs(
        self,
        raster_layer_id: int,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Analyze all inlet points from site pairs and generate diversion zones.
        
        This is the main entry point for batch processing.
        
        Args:
            raster_layer_id: ID of the RasterLayer to process
            limit: Maximum number of site pairs to process
        
        Returns:
            List of zone result dictionaries
        """
        from hydropower.models import SitePair, SitePoint, RasterLayer
        
        # Get the raster layer and load DEM
        raster_layer = RasterLayer.objects.select_related('dataset').get(id=raster_layer_id)
        dem_path = raster_layer.dataset.file.path
        self.load_dem(dem_path)
        
        # Get unique inlet points from site pairs
        site_pairs = SitePair.objects.filter(
            raster_layer_id=raster_layer_id,
            is_feasible=True
        ).select_related('inlet')
        
        if limit:
            site_pairs = site_pairs[:limit]
        
        # Get unique inlets
        processed_inlets = set()
        results = []
        
        for pair in site_pairs:
            inlet = pair.inlet
            if inlet.id in processed_inlets:
                continue
            processed_inlets.add(inlet.id)
            
            # Create shapely point from GEOS point
            inlet_point = Point(inlet.geometry.x, inlet.geometry.y)
            inlet_elevation = inlet.elevation
            
            # Find diversion zone
            zone_result = self.find_similar_elevation_zone(
                inlet_point=inlet_point,
                inlet_elevation=inlet_elevation
            )
            
            if zone_result:
                zone_result['inlet_point_id'] = inlet.id
                zone_result['site_id'] = inlet.site_id
                zone_result['raster_layer_id'] = raster_layer_id
                results.append(zone_result)
                logger.info(f"Generated zone for {inlet.site_id}: {zone_result['area_hectares']:.2f} ha")
            
            # Check limit
            if len(results) >= self.config.max_zones_per_run:
                logger.warning(f"Reached max zones limit ({self.config.max_zones_per_run})")
                break
        
        logger.info(f"Generated {len(results)} diversion zones from {len(processed_inlets)} inlet points")
        return results
    
    def save_zones_to_database(
        self,
        zones: List[Dict],
        raster_layer_id: int
    ) -> int:
        """
        Save generated diversion zones to the database.
        
        Args:
            zones: List of zone result dictionaries
            raster_layer_id: ID of the source RasterLayer
        
        Returns:
            Number of zones saved
        """
        from hydropower.models import DiversionZone, SitePoint, RasterLayer, SitePair
        from django.contrib.gis.geos import GEOSGeometry
        
        saved_count = 0
        
        for zone_data in zones:
            try:
                # Get the inlet point
                inlet_point = SitePoint.objects.get(id=zone_data['inlet_point_id'])
                
                # Convert shapely geometry to GEOS
                geom_wkt = zone_data['geometry'].wkt
                geos_geom = GEOSGeometry(geom_wkt, srid=32651)
                
                # Ensure it's a MultiPolygon
                if geos_geom.geom_type == 'Polygon':
                    geos_geom = GEOSMultiPolygon(geos_geom, srid=32651)
                
                # Create centroid
                centroid_wkt = zone_data['centroid'].wkt
                centroid_geos = GEOSGeometry(centroid_wkt, srid=32651)
                
                # Generate zone ID
                zone_id = f"DZ_{zone_data['site_id']}"
                
                # Create or update the zone
                zone, created = DiversionZone.objects.update_or_create(
                    zone_id=zone_id,
                    defaults={
                        'raster_layer_id': raster_layer_id,
                        'inlet_point': inlet_point,
                        'geometry': geos_geom,
                        'centroid': centroid_geos,
                        'reference_elevation': zone_data['reference_elevation'],
                        'elevation_tolerance': zone_data['elevation_tolerance'],
                        'search_radius': zone_data['search_radius'],
                        'min_elevation': zone_data['min_elevation'],
                        'max_elevation': zone_data['max_elevation'],
                        'mean_elevation': zone_data['mean_elevation'],
                        'area_m2': zone_data['area_m2'],
                        'area_hectares': zone_data['area_hectares'],
                        'perimeter_m': zone_data['perimeter_m'],
                        'pixel_count': zone_data['pixel_count'],
                        'mean_slope': zone_data.get('mean_slope'),
                        'max_slope': zone_data.get('max_slope'),
                        'is_contiguous': zone_data['is_contiguous'],
                        'fragment_count': zone_data['fragment_count'],
                        'largest_fragment_area_m2': zone_data['largest_fragment_area_m2'],
                        'processing_time_seconds': zone_data.get('processing_time_seconds'),
                    }
                )
                
                # Link to site pair(s) that use this inlet
                site_pairs = SitePair.objects.filter(inlet=inlet_point)
                for pair in site_pairs:
                    pair.diversion_zone = zone
                    pair.save(update_fields=['diversion_zone'])
                
                saved_count += 1
                logger.debug(f"Saved zone {zone_id} (created={created})")
                
            except Exception as e:
                logger.error(f"Error saving zone for inlet {zone_data.get('site_id', 'unknown')}: {e}")
                continue
        
        logger.info(f"Saved {saved_count} diversion zones to database")
        return saved_count


def generate_diversion_zones(
    raster_layer_id: int,
    search_radius: float = 500.0,
    elevation_tolerance: float = 2.0,
    limit: Optional[int] = None
) -> Dict:
    """
    Main entry point for generating diversion zones for a raster layer.
    
    This function:
    1. Loads the DEM from the raster layer
    2. Gets all inlet points from feasible site pairs
    3. Searches for similar elevation zones around each inlet
    4. Saves the zones to the database
    5. Links zones to their site pairs
    
    Args:
        raster_layer_id: ID of the RasterLayer (DEM) to process
        search_radius: Radius around inlet to search (meters)
        elevation_tolerance: Allowed elevation difference (meters)
        limit: Maximum number of zones to generate
    
    Returns:
        Summary dictionary with counts and statistics
    """
    start_time = time.time()
    
    # Configure analyzer
    config = DiversionZoneConfig(
        search_radius=search_radius,
        elevation_tolerance=elevation_tolerance,
        max_zones_per_run=limit or 1000
    )
    
    analyzer = DiversionZoneAnalyzer(config)
    
    # Generate zones
    zones = analyzer.analyze_site_pairs(raster_layer_id, limit=limit)
    
    # Save to database
    saved_count = analyzer.save_zones_to_database(zones, raster_layer_id)
    
    # Calculate summary statistics
    total_area = sum(z['area_m2'] for z in zones)
    avg_area = total_area / len(zones) if zones else 0
    
    processing_time = time.time() - start_time
    
    return {
        'zones_generated': len(zones),
        'zones_saved': saved_count,
        'total_area_m2': total_area,
        'total_area_hectares': total_area / 10000.0,
        'average_area_m2': avg_area,
        'average_area_hectares': avg_area / 10000.0,
        'search_radius': search_radius,
        'elevation_tolerance': elevation_tolerance,
        'processing_time_seconds': processing_time
    }
