"""
Discharge Association Module for Hydropower Site Selection

This module links HEC-HMS discharge data to site pairs based on:
1. Extraction of representative discharge values (peak, average, percentile)
2. Assignment based on watershed/stream characteristics
3. Multiple return periods/scenarios support
4. Configurable discharge extraction methods

Note: Current implementation assigns discharge based on HMS element statistics
rather than spatial matching, as HMS elements lack geometry data in the database.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from django.conf import settings
from django.db.models import Max, Avg, Count
from scipy.spatial import cKDTree
import geopandas as gpd
from shapely.geometry import Point, Polygon
import logging
import os

logger = logging.getLogger(__name__)


@dataclass
class DischargeConfig:
    """Configuration for discharge extraction and association"""
    
    # Extraction method
    extraction_method: str = 'peak'  # Options: 'peak', 'average', 'median', 'flow_duration'
    percentile: float = 95.0  # For percentile method (e.g., 95th percentile)
    
    # Flow Duration Curve parameters (when extraction_method='flow_duration')
    exceedance_percentile: float = 30.0  # Q30 = discharge exceeded 30% of the time (design flow)
    
    # Discharge assignment strategy
    # 'drainage_area': Scale by drainage area (RECOMMENDED - most accurate)
    # 'spatial': Spatial matching with watershed/HMS elements
    # 'max': Use maximum discharge from all HMS elements
    # 'avg': Use average discharge across HMS elements
    # 'scaled': Scale discharge based on head/elevation (legacy method)
    assignment_strategy: str = 'drainage_area'
    
    # Spatial matching parameters
    max_search_distance: float = 2000.0  # Maximum distance to search for HMS elements (meters)
    use_watershed_overlay: bool = True  # Use watershed boundary overlay for matching
    drainage_area_tolerance: float = 0.3  # Tolerance for drainage area ratio (30%)
    
    # Return period filtering
    selected_return_period: Optional[str] = None  # e.g., '100-year', '2-year', None = use all
    available_return_periods: List[str] = None  # List of available return periods in HMS data
    
    # Drainage area weighted discharge parameters
    use_flow_accumulation: bool = True  # Use flow accumulation as proxy for drainage area
    basin_total_area_km2: Optional[float] = None  # Total basin area for scaling
    basin_outlet_discharge: Optional[float] = None  # Discharge at basin outlet for scaling
    
    # Units
    discharge_units: str = 'm³/s'
    
    def __post_init__(self):
        if self.available_return_periods is None:
            self.available_return_periods = []


class DischargeAssociator:
    """
    Main class for associating HEC-HMS discharge data with site pairs.
    
    Workflow:
    1. Load HMS run data from database (TimeSeries table)
    2. Extract representative discharge values (peak, average, etc.) per station
    3. Calculate discharge statistics across all HMS elements
    4. Assign discharge to site pairs based on scaling strategy
    
    Note: Since HMS elements (stations) don't have geometry in the database,
    we use a statistical approach rather than spatial proximity matching.
    """
    
    def __init__(self, config: Optional[DischargeConfig] = None):
        """
        Initialize discharge associator.
        
        Args:
            config: DischargeConfig object with extraction parameters
        """
        self.config = config or DischargeConfig()
        self.hms_run = None
        self.discharge_summary = None
        self.hms_spatial_index = None  # KDTree for spatial queries
        self.watershed_geometries = None  # Watershed polygons for overlay
        
    def extract_discharge_from_timeseries(self, hms_run_id: int, 
                                          station_id: str) -> Optional[float]:
        """
        Extract representative discharge value from time series data.
        
        Args:
            hms_run_id: HMSRun ID
            station_id: Station/element ID from TimeSeries
        
        Returns:
            Discharge value in m³/s, or None if not found
        """
        from hydropower.models import HMSRun, TimeSeries
        
        try:
            hms_run = HMSRun.objects.get(id=hms_run_id)
            
            # Query time series data for this station
            timeseries = TimeSeries.objects.filter(
                dataset=hms_run.dataset,
                station_id=station_id,
                data_type='DISCHARGE'
            ).order_by('datetime')
            
            if not timeseries.exists():
                logger.warning(f"No discharge data found for station {station_id}")
                return None
            
            # Extract values
            values = [ts.value for ts in timeseries]
            
            # Apply extraction method
            if self.config.extraction_method == 'peak':
                discharge = max(values)
            elif self.config.extraction_method == 'average':
                discharge = sum(values) / len(values)
            elif self.config.extraction_method == 'median':
                discharge = np.median(values)
            elif self.config.extraction_method == 'percentile':
                discharge = np.percentile(values, self.config.percentile)
            else:
                # Default to peak
                discharge = max(values)
            
            logger.debug(f"Extracted discharge for {station_id}: {discharge:.2f} m³/s ({self.config.extraction_method})")
            return discharge
            
        except Exception as e:
            logger.error(f"Error extracting discharge for station {station_id}: {e}")
            return None
    
    def create_discharge_summary(self, hms_run_id: int) -> Dict[str, float]:
        """
        Create summary of discharge values for all stations in HMS run.
        
        Args:
            hms_run_id: HMSRun ID
        
        Returns:
            Dictionary mapping station_id to discharge value
        """
        from hydropower.models import HMSRun, TimeSeries
        
        discharge_summary = {}
        
        try:
            hms_run = HMSRun.objects.get(id=hms_run_id)
            self.hms_run = hms_run
            
            # Get all unique station IDs from time series
            station_ids = TimeSeries.objects.filter(
                dataset=hms_run.dataset,
                data_type='DISCHARGE'
            ).values_list('station_id', flat=True).distinct()
            
            logger.info(f"Found {len(station_ids)} unique stations in HMS run")
            
            # Extract discharge for each station
            for station_id in station_ids:
                discharge = self.extract_discharge_from_timeseries(hms_run_id, station_id)
                if discharge is not None:
                    discharge_summary[station_id] = discharge
            
            self.discharge_summary = discharge_summary
            logger.info(f"Created discharge summary for {len(discharge_summary)} stations")
            logger.info(f"Discharge range: {min(discharge_summary.values()):.2f} - {max(discharge_summary.values()):.2f} m³/s")
            
        except Exception as e:
            logger.error(f"Error creating discharge summary: {e}")
        
        return discharge_summary
    
    def calculate_scaled_discharge(self, head: float, discharge_summary: Dict[str, float],
                                     flow_accumulation: Optional[float] = None,
                                     max_flow_accumulation: Optional[float] = None) -> float:
        """
        Calculate scaled discharge based on drainage area or head.
        
        PRIMARY METHOD: Drainage-area weighted (Q_site = Q_basin × A_site/A_basin)
        FALLBACK: Head-based scaling if drainage area not available
        
        Args:
            head: Head value for site pair (meters)
            discharge_summary: Dictionary of HMS discharge values
            flow_accumulation: Flow accumulation value at site (proxy for drainage area)
            max_flow_accumulation: Maximum flow accumulation in basin (at outlet)
        
        Returns:
            Scaled discharge value (m³/s)
        """
        if not discharge_summary:
            return None
        
        discharges = list(discharge_summary.values())
        max_q = max(discharges)  # Basin outlet discharge
        avg_q = sum(discharges) / len(discharges)
        
        # PRIMARY: Drainage-area weighted scaling using flow accumulation
        if (flow_accumulation is not None and max_flow_accumulation is not None 
            and max_flow_accumulation > 0):
            # Q_site = Q_basin × (A_site / A_basin)
            # Using flow accumulation as proxy for drainage area
            area_ratio = flow_accumulation / max_flow_accumulation
            area_ratio = np.clip(area_ratio, 0.01, 1.0)  # Ensure sensible range
            
            # Use basin outlet (max) discharge for scaling
            scaled_q = max_q * area_ratio
            
            logger.debug(f"Drainage-area weighted Q: {scaled_q:.2f} m³/s "
                        f"(area_ratio={area_ratio:.3f}, Q_basin={max_q:.2f})")
            return scaled_q
        
        # FALLBACK: Use configured basin values if available
        if (self.config.basin_outlet_discharge is not None and 
            self.config.basin_total_area_km2 is not None):
            # If we have basin-level info but not site-specific flow accumulation
            # Use average discharge as approximation
            return avg_q
        
        # LEGACY FALLBACK: Head-based scaling
        # For high head sites (>100m), use lower percentile of discharge
        # For low head sites (<50m), use higher percentile of discharge
        min_q = min(discharges)
        
        if head > 200:
            # High head sites: lower discharge adequate
            return min_q + (avg_q - min_q) * 0.5
        elif head > 100:
            # Medium-high head: average discharge
            return avg_q
        elif head > 50:
            # Medium head: above average
            return avg_q + (max_q - avg_q) * 0.3
        else:
            # Low head sites: higher discharge needed
            return avg_q + (max_q - avg_q) * 0.6
    
    def calculate_flow_duration_discharge(self, hms_run_id: int, 
                                          station_id: str) -> Optional[float]:
        """
        Extract flow duration curve discharge (Q30, Q50, etc.).
        
        Q30 = discharge exceeded 30% of the time (typical design flow for run-of-river)
        
        Args:
            hms_run_id: HMSRun ID
            station_id: Station/element ID
        
        Returns:
            Discharge at configured exceedance percentile
        """
        from hydropower.models import HMSRun, TimeSeries
        
        try:
            hms_run = HMSRun.objects.get(id=hms_run_id)
            
            timeseries = TimeSeries.objects.filter(
                dataset=hms_run.dataset,
                station_id=station_id,
                data_type='DISCHARGE'
            ).order_by('datetime')
            
            if not timeseries.exists():
                return None
            
            values = sorted([ts.value for ts in timeseries], reverse=True)
            
            if len(values) < 2:
                return values[0] if values else None
            
            # Calculate percentile (Q30 = 30th percentile from top)
            percentile_idx = int(len(values) * (self.config.exceedance_percentile / 100.0))
            percentile_idx = min(percentile_idx, len(values) - 1)
            
            q_exceedance = values[percentile_idx]
            
            logger.debug(f"Q{self.config.exceedance_percentile:.0f} for {station_id}: {q_exceedance:.2f} m³/s")
            return q_exceedance
            
        except Exception as e:
            logger.warning(f"Error calculating flow duration discharge: {e}")
            return None
    
    def assign_discharge_to_pair(self, pair_dict: Dict, discharge_summary: Dict[str, float],
                                   flow_accumulation: Optional[float] = None,
                                   max_flow_accumulation: Optional[float] = None) -> Dict:
        """
        Assign discharge to a site pair dictionary (before saving to database).
        
        Priority order:
        1. Drainage-area weighted (if flow_accumulation provided)
        2. Spatial matching (if configured)
        3. Head-based scaling (fallback)
        
        Args:
            pair_dict: Dictionary containing pair data (head, geometry, etc.)
            discharge_summary: Dictionary of HMS discharge values
            flow_accumulation: Flow accumulation at site inlet (drainage area proxy)
            max_flow_accumulation: Maximum flow accumulation in basin
        
        Returns:
            Updated pair dictionary with discharge assigned
        """
        if not discharge_summary:
            pair_dict['discharge'] = None
            pair_dict['discharge_method'] = 'none'
            return pair_dict
        
        head = pair_dict.get('head', 0)
        
        # PRIORITY 1: Drainage-area weighted (most accurate for hydrological scaling)
        if self.config.assignment_strategy == 'drainage_area':
            if flow_accumulation is not None and max_flow_accumulation is not None:
                discharge = self.calculate_scaled_discharge(
                    head, discharge_summary, flow_accumulation, max_flow_accumulation
                )
                if discharge is not None:
                    pair_dict['discharge'] = discharge
                    pair_dict['discharge_method'] = 'drainage_area_weighted'
                    pair_dict['area_ratio'] = flow_accumulation / max_flow_accumulation if max_flow_accumulation > 0 else None
                    return pair_dict
            else:
                logger.debug("Flow accumulation not available, falling back to spatial method")
        
        # PRIORITY 2: Spatial matching
        if self.config.assignment_strategy in ['spatial', 'drainage_area']:
            discharge = self._spatial_discharge_matching(pair_dict, discharge_summary)
            if discharge is not None:
                pair_dict['discharge'] = discharge
                pair_dict['discharge_method'] = 'spatial_match'
                return pair_dict
            else:
                logger.debug("Spatial matching failed, falling back to scaled method")
        
        # FALLBACK strategies
        if self.config.assignment_strategy == 'max':
            discharge = max(discharge_summary.values())
            pair_dict['discharge_method'] = 'max'
        elif self.config.assignment_strategy == 'avg':
            discharge = sum(discharge_summary.values()) / len(discharge_summary)
            pair_dict['discharge_method'] = 'avg'
        else:  # 'scaled' or final fallback
            discharge = self.calculate_scaled_discharge(head, discharge_summary)
            pair_dict['discharge_method'] = 'head_scaled'
        
        pair_dict['discharge'] = discharge
        
        return pair_dict
    
    def _spatial_discharge_matching(self, pair_dict: Dict, discharge_summary: Dict[str, float]) -> Optional[float]:
        """
        Match discharge to site pair using spatial analysis.
        
        Method:
        1. Find HMS element nearest to site inlet/outlet
        2. Verify it's within watershed contributing to the site
        3. Match discharge based on proximity and drainage area
        
        Args:
            pair_dict: Site pair dictionary with geometry
            discharge_summary: HMS discharge values by station
        
        Returns:
            Matched discharge value or None if no match found
        """
        try:
            # Get inlet geometry
            inlet_geom = pair_dict.get('inlet_geom')
            if inlet_geom is None:
                return None
            
            # Build spatial index if not already built
            if self.hms_spatial_index is None:
                self._build_hms_spatial_index(discharge_summary)
            
            if self.hms_spatial_index is None:
                return None
            
            # Find nearest HMS elements to inlet
            inlet_coords = np.array([inlet_geom.x, inlet_geom.y])
            distances, indices = self.hms_spatial_index.query(inlet_coords, k=min(5, len(self.hms_locations)))
            
            # Ensure distances and indices are arrays
            if not isinstance(distances, np.ndarray):
                distances = np.array([distances])
                indices = np.array([indices])
            
            # Find best match within search distance
            for dist, idx in zip(distances, indices):
                if dist <= self.config.max_search_distance:
                    station_id = self.hms_station_ids[idx]
                    discharge = discharge_summary.get(station_id)
                    
                    if discharge is not None:
                        logger.debug(f"Spatial match: HMS element '{station_id}' at {dist:.1f}m from inlet, Q={discharge:.2f}m³/s")
                        return discharge
            
            logger.debug(f"No HMS element within {self.config.max_search_distance}m of inlet")
            return None
            
        except Exception as e:
            logger.warning(f"Error in spatial discharge matching: {e}")
            return None
    
    def load_hms_basin_geometry(self, basin_file_path: str):
        """
        Load HMS element geometry from .basin file.
        
        Args:
            basin_file_path: Path to HEC-HMS .basin file
        """
        try:
            from hydropower.utils import parse_hms_basin_file
            
            # Parse basin file
            basin_data = parse_hms_basin_file(basin_file_path)
            
            # Extract all elements with centroids
            elements = []
            
            # Add subbasins
            for subbasin in basin_data.get('subbasins', []):
                if subbasin.get('centroid'):
                    elements.append({
                        'name': subbasin['name'],
                        'type': 'Subbasin',
                        'centroid': subbasin['centroid'],
                        'area_km2': subbasin.get('area_km2'),
                        'downstream': subbasin.get('downstream')
                    })
            
            # Add junctions
            for junction in basin_data.get('junctions', []):
                if junction.get('centroid'):
                    elements.append({
                        'name': junction['name'],
                        'type': 'Junction',
                        'centroid': junction['centroid'],
                        'area_km2': None,
                        'downstream': junction.get('downstream')
                    })
            
            # Add reaches (use to-point as centroid)
            for reach in basin_data.get('reaches', []):
                if reach.get('centroid'):
                    elements.append({
                        'name': reach['name'],
                        'type': 'Reach',
                        'centroid': reach['centroid'],
                        'area_km2': None,
                        'downstream': reach.get('downstream')
                    })
            
            # Add sinks
            for sink in basin_data.get('sinks', []):
                if sink.get('centroid'):
                    elements.append({
                        'name': sink['name'],
                        'type': 'Sink',
                        'centroid': sink['centroid'],
                        'area_km2': None,
                        'downstream': None
                    })
            
            self.hms_elements = elements
            logger.info(f"Loaded {len(elements)} HMS elements with geometry from .basin file")
            
        except Exception as e:
            logger.warning(f"Could not load HMS basin geometry: {e}")
            self.hms_elements = None
    
    def _build_hms_spatial_index(self, discharge_summary: Dict[str, float]):
        """Build spatial index of HMS element locations for fast queries"""
        try:
            # First, try to use loaded HMS basin geometry if available
            if hasattr(self, 'hms_elements') and self.hms_elements:
                # Filter elements that have discharge data
                valid_elements = [
                    elem for elem in self.hms_elements
                    if elem['name'] in discharge_summary
                ]
                
                if len(valid_elements) == 0:
                    logger.warning("No HMS elements with discharge data found")
                    return
                
                # Build spatial index from HMS element centroids
                self.hms_locations = np.array([elem['centroid'] for elem in valid_elements])
                self.hms_station_ids = [elem['name'] for elem in valid_elements]
                self.hms_drainage_areas = [elem['area_km2'] for elem in valid_elements]
                
                self.hms_spatial_index = cKDTree(self.hms_locations)
                logger.info(f"Built HMS spatial index with {len(self.hms_locations)} elements from .basin file")
                return
            
            # Fallback to old method using stream nodes as proxy
            from hydropower.models import TimeSeries
            
            if self.hms_run is None:
                return
            
            # Get unique station IDs
            station_ids = list(discharge_summary.keys())
            
            # Use stream network junctions as proxy locations
            from hydropower.models import StreamNode, StreamNetwork
            
            try:
                stream_nodes = StreamNode.objects.filter(
                    node_type__in=['confluence', 'outlet']
                )[:100]
                
                if len(stream_nodes) == 0:
                    logger.warning("No stream nodes found for spatial matching")
                    return
                
                # Create spatial index using stream node locations as proxy
                locations = np.array([[node.geometry.x, node.geometry.y] for node in stream_nodes])
                
                num_stations = len(station_ids)
                num_locations = len(locations)
                
                if num_locations < num_stations:
                    self.hms_locations = locations
                    self.hms_station_ids = station_ids[:num_locations]
                else:
                    self.hms_locations = locations[:num_stations]
                    self.hms_station_ids = station_ids
                
                self.hms_spatial_index = cKDTree(self.hms_locations)
                logger.info(f"Built HMS spatial index with {len(self.hms_locations)} locations (using stream nodes as proxy)")
                
            except Exception as e:
                logger.warning(f"Could not build HMS spatial index from stream nodes: {e}")
                return
            
        except Exception as e:
            logger.warning(f"Error building HMS spatial index: {e}")
            self.hms_spatial_index = None
    
    def get_discharge_statistics(self, hms_run_id: int) -> Dict[str, float]:
        """
        Get statistics on discharge values in HMS run.
        
        Args:
            hms_run_id: HMSRun ID
        
        Returns:
            Dictionary with min, max, mean, median discharge
        """
        discharge_summary = self.create_discharge_summary(hms_run_id)
        
        if not discharge_summary:
            return {}
        
        values = list(discharge_summary.values())
        
        stats = {
            'min': min(values),
            'max': max(values),
            'mean': sum(values) / len(values),
            'median': np.median(values),
            'count': len(values)
        }
        
        return stats


def associate_discharge_simple(hms_run_id: int, extraction_method: str = 'peak',
                               assignment_strategy: str = 'scaled') -> Dict[str, int]:
    """
    Simplified function for discharge association with default settings.
    
    This function is maintained for backward compatibility but is not the
    recommended approach. Use DischargeAssociator directly within the
    site pairing workflow instead.
    
    Args:
        hms_run_id: HMSRun ID to use for discharge data
        extraction_method: 'peak', 'average', or 'median'
        assignment_strategy: 'max', 'avg', or 'scaled'
    
    Returns:
        Dictionary with statistics: {'total': N, 'assigned': M, 'failed': K}
    
    Example:
        >>> stats = associate_discharge_simple(hms_run_id=1, extraction_method='peak')
        >>> print(f"Discharge data prepared from {stats['stations']} stations")
    """
    config = DischargeConfig(
        extraction_method=extraction_method,
        assignment_strategy=assignment_strategy
    )
    
    associator = DischargeAssociator(config=config)
    discharge_summary = associator.create_discharge_summary(hms_run_id)
    
    stats = {
        'stations': len(discharge_summary),
        'min_discharge': min(discharge_summary.values()) if discharge_summary else 0,
        'max_discharge': max(discharge_summary.values()) if discharge_summary else 0,
        'avg_discharge': sum(discharge_summary.values()) / len(discharge_summary) if discharge_summary else 0
    }
    
    logger.info(f"Discharge summary: {stats['stations']} stations, range {stats['min_discharge']:.2f}-{stats['max_discharge']:.2f} m³/s")
    
    return stats


def get_flow_accumulation_at_point(raster_layer, x: float, y: float) -> Optional[float]:
    """
    Extract flow accumulation value at a specific point from preprocessed DEM.
    
    Flow accumulation serves as a proxy for upstream drainage area.
    Q_site = Q_basin × (FlowAcc_site / FlowAcc_outlet)
    
    Args:
        raster_layer: RasterLayer model instance with flow_accumulation_path
        x, y: Coordinates of the point (in raster CRS)
    
    Returns:
        Flow accumulation value (number of upstream cells), or None if not available
    """
    import rasterio
    from rasterio.transform import rowcol
    
    try:
        # Check if flow accumulation raster exists
        if not raster_layer or not raster_layer.flow_accumulation_path:
            return None
        
        flow_acc_path = os.path.join(settings.MEDIA_ROOT, raster_layer.flow_accumulation_path)
        
        with rasterio.open(flow_acc_path) as src:
            # Convert coordinates to pixel indices
            row, col = rowcol(src.transform, x, y)
            
            # Check bounds
            if not (0 <= row < src.height and 0 <= col < src.width):
                return None
            
            # Read value at point
            flow_acc = src.read(1)[row, col]
            
            # Handle nodata
            if flow_acc == src.nodata or np.isnan(flow_acc):
                return None
            
            return float(flow_acc)
            
    except Exception as e:
        logger.warning(f"Error extracting flow accumulation at ({x}, {y}): {e}")
        return None


def get_max_flow_accumulation(raster_layer) -> Optional[float]:
    """
    Get maximum flow accumulation in the raster (basin outlet value).
    
    Args:
        raster_layer: RasterLayer model instance
    
    Returns:
        Maximum flow accumulation value
    """
    import rasterio
    
    try:
        if not raster_layer or not raster_layer.flow_accumulation_path:
            return None
        
        flow_acc_path = os.path.join(settings.MEDIA_ROOT, raster_layer.flow_accumulation_path)
        with rasterio.open(flow_acc_path) as src:
            flow_acc = src.read(1)
            
            # Mask nodata
            if src.nodata is not None:
                flow_acc = np.ma.masked_equal(flow_acc, src.nodata)
            
            max_val = float(np.max(flow_acc))
            return max_val
            
    except Exception as e:
        logger.warning(f"Error getting max flow accumulation: {e}")
        return None


def get_available_return_periods(raster_layer_id: int) -> List[Dict]:
    """
    Get available HMS runs and their return periods for a raster layer.
    
    Args:
        raster_layer_id: RasterLayer ID
    
    Returns:
        List of dicts with HMS run info: [{'id': 1, 'name': '...', 'return_period': '100-year'}]
    """
    from hydropower.models import HMSRun, RasterLayer
    
    try:
        raster_layer = RasterLayer.objects.get(id=raster_layer_id)
        
        # Find HMS runs associated with this raster's datasets
        hms_runs = HMSRun.objects.filter(
            dataset__id__in=[raster_layer.dataset_id]
        ).values('id', 'event_name', 'return_period', 'peak_discharge')
        
        return_periods = []
        for run in hms_runs:
            return_periods.append({
                'id': run['id'],
                'event_name': run['event_name'],
                'return_period': run['return_period'],
                'peak_discharge': run['peak_discharge']
            })
        
        logger.info(f"Found {len(return_periods)} HMS runs for raster {raster_layer_id}")
        return return_periods
        
    except Exception as e:
        logger.warning(f"Error getting return periods: {e}")
        return []


def associate_discharge_with_drainage_area(
    site_pairs_queryset,
    hms_run_id: int,
    raster_layer,
    return_period: Optional[str] = None,
    use_flow_duration: bool = False,
    exceedance_percentile: float = 30.0
) -> Dict[str, any]:
    """
    Associate discharge to site pairs using drainage-area weighted method.
    
    This is the RECOMMENDED method for accurate hydrological scaling.
    
    Formula: Q_site = Q_basin × (A_site / A_basin)
    Where A is approximated by flow accumulation values.
    
    Args:
        site_pairs_queryset: QuerySet of SitePair objects
        hms_run_id: HMSRun ID for discharge data
        raster_layer: RasterLayer with flow accumulation data
        return_period: Optional return period filter (e.g., '100-year')
        use_flow_duration: Use flow duration curve (Q30) instead of peak
        exceedance_percentile: Exceedance percentile for flow duration curve
    
    Returns:
        Statistics dict with counts and discharge range
    """
    from hydropower.models import SitePair
    
    # Configure discharge associator
    config = DischargeConfig(
        extraction_method='flow_duration' if use_flow_duration else 'peak',
        exceedance_percentile=exceedance_percentile,
        assignment_strategy='drainage_area',
        selected_return_period=return_period,
        use_flow_accumulation=True
    )
    
    associator = DischargeAssociator(config=config)
    
    # Get discharge summary from HMS
    discharge_summary = associator.create_discharge_summary(hms_run_id)
    
    if not discharge_summary:
        logger.warning("No discharge data available from HMS run")
        return {'updated': 0, 'failed': 0, 'no_data': True}
    
    # Get max flow accumulation (basin outlet)
    max_flow_acc = get_max_flow_accumulation(raster_layer)
    
    if max_flow_acc is None:
        logger.warning("Flow accumulation data not available, falling back to spatial method")
        max_flow_acc = 1.0  # Will trigger fallback in assign_discharge_to_pair
    
    # Process each site pair
    updated = 0
    failed = 0
    
    for site_pair in site_pairs_queryset:
        try:
            # Get flow accumulation at inlet
            inlet_x = site_pair.inlet.geometry.x
            inlet_y = site_pair.inlet.geometry.y
            flow_acc = get_flow_accumulation_at_point(raster_layer, inlet_x, inlet_y)
            
            # Build pair dict for association
            pair_dict = {
                'head': site_pair.head or 0,
                'inlet_geom': site_pair.inlet.geometry
            }
            
            # Assign discharge
            pair_dict = associator.assign_discharge_to_pair(
                pair_dict, discharge_summary, flow_acc, max_flow_acc
            )
            
            if pair_dict.get('discharge') is not None:
                site_pair.discharge = pair_dict['discharge']
                
                # Calculate power with net head (using head losses if available)
                gross_head = site_pair.head or 0
                
                # Apply head losses if penstock exists
                if site_pair.penstock_length and site_pair.penstock_length > 0:
                    from hydropower.site_pairing import calculate_head_losses
                    losses = calculate_head_losses(
                        site_pair.discharge, 
                        site_pair.penstock_length,
                        gross_head
                    )
                    net_head = gross_head - losses.get('total_head_loss', 0)
                else:
                    net_head = gross_head
                
                # Calculate power: P = ρ × g × Q × H × η
                efficiency = site_pair.efficiency or 0.7
                rho = 1000.0
                g = 9.81
                power = rho * g * site_pair.discharge * net_head * efficiency / 1000  # kW
                
                site_pair.power = max(0, power)  # Ensure non-negative
                site_pair.save(update_fields=['discharge', 'power'])
                
                updated += 1
                logger.debug(f"Site {site_pair.id}: Q={site_pair.discharge:.2f} m³/s, P={power:.1f} kW")
            else:
                failed += 1
                
        except Exception as e:
            logger.error(f"Error updating site pair {site_pair.id}: {e}")
            failed += 1
    
    stats = {
        'updated': updated,
        'failed': failed,
        'total': updated + failed,
        'max_flow_acc': max_flow_acc,
        'discharge_range': (
            min(discharge_summary.values()),
            max(discharge_summary.values())
        ) if discharge_summary else (0, 0)
    }
    
    logger.info(f"Drainage-area weighted discharge: {updated} updated, {failed} failed")
    
    return stats
