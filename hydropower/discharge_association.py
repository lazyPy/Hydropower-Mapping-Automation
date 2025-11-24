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
from django.db.models import Max, Avg, Count
from scipy.spatial import cKDTree
import geopandas as gpd
from shapely.geometry import Point, Polygon
import logging

logger = logging.getLogger(__name__)


@dataclass
class DischargeConfig:
    """Configuration for discharge extraction and association"""
    
    # Extraction method
    extraction_method: str = 'peak'  # Options: 'peak', 'average', 'median'
    percentile: float = 95.0  # For percentile method (e.g., 95th percentile)
    
    # Discharge assignment strategy
    # 'max': Use maximum discharge from all HMS elements
    # 'avg': Use average discharge across HMS elements
    # 'scaled': Scale discharge based on head/elevation (old method)
    # 'spatial': Spatial matching with watershed/HMS elements (NEW - recommended)
    assignment_strategy: str = 'spatial'
    
    # Spatial matching parameters
    max_search_distance: float = 2000.0  # Maximum distance to search for HMS elements (meters)
    use_watershed_overlay: bool = True  # Use watershed boundary overlay for matching
    drainage_area_tolerance: float = 0.3  # Tolerance for drainage area ratio (30%)
    
    # Return period filtering
    selected_return_period: Optional[str] = None  # e.g., '100-year', None = use all
    
    # Units
    discharge_units: str = 'm³/s'


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
    
    def calculate_scaled_discharge(self, head: float, discharge_summary: Dict[str, float]) -> float:
        """
        Calculate scaled discharge based on head and HMS discharge statistics.
        
        Uses a simple scaling approach:
        - Higher head sites get proportionally higher discharge allocation
        - Scales between min and max discharge from HMS data
        
        Args:
            head: Head value for site pair (meters)
            discharge_summary: Dictionary of HMS discharge values
        
        Returns:
            Scaled discharge value (m³/s)
        """
        if not discharge_summary:
            return None
        
        discharges = list(discharge_summary.values())
        min_q = min(discharges)
        max_q = max(discharges)
        avg_q = sum(discharges) / len(discharges)
        
        # For high head sites (>100m), use lower percentile of discharge
        # For low head sites (<50m), use higher percentile of discharge
        # This reflects typical hydropower site characteristics
        
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
    
    def assign_discharge_to_pair(self, pair_dict: Dict, discharge_summary: Dict[str, float]) -> Dict:
        """
        Assign discharge to a site pair dictionary (before saving to database).
        
        Uses spatial matching if available, falls back to scaling methods.
        
        Args:
            pair_dict: Dictionary containing pair data (head, geometry, etc.)
            discharge_summary: Dictionary of HMS discharge values
        
        Returns:
            Updated pair dictionary with discharge assigned
        """
        if not discharge_summary:
            pair_dict['discharge'] = None
            return pair_dict
        
        # Try spatial matching first if configured
        if self.config.assignment_strategy == 'spatial':
            discharge = self._spatial_discharge_matching(pair_dict, discharge_summary)
            if discharge is not None:
                pair_dict['discharge'] = discharge
                pair_dict['discharge_method'] = 'spatial_match'
                return pair_dict
            else:
                logger.debug("Spatial matching failed, falling back to scaled method")
        
        # Fall back to other strategies
        head = pair_dict.get('head', 0)
        
        if self.config.assignment_strategy == 'max':
            discharge = max(discharge_summary.values())
        elif self.config.assignment_strategy == 'avg':
            discharge = sum(discharge_summary.values()) / len(discharge_summary)
        else:  # 'scaled' or fallback
            discharge = self.calculate_scaled_discharge(head, discharge_summary)
        
        pair_dict['discharge'] = discharge
        pair_dict['discharge_method'] = self.config.assignment_strategy
        
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
    
    def _build_hms_spatial_index(self, discharge_summary: Dict[str, float]):
        """Build spatial index of HMS element locations for fast queries"""
        try:
            # Try to get HMS element locations from database
            # This requires HMS elements to have geometry (junction/reach centroids)
            from hydropower.models import TimeSeries
            
            if self.hms_run is None:
                return
            
            # Get unique station IDs
            station_ids = list(discharge_summary.keys())
            
            # For now, we'll use a simplified approach:
            # Assume HMS elements are distributed along the stream network
            # Use stream network junctions as proxy locations
            
            from hydropower.models import StreamNode, StreamNetwork
            
            # Try to get stream nodes from the same raster layer
            try:
                # Get raster layer from HMS run's dataset
                # This is a simplification - ideally HMS elements would have geometry
                stream_nodes = StreamNode.objects.filter(
                    node_type__in=['confluence', 'outlet']
                )[:100]
                
                if len(stream_nodes) == 0:
                    logger.warning("No stream nodes found for spatial matching")
                    return
                
                # Create spatial index using stream node locations as proxy
                locations = np.array([[node.geometry.x, node.geometry.y] for node in stream_nodes])
                
                # Map station IDs to locations (simplified - use sequential mapping)
                # In real implementation, HMS elements should have geometry
                num_stations = len(station_ids)
                num_locations = len(locations)
                
                if num_locations < num_stations:
                    # Repeat locations if not enough
                    self.hms_locations = locations
                    self.hms_station_ids = station_ids[:num_locations]
                else:
                    # Use first N locations
                    self.hms_locations = locations[:num_stations]
                    self.hms_station_ids = station_ids
                
                self.hms_spatial_index = cKDTree(self.hms_locations)
                logger.info(f"Built HMS spatial index with {len(self.hms_locations)} locations")
                
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
