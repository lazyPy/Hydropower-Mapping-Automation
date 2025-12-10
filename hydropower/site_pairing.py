"""
Inlet-Outlet Pairing Algorithm for Hydropower Site Selection

This module implements the core algorithm for identifying and pairing
inlet and outlet points along a stream network to create candidate
hydropower sites. It includes head calculations, constraint filtering,
and multi-criteria scoring.

Based on research methodology using DEM analysis and HEC-HMS discharge data.
"""

import rasterio
from rasterio.transform import rowcol
import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging
from django.contrib.gis.geos import Point as GEOSPoint, LineString as GEOSLineString
import networkx as nx
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


def calculate_head_losses(discharge: float, penstock_length: float, gross_head: float,
                         penstock_roughness: float = 0.012, velocity_factor: float = 4.0,
                         entrance_loss_coeff: float = 0.5, exit_loss_coeff: float = 1.0,
                         bend_loss_coeff: float = 0.25, num_bends: int = 2) -> Dict[str, float]:
    """
    Calculate head losses in penstock and fittings (standalone function).
    
    This is the recommended function for calculating net head from gross head.
    Uses Hazen-Williams formula for friction losses and standard coefficients
    for minor losses.
    
    Formula: Net Head = Gross Head - Friction Loss - Minor Losses
    
    Args:
        discharge: Flow rate (m³/s)
        penstock_length: Penstock pipe length (m)
        gross_head: Gross head (m)
        penstock_roughness: Manning's n for pipe material (default 0.012 for steel)
        velocity_factor: Design velocity in penstock (m/s, default 4.0)
        entrance_loss_coeff: Entrance loss coefficient (default 0.5)
        exit_loss_coeff: Exit loss coefficient (default 1.0)
        bend_loss_coeff: Loss per 90° bend (default 0.25)
        num_bends: Number of 90° bends in penstock (default 2)
    
    Returns:
        Dictionary with loss components and net head:
        - friction_loss: Friction head loss (m)
        - entrance_loss: Entrance head loss (m)
        - exit_loss: Exit head loss (m)
        - bend_loss: Bend head losses (m)
        - total_minor_loss: Sum of minor losses (m)
        - total_head_loss: Total head loss (m)
        - net_head: Net head after losses (m)
        - gross_head: Original gross head (m)
        - efficiency_factor: Net/Gross ratio (0-1)
    
    Example:
        >>> losses = calculate_head_losses(discharge=5.0, penstock_length=200, gross_head=100)
        >>> print(f"Net head: {losses['net_head']:.1f}m (efficiency: {losses['efficiency_factor']*100:.1f}%)")
    """
    g = 9.81  # gravitational acceleration
    
    if discharge <= 0 or gross_head <= 0:
        return {
            'friction_loss': 0, 
            'entrance_loss': 0,
            'exit_loss': 0,
            'bend_loss': 0,
            'total_minor_loss': 0,
            'total_head_loss': 0,
            'net_head': gross_head,
            'gross_head': gross_head,
            'efficiency_factor': 1.0
        }
    
    # Calculate penstock diameter based on velocity
    velocity = velocity_factor
    diameter = np.sqrt((4 * discharge) / (np.pi * velocity))
    
    # Friction loss using Hazen-Williams formula (common in hydropower)
    # hf = 10.67 * L * Q^1.852 / (C^1.852 * D^4.87)
    C = 140  # Hazen-Williams coefficient for steel pipe
    friction_loss = 10.67 * penstock_length * (discharge ** 1.852) / ((C ** 1.852) * (diameter ** 4.87))
    
    # Minor losses
    entrance_loss = entrance_loss_coeff * (velocity ** 2) / (2 * g)
    exit_loss = exit_loss_coeff * (velocity ** 2) / (2 * g)
    bend_loss = num_bends * bend_loss_coeff * (velocity ** 2) / (2 * g)
    
    total_minor_loss = entrance_loss + exit_loss + bend_loss
    total_head_loss = friction_loss + total_minor_loss
    
    # Net head (ensure non-negative)
    net_head = max(0, gross_head - total_head_loss)
    
    # Efficiency factor
    efficiency_factor = net_head / gross_head if gross_head > 0 else 0
    
    return {
        'friction_loss': friction_loss,
        'entrance_loss': entrance_loss,
        'exit_loss': exit_loss,
        'bend_loss': bend_loss,
        'total_minor_loss': total_minor_loss,
        'total_head_loss': total_head_loss,
        'net_head': net_head,
        'gross_head': gross_head,
        'efficiency_factor': efficiency_factor
    }


@dataclass
class PairingConfig:
    """Configuration parameters for inlet-outlet pairing algorithm"""
    
    # Head constraints - LOWERED for micro/small hydro potential
    min_head: float = 5.0  # meters (was 10.0, lowered for micro-hydro)
    max_head: float = 500.0  # meters
    
    # Distance constraints - EXPANDED for more site discovery
    min_river_distance: float = 20.0  # meters (was 50.0, lowered for short streams)
    max_river_distance: float = 8000.0  # meters (was 5000.0)
    
    # Spacing and buffer
    spacing_buffer: float = 100.0  # meters (was 200.0, allow closer sites)
    land_buffer: float = 50.0  # meters (proximity to riverbanks)
    
    # Scoring weights (now configurable with validation)
    weight_head: float = 0.4
    weight_discharge: float = 0.3
    weight_distance: float = 0.2
    weight_accessibility: float = 0.1
    
    # Efficiency factor
    efficiency: float = 0.7  # 70% turbine efficiency (range: 0.6-0.85)
    
    # Head loss coefficients (for accurate power calculation)
    penstock_roughness: float = 0.012  # Manning's n for steel pipe
    entrance_loss_coeff: float = 0.5  # Entrance loss coefficient
    exit_loss_coeff: float = 1.0  # Exit loss coefficient
    bend_loss_coeff: float = 0.25  # Loss per 90° bend
    velocity_factor: float = 4.0  # Design velocity in penstock (m/s)
    
    # Performance limits (to avoid excessive computation)
    max_stream_segments: int = 5000  # Maximum number of stream segments to process
    max_candidates_per_type: int = 2000  # Maximum inlet/outlet candidates (was 1000)
    
    # Physical constants
    rho: float = 1000.0  # kg/m³ (water density)
    g: float = 9.81  # m/s² (gravitational acceleration)
    
    # Search parameters
    max_outlets_per_inlet: int = 15  # Maximum outlets per inlet (was 10)
    min_stream_order: int = 1  # Minimum Strahler order (was 2, now captures all streams)
    main_river_only: bool = True  # Focus on main river stem/major reaches (practical sites)
    min_main_river_order: int = 2  # Minimum stream order to consider as "main river" (when main_river_only=True)
    min_main_river_length_m: float = 12.0  # Minimum stream length (meters) - will use top 20% if no streams meet threshold
    
    # Node densification parameters (for systematic IO pairing)
    node_spacing: float = 50.0  # Spacing between nodes along channel (meters, was 100.0)
    use_node_densification: bool = True  # Enable systematic node-based pairing
    
    # Flow accumulation parameters (for pair-specific discharge)
    use_flow_accumulation: bool = False  # Use FA raster for discharge calculation
    q_outlet_design: Optional[float] = None  # Design Q at outlet for FA calibration (m³/s)
    constant_q_fallback: float = 7.0  # Fallback Q if FA not available (m³/s)
    
    # Land feasibility parameters
    max_slope_percent: float = 45.0  # Maximum terrain slope for infrastructure (was 30%)
    min_road_distance: float = 10000.0  # Maximum distance from road (was 5000m)
    check_land_use: bool = True  # Whether to validate land use suitability
    
    # Infrastructure validation parameters
    validate_infrastructure_slope: bool = True  # Validate penstock/channel slope feasibility
    max_penstock_slope: float = 80.0  # Maximum penstock slope in percent (near-vertical OK)
    max_channel_slope: float = 5.0  # Maximum headrace channel slope in percent
    
    def validate_weights(self):
        """Validate that scoring weights sum to 1.0"""
        total = self.weight_head + self.weight_discharge + self.weight_distance + self.weight_accessibility
        if not np.isclose(total, 1.0, atol=0.01):
            logger.warning(f"Scoring weights sum to {total:.3f}, not 1.0. Normalizing...")
            # Auto-normalize
            self.weight_head /= total
            self.weight_discharge /= total
            self.weight_distance /= total
            self.weight_accessibility /= total


class InletOutletPairing:
    """
    Main class for inlet-outlet pairing algorithm.
    
    Workflow:
    1. Load stream network and DEM
    2. Identify inlet candidates (upstream points)
    3. Identify outlet candidates (downstream points)
    4. For each inlet, search downstream for feasible outlets
    5. Calculate head from DEM elevations
    6. Apply constraint filters
    7. Score and rank site pairs
    8. Deduplicate and enforce spacing
    9. Save to PostGIS
    """
    
    def __init__(self, config: Optional[PairingConfig] = None, raster_layer_id: Optional[int] = None):
        """
        Initialize the pairing algorithm.
        
        Args:
            config: PairingConfig object with algorithm parameters
            raster_layer_id: ID of RasterLayer in database (for persisting main river flags)
        """
        self.config = config or PairingConfig()
        self.config.validate_weights()  # Validate and normalize weights
        self.dem_path = None
        self.dem_array = None
        self.dem_transform = None
        self.dem_crs = None
        self.stream_network = None
        self.stream_nodes = None
        self.stream_graph = None  # NetworkX graph for connectivity
        self.node_spatial_index = None  # KDTree for spatial queries
        self.stream_spatial_index = None  # STRtree for stream distance queries
        self.watershed_polygons = None  # Watershed polygons from database
        self.watershed_gdf = None  # GeoDataFrame of watershed polygons for spatial queries
        self.flowacc_path = None  # Flow accumulation raster path
        self.flowacc_array = None  # Flow accumulation array
        self.flowacc_transform = None  # Flow accumulation transform
        self.q_scale = None  # Q scale factor (m³/s per FA unit)
        self.raster_layer_id = raster_layer_id  # For persisting main river flags to database
        
    def load_dem(self, dem_path: str):
        """
        Load DEM for elevation extraction.
        
        Args:
            dem_path: Path to DEM GeoTIFF file
        """
        self.dem_path = dem_path
        with rasterio.open(dem_path) as src:
            self.dem_array = src.read(1)
            self.dem_transform = src.transform
            self.dem_crs = src.crs
            logger.info(f"Loaded DEM: {src.width}x{src.height}, CRS={src.crs}")
    
    def load_flow_accumulation(self, flowacc_path: str, outlet_point: Optional[Point] = None):
        """
        Load flow accumulation raster and calibrate Q scale factor.
        
        Args:
            flowacc_path: Path to flow accumulation GeoTIFF
            outlet_point: Outlet point for calibration (optional)
        """
        self.flowacc_path = flowacc_path
        with rasterio.open(flowacc_path) as src:
            self.flowacc_array = src.read(1)
            self.flowacc_transform = src.transform
            logger.info(f"Loaded Flow Accumulation: {src.width}x{src.height}")
        
        # Calibrate Q scale from outlet if provided
        if outlet_point and self.config.q_outlet_design:
            fa_outlet = self._extract_from_raster(
                outlet_point.x, outlet_point.y,
                self.flowacc_array, self.flowacc_transform
            )
            if fa_outlet and fa_outlet > 0:
                self.q_scale = self.config.q_outlet_design / fa_outlet
                logger.info(
                    f"Calibrated Q scale: FA_outlet={fa_outlet:.1f}, "
                    f"Q_outlet={self.config.q_outlet_design:.3f} m³/s, "
                    f"Q_scale={self.q_scale:.6f} m³/s per FA unit"
                )
            else:
                logger.warning("Failed to sample FA at outlet, using constant Q fallback")
                self.q_scale = None
        else:
            logger.info("No outlet calibration provided, will use constant Q fallback")
            self.q_scale = None
    
    def _extract_from_raster(self, x: float, y: float, 
                            array: np.ndarray, transform) -> Optional[float]:
        """
        Extract value from raster at given coordinates.
        
        Args:
            x: X coordinate
            y: Y coordinate
            array: Raster array
            transform: Raster transform
        
        Returns:
            Raster value, or None if outside bounds
        """
        try:
            row, col = rowcol(transform, x, y)
            if 0 <= row < array.shape[0] and 0 <= col < array.shape[1]:
                value = float(array[row, col])
                if value < -9999:  # Nodata
                    return None
                return value
            return None
        except Exception as e:
            logger.debug(f"Error extracting from raster: {e}")
            return None
    
    def compute_discharge_at_point(self, point: Point) -> float:
        """
        Compute discharge at a point using flow accumulation and calibrated Q scale.
        
        Falls back to constant Q if FA not available or calibration failed.
        
        Args:
            point: Shapely Point geometry
        
        Returns:
            Discharge in m³/s
        """
        if self.config.use_flow_accumulation and self.q_scale and self.flowacc_array is not None:
            fa_val = self._extract_from_raster(
                point.x, point.y,
                self.flowacc_array, self.flowacc_transform
            )
            if fa_val and fa_val > 0:
                q_computed = fa_val * self.q_scale
                logger.debug(f"Computed Q at ({point.x:.1f}, {point.y:.1f}): FA={fa_val:.1f}, Q={q_computed:.3f} m³/s")
                return max(0.1, q_computed)  # Minimum 0.1 m³/s
        
        # Fallback to constant Q
        return self.config.constant_q_fallback
    
    def load_stream_network(self, streams_path: str, nodes_path: Optional[str] = None):
        """
        Load stream network and nodes from shapefiles.
        
        Args:
            streams_path: Path to stream network shapefile
            nodes_path: Path to stream nodes shapefile (optional)
        """
        self.stream_network = gpd.read_file(streams_path)
        logger.info(f"Loaded {len(self.stream_network)} stream segments")
        
        if nodes_path:
            self.stream_nodes = gpd.read_file(nodes_path)
            logger.info(f"Loaded {len(self.stream_nodes)} stream nodes")
        
        # Build network graph for connectivity analysis
        self._build_stream_graph()
        
        # Calculate stream order if missing (skip if causing issues with circular graphs)
        try:
            self._calculate_stream_order_if_missing()
        except RecursionError:
            logger.warning("Stream order calculation failed (circular graph), using existing values or defaults")
            if 'stream_order' not in self.stream_network.columns:
                self.stream_network['stream_order'] = 1
        
        # Build spatial index for efficient queries
        self._build_spatial_index()
        
        # Identify main river segments if main_river_only is enabled
        if self.config.main_river_only:
            self._identify_main_river_segments()
    
    def load_watershed_polygons(self, raster_layer_id: int):
        """
        Load watershed polygons from PostGIS for boundary validation.
        
        Creates a UNION of all watershed polygons to form the overall basin boundary.
        This ensures sites are within the overall river basin, not individual sub-basins.
        
        Args:
            raster_layer_id: ID of the RasterLayer with watershed polygons
        """
        from hydropower.models import WatershedPolygon
        from django.contrib.gis.geos import GEOSGeometry
        from shapely.ops import unary_union
        
        # Load watershed polygons from database
        watersheds = WatershedPolygon.objects.filter(raster_layer_id=raster_layer_id)
        
        if not watersheds.exists():
            logger.warning(f"No watershed polygons found for RasterLayer {raster_layer_id}")
            return
        
        # Convert to GeoDataFrame for spatial operations
        watershed_data = []
        for ws in watersheds:
            watershed_data.append({
                'watershed_id': ws.watershed_id,
                'geometry': ws.geometry,
                'area_km2': ws.area_km2
            })
        
        # Create GeoDataFrame with shapely geometries
        from shapely.geometry import shape
        from shapely import wkt
        from shapely.ops import unary_union
        
        watershed_geoms = []
        all_polygons = []
        for ws_data in watershed_data:
            # Convert GEOS geometry to shapely
            geom_wkt = ws_data['geometry'].wkt
            shapely_geom = wkt.loads(geom_wkt)
            watershed_geoms.append({
                'watershed_id': ws_data['watershed_id'],
                'area_km2': ws_data['area_km2'],
                'geometry': shapely_geom
            })
            all_polygons.append(shapely_geom)
        
        # Create UNION of all watershed polygons to form the overall basin boundary
        # This is critical - we want sites within the OVERALL basin, not individual sub-basins
        basin_union = unary_union(all_polygons)
        
        # Store both individual watersheds and the unified basin
        self.watershed_gdf = gpd.GeoDataFrame([{
            'watershed_id': 0,
            'area_km2': sum(ws['area_km2'] for ws in watershed_geoms),
            'geometry': basin_union
        }], crs='EPSG:32651')
        self.watershed_polygons = watersheds
        
        logger.info(f"Created unified basin boundary from {len(watershed_geoms)} sub-watersheds (total area: {self.watershed_gdf['area_km2'].sum():.2f} km²)")
    
    def load_watershed_from_shapefile(self, shapefile_path: str):
        """
        Load watershed boundary from a shapefile (subbasin/basin boundary).
        
        This is the preferred method when you have a known basin boundary shapefile
        (e.g., from HEC-HMS subbasin shapefile) which is more accurate than 
        auto-delineated micro-watersheds.
        
        Args:
            shapefile_path: Path to shapefile or GeoJSON with basin boundary
        """
        from shapely.ops import unary_union
        
        # Load the shapefile
        gdf = gpd.read_file(shapefile_path)
        
        # Ensure CRS is EPSG:32651
        if gdf.crs is None:
            gdf.set_crs(epsg=32651, inplace=True)
        elif gdf.crs.to_epsg() != 32651:
            gdf = gdf.to_crs(epsg=32651)
        
        # Union all polygons to get overall basin
        basin_union = unary_union(gdf.geometry.tolist())
        total_area = gdf.area.sum() / 1e6  # km²
        
        self.watershed_gdf = gpd.GeoDataFrame([{
            'watershed_id': 0,
            'area_km2': total_area,
            'geometry': basin_union
        }], crs='EPSG:32651')
        
        logger.info(f"Loaded watershed boundary from shapefile: {total_area:.2f} km²")
    
    def _build_stream_graph(self):
        """Build NetworkX directed graph from stream network for connectivity analysis"""
        self.stream_graph = nx.DiGraph()
        
        for idx, stream in self.stream_network.iterrows():
            coords = list(stream.geometry.coords)
            if len(coords) < 2:
                continue
            
            # Get from_node and to_node (use existing or create from coordinates)
            from_node = stream.get('from_node', coords[0])
            to_node = stream.get('to_node', coords[-1])
            
            # Add edge with attributes
            self.stream_graph.add_edge(
                from_node,
                to_node,
                length=stream.geometry.length,
                stream_id=idx,
                geometry=stream.geometry,
                stream_order=stream.get('stream_order', 1)
            )
        
        logger.info(f"Built stream graph: {self.stream_graph.number_of_nodes()} nodes, {self.stream_graph.number_of_edges()} edges")
    
    def _identify_main_river_segments(self):
        """
        Identify main river segments (longest flow path + high-order reaches).
        
        Strategy:
        1. Find outlet node (most downstream point)
        2. Trace longest path from outlet to source
        3. Mark all segments on longest path as is_main_river=True
        4. Also mark high-order reaches (stream_order >= min_main_river_order)
        
        This focuses IO pairing on practical locations where real plants would be built:
        - Main river stem has reliable discharge
        - Geomorphically stable
        - Better access for construction
        - More realistic for investment
        """
        logger.info("Identifying main river segments...")
        
        # Find outlet node (node with no outgoing edges)
        outlet_nodes = [n for n in self.stream_graph.nodes() if self.stream_graph.out_degree(n) == 0]
        
        if not outlet_nodes:
            logger.warning("No outlet node found in stream graph, marking high-order streams as main river")
            self._mark_high_order_as_main_river()
            return
        
        # If multiple outlets, use the one with highest contributing area (longest path)
        main_outlet = outlet_nodes[0]
        if len(outlet_nodes) > 1:
            logger.info(f"Multiple outlet nodes found ({len(outlet_nodes)}), selecting main outlet by longest path")
            # Select outlet with longest upstream path
            max_path_length = 0
            for outlet in outlet_nodes:
                try:
                    # Find longest path to this outlet
                    source_nodes = [n for n in self.stream_graph.nodes() if self.stream_graph.in_degree(n) == 0]
                    for source in source_nodes:
                        if nx.has_path(self.stream_graph, source, outlet):
                            path_length = nx.shortest_path_length(self.stream_graph, source, outlet, weight='length')
                            if path_length > max_path_length:
                                max_path_length = path_length
                                main_outlet = outlet
                except (nx.NetworkXNoPath, nx.NetworkXError):
                    continue
        
        logger.info(f"Main outlet node: {main_outlet}")
        
        # Find longest path from source to outlet
        source_nodes = [n for n in self.stream_graph.nodes() if self.stream_graph.in_degree(n) == 0]
        longest_path = []
        max_length = 0
        
        for source in source_nodes:
            try:
                if nx.has_path(self.stream_graph, source, main_outlet):
                    # Use all simple paths and find longest
                    for path in nx.all_simple_paths(self.stream_graph, source, main_outlet):
                        path_length = sum(
                            self.stream_graph[path[i]][path[i+1]]['length'] 
                            for i in range(len(path)-1)
                        )
                        if path_length > max_length:
                            max_length = path_length
                            longest_path = path
            except (nx.NetworkXNoPath, nx.NetworkXError, RecursionError) as e:
                logger.debug(f"Path finding failed for source {source}: {e}")
                continue
        
        if not longest_path:
            logger.warning("Could not find longest path, marking high-order streams as main river")
            self._mark_high_order_as_main_river()
            return
        
        logger.info(f"Longest flow path: {len(longest_path)-1} segments, {max_length:.1f} m")
        
        # Mark segments on longest path as is_main_river=True
        main_river_stream_ids = set()
        for i in range(len(longest_path)-1):
            edge_data = self.stream_graph[longest_path[i]][longest_path[i+1]]
            stream_id = edge_data.get('stream_id')
            if stream_id is not None:
                main_river_stream_ids.add(stream_id)
        
        # Also mark high-order reaches as main river (major tributaries)
        if 'stream_order' in self.stream_network.columns:
            high_order_mask = self.stream_network['stream_order'] >= self.config.min_main_river_order
            high_order_ids = set(self.stream_network[high_order_mask].index)
            main_river_stream_ids.update(high_order_ids)
            logger.info(f"Added {len(high_order_ids)} high-order reaches (order >= {self.config.min_main_river_order})")
        
        # Also mark long stream segments as main river (significant channels)
        if 'length_m' in self.stream_network.columns:
            long_stream_mask = self.stream_network['length_m'] >= self.config.min_main_river_length_m
            long_stream_ids = set(self.stream_network[long_stream_mask].index)
            added_long = long_stream_ids - main_river_stream_ids  # Only count new additions
            main_river_stream_ids.update(long_stream_ids)
            logger.info(f"Added {len(added_long)} long reaches (length >= {self.config.min_main_river_length_m}m)")
        
        # Set is_main_river column
        self.stream_network['is_main_river'] = False
        self.stream_network.loc[list(main_river_stream_ids), 'is_main_river'] = True
        
        main_river_count = self.stream_network['is_main_river'].sum()
        main_river_length = self.stream_network[self.stream_network['is_main_river']]['length_m'].sum()
        total_length = self.stream_network['length_m'].sum()
        
        logger.info(f"Identified main river network: {main_river_count} segments ({main_river_count/len(self.stream_network)*100:.1f}%), "
                   f"{main_river_length/1000:.1f} km ({main_river_length/total_length*100:.1f}% of total)")
        
        # Persist is_main_river flag to database
        self._persist_main_river_to_db()
    
    def _mark_high_order_as_main_river(self):
        """Fallback: mark high-order or long streams as main river if path finding fails"""
        import pandas as pd
        main_river_mask = pd.Series([False] * len(self.stream_network), index=self.stream_network.index)
        
        # Try stream order first
        if 'stream_order' in self.stream_network.columns:
            order_mask = self.stream_network['stream_order'] >= self.config.min_main_river_order
            main_river_mask |= order_mask
            logger.info(f"Marked {order_mask.sum()} high-order streams (order >= {self.config.min_main_river_order})")
        
        # Use length threshold as additional criterion
        if 'length_m' in self.stream_network.columns:
            length_mask = self.stream_network['length_m'] >= self.config.min_main_river_length_m
            added_by_length = length_mask & ~main_river_mask
            main_river_mask |= length_mask
            logger.info(f"Marked {added_by_length.sum()} long streams (length >= {self.config.min_main_river_length_m}m)")
        
        if not main_river_mask.any():
            logger.warning("No main river segments identified by order or length, using top 20% longest segments")
            # Fallback to longest 20% of streams
            length_threshold = self.stream_network['length_m'].quantile(0.8)
            main_river_mask = self.stream_network['length_m'] >= length_threshold
        
        self.stream_network['is_main_river'] = main_river_mask
        main_river_count = main_river_mask.sum()
        logger.info(f"Total main river segments: {main_river_count} ({main_river_count/len(self.stream_network)*100:.1f}%)")
        
        # Persist is_main_river flag to database
        self._persist_main_river_to_db()
    
    def _persist_main_river_to_db(self):
        """
        Persist is_main_river flags from GeoDataFrame back to StreamNetwork models in database.
        
        This updates the database so the main river identification is saved for future use.
        """
        from .models import StreamNetwork
        from django.db import transaction
        
        if 'is_main_river' not in self.stream_network.columns:
            logger.warning("is_main_river column not found in stream network, skipping database update")
            return
        
        if not hasattr(self, 'raster_layer_id') or self.raster_layer_id is None:
            logger.warning("No raster_layer_id set, cannot persist is_main_river to database")
            return
        
        logger.info("Persisting is_main_river flags to database...")
        
        try:
            with transaction.atomic():
                # Get all streams for this raster layer
                streams_qs = StreamNetwork.objects.filter(raster_layer_id=self.raster_layer_id)
                
                # Build lookup dict from GeoDataFrame
                is_main_river_lookup = {}
                for idx, row in self.stream_network.iterrows():
                    # Use from_node and to_node to match streams
                    from_node = row.get('from_node')
                    to_node = row.get('to_node')
                    is_main = row.get('is_main_river', False)
                    if from_node is not None and to_node is not None:
                        key = (from_node, to_node)
                        is_main_river_lookup[key] = is_main
                
                # Update database models
                updated_count = 0
                for stream in streams_qs:
                    key = (stream.from_node, stream.to_node)
                    if key in is_main_river_lookup:
                        new_value = is_main_river_lookup[key]
                        if stream.is_main_river != new_value:
                            stream.is_main_river = new_value
                            stream.save(update_fields=['is_main_river'])
                            updated_count += 1
                
                logger.info(f"Updated {updated_count} StreamNetwork records in database")
                
        except Exception as e:
            logger.error(f"Failed to persist is_main_river to database: {e}")
    
    def _calculate_stream_order_if_missing(self):
        """Calculate Strahler stream order if not present in data"""
        if 'stream_order' in self.stream_network.columns:
            non_null_count = self.stream_network['stream_order'].notna().sum()
            total_count = len(self.stream_network)
            if non_null_count > 0:
                logger.info(f"Stream order already present in data ({non_null_count}/{total_count} segments)")
                return
        
        logger.info("Calculating Strahler stream order...")
        
        # Calculate stream order using graph topology
        stream_orders = {}
        
        # Find source nodes (no incoming edges)
        source_nodes = [n for n in self.stream_graph.nodes() if self.stream_graph.in_degree(n) == 0]
        
        def calculate_order(node):
            """Recursively calculate Strahler order"""
            if node in stream_orders:
                return stream_orders[node]
            
            # Get upstream nodes
            upstream = list(self.stream_graph.predecessors(node))
            
            if not upstream:
                # Source node: order 1
                stream_orders[node] = 1
                return 1
            
            # Calculate orders of all upstream branches
            upstream_orders = [calculate_order(u) for u in upstream]
            
            # Strahler rules:
            # - If all upstream orders are the same, order = max + 1
            # - Otherwise, order = max of upstream orders
            max_order = max(upstream_orders)
            if all(o == max_order for o in upstream_orders) and len(upstream_orders) > 1:
                order = max_order + 1
            else:
                order = max_order
            
            stream_orders[node] = order
            return order
        
        # Calculate for all nodes
        for node in self.stream_graph.nodes():
            calculate_order(node)
        
        # Assign to stream segments
        for idx, stream in self.stream_network.iterrows():
            coords = list(stream.geometry.coords)
            to_node = stream.get('to_node', coords[-1])
            self.stream_network.at[idx, 'stream_order'] = stream_orders.get(to_node, 1)
        
        logger.info(f"Calculated stream orders: range {self.stream_network['stream_order'].min():.0f} - {self.stream_network['stream_order'].max():.0f}")
    
    def _build_spatial_index(self):
        """Build spatial indexes for fast queries"""
        # Build KDTree for stream nodes
        if self.stream_nodes is not None:
            coords = np.array([[node.geometry.x, node.geometry.y] for _, node in self.stream_nodes.iterrows()])
            self.node_spatial_index = cKDTree(coords)
            logger.info(f"Built node spatial index with {len(coords)} nodes")
        
        # Build STRtree spatial index for stream geometries (for distance queries)
        if self.stream_network is not None and len(self.stream_network) > 0:
            from shapely import STRtree
            self.stream_spatial_index = STRtree(self.stream_network.geometry.values)
            logger.info(f"Built stream spatial index with {len(self.stream_network)} segments")
    
    def _is_within_watershed(self, point: Point) -> bool:
        """
        Check if a point is within any watershed boundary.
        
        Args:
            point: Shapely Point geometry
        
        Returns:
            True if point is within watershed boundary, False otherwise
        """
        if self.watershed_gdf is None:
            # No watershed boundaries loaded - assume all points are valid
            logger.debug("No watershed boundaries loaded, skipping boundary check")
            return True
        
        # Check if point intersects with any watershed polygon
        for idx, watershed in self.watershed_gdf.iterrows():
            if watershed.geometry.contains(point) or watershed.geometry.intersects(point):
                return True
        
        return False
    
    def extract_elevation(self, x: float, y: float) -> Optional[float]:
        """
        Extract elevation from DEM at given coordinates.
        
        Args:
            x: X coordinate (longitude/easting)
            y: Y coordinate (latitude/northing)
        
        Returns:
            Elevation in meters, or None if outside DEM bounds
        """
        if self.dem_array is None:
            raise ValueError("DEM not loaded. Call load_dem() first.")
        
        try:
            row, col = rowcol(self.dem_transform, x, y)
            
            # Check bounds
            if 0 <= row < self.dem_array.shape[0] and 0 <= col < self.dem_array.shape[1]:
                elevation = float(self.dem_array[row, col])
                # Check for nodata values
                if elevation < -9999:  # Common nodata threshold
                    return None
                return elevation
            return None
        except Exception as e:
            logger.warning(f"Error extracting elevation at ({x}, {y}): {e}")
            return None
    
    def identify_inlet_candidates(self) -> gpd.GeoDataFrame:
        """
        Identify inlet candidates (upstream points on stream network).
        
        Inlets are typically:
        - Source nodes (headwaters)
        - High-order confluences
        - Points with sufficient upstream drainage area
        
        Returns:
            GeoDataFrame of inlet candidate points
        """
        if self.stream_nodes is None:
            # If no nodes, use upstream endpoints of high-order streams
            inlets = []
            # Check if stream_order column exists
            has_stream_order = 'stream_order' in self.stream_network.columns
            
            # Sample streams if too many
            stream_sample = self.stream_network
            if len(stream_sample) > self.config.max_stream_segments:
                logger.warning(f"Too many streams ({len(stream_sample)}), sampling {self.config.max_stream_segments} uniformly")
                stream_sample = stream_sample.sample(n=self.config.max_stream_segments, random_state=42)
            
            for idx, stream in stream_sample.iterrows():
                # If stream_order exists, filter by min_stream_order; otherwise accept all
                stream_order = stream.get('stream_order', None) if has_stream_order else None
                if stream_order is None or stream_order >= self.config.min_stream_order:
                    # Get first point of LineString (upstream end)
                    coords = list(stream.geometry.coords)
                    point = Point(coords[0])
                    
                    # Check if point is within watershed boundary
                    if not self._is_within_watershed(point):
                        logger.debug(f"Inlet candidate at ({point.x:.1f}, {point.y:.1f}) outside watershed boundary, skipping")
                        continue
                    
                    elevation = self.extract_elevation(point.x, point.y)
                    if elevation is not None:
                        inlets.append({
                            'geometry': point,
                            'stream_order': stream_order,
                            'elevation': elevation,
                            'type': 'stream_endpoint'
                        })
                        
                        # Limit candidates
                        if len(inlets) >= self.config.max_candidates_per_type:
                            logger.info(f"Reached max inlet candidates limit ({self.config.max_candidates_per_type})")
                            break
            
            logger.info(f"Identified {len(inlets)} inlet candidates (stream_order column present: {has_stream_order})")
            
            # Handle empty inlets case
            if len(inlets) == 0:
                logger.warning("No inlet candidates found using stream order criteria, falling back to node-based method")
                return gpd.GeoDataFrame(columns=['geometry', 'elevation', 'stream_order', 'node_type'], crs=self.stream_network.crs)
            
            return gpd.GeoDataFrame(inlets, crs=self.stream_network.crs)
        else:
            # Use source and confluence nodes
            inlet_nodes = self.stream_nodes[
                (self.stream_nodes['node_type'].isin(['source', 'confluence'])) &
                (self.stream_nodes.get('stream_order', 0) >= self.config.min_stream_order)
            ].copy()
            
            # Extract elevations
            inlet_nodes['elevation'] = inlet_nodes.geometry.apply(
                lambda geom: self.extract_elevation(geom.x, geom.y)
            )
            
            # Remove nodes with no elevation
            inlet_nodes = inlet_nodes[inlet_nodes['elevation'].notna()]
            
            logger.info(f"Identified {len(inlet_nodes)} inlet candidates")
            return inlet_nodes
    
    def identify_outlet_candidates(self) -> gpd.GeoDataFrame:
        """
        Identify outlet candidates (downstream points on stream network).
        
        Outlets are typically:
        - Outlet nodes (basin outlets)
        - Confluences
        - Points downstream of inlets
        
        Returns:
            GeoDataFrame of outlet candidate points
        """
        if self.stream_nodes is None:
            # If no nodes, use downstream endpoints of streams
            outlets = []
            # Check if stream_order column exists
            has_stream_order = 'stream_order' in self.stream_network.columns
            
            # Sample streams if too many
            stream_sample = self.stream_network
            if len(stream_sample) > self.config.max_stream_segments:
                logger.warning(f"Too many streams ({len(stream_sample)}), sampling {self.config.max_stream_segments} uniformly")
                stream_sample = stream_sample.sample(n=self.config.max_stream_segments, random_state=42)
            
            for idx, stream in stream_sample.iterrows():
                # If stream_order exists, filter by min_stream_order; otherwise accept all
                stream_order = stream.get('stream_order', None) if has_stream_order else None
                
                # Filter by main_river_only if enabled
                if self.config.main_river_only and 'is_main_river' in stream:
                    if not stream.get('is_main_river', False):
                        continue
                
                if stream_order is None or stream_order >= self.config.min_stream_order:
                    # Get last point of LineString (downstream end)
                    coords = list(stream.geometry.coords)
                    point = Point(coords[-1])
                    
                    # Check if point is within watershed boundary
                    if not self._is_within_watershed(point):
                        logger.debug(f"Outlet candidate at ({point.x:.1f}, {point.y:.1f}) outside watershed boundary, skipping")
                        continue
                    
                    elevation = self.extract_elevation(point.x, point.y)
                    if elevation is not None:
                        outlets.append({
                            'geometry': point,
                            'stream_order': stream_order,
                            'elevation': elevation,
                            'type': 'stream_endpoint'
                        })
                        
                        # Limit candidates
                        if len(outlets) >= self.config.max_candidates_per_type:
                            logger.info(f"Reached max outlet candidates limit ({self.config.max_candidates_per_type})")
                            break
            
            logger.info(f"Identified {len(outlets)} outlet candidates (stream_order column present: {has_stream_order})")
            
            # Handle empty outlets case
            if len(outlets) == 0:
                logger.warning("No outlet candidates found using stream order criteria, falling back to node-based method")
                return gpd.GeoDataFrame(columns=['geometry', 'elevation', 'stream_order', 'type'], crs=self.stream_network.crs)
            
            return gpd.GeoDataFrame(outlets, crs=self.stream_network.crs)
        else:
            # Use outlet and confluence nodes
            outlet_nodes = self.stream_nodes[
                (self.stream_nodes['node_type'].isin(['outlet', 'confluence'])) &
                (self.stream_nodes.get('stream_order', 0) >= self.config.min_stream_order)
            ].copy()
            
            # Extract elevations
            outlet_nodes['elevation'] = outlet_nodes.geometry.apply(
                lambda geom: self.extract_elevation(geom.x, geom.y)
            )
            
            # Remove nodes with no elevation
            outlet_nodes = outlet_nodes[outlet_nodes['elevation'].notna()]
            
            logger.info(f"Identified {len(outlet_nodes)} outlet candidates")
            return outlet_nodes
    
    def create_densified_nodes(self) -> gpd.GeoDataFrame:
        """
        Create densified nodes along stream network at regular spacing.
        
        This mirrors the reference QGIS code approach of placing nodes every
        NODE_SPACING_M meters along each channel segment, with chainage tracking.
        
        Returns:
            GeoDataFrame with columns: [node_id, seg_id, chainage, geometry, elevation, discharge]
        """
        if not self.config.use_node_densification:
            logger.info("Node densification disabled, using original candidate methods")
            return None
        
        logger.info(f"Creating densified nodes at {self.config.node_spacing}m spacing...")
        
        all_nodes = []
        node_id = 1
        
        # Sample streams if too many
        stream_sample = self.stream_network
        if len(stream_sample) > self.config.max_stream_segments:
            logger.warning(
                f"Sampling {self.config.max_stream_segments} streams "
                f"from {len(stream_sample)} total"
            )
            stream_sample = stream_sample.sample(
                n=self.config.max_stream_segments, 
                random_state=42
            )
        
        for seg_idx, stream in stream_sample.iterrows():
            geom = stream.geometry
            if geom is None or geom.is_empty:
                continue
            
            seg_length = geom.length
            if seg_length <= 0:
                continue
            
            # Create nodes at regular intervals
            chainage = 0.0
            nodes_on_segment = []
            
            while chainage <= seg_length:
                # Interpolate point at chainage
                pt_geom = geom.interpolate(chainage)
                if pt_geom.is_empty:
                    chainage += self.config.node_spacing
                    continue
                
                pt = Point(pt_geom.coords[0])
                
                # Check watershed boundary
                if not self._is_within_watershed(pt):
                    chainage += self.config.node_spacing
                    continue
                
                # Extract elevation
                elevation = self.extract_elevation(pt.x, pt.y)
                if elevation is None:
                    chainage += self.config.node_spacing
                    continue
                
                # Compute discharge at this node
                discharge = self.compute_discharge_at_point(pt)
                
                node_data = {
                    'node_id': node_id,
                    'seg_id': int(seg_idx),
                    'chainage': chainage,
                    'geometry': pt,
                    'elevation': elevation,
                    'discharge': discharge,
                    'x': pt.x,
                    'y': pt.y
                }
                
                nodes_on_segment.append(node_data)
                all_nodes.append(node_data)
                node_id += 1
                
                chainage += self.config.node_spacing
            
            # Ensure node at downstream end
            if nodes_on_segment:
                last_chainage = nodes_on_segment[-1]['chainage']
                if abs(last_chainage - seg_length) > 0.01:  # More than 1cm difference
                    # Add final node at segment end
                    pt_geom = geom.interpolate(seg_length)
                    if not pt_geom.is_empty:
                        pt = Point(pt_geom.coords[0])
                        elevation = self.extract_elevation(pt.x, pt.y)
                        if elevation is not None and self._is_within_watershed(pt):
                            discharge = self.compute_discharge_at_point(pt)
                            node_data = {
                                'node_id': node_id,
                                'seg_id': int(seg_idx),
                                'chainage': seg_length,
                                'geometry': pt,
                                'elevation': elevation,
                                'discharge': discharge,
                                'x': pt.x,
                                'y': pt.y
                            }
                            all_nodes.append(node_data)
                            node_id += 1
        
        if not all_nodes:
            logger.warning("No nodes created during densification")
            return None
        
        nodes_gdf = gpd.GeoDataFrame(all_nodes, crs=self.stream_network.crs)
        logger.info(f"Created {len(nodes_gdf)} densified nodes on {len(stream_sample)} stream segments")
        
        return nodes_gdf
    
    def calculate_river_distance(self, inlet: Point, outlet: Point) -> float:
        """
        Calculate distance along river between inlet and outlet using network analysis.
        
        Uses NetworkX shortest path algorithm to trace actual stream path.
        Falls back to sinuosity estimation if network path not found.
        
        Args:
            inlet: Inlet point geometry
            outlet: Outlet point geometry
        
        Returns:
            River distance in meters along stream network
        """
        # OPTIMIZATION: For large networks (>10k edges), skip network path search
        # Use sinuosity estimation to avoid performance issues
        if self.stream_graph is None or self.stream_graph.number_of_edges() > 10000:
            euclidean = inlet.distance(outlet)
            sinuosity_factor = 1.3
            if self.stream_graph is None:
                logger.debug("Stream graph not available, using sinuosity estimation")
            return euclidean * sinuosity_factor
        
        try:
            # Find nearest nodes in stream network
            inlet_node = self._find_nearest_stream_node(inlet)
            outlet_node = self._find_nearest_stream_node(outlet)
            
            if inlet_node is None or outlet_node is None:
                euclidean = inlet.distance(outlet)
                return euclidean * 1.3
            
            # Quick connectivity check first (faster than finding full path)
            if not nx.has_path(self.stream_graph, inlet_node, outlet_node):
                # No path - use sinuosity estimate
                euclidean = inlet.distance(outlet)
                return euclidean * 1.3
            
            # Find shortest path along stream network
            path = nx.shortest_path(self.stream_graph, inlet_node, outlet_node, weight='length')
            
            # Sum distances along path
            total_distance = 0.0
            for i in range(len(path) - 1):
                edge_data = self.stream_graph.get_edge_data(path[i], path[i+1])
                if edge_data:
                    total_distance += edge_data.get('length', 0)
            
            logger.debug(f"Network path distance: {total_distance:.1f}m (vs Euclidean: {inlet.distance(outlet):.1f}m)")
            return total_distance
                
        except Exception as e:
            logger.debug(f"Error calculating network distance: {e}, using fallback")
            euclidean = inlet.distance(outlet)
            return euclidean * 1.3
    
    def _find_nearest_stream_node(self, point: Point) -> Optional[tuple]:
        """
        Find nearest stream network node to a point.
        
        Uses cached KDTree spatial index for fast O(log n) queries.
        
        Returns:
            Tuple of (x, y) coordinates of nearest node, or None if too far
        """
        if self.node_spatial_index is None:
            # Build spatial index from stream endpoints (cached for reuse)
            if not hasattr(self, '_cached_stream_nodes') or self._cached_stream_nodes is None:
                stream_nodes = []
                seen_coords = set()  # Deduplicate nodes at same location
                
                for geom in self.stream_network.geometry:
                    coords = list(geom.coords)
                    if len(coords) < 2:
                        continue
                    
                    # Add start point (inlet/source)
                    start = (round(coords[0][0], 2), round(coords[0][1], 2))
                    if start not in seen_coords:
                        stream_nodes.append(coords[0])
                        seen_coords.add(start)
                    
                    # Add end point (outlet/confluence)
                    end = (round(coords[-1][0], 2), round(coords[-1][1], 2))
                    if end not in seen_coords:
                        stream_nodes.append(coords[-1])
                        seen_coords.add(end)
                
                if not stream_nodes:
                    logger.warning("No stream nodes found for spatial indexing")
                    return None
                
                self._cached_stream_nodes = np.array(stream_nodes)
                self._cached_spatial_index = cKDTree(self._cached_stream_nodes)
                logger.debug(f"Built cached spatial index with {len(stream_nodes)} unique nodes")
            
            # Query nearest node
            try:
                dist, idx = self._cached_spatial_index.query([point.x, point.y])
            except Exception as e:
                logger.warning(f"Spatial index query failed: {e}")
                return None
            
            # Adaptive threshold based on stream network density
            # For dense networks (many nodes), use tighter threshold
            # For sparse networks, allow more distance
            num_nodes = len(self._cached_stream_nodes)
            if num_nodes > 1000:
                max_dist = 200  # Dense network: 200m max
            elif num_nodes > 100:
                max_dist = 500  # Medium network: 500m max
            else:
                max_dist = 1000  # Sparse network: 1km max
            
            if dist > max_dist:
                logger.debug(f"Point too far from stream network: {dist:.1f}m (threshold: {max_dist}m)")
                return None
            
            return tuple(self._cached_stream_nodes[idx])
        else:
            # Use existing spatial index from stream_nodes
            try:
                dist, idx = self.node_spatial_index.query([point.x, point.y])
            except Exception as e:
                logger.warning(f"Spatial index query failed: {e}")
                return None
            
            # Same adaptive threshold
            num_nodes = len(self.stream_nodes)
            if num_nodes > 1000:
                max_dist = 200
            elif num_nodes > 100:
                max_dist = 500
            else:
                max_dist = 1000
            
            if dist > max_dist:
                logger.debug(f"Point too far from stream network: {dist:.1f}m (threshold: {max_dist}m)")
                return None
            
            node = self.stream_nodes.iloc[idx]
            return (node.geometry.x, node.geometry.y)
    
    def check_downstream_relationship(self, inlet: Point, outlet: Point) -> bool:
        """
        Check if outlet is downstream of inlet using stream network connectivity.
        
        Verifies both elevation gradient and network connectivity to ensure
        sites are on the same stream path.
        
        Args:
            inlet: Inlet point geometry with elevation
            outlet: Outlet point geometry with elevation
        
        Returns:
            True if outlet is downstream and connected, False otherwise
        """
        inlet_elev = self.extract_elevation(inlet.x, inlet.y)
        outlet_elev = self.extract_elevation(outlet.x, outlet.y)
        
        if inlet_elev is None or outlet_elev is None:
            return False
        
        # First check: Outlet must be lower than inlet
        if outlet_elev >= inlet_elev:
            return False
        
        # OPTIMIZATION: Skip expensive network connectivity check for large networks
        # For networks > 10k edges, rely on elevation check only to avoid performance issues
        if self.stream_graph is not None and self.stream_graph.number_of_edges() > 10000:
            # Large network - skip connectivity check, use elevation only
            return True
        
        # Second check: Verify network connectivity if graph available (small networks only)
        if self.stream_graph is not None:
            try:
                inlet_node = self._find_nearest_stream_node(inlet)
                outlet_node = self._find_nearest_stream_node(outlet)
                
                if inlet_node and outlet_node:
                    # Use faster connectivity check with cutoff
                    try:
                        # Check path existence with length limit (faster than full path search)
                        has_path = nx.has_path(self.stream_graph, inlet_node, outlet_node)
                    except:
                        # If path check fails, assume valid (fall back to elevation)
                        has_path = True
                    
                    if not has_path:
                        logger.debug(f"No stream connectivity from inlet to outlet")
                        return False
                    
                    return True
                else:
                    # Node lookup failed, fall back to elevation check only
                    return True
            except Exception as e:
                # Fall back to elevation check on any error
                return True
        
        # If no graph, rely on elevation check
        return True
    
    def calculate_head(self, inlet_elevation: float, outlet_elevation: float) -> float:
        """
        Calculate gross hydraulic head (before losses).
        
        Args:
            inlet_elevation: Elevation at inlet (m)
            outlet_elevation: Elevation at outlet (m)
        
        Returns:
            Gross head in meters (H = z_inlet - z_outlet)
        """
        return inlet_elevation - outlet_elevation
    
    def calculate_head_losses(self, gross_head: float, penstock_length: float, 
                             discharge: float, num_bends: int = 2) -> Dict[str, float]:
        """
        Calculate head losses in penstock and fittings.
        
        Includes:
        - Friction losses (Darcy-Weisbach equation)
        - Entrance and exit losses
        - Bend losses
        - Velocity head
        
        Args:
            gross_head: Gross head (m)
            penstock_length: Penstock pipe length (m)
            discharge: Discharge (m³/s)
            num_bends: Number of 90° bends in penstock
        
        Returns:
            Dictionary with loss components and net head
        """
        if discharge <= 0:
            return {'friction_loss': 0, 'minor_loss': 0, 'net_head': gross_head}
        
        # Calculate penstock diameter based on velocity
        velocity = self.config.velocity_factor
        diameter = np.sqrt((4 * discharge) / (np.pi * velocity))
        
        # Friction loss using Hazen-Williams formula (more common in hydropower)
        # hf = 10.67 * L * Q^1.852 / (C^1.852 * D^4.87)
        C = 140  # Hazen-Williams coefficient for steel pipe
        friction_loss = 10.67 * penstock_length * (discharge ** 1.852) / ((C ** 1.852) * (diameter ** 4.87))
        
        # Minor losses
        entrance_loss = self.config.entrance_loss_coeff * (velocity ** 2) / (2 * self.config.g)
        exit_loss = self.config.exit_loss_coeff * (velocity ** 2) / (2 * self.config.g)
        bend_loss = num_bends * self.config.bend_loss_coeff * (velocity ** 2) / (2 * self.config.g)
        
        total_minor_loss = entrance_loss + exit_loss + bend_loss
        total_head_loss = friction_loss + total_minor_loss
        
        # Net head
        net_head = gross_head - total_head_loss
        
        # Ensure net head is positive
        net_head = max(0, net_head)
        
        return {
            'friction_loss': friction_loss,
            'entrance_loss': entrance_loss,
            'exit_loss': exit_loss,
            'bend_loss': bend_loss,
            'total_minor_loss': total_minor_loss,
            'total_head_loss': total_head_loss,
            'net_head': net_head,
            'gross_head': gross_head,
            'efficiency_factor': net_head / gross_head if gross_head > 0 else 0
        }
    
    def apply_constraints(self, head: float, river_distance: float, 
                         inlet: Point, outlet: Point) -> Tuple[bool, Dict[str, bool]]:
        """
        Apply constraint filters to site pair.
        
        Includes validation for:
        - Head constraints (min/max)
        - Distance constraints (min/max river distance)
        - Watershed boundary (both inlet and outlet must be within watershed)
        - Land constraints (terrain slope, stream proximity)
        
        Args:
            head: Hydraulic head (m)
            river_distance: Distance along river (m)
            inlet: Inlet point geometry
            outlet: Outlet point geometry
        
        Returns:
            Tuple of (is_feasible, constraint_flags)
        """
        # Watershed boundary constraint - CRITICAL for Claveria issue
        inlet_in_watershed = self._is_within_watershed(inlet)
        outlet_in_watershed = self._is_within_watershed(outlet)
        meets_watershed_constraint = inlet_in_watershed and outlet_in_watershed
        
        if not meets_watershed_constraint:
            logger.debug(f"Site pair outside watershed boundary: inlet={inlet_in_watershed}, outlet={outlet_in_watershed}")
        
        constraints = {
            'meets_head_constraint': self.config.min_head <= head <= self.config.max_head,
            'meets_distance_constraint': self.config.min_river_distance <= river_distance <= self.config.max_river_distance,
            'meets_watershed_constraint': meets_watershed_constraint,
            'meets_land_constraint': self.check_land_proximity(inlet, outlet)
        }
        
        # Add infrastructure slope validation if enabled
        if self.config.validate_infrastructure_slope:
            constraints['meets_infrastructure_slope'] = self.check_infrastructure_slope(
                inlet, outlet, head, river_distance
            )
        
        is_feasible = all(constraints.values())
        return is_feasible, constraints
    
    def check_infrastructure_slope(self, inlet: Point, outlet: Point, 
                                   head: float, river_distance: float) -> bool:
        """
        Validate infrastructure slope feasibility.
        
        Checks:
        - Penstock slope (can be steep, up to 80%)
        - Headrace channel slope (must be gentle, <5%)
        
        Args:
            inlet: Inlet point geometry
            outlet: Outlet point geometry
            head: Hydraulic head (meters)
            river_distance: River distance (meters)
        
        Returns:
            True if infrastructure slopes are feasible
        """
        # Calculate euclidean distance
        euclidean_dist = inlet.distance(outlet)
        
        if euclidean_dist < 1:
            return False  # Too short
        
        # Calculate average slope (simplified)
        avg_slope = (head / euclidean_dist) * 100  # in percent
        
        # Penstock slope (direct line) - can be steep
        penstock_slope = avg_slope  # Simplified: direct inlet to outlet
        if penstock_slope > self.config.max_penstock_slope:
            logger.debug(f"Penstock slope {penstock_slope:.1f}% exceeds max {self.config.max_penstock_slope}%")
            return False
        
        # For longer systems, estimate channel slope
        # Channel typically follows 70% of the river distance
        if river_distance > 200:  # Only for longer systems
            # Estimate elevation drop along channel (about 20% of total head)
            channel_head = head * 0.2
            channel_length = river_distance * 0.7
            
            if channel_length > 0:
                channel_slope = (channel_head / channel_length) * 100
                if channel_slope > self.config.max_channel_slope:
                    logger.debug(f"Channel slope {channel_slope:.1f}% exceeds max {self.config.max_channel_slope}%")
                    return False
        
        return True
    
    def check_land_proximity(self, inlet: Point, outlet: Point) -> bool:
        """
        Check land proximity and feasibility constraints.
        
        Validates:
        - Distance to river centerline (should be near stream)
        - Terrain slope at site locations
        - Distance to road network for accessibility
        
        Args:
            inlet: Inlet point geometry
            outlet: Outlet point geometry
        
        Returns:
            True if all feasibility constraints are met
        """
        # Check 1: Distance to stream network
        inlet_dist = self._distance_to_nearest_stream(inlet)
        outlet_dist = self._distance_to_nearest_stream(outlet)
        
        if inlet_dist > self.config.land_buffer or outlet_dist > self.config.land_buffer:
            logger.debug(f"Site too far from stream: inlet={inlet_dist:.1f}m, outlet={outlet_dist:.1f}m")
            return False
        
        # Check 2: Terrain slope at site locations (optional - requires slope raster)
        if self.config.check_land_use:
            inlet_slope = self._get_terrain_slope(inlet)
            outlet_slope = self._get_terrain_slope(outlet)
            
            if inlet_slope and inlet_slope > self.config.max_slope_percent:
                logger.debug(f"Inlet slope too steep: {inlet_slope:.1f}%")
                return False
            
            if outlet_slope and outlet_slope > self.config.max_slope_percent:
                logger.debug(f"Outlet slope too steep: {outlet_slope:.1f}%")
                return False
        
        # All checks passed
        return True
    
    def _distance_to_nearest_stream(self, point: Point) -> float:
        """Calculate distance from point to nearest stream segment using spatial index"""
        if self.stream_network is None or len(self.stream_network) == 0:
            return 0.0  # Assume valid if no stream network
        
        # Use STRtree spatial index for fast nearest neighbor query
        if self.stream_spatial_index is not None:
            nearest_idx = self.stream_spatial_index.nearest(point)
            nearest_geom = self.stream_network.geometry.iloc[nearest_idx]
            return point.distance(nearest_geom)
        
        # Fallback to brute force (should not happen)
        min_dist = float('inf')
        for _, stream in self.stream_network.iterrows():
            dist = point.distance(stream.geometry)
            if dist < min_dist:
                min_dist = dist
        
        return min_dist
    
    def _get_terrain_slope(self, point: Point) -> Optional[float]:
        """Calculate terrain slope at point location (rise/run × 100%)"""
        if self.dem_array is None:
            return None
        
        try:
            # Get elevation at point and surrounding 8 neighbors
            row, col = rowcol(self.dem_transform, point.x, point.y)
            
            if row <= 0 or row >= self.dem_array.shape[0] - 1:
                return None
            if col <= 0 or col >= self.dem_array.shape[1] - 1:
                return None
            
            # 3x3 window
            window = self.dem_array[row-1:row+2, col-1:col+2]
            
            # Calculate max slope using Sobel operator
            dz_dx = (window[0, 2] + 2*window[1, 2] + window[2, 2] - 
                     window[0, 0] - 2*window[1, 0] - window[2, 0]) / (8 * abs(self.dem_transform[0]))
            dz_dy = (window[2, 0] + 2*window[2, 1] + window[2, 2] - 
                     window[0, 0] - 2*window[0, 1] - window[0, 2]) / (8 * abs(self.dem_transform[4]))
            
            slope_radians = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
            slope_percent = np.tan(slope_radians) * 100
            
            return float(slope_percent)
            
        except Exception as e:
            logger.debug(f"Error calculating slope: {e}")
            return None
    
    def calculate_score(self, head: float, discharge: Optional[float], 
                       river_distance: float, inlet: Point) -> float:
        """
        Calculate multi-criteria score for site pair.
        
        Args:
            head: Hydraulic head (m)
            discharge: Discharge (m³/s)
            river_distance: Distance along river (m)
            inlet: Inlet point geometry
        
        Returns:
            Normalized score (0-100)
        """
        # Normalize head (0-100 scale)
        head_score = min(100, (head / self.config.max_head) * 100)
        
        # Normalize discharge (0-100 scale, assume max 100 m³/s)
        discharge_score = 0
        if discharge:
            discharge_score = min(100, (discharge / 100.0) * 100)
        
        # Normalize distance (inverse: shorter is better, 0-100 scale)
        distance_score = max(0, 100 - (river_distance / self.config.max_river_distance) * 100)
        
        # Accessibility score (simplified: based on elevation, lower is more accessible)
        inlet_elev = self.extract_elevation(inlet.x, inlet.y) or 0
        accessibility_score = max(0, 100 - (inlet_elev / 1000.0) * 100)
        
        # Weighted sum
        score = (
            self.config.weight_head * head_score +
            self.config.weight_discharge * discharge_score +
            self.config.weight_distance * distance_score +
            self.config.weight_accessibility * accessibility_score
        )
        
        return score
    
    def search_downstream_outlets(self, inlet: gpd.GeoSeries, outlets: gpd.GeoDataFrame) -> List[Dict]:
        """
        Search for feasible outlets downstream of an inlet.
        
        Args:
            inlet: Inlet point (GeoDataFrame row)
            outlets: GeoDataFrame of outlet candidates
        
        Returns:
            List of feasible site pairs with metadata
        """
        inlet_point = inlet.geometry
        inlet_elevation = inlet['elevation']
        
        feasible_pairs = []
        
        # Find outlets within search radius and downstream
        for idx, outlet in outlets.iterrows():
            outlet_point = outlet.geometry
            outlet_elevation = outlet['elevation']
            
            # Check downstream relationship
            if outlet_elevation >= inlet_elevation:
                continue  # Skip if not downstream
            
            # Calculate head
            head = self.calculate_head(inlet_elevation, outlet_elevation)
            
            # Calculate river distance
            river_distance = self.calculate_river_distance(inlet_point, outlet_point)
            euclidean_distance = inlet_point.distance(outlet_point)
            
            # Apply constraints
            is_feasible, constraints = self.apply_constraints(head, river_distance, inlet_point, outlet_point)
            
            if not is_feasible:
                continue
            
            # Calculate score
            discharge = outlet.get('discharge', None)  # If available
            score = self.calculate_score(head, discharge, river_distance, inlet_point)
            
            # Create pair record
            pair = {
                'inlet_geom': inlet_point,
                'outlet_geom': outlet_point,
                'inlet_elevation': inlet_elevation,
                'outlet_elevation': outlet_elevation,
                'head': head,
                'river_distance': river_distance,
                'euclidean_distance': euclidean_distance,
                'discharge': discharge,
                'score': score,
                'is_feasible': is_feasible,
                **constraints
            }
            
            feasible_pairs.append(pair)
        
        # Sort by score (descending) and take top N
        feasible_pairs.sort(key=lambda x: x['score'], reverse=True)
        return feasible_pairs[:self.config.max_outlets_per_inlet]
    
    def deduplicate_pairs(self, pairs: List[Dict]) -> List[Dict]:
        """
        Remove overlapping/duplicate site pairs.
        
        Args:
            pairs: List of site pair dictionaries
        
        Returns:
            Deduplicated list of site pairs
        """
        if not pairs:
            return []
        
        # Sort by score (descending)
        pairs.sort(key=lambda x: x['score'], reverse=True)
        
        selected_pairs = []
        used_inlets = set()
        used_outlets = set()
        
        for pair in pairs:
            inlet_coords = (pair['inlet_geom'].x, pair['inlet_geom'].y)
            outlet_coords = (pair['outlet_geom'].x, pair['outlet_geom'].y)
            
            # Check if inlet or outlet already used
            if inlet_coords in used_inlets or outlet_coords in used_outlets:
                continue
            
            # Check spacing buffer against already selected pairs
            too_close = False
            for selected in selected_pairs:
                inlet_dist = pair['inlet_geom'].distance(selected['inlet_geom'])
                outlet_dist = pair['outlet_geom'].distance(selected['outlet_geom'])
                
                if inlet_dist < self.config.spacing_buffer or outlet_dist < self.config.spacing_buffer:
                    too_close = True
                    break
            
            if too_close:
                continue
            
            # Add to selected pairs
            selected_pairs.append(pair)
            used_inlets.add(inlet_coords)
            used_outlets.add(outlet_coords)
        
        logger.info(f"Deduplicated {len(pairs)} pairs to {len(selected_pairs)} unique pairs")
        return selected_pairs
    
    def systematic_io_pairing(self, densified_nodes: gpd.GeoDataFrame) -> List[Dict]:
        """
        Perform systematic inlet-outlet pairing using densified nodes.
        
        This implements the reference QGIS algorithm:
        1. Group nodes by stream segment
        2. For each segment, nest loop through nodes (inlet i, outlet j where j > i)
        3. Check three conditions:
           a) outlet elevation < inlet elevation (positive head)
           b) along-river distance (outlet.chainage - inlet.chainage) within bounds
           c) head >= minimum head threshold
        4. For each valid pair, compute pair-specific discharge and power
        5. Deduplicate by (inlet_node, outlet_node), keeping best P_kW
        6. Rank by power output
        
        Args:
            densified_nodes: GeoDataFrame from create_densified_nodes()
        
        Returns:
            List of valid IO pairs with metadata
        """
        if densified_nodes is None or len(densified_nodes) == 0:
            logger.warning("No densified nodes provided for systematic pairing")
            return []
        
        logger.info("Starting systematic IO pairing with densified nodes...")
        
        # Adaptive constraints based on stream characteristics
        # Check if we have very short streams that need relaxed constraints
        if 'length_m' in self.stream_network.columns:
            median_length = self.stream_network['length_m'].median()
            if median_length < 50:
                logger.warning(f"Short streams detected (median length: {median_length:.1f}m)")
                logger.warning(f"Relaxing min_river_distance from {self.config.min_river_distance}m to {max(10, median_length * 0.3):.1f}m")
                adaptive_min_distance = max(10, median_length * 0.3)
            else:
                adaptive_min_distance = self.config.min_river_distance
        else:
            adaptive_min_distance = self.config.min_river_distance
        
        # Group nodes by segment
        segments = densified_nodes.groupby('seg_id')
        
        all_pairs = []
        pair_id_counter = 1
        
        # Count segments with multiple nodes
        multi_node_segments = sum(1 for _, nodes in segments if len(nodes) >= 2)
        logger.info(f"Found {len(segments)} segments, {multi_node_segments} with 2+ nodes")
        
        # Strategy: If most segments have <2 nodes, pair across all nodes globally
        # Otherwise, pair within segments
        if multi_node_segments < len(segments) * 0.1:  # Less than 10% have multiple nodes
            logger.info("Most segments have single nodes - pairing across all nodes globally")
            
            # Sort all nodes by elevation (highest first = upstream)
            all_nodes_sorted = densified_nodes.sort_values('elevation', ascending=False).reset_index(drop=True)
            n_total = len(all_nodes_sorted)
            
            logger.info(f"Pairing {n_total} nodes globally...")
            
            # Nested loop: inlet i (higher elevation), outlet j (lower elevation)
            for i in range(n_total - 1):
                inlet_node = all_nodes_sorted.iloc[i]
                inlet_pt = inlet_node.geometry
                inlet_z = inlet_node['elevation']
                inlet_discharge = inlet_node['discharge']
                
                # Only check nearby outlets to avoid O(n²) explosion
                max_outlets_to_check = min(50, n_total - i - 1)
                
                for j in range(i + 1, i + 1 + max_outlets_to_check):
                    outlet_node = all_nodes_sorted.iloc[j]
                    outlet_pt = outlet_node.geometry
                    outlet_z = outlet_node['elevation']
                    outlet_discharge = outlet_node['discharge']
                    
                    # Condition 1: Positive head
                    head = inlet_z - outlet_z
                    if head <= 0:
                        continue
                    
                    # Condition 2: Euclidean distance (no chainage across segments)
                    euclidean_distance = inlet_pt.distance(outlet_pt)
                    river_distance = euclidean_distance * 1.3  # Sinuosity estimate
                    
                    if river_distance < adaptive_min_distance:
                        continue
                    
                    if river_distance > self.config.max_river_distance:
                        continue
                    
                    # Condition 3: Minimum head
                    if head < self.config.min_head:
                        continue
                    
                    # Compute pair-specific discharge
                    pair_discharge = min(inlet_discharge, outlet_discharge)
                    
                    # Calculate power
                    power_kw = (
                        self.config.rho * self.config.g * 
                        pair_discharge * head * self.config.efficiency
                    ) / 1000.0
                    
                    # Calculate score
                    score = self.calculate_score(
                        head, pair_discharge, river_distance, inlet_pt
                    )
                    
                    # Apply all other constraints
                    is_feasible, constraints = self.apply_constraints(
                        head, river_distance, inlet_pt, outlet_pt
                    )
                    
                    if not is_feasible:
                        continue
                    
                    # Create pair record
                    pair = {
                        'pair_id': pair_id_counter,
                        'inlet_node_id': int(inlet_node['node_id']),
                        'outlet_node_id': int(outlet_node['node_id']),
                        'seg_id': int(inlet_node['seg_id']),  # Inlet segment
                        'inlet_geom': inlet_pt,
                        'outlet_geom': outlet_pt,
                        'inlet_elevation': inlet_z,
                        'outlet_elevation': outlet_z,
                        'head': head,
                        'river_distance': river_distance,
                        'euclidean_distance': euclidean_distance,
                        'discharge': pair_discharge,
                        'power_kw': power_kw,
                        'score': score,
                        'is_feasible': is_feasible,
                        **constraints
                    }
                    
                    all_pairs.append(pair)
                    pair_id_counter += 1
        else:
            # Original within-segment pairing for longer segments
            logger.info("Segments have multiple nodes - pairing within segments")
            
            for seg_id, nodes_group in segments:
                # Sort nodes by chainage (upstream to downstream)
                nodes_sorted = nodes_group.sort_values('chainage').reset_index(drop=True)
                n = len(nodes_sorted)
                
                if n < 2:
                    continue
                
                logger.debug(f"Segment {seg_id}: {n} nodes, pairing...")
                
                # Nested loop: inlet i, outlet j (where j > i)
                for i in range(n - 1):
                    inlet_node = nodes_sorted.iloc[i]
                    inlet_pt = inlet_node.geometry
                    inlet_z = inlet_node['elevation']
                    inlet_chainage = inlet_node['chainage']
                    inlet_discharge = inlet_node['discharge']
                    
                    for j in range(i + 1, n):
                        outlet_node = nodes_sorted.iloc[j]
                        outlet_pt = outlet_node.geometry
                        outlet_z = outlet_node['elevation']
                        outlet_chainage = outlet_node['chainage']
                        outlet_discharge = outlet_node['discharge']
                        
                        # Condition 1: Positive head (outlet lower than inlet)
                        head = inlet_z - outlet_z
                        if head <= 0:
                            continue
                        
                        # Condition 2: Along-river distance via chainage
                        river_distance = outlet_chainage - inlet_chainage
                        if river_distance <= 0:
                            continue
                        
                        # Check distance bounds (use adaptive minimum)
                        if river_distance < adaptive_min_distance:
                            continue
                        
                        if river_distance > self.config.max_river_distance:
                            break  # No point checking further outlets (too far)
                        
                        # Condition 3: Minimum head threshold
                        if head < self.config.min_head:
                            continue
                        
                        # Compute pair-specific discharge
                        pair_discharge = min(inlet_discharge, outlet_discharge)
                        
                        # Calculate power: P = ρ × g × Q × H × η (in kW)
                        power_kw = (
                            self.config.rho * self.config.g * 
                            pair_discharge * head * self.config.efficiency
                        ) / 1000.0
                        
                        # Euclidean distance
                        euclidean_distance = inlet_pt.distance(outlet_pt)
                        
                        # Calculate score
                        score = self.calculate_score(
                            head, pair_discharge, river_distance, inlet_pt
                        )
                        
                        # Apply all other constraints
                        is_feasible, constraints = self.apply_constraints(
                            head, river_distance, inlet_pt, outlet_pt
                        )
                        
                        if not is_feasible:
                            continue
                        
                        # Create pair record
                        pair = {
                            'pair_id': pair_id_counter,
                            'inlet_node_id': int(inlet_node['node_id']),
                            'outlet_node_id': int(outlet_node['node_id']),
                            'seg_id': int(seg_id),
                            'inlet_geom': inlet_pt,
                            'outlet_geom': outlet_pt,
                            'inlet_elevation': inlet_z,
                            'outlet_elevation': outlet_z,
                            'head': head,
                            'river_distance': river_distance,
                            'euclidean_distance': euclidean_distance,
                            'discharge': pair_discharge,
                            'power_kw': power_kw,
                            'score': score,
                            'is_feasible': is_feasible,
                            **constraints
                        }
                        
                        all_pairs.append(pair)
                        pair_id_counter += 1
        
        logger.info(f"Systematic pairing found {len(all_pairs)} raw IO pairs")
        
        # Deduplicate by (inlet_node_id, outlet_node_id): keep best power
        unique_pairs = {}
        for pair in all_pairs:
            key = (pair['inlet_node_id'], pair['outlet_node_id'])
            if key not in unique_pairs or pair['power_kw'] > unique_pairs[key]['power_kw']:
                unique_pairs[key] = pair
        
        dedup_pairs = list(unique_pairs.values())
        logger.info(f"After deduplication: {len(dedup_pairs)} unique IO pairs")
        
        # Sort by power (descending) and assign rank
        dedup_pairs.sort(key=lambda p: p['power_kw'], reverse=True)
        for rank, pair in enumerate(dedup_pairs, start=1):
            pair['rank'] = rank
        
        return dedup_pairs
    
    def run_pairing(self) -> List[Dict]:
        """
        Execute the complete inlet-outlet pairing algorithm.
        
        Returns:
            List of feasible, scored, and deduplicated site pairs
        """
        logger.info("Starting inlet-outlet pairing algorithm")
        
        # Check if systematic pairing is suitable for this network
        # Disable it for networks with very short stream segments
        use_systematic = self.config.use_node_densification
        if use_systematic and 'length_m' in self.stream_network.columns:
            median_length = self.stream_network['length_m'].median()
            if median_length < 50.0:
                logger.warning(f"Stream network has very short segments (median: {median_length:.1f}m)")
                logger.warning("Systematic pairing is not suitable - using original endpoint-based method")
                use_systematic = False
        
        # Check if systematic pairing with node densification is enabled
        if use_systematic:
            logger.info("Using systematic IO pairing with densified nodes")
            
            # Create densified nodes
            densified_nodes = self.create_densified_nodes()
            
            if densified_nodes is None or len(densified_nodes) == 0:
                logger.warning("Node densification failed, falling back to original method")
                # Fall through to original method below
            else:
                # Perform systematic pairing
                pairs = self.systematic_io_pairing(densified_nodes)
                
                # Additional deduplication by spatial proximity (optional)
                if len(pairs) > 0:
                    final_pairs = self.deduplicate_pairs(pairs)
                    logger.info(f"Final result: {len(final_pairs)} unique site pairs (after spatial dedup)")
                    return final_pairs
                else:
                    logger.warning("Systematic pairing produced no results, falling back to original method")
                    # Fall through to original method
        
        # Original pairing method (fallback or when densification disabled)
        logger.info("Using original pairing method (endpoint-based)")
        
        # Step 1: Identify inlet candidates
        inlets = self.identify_inlet_candidates()
        if len(inlets) == 0:
            logger.warning("No inlet candidates found")
            return []
        
        # Step 2: Identify outlet candidates
        outlets = self.identify_outlet_candidates()
        if len(outlets) == 0:
            logger.warning("No outlet candidates found")
            return []
        
        # Step 3: For each inlet, search downstream for outlets
        all_pairs = []
        for idx, inlet in inlets.iterrows():
            pairs = self.search_downstream_outlets(inlet, outlets)
            all_pairs.extend(pairs)
        
        logger.info(f"Found {len(all_pairs)} feasible site pairs before deduplication")
        
        # Step 4: Deduplicate and enforce spacing
        unique_pairs = self.deduplicate_pairs(all_pairs)
        
        # Step 5: Rank pairs
        for rank, pair in enumerate(unique_pairs, start=1):
            pair['rank'] = rank
        
        logger.info(f"Final result: {len(unique_pairs)} unique site pairs")
        return unique_pairs
    
    def save_to_postgis(self, pairs: List[Dict], raster_layer_id: int, 
                       hms_run_id: Optional[int] = None) -> Tuple[List[int], List[int]]:
        """
        Save site pairs to PostGIS database.
        
        Args:
            pairs: List of site pair dictionaries
            raster_layer_id: ID of source RasterLayer model
            hms_run_id: Optional HMSRun ID for discharge association
        
        Returns:
            Tuple of (site_point_ids, site_pair_ids)
        """
        from hydropower.models import RasterLayer, SitePoint, SitePair, HMSRun
        import time
        
        raster_layer = RasterLayer.objects.get(id=raster_layer_id)
        hms_run = None
        if hms_run_id:
            hms_run = HMSRun.objects.get(id=hms_run_id)
        
        # Get existing max site IDs to avoid duplicates
        existing_inlets = SitePoint.objects.filter(
            raster_layer=raster_layer, 
            site_type='INLET'
        ).count()
        existing_outlets = SitePoint.objects.filter(
            raster_layer=raster_layer,
            site_type='OUTLET'
        ).count()
        
        inlet_counter = existing_inlets + 1
        outlet_counter = existing_outlets + 1
        
        site_point_ids = []
        site_pair_ids = []
        
        for pair in pairs:
            # Create inlet point with unique ID
            inlet_id = f"IN_{raster_layer_id}_{inlet_counter}"
            inlet = SitePoint.objects.create(
                raster_layer=raster_layer,
                site_id=inlet_id,
                site_type='INLET',
                geometry=GEOSPoint(pair['inlet_geom'].x, pair['inlet_geom'].y, srid=32651),
                elevation=pair['inlet_elevation']
            )
            site_point_ids.append(inlet.id)
            inlet_counter += 1
            
            # Create outlet point with unique ID
            outlet_id = f"OUT_{raster_layer_id}_{outlet_counter}"
            outlet = SitePoint.objects.create(
                raster_layer=raster_layer,
                site_id=outlet_id,
                site_type='OUTLET',
                geometry=GEOSPoint(pair['outlet_geom'].x, pair['outlet_geom'].y, srid=32651),
                elevation=pair['outlet_elevation']
            )
            site_point_ids.append(outlet.id)
            outlet_counter += 1
            
            # Create site pair
            pair_id = f"{inlet_id}-{outlet_id}"
            line_geom = GEOSLineString(
                (pair['inlet_geom'].x, pair['inlet_geom'].y),
                (pair['outlet_geom'].x, pair['outlet_geom'].y),
                srid=32651
            )
            
            site_pair = SitePair.objects.create(
                raster_layer=raster_layer,
                inlet=inlet,
                outlet=outlet,
                pair_id=pair_id,
                geometry=line_geom,
                river_distance=pair['river_distance'],
                euclidean_distance=pair['euclidean_distance'],
                head=pair['head'],
                discharge=pair.get('discharge'),
                efficiency=self.config.efficiency,
                score=pair['score'],
                rank=pair['rank'],
                hms_run=hms_run,
                meets_head_constraint=pair['meets_head_constraint'],
                meets_distance_constraint=pair['meets_distance_constraint'],
                meets_watershed_constraint=pair.get('meets_watershed_constraint', True),
                meets_land_constraint=pair['meets_land_constraint'],
                is_feasible=pair['is_feasible']
            )
            
            # Calculate power with head losses if discharge available
            if pair.get('discharge'):
                # Use pre-calculated power if available (from systematic_io_pairing)
                if pair.get('power_kw'):
                    site_pair.power = pair['power_kw']
                    logger.debug(f"{pair_id}: Using pre-calculated power={site_pair.power:.1f}kW")
                else:
                    # Calculate head losses
                    penstock_length = pair.get('euclidean_distance', pair['river_distance'] * 0.3)
                    head_loss_data = self.calculate_head_losses(
                        gross_head=pair['head'],
                        penstock_length=penstock_length,
                        discharge=pair['discharge'],
                        num_bends=2
                    )
                    
                    # Calculate power using net head (after losses)
                    net_head = head_loss_data['net_head']
                    power_watts = self.config.rho * self.config.g * pair['discharge'] * net_head * self.config.efficiency
                    site_pair.power = power_watts / 1000.0  # Convert to kW
                    
                    # Store head loss information in metadata
                    site_pair.meets_head_constraint = net_head >= self.config.min_head
                    
                    logger.debug(f"{pair_id}: Gross head={pair['head']:.1f}m, Net head={net_head:.1f}m, "
                               f"Head loss={head_loss_data['total_head_loss']:.1f}m ({head_loss_data['efficiency_factor']*100:.1f}%), "
                               f"Power={site_pair.power:.1f}kW")
                
                site_pair.save()
            
            # Calculate infrastructure layout for top-ranked sites (rank <= 5)
            if pair.get('rank') and pair['rank'] <= 5:
                infrastructure = self.calculate_infrastructure_layout(site_pair)
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
                logger.info(f"Infrastructure layout calculated for {pair_id} (rank {pair['rank']})")
            
            site_pair_ids.append(site_pair.id)
        
        logger.info(f"Saved {len(site_point_ids)} site points and {len(site_pair_ids)} site pairs to PostGIS")
        return site_point_ids, site_pair_ids
    
    def run_pairing_with_discharge(self, hms_run_id: int, 
                                   extraction_method: str = 'peak',
                                   max_search_distance: float = 1000.0) -> Tuple[List[Dict], Dict[str, int]]:
        """
        Execute inlet-outlet pairing with automatic discharge association.
        
        This is a convenience method that combines the pairing algorithm with
        discharge association in a single workflow.
        
        Args:
            hms_run_id: HMSRun ID for discharge data
            extraction_method: 'peak', 'average', or 'percentile'
            max_search_distance: Maximum distance in meters to search for HMS elements
        
        Returns:
            Tuple of (site_pairs_list, discharge_stats)
        """
        from hydropower.discharge_association import DischargeAssociator, DischargeConfig
        
        # Step 1: Run pairing algorithm
        logger.info("Step 1/2: Running inlet-outlet pairing algorithm")
        pairs = self.run_pairing()
        
        if not pairs:
            logger.warning("No site pairs generated")
            return [], {'total': 0, 'assigned': 0, 'failed': 0}
        
        # Step 2: Associate discharge data
        logger.info("Step 2/2: Associating discharge data")
        
        # Create discharge associator
        config = DischargeConfig(
            extraction_method=extraction_method,
            assignment_strategy='scaled'  # Use head-based scaling
        )
        associator = DischargeAssociator(config=config)
        
        # Create discharge summary from HMS TimeSeries data
        discharge_summary = associator.create_discharge_summary(hms_run_id)
        
        if not discharge_summary:
            logger.warning("No discharge data available in HMS run")
            return pairs, {'total': len(pairs), 'assigned': 0, 'failed': len(pairs)}
        
        # Assign discharge to each pair based on head scaling
        assigned_count = 0
        for pair in pairs:
            # Use head-based scaling to assign discharge
            pair_with_discharge = associator.assign_discharge_to_pair(pair, discharge_summary)
            
            if pair_with_discharge.get('discharge') is not None:
                assigned_count += 1
                logger.debug(f"Assigned Q={pair['discharge']:.2f} m³/s (head={pair['head']:.1f}m)")
        
        discharge_stats = {
            'total': len(pairs),
            'assigned': assigned_count,
            'failed': len(pairs) - assigned_count
        }
        
        logger.info(f"Discharge association complete: {assigned_count}/{len(pairs)} pairs have discharge data")
        
        return pairs, discharge_stats
    
    @staticmethod
    def calculate_infrastructure_layout(site_pair):
        """
        Calculate run-of-river hydropower infrastructure component positions.
        
        Uses terrain-based routing with least-cost path analysis for realistic layouts.
        
        Generates geometry for:
        - Intake/Weir basin (at inlet)
        - Settling basin (near intake, 20m downstream)
        - Channel (follows terrain and stream network with least-cost path)
        - Forebay tank (before penstock, at elevated position)
        - Penstock (optimized path considering slope and accessibility)
        - Powerhouse (at outlet, near turbine discharge point)
        
        Args:
            site_pair: SitePair model instance
        
        Returns:
            Dict with infrastructure geometries and dimensions
        """
        from shapely.geometry import Point as ShapelyPoint, LineString as ShapelyLineString
        from shapely.ops import transform, nearest_points
        from shapely.affinity import translate
        import numpy as np
        from django.contrib.gis.geos import GEOSGeometry, fromstr
        from scipy.ndimage import distance_transform_edt
        from skimage.graph import route_through_array
        
        inlet_x, inlet_y = site_pair.inlet.geometry.x, site_pair.inlet.geometry.y
        outlet_x, outlet_y = site_pair.outlet.geometry.x, site_pair.outlet.geometry.y
        inlet_elev = site_pair.inlet.elevation
        outlet_elev = site_pair.outlet.elevation
        
        # 1. Intake basin (water weir) - At inlet point
        intake_basin = GEOSPoint(inlet_x, inlet_y, srid=32651)
        
        # Get stream network segments near the inlet-outlet path
        from hydropower.models import StreamNetwork
        
        # Query streams within buffer around site pair
        buffer_distance = max(100.0, site_pair.euclidean_distance * 0.3)  # 30% buffer or 100m minimum
        site_line = fromstr(f'LINESTRING({inlet_x} {inlet_y}, {outlet_x} {outlet_y})', srid=32651)
        
        stream_segments = StreamNetwork.objects.filter(
            raster_layer=site_pair.raster_layer,
            geometry__dwithin=(site_line, buffer_distance)
        ).order_by('-stream_order', '-length_m')
        
        # Try to trace along stream network if available
        stream_path_coords = []
        if stream_segments.exists():
            # Build a continuous path by connecting stream segments
            # Start from segment closest to inlet
            inlet_shapely = ShapelyPoint(inlet_x, inlet_y)
            outlet_shapely = ShapelyPoint(outlet_x, outlet_y)
            
            # Convert all segments to list with distance from inlet
            segments_data = []
            for stream in stream_segments:
                stream_shapely = ShapelyLineString(stream.geometry.coords)
                # Distance from inlet to start of segment
                dist_to_inlet = inlet_shapely.distance(ShapelyPoint(stream.geometry.coords[0]))
                segments_data.append({
                    'geometry': stream_shapely,
                    'coords': list(stream.geometry.coords),
                    'dist_to_inlet': dist_to_inlet,
                    'used': False
                })
            
            # Sort by distance from inlet (nearest first)
            segments_data.sort(key=lambda x: x['dist_to_inlet'])
            
            # Build connected path starting from inlet
            if segments_data:
                # Start with segment closest to inlet
                current_segment = segments_data[0]
                current_segment['used'] = True
                stream_path_coords.extend(current_segment['coords'])
                current_end = ShapelyPoint(current_segment['coords'][-1])
                
                # Connect subsequent segments
                max_iterations = len(segments_data)
                for _ in range(max_iterations):
                    # Find unused segment closest to current end point
                    min_dist = float('inf')
                    next_segment = None
                    
                    for seg in segments_data:
                        if seg['used']:
                            continue
                        
                        seg_start = ShapelyPoint(seg['coords'][0])
                        seg_end = ShapelyPoint(seg['coords'][-1])
                        
                        dist_start = current_end.distance(seg_start)
                        dist_end = current_end.distance(seg_end)
                        
                        # Check both directions
                        if dist_start < min_dist:
                            min_dist = dist_start
                            next_segment = seg
                            seg['reverse'] = False
                        if dist_end < min_dist:
                            min_dist = dist_end
                            next_segment = seg
                            seg['reverse'] = True
                    
                    # Stop if no nearby segment found or too far away
                    if next_segment is None or min_dist > 100:  # 100m max gap
                        break
                    
                    # Add segment coordinates (reversed if needed)
                    next_segment['used'] = True
                    coords_to_add = next_segment['coords']
                    if next_segment.get('reverse', False):
                        coords_to_add = list(reversed(coords_to_add))
                    
                    stream_path_coords.extend(coords_to_add[1:])  # Skip duplicate point
                    current_end = ShapelyPoint(coords_to_add[-1])
        
        # 2. Settling basin - 20m downstream from intake
        if stream_path_coords and len(stream_path_coords) > 1:
            # Follow stream path for first 20m
            accumulated_dist = 0.0
            settling_x, settling_y = inlet_x, inlet_y
            
            for i in range(1, len(stream_path_coords)):
                x1, y1 = stream_path_coords[i-1]
                x2, y2 = stream_path_coords[i]
                segment_dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                if accumulated_dist + segment_dist >= 20.0:
                    # Interpolate to exactly 20m
                    remaining = 20.0 - accumulated_dist
                    ratio = remaining / segment_dist
                    settling_x = x1 + (x2 - x1) * ratio
                    settling_y = y1 + (y2 - y1) * ratio
                    break
                
                accumulated_dist += segment_dist
                settling_x, settling_y = x2, y2
            
            # Offset 10m perpendicular to avoid blocking flow
            dx = settling_x - inlet_x
            dy = settling_y - inlet_y
            dist = np.sqrt(dx**2 + dy**2)
            if dist > 0:
                unit_dx = dx / dist
                unit_dy = dy / dist
                settling_x += (-unit_dy) * 10.0
                settling_y += (unit_dx) * 10.0
        else:
            # Fallback: simple linear offset
            dx = outlet_x - inlet_x
            dy = outlet_y - inlet_y
            distance_total = np.sqrt(dx**2 + dy**2)
            unit_dx = dx / distance_total
            unit_dy = dy / distance_total
            settling_x = inlet_x + unit_dx * 20.0 + (-unit_dy) * 10.0
            settling_y = inlet_y + unit_dy * 20.0 + (unit_dx) * 10.0
        
        settling_basin = GEOSPoint(settling_x, settling_y, srid=32651)
        
        # 3. Channel path - From settling basin to forebay
        # Uses terrain-based routing with least-cost path considering:
        # - Stream network proximity (follow natural drainage)
        # - Terrain slope (avoid steep gradients)
        # - Distance efficiency (balance natural path vs. construction cost)
        target_channel_length = site_pair.river_distance * 0.7  # Target 70% of river distance
        
        # Try terrain-based routing if DEM available
        channel_points, forebay_x, forebay_y = InletOutletPairing._calculate_terrain_based_channel(
            settling_x, settling_y,
            outlet_x, outlet_y,
            stream_path_coords,
            target_channel_length,
            site_pair.raster_layer
        )
        
        # Fallback: use stream path tracing if terrain routing not available
        if channel_points is None and stream_path_coords and len(stream_path_coords) > 2:
            accumulated_dist = 0.0
            forebay_x, forebay_y = settling_x, settling_y
            channel_points = [(settling_x, settling_y)]
            
            # Find starting point in stream path closest to settling basin
            min_dist = float('inf')
            start_idx = 0
            for i, (x, y) in enumerate(stream_path_coords):
                dist = np.sqrt((x - settling_x)**2 + (y - settling_y)**2)
                if dist < min_dist:
                    min_dist = dist
                    start_idx = i
            
            # Trace along stream from settling basin
            for i in range(start_idx + 1, len(stream_path_coords)):
                x1, y1 = stream_path_coords[i-1]
                x2, y2 = stream_path_coords[i]
                segment_dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                if accumulated_dist + segment_dist >= target_channel_length:
                    # Interpolate to target length
                    remaining = target_channel_length - accumulated_dist
                    ratio = remaining / segment_dist if segment_dist > 0 else 0
                    forebay_x = x1 + (x2 - x1) * ratio
                    forebay_y = y1 + (y2 - y1) * ratio
                    channel_points.append((forebay_x, forebay_y))
                    break
                
                accumulated_dist += segment_dist
                channel_points.append((x2, y2))
                forebay_x, forebay_y = x2, y2
            
            # Use traced path for channel
            if len(channel_points) >= 2:
                channel = GEOSLineString(*channel_points, srid=32651)
                channel_length = sum(
                    np.sqrt((channel_points[i][0] - channel_points[i-1][0])**2 + 
                           (channel_points[i][1] - channel_points[i-1][1])**2)
                    for i in range(1, len(channel_points))
                )
            else:
                # Fallback to straight line
                channel_points = [(settling_x, settling_y), (forebay_x, forebay_y)]
                channel = GEOSLineString(*channel_points, srid=32651)
                channel_length = target_channel_length
        else:
            # Fallback: curved path with multiple points for natural appearance
            dx = outlet_x - inlet_x
            dy = outlet_y - inlet_y
            distance_total = np.sqrt(dx**2 + dy**2)
            unit_dx = dx / distance_total
            unit_dy = dy / distance_total
            
            channel_length = target_channel_length
            
            # Create a meandering path with 5-7 curve points
            num_points = 6
            channel_points = [(settling_x, settling_y)]
            
            for i in range(1, num_points):
                progress = i / (num_points - 1)  # 0.0 to 1.0
                
                # Base position along straight line
                base_x = settling_x + unit_dx * channel_length * progress
                base_y = settling_y + unit_dy * channel_length * progress
                
                # Add sinusoidal offset perpendicular to flow
                # Creates natural-looking meanders
                offset_magnitude = 20.0 * np.sin(progress * np.pi * 2)  # Oscillate 2 times
                offset_x = base_x + (-unit_dy) * offset_magnitude
                offset_y = base_y + (unit_dx) * offset_magnitude
                
                channel_points.append((offset_x, offset_y))
            
            forebay_x, forebay_y = channel_points[-1]
            channel = GEOSLineString(*channel_points, srid=32651)
            channel_length = sum(
                np.sqrt((channel_points[i][0] - channel_points[i-1][0])**2 + 
                       (channel_points[i][1] - channel_points[i-1][1])**2)
                for i in range(1, len(channel_points))
            )
            
            # Adjust forebay position (skip the old linear calculation below)
        
        # 4. Forebay tank - At the end of channel, before penstock
        forebay_tank = GEOSPoint(forebay_x, forebay_y, srid=32651)
        
        # 5. Penstock - Straight pressure pipe from forebay to powerhouse
        penstock_coords = [(forebay_x, forebay_y), (outlet_x, outlet_y)]
        penstock = GEOSLineString(*penstock_coords, srid=32651)
        penstock_length = np.sqrt((outlet_x - forebay_x)**2 + (outlet_y - forebay_y)**2)
        
        # Estimate penstock diameter based on discharge
        penstock_diameter = None
        if site_pair.discharge:
            velocity = 4.0  # m/s (typical for run-of-river)
            penstock_diameter = np.sqrt((4.0 * site_pair.discharge) / (np.pi * velocity))
        
        # 6. Powerhouse - At outlet point (turbine and generator building)
        powerhouse = GEOSPoint(outlet_x, outlet_y, srid=32651)
        
        return {
            'intake_basin_geom': intake_basin,
            'settling_basin_geom': settling_basin,
            'channel_geom': channel,
            'channel_length': channel_length,
            'forebay_tank_geom': forebay_tank,
            'penstock_geom': penstock,
            'penstock_length': penstock_length,
            'penstock_diameter': penstock_diameter,
            'powerhouse_geom': powerhouse
        }
    
    @staticmethod
    def _calculate_terrain_based_channel(start_x, start_y, end_x, end_y, 
                                         stream_coords, target_length, raster_layer):
        """
        Calculate channel path using terrain-based least-cost path analysis.
        
        Considers:
        - Proximity to stream network (lower cost)
        - Terrain slope (avoid steep areas)
        - Distance from start to end
        
        Args:
            start_x, start_y: Starting coordinates (settling basin)
            end_x, end_y: Target area coordinates (near outlet)
            stream_coords: List of stream path coordinates
            target_length: Target channel length (meters)
            raster_layer: RasterLayer model with DEM data
        
        Returns:
            Tuple of (channel_points, forebay_x, forebay_y) or (None, None, None) if failed
        """
        try:
            # Check if DEM is available
            if not raster_layer or not raster_layer.dataset or not raster_layer.dataset.file:
                return None, None, None
            
            import rasterio
            from scipy.ndimage import distance_transform_edt, generic_gradient_magnitude
            from skimage.graph import route_through_array
            
            # Load DEM
            dem_path = raster_layer.dataset.file.path
            
            with rasterio.open(dem_path) as src:
                dem = src.read(1)
                transform = src.transform
                
                # Convert coordinates to pixel indices
                from rasterio.transform import rowcol
                start_row, start_col = rowcol(transform, start_x, start_y)
                end_row, end_col = rowcol(transform, end_x, end_y)
                
                # Check bounds
                if not (0 <= start_row < dem.shape[0] and 0 <= start_col < dem.shape[1]):
                    return None, None, None
                if not (0 <= end_row < dem.shape[0] and 0 <= end_col < dem.shape[1]):
                    return None, None, None
                
                # Create cost surface
                # Factor 1: Slope cost (higher slope = higher cost)
                # Compute gradient magnitude using numpy gradient
                grad_y, grad_x = np.gradient(dem.astype(float))
                gradient = np.sqrt(grad_x**2 + grad_y**2)
                slope_cost = np.clip(gradient / np.percentile(gradient, 95), 0, 10)
                
                # Factor 2: Distance from stream network (farther = higher cost)
                stream_raster = np.zeros_like(dem, dtype=bool)
                if stream_coords:
                    for sx, sy in stream_coords:
                        sr, sc = rowcol(transform, sx, sy)
                        if 0 <= sr < dem.shape[0] and 0 <= sc < dem.shape[1]:
                            stream_raster[sr, sc] = True
                    
                    # Distance transform (closer to stream = lower cost)
                    stream_distance = distance_transform_edt(~stream_raster)
                    stream_cost = np.clip(stream_distance / 50, 0, 5)  # Normalize to 0-5 range
                else:
                    stream_cost = np.ones_like(dem) * 0.5
                
                # Combined cost surface (weighted combination)
                cost_surface = 0.6 * slope_cost + 0.4 * stream_cost + 0.1  # Add base cost
                
                # Find least-cost path
                indices, weight = route_through_array(
                    cost_surface,
                    (start_row, start_col),
                    (end_row, end_col),
                    fully_connected=True,
                    geometric=True
                )
                
                # Convert indices back to coordinates
                channel_points = []
                for row, col in indices:
                    x, y = transform * (col, row)
                    channel_points.append((x, y))
                
                # Resample to get target number of points
                if len(channel_points) > 3:
                    # Calculate forebay position at 70% along path
                    path_length = sum(
                        np.sqrt((channel_points[i][0] - channel_points[i-1][0])**2 + 
                               (channel_points[i][1] - channel_points[i-1][1])**2)
                        for i in range(1, len(channel_points))
                    )
                    
                    # Find point at 70% of path
                    target_dist = path_length * 0.7
                    accumulated = 0.0
                    forebay_idx = len(channel_points) - 1
                    
                    for i in range(1, len(channel_points)):
                        seg_dist = np.sqrt(
                            (channel_points[i][0] - channel_points[i-1][0])**2 + 
                            (channel_points[i][1] - channel_points[i-1][1])**2
                        )
                        if accumulated + seg_dist >= target_dist:
                            # Interpolate
                            ratio = (target_dist - accumulated) / seg_dist if seg_dist > 0 else 0
                            forebay_x = channel_points[i-1][0] + ratio * (channel_points[i][0] - channel_points[i-1][0])
                            forebay_y = channel_points[i-1][1] + ratio * (channel_points[i][1] - channel_points[i-1][1])
                            forebay_idx = i
                            break
                        accumulated += seg_dist
                    else:
                        forebay_x, forebay_y = channel_points[-1]
                    
                    # Trim channel to forebay point
                    channel_points = channel_points[:forebay_idx+1]
                    
                    # Resample to reasonable number of points (10-20)
                    if len(channel_points) > 20:
                        step = len(channel_points) // 15
                        channel_points = [channel_points[i] for i in range(0, len(channel_points), step)]
                        channel_points.append((forebay_x, forebay_y))  # Ensure forebay is included
                    
                    logger.info(f"Terrain-based routing: {len(channel_points)} points, path length={path_length:.1f}m")
                    return channel_points, forebay_x, forebay_y
                else:
                    return None, None, None
                
        except Exception as e:
            logger.warning(f"Terrain-based channel routing failed: {e}")
            return None, None, None
