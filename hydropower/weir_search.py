"""
Weir/Diversion Search Algorithm for Hydropower Site Selection

This module implements Objective 3: HPP Location (Weir Identification)
- Focuses on main river inlet points (from high-ranked IO pairs on main stem)
- Searches DEM around top-ranked inlet points for candidate weir locations
- Applies directional constraint toward outlet nodes
- Ensures hydraulic feasibility for water diversion

Main River Focus:
- Most practical hydropower plants are on main rivers, not tiny tributaries
- Main stem has reliable discharge, stable geomorphology, better access
- Minor ephemeral streams are excluded (too small, too seasonal, impractical)

Based on Full-Code_Pairing_to_Weir.py reference implementation.
"""

import rasterio
from rasterio.transform import rowcol
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging
import math
from django.contrib.gis.geos import Point as GEOSPoint
from shapely.geometry import Point as ShapelyPoint

logger = logging.getLogger(__name__)


@dataclass
class WeirSearchConfig:
    """Configuration for weir/diversion search - ALIGNED WITH REFERENCE (reference_here.py)"""
    search_radius_m: float = 1500.0         # Search radius around inlet (m) - WEIR_SEARCH_RADIUS_M from reference
    min_distance_m: float = 500.0           # Minimum distance from inlet (m) - WEIR_MIN_DIST_M from reference
    elevation_tolerance_m: float = 20.0     # |DEM_z - inlet_z| <= this (m) - WEIR_ELEV_TOLERANCE_M from reference
    max_candidates_per_inlet: int = 3       # Max candidates per inlet - WEIR_MAX_CANDIDATES_PER_IN from reference
    cone_angle_deg: float = 50.0            # Directional cone half-angle (degrees) - WEIR_ANGLE_LIMIT_DEG from reference
    top_n_pairs: int = 200                  # Number of top IO pairs to consider - TOP_N_PAIRS from reference


class WeirSearch:
    """
    Weir/Diversion search algorithm for hydropower intake locations.
    
    This class implements the weir identification workflow:
    1. Extract inlet nodes from top-ranked IO pairs
    2. For each inlet, search DEM within radius for candidate locations
    3. Filter candidates by elevation tolerance and minimum distance
    4. Apply directional constraint (toward outlet) using cone angle
    5. Rank candidates by suitability (proximity, elevation match)
    
    Output:
    - TopInletNodes: Filtered inlet nodes from top pairs with metadata
    - WeirCandidates: Candidate weir/diversion locations
    - Links: Geometric connections from inlet to candidate weirs
    """
    
    def __init__(self, config: WeirSearchConfig = None):
        """
        Initialize weir search algorithm.
        
        Args:
            config: Configuration parameters (uses defaults if None)
        """
        self.config = config or WeirSearchConfig()
        self.dem = None
        self.dem_transform = None
        self.dem_array = None
        self.dem_nodata = None
        
    def load_dem(self, dem_path: str):
        """
        Load DEM raster for elevation sampling.
        
        Args:
            dem_path: Path to DEM GeoTIFF file
        """
        try:
            self.dem = rasterio.open(dem_path)
            self.dem_array = self.dem.read(1)
            self.dem_transform = self.dem.transform
            self.dem_nodata = self.dem.nodata
            
            logger.info(f"Loaded DEM for weir search: {self.dem.width}x{self.dem.height}, "
                       f"CRS={self.dem.crs}")
        except Exception as e:
            logger.error(f"Failed to load DEM: {e}")
            raise
    
    def extract_top_inlets(self, site_pairs: List[Dict]) -> List[Dict]:
        """
        Extract unique inlet nodes from top-ranked IO pairs.
        
        Groups all high-ranked pairs by inlet node and collects metadata:
        - How many pairs use this inlet
        - Which outlet nodes are paired with it
        - Inlet elevation and coordinates
        
        Args:
            site_pairs: List of top-ranked site pair dictionaries
        
        Returns:
            List of inlet node dictionaries with metadata
        """
        inlet_map = {}  # inlet_node_id -> metadata
        
        for pair in site_pairs:
            inlet_id = pair.get('inlet_node_id') or pair.get('pair_id', '').split('-')[0]
            inlet_x = pair.get('inlet_x')
            inlet_y = pair.get('inlet_y')
            inlet_z = pair.get('inlet_elevation')
            outlet_id = pair.get('outlet_node_id') or pair.get('pair_id', '').split('-')[1]
            outlet_x = pair.get('outlet_x')
            outlet_y = pair.get('outlet_y')
            pair_id = pair.get('pair_id')
            
            if inlet_id not in inlet_map:
                inlet_map[inlet_id] = {
                    'inlet_node_id': inlet_id,
                    'inlet_x': inlet_x,
                    'inlet_y': inlet_y,
                    'inlet_z': inlet_z,
                    'pair_count': 0,
                    'pair_ids': [],
                    'outlet_count': 0,
                    'outlet_ids': [],
                    'outlet_coords': []  # [(x, y), ...]
                }
            
            # Accumulate metadata
            inlet_map[inlet_id]['pair_count'] += 1
            inlet_map[inlet_id]['pair_ids'].append(pair_id)
            
            # Track unique outlets
            if outlet_id not in inlet_map[inlet_id]['outlet_ids']:
                inlet_map[inlet_id]['outlet_ids'].append(outlet_id)
                inlet_map[inlet_id]['outlet_coords'].append((outlet_x, outlet_y))
                inlet_map[inlet_id]['outlet_count'] += 1
        
        inlet_nodes = list(inlet_map.values())
        logger.info(f"Extracted {len(inlet_nodes)} unique inlet nodes from {len(site_pairs)} top pairs")
        
        return inlet_nodes
    
    def search_weir_candidates(self, inlet_nodes: List[Dict]) -> List[Dict]:
        """
        Search DEM around each inlet for candidate weir/diversion locations.
        
        For each inlet:
        1. Define search area (circle with radius)
        2. Sample DEM cells within search area
        3. Filter by elevation tolerance and minimum distance
        4. Apply directional constraint (toward outlets)
        5. Rank candidates by suitability
        
        Args:
            inlet_nodes: List of inlet node dictionaries from extract_top_inlets()
        
        Returns:
            List of weir candidate dictionaries
        """
        if self.dem is None:
            raise ValueError("DEM not loaded. Call load_dem() first.")
        
        all_candidates = []
        
        # DEM resolution
        x_res = abs(self.dem_transform[0])
        y_res = abs(self.dem_transform[4])
        
        # Cosine of cone angle for directional constraint
        cos_theta_min = math.cos(math.radians(self.config.cone_angle_deg))
        
        logger.info(f"Searching DEM for weir candidates...")
        logger.info(f"  Radius: {self.config.search_radius_m}m, "
                   f"Min dist: {self.config.min_distance_m}m, "
                   f"Elev tol: {self.config.elevation_tolerance_m}m")
        logger.info(f"  Directional cone: {self.config.cone_angle_deg}° half-angle")
        
        for inlet in inlet_nodes:
            inlet_x = inlet['inlet_x']
            inlet_y = inlet['inlet_y']
            inlet_z = inlet['inlet_z']
            inlet_id = inlet['inlet_node_id']
            outlet_coords = inlet['outlet_coords']
            
            # Define search bounding box
            search_bbox = {
                'minx': inlet_x - self.config.search_radius_m,
                'maxx': inlet_x + self.config.search_radius_m,
                'miny': inlet_y - self.config.search_radius_m,
                'maxy': inlet_y + self.config.search_radius_m
            }
            
            # Convert to DEM pixel coordinates
            # rowcol returns (row, col) for given (x, y)
            row_max, col_min = rowcol(self.dem_transform, search_bbox['minx'], search_bbox['miny'])
            row_min, col_max = rowcol(self.dem_transform, search_bbox['maxx'], search_bbox['maxy'])
            
            # Clamp to DEM bounds
            row_min = max(0, row_min)
            row_max = min(self.dem_array.shape[0] - 1, row_max)
            col_min = max(0, col_min)
            col_max = min(self.dem_array.shape[1] - 1, col_max)
            
            # Extract DEM subset
            dem_subset = self.dem_array[row_min:row_max+1, col_min:col_max+1]
            
            # Debug logging for first inlet
            if inlet_id == inlet_nodes[0]['inlet_node_id']:
                valid_pixels = (dem_subset != self.dem_nodata).sum() if self.dem_nodata is not None else dem_subset.size
                logger.info(f"DEBUG first inlet {inlet_id}: subset shape={dem_subset.shape}, valid={valid_pixels}/{dem_subset.size}")
                logger.info(f"  Inlet elev={inlet_z:.1f}m, tolerance=±{self.config.elevation_tolerance_m}m")
                logger.info(f"  Elev range: {inlet_z - self.config.elevation_tolerance_m:.1f} to {inlet_z + self.config.elevation_tolerance_m:.1f}m")
                logger.info(f"  DEM subset range: {dem_subset.min():.1f} to {dem_subset.max():.1f}m")
            
            # Find candidate cells
            candidate_cells = []
            
            for r in range(dem_subset.shape[0]):
                for c in range(dem_subset.shape[1]):
                    cell_z = dem_subset[r, c]
                    
                    # Skip nodata
                    if self.dem_nodata is not None and cell_z == self.dem_nodata:
                        continue
                    
                    # Check elevation tolerance
                    dz = abs(cell_z - inlet_z)
                    if dz > self.config.elevation_tolerance_m:
                        continue
                    
                    # Convert back to geographic coordinates
                    cell_row = row_min + r
                    cell_col = col_min + c
                    # Use rasterio's affine transform to convert pixel to coordinates
                    cell_x, cell_y = self.dem_transform * (cell_col, cell_row)
                    
                    # Distance from inlet
                    dx = cell_x - inlet_x
                    dy = cell_y - inlet_y
                    dist = math.sqrt(dx**2 + dy**2)
                    
                    # Check minimum distance
                    if dist < self.config.min_distance_m:
                        continue
                    
                    # Check if within search radius (circular constraint)
                    if dist > self.config.search_radius_m:
                        continue
                    
                    # Directional constraint: check if candidate is toward ANY outlet
                    is_toward_outlet = False
                    min_angle = 180.0
                    
                    if len(outlet_coords) > 0:
                        for outlet_x, outlet_y in outlet_coords:
                            # Vector from inlet to outlet
                            vec_out_x = outlet_x - inlet_x
                            vec_out_y = outlet_y - inlet_y
                            len_out = math.sqrt(vec_out_x**2 + vec_out_y**2)
                            
                            if len_out < 1e-6:
                                continue  # Outlet same as inlet (shouldn't happen)
                            
                            # Vector from inlet to candidate cell
                            vec_cand_x = cell_x - inlet_x
                            vec_cand_y = cell_y - inlet_y
                            len_cand = dist
                            
                            # Dot product to get cosine of angle
                            dot_product = (vec_out_x * vec_cand_x + vec_out_y * vec_cand_y) / (len_out * len_cand)
                            
                            # Clamp to [-1, 1] to avoid numerical errors
                            dot_product = max(-1.0, min(1.0, dot_product))
                            
                            # Check if within cone
                            if dot_product >= cos_theta_min:
                                is_toward_outlet = True
                                angle_deg = math.degrees(math.acos(dot_product))
                                min_angle = min(min_angle, angle_deg)
                                break
                    else:
                        # No outlets (shouldn't happen), accept all candidates
                        is_toward_outlet = True
                    
                    if not is_toward_outlet:
                        continue
                    
                    # Valid candidate
                    candidate_cells.append({
                        'cell_x': cell_x,
                        'cell_y': cell_y,
                        'cell_z': cell_z,
                        'distance': dist,
                        'dz': dz,
                        'angle_to_outlet': min_angle if min_angle < 180.0 else None
                    })
            
            # Rank candidates by distance (closer is better)
            candidate_cells.sort(key=lambda c: c['distance'])
            
            # Limit to max candidates per inlet
            candidate_cells = candidate_cells[:self.config.max_candidates_per_inlet]
            
            # Create candidate records
            for rank, cand in enumerate(candidate_cells, start=1):
                all_candidates.append({
                    'candidate_id': f"WEIR_{inlet_id}_{rank:03d}",
                    'inlet_node_id': inlet_id,
                    'inlet_x': inlet_x,
                    'inlet_y': inlet_y,
                    'inlet_z': inlet_z,
                    'weir_x': cand['cell_x'],
                    'weir_y': cand['cell_y'],
                    'weir_z': cand['cell_z'],
                    'distance_from_inlet': cand['distance'],
                    'elevation_difference': cand['cell_z'] - inlet_z,
                    'angle_to_outlet_deg': cand.get('angle_to_outlet'),
                    'outlet_count': inlet['outlet_count'],
                    'outlet_ids': ','.join(map(str, inlet['outlet_ids'])),
                    'pair_count': inlet['pair_count'],
                    'pair_ids': ','.join(map(str, inlet['pair_ids'])),
                    'rank_within_inlet': rank,
                    'suitability_score': 100.0 - (cand['distance'] / self.config.search_radius_m) * 50.0  # Simple score
                })
            
            if candidate_cells:
                logger.info(f"  Inlet {inlet_id}: {len(candidate_cells)} candidates found "
                           f"(best: {candidate_cells[0]['distance']:.1f}m, dz={candidate_cells[0]['dz']:.1f}m)")
        
        logger.info(f"Total weir candidates found: {len(all_candidates)}")
        
        return all_candidates
    
    def save_to_postgis(self, inlet_nodes: List[Dict], weir_candidates: List[Dict], 
                       raster_layer_id: int) -> Tuple[List[int], List[int]]:
        """
        Save inlet nodes and weir candidates to PostgreSQL/PostGIS database.
        
        Args:
            inlet_nodes: List of inlet node dictionaries
            weir_candidates: List of weir candidate dictionaries
            raster_layer_id: RasterLayer ID (foreign key)
        
        Returns:
            Tuple of (inlet_site_point_ids, weir_candidate_ids)
        """
        from hydropower.models import RasterLayer, SitePoint, SitePair, WeirCandidate
        from django.contrib.gis.geos import Point
        
        try:
            raster_layer = RasterLayer.objects.get(id=raster_layer_id)
        except RasterLayer.DoesNotExist:
            raise ValueError(f"RasterLayer {raster_layer_id} not found")
        
        # Note: Inlet SitePoints are already created by IO pairing
        # We'll look them up and update metadata if needed
        
        weir_candidate_ids = []
        
        logger.info("Saving weir candidates to database...")
        
        for cand in weir_candidates:
            inlet_node_id = cand['inlet_node_id']
            
            # Find inlet SitePoint
            try:
                inlet_point = SitePoint.objects.get(
                    id=inlet_node_id,
                    raster_layer=raster_layer,
                    site_type='INLET'
                )
            except SitePoint.DoesNotExist:
                logger.warning(f"Inlet SitePoint {inlet_node_id} not found, skipping candidate")
                continue
            
            # Get associated SitePairs
            pair_ids_list = cand['pair_ids'].split(',')
            site_pairs = SitePair.objects.filter(
                raster_layer=raster_layer,
                pair_id__in=pair_ids_list
            )
            
            # Create WeirCandidate
            weir_candidate = WeirCandidate(
                raster_layer=raster_layer,
                inlet_point=inlet_point,
                candidate_id=cand['candidate_id'],
                geometry=Point(cand['weir_x'], cand['weir_y'], srid=32651),
                elevation=cand['weir_z'],
                distance_from_inlet=cand['distance_from_inlet'],
                elevation_difference=cand['elevation_difference'],
                inlet_elevation=cand['inlet_z'],
                inlet_node_id=inlet_node_id,
                outlet_count=cand['outlet_count'],
                outlet_node_ids=cand['outlet_ids'],
                is_toward_outlet=True,
                angle_to_outlet_deg=cand.get('angle_to_outlet_deg'),
                pair_count=cand['pair_count'],
                pair_ids_list=cand['pair_ids'],
                suitability_score=cand.get('suitability_score'),
                rank_within_inlet=cand['rank_within_inlet'],
                search_radius=self.config.search_radius_m,
                elevation_tolerance=self.config.elevation_tolerance_m,
                min_distance=self.config.min_distance_m,
                cone_angle_deg=self.config.cone_angle_deg
            )
            weir_candidate.save()
            
            # Associate with site pairs
            weir_candidate.site_pairs.set(site_pairs)
            
            weir_candidate_ids.append(weir_candidate.id)
        
        logger.info(f"Saved {len(weir_candidate_ids)} weir candidates to database")
        
        return [], weir_candidate_ids  # Return empty inlet IDs (already exist)
