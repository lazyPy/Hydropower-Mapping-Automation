"""
Main Channel Weir Search Algorithm

This module implements weir/diversion location search for inlet points
from the main channel site pairing workflow. It identifies optimal
intake/weir structures near each inlet point.

Workflow:
1. Load top-ranked site pairs from database
2. Extract unique inlet points
3. For each inlet, search DEM for candidate weir locations
4. Apply hydraulic constraints (elevation, distance, direction toward outlet)
5. Rank candidates by suitability
6. Highlight best weir location for each inlet

Based on weir identification methodology for run-of-river hydropower.
"""

import rasterio
from rasterio.transform import rowcol
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging
import math
from django.contrib.gis.geos import Point as GEOSPoint
from django.db import transaction

logger = logging.getLogger(__name__)


@dataclass
class WeirSearchConfig:
    """Configuration for weir/diversion search"""
    search_radius_m: float = 500.0          # Search radius around inlet (m)
    min_distance_m: float = 100.0           # Minimum distance from inlet (m)
    elevation_tolerance_m: float = 20.0     # |DEM_z - inlet_z| <= this (m)
    max_candidates_per_inlet: int = 10      # Max candidates per inlet
    cone_angle_deg: float = 90.0            # Directional cone half-angle (degrees)
    pixel_sampling_factor: int = 4          # Sample every Nth pixel (for performance)


class MainChannelWeirSearch:
    """
    Weir/diversion search algorithm for main channel inlet points.
    
    Searches DEM around inlet points to identify optimal water intake/weir
    locations with hydraulic feasibility constraints.
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
    
    def load_top_site_pairs(self, raster_layer_id: int, top_n: int = 50) -> List[Dict]:
        """
        Load top-ranked site pairs from database.
        
        Args:
            raster_layer_id: Database ID of RasterLayer
            top_n: Number of top pairs to load
        
        Returns:
            List of site pair dictionaries
        """
        from hydropower.models import SitePair
        
        queryset = SitePair.objects.filter(
            raster_layer_id=raster_layer_id,
            is_feasible=True,
            pair_id__startswith='HP_'  # Only main channel pairs
        ).select_related('inlet', 'outlet').order_by('rank')[:top_n]
        
        pairs = []
        for pair in queryset:
            pairs.append({
                'id': pair.id,
                'pair_id': pair.pair_id,
                'rank': pair.rank,
                'inlet_id': pair.inlet.id,
                'inlet_site_id': pair.inlet.site_id,
                'inlet_geometry': pair.inlet.geometry,
                'inlet_elevation': pair.inlet.elevation,
                'outlet_id': pair.outlet.id,
                'outlet_site_id': pair.outlet.site_id,
                'outlet_geometry': pair.outlet.geometry,
                'outlet_elevation': pair.outlet.elevation,
                'head': pair.head,
                'power': pair.power,
            })
        
        logger.info(f"Loaded {len(pairs)} top-ranked site pairs for weir search")
        return pairs
    
    def extract_unique_inlets(self, pairs: List[Dict]) -> List[Dict]:
        """
        Extract unique inlet points from site pairs.
        
        Multiple pairs may share the same inlet, so we group them.
        
        Args:
            pairs: List of site pair dictionaries
        
        Returns:
            List of unique inlet dictionaries with associated pairs
        """
        inlets_map = {}
        
        for pair in pairs:
            inlet_id = pair['inlet_id']
            
            if inlet_id not in inlets_map:
                inlets_map[inlet_id] = {
                    'inlet_id': inlet_id,
                    'inlet_site_id': pair['inlet_site_id'],
                    'geometry': pair['inlet_geometry'],
                    'elevation': pair['inlet_elevation'],
                    'pairs': [],
                    'outlets': [],
                }
            
            inlets_map[inlet_id]['pairs'].append(pair)
            inlets_map[inlet_id]['outlets'].append({
                'outlet_id': pair['outlet_id'],
                'geometry': pair['outlet_geometry'],
                'elevation': pair['outlet_elevation'],
            })
        
        inlets = list(inlets_map.values())
        logger.info(f"Extracted {len(inlets)} unique inlet points from site pairs")
        
        return inlets
    
    def search_weir_candidates(self, inlet: Dict) -> List[Dict]:
        """
        Search for candidate weir locations around an inlet point.
        
        Args:
            inlet: Inlet dictionary with geometry, elevation, outlets
        
        Returns:
            List of weir candidate dictionaries
        """
        if self.dem is None:
            raise RuntimeError("DEM not loaded. Call load_dem() first.")
        
        inlet_x = inlet['geometry'].x
        inlet_y = inlet['geometry'].y
        inlet_elevation = inlet['elevation']
        
        # Calculate search bounding box
        search_radius = self.config.search_radius_m
        min_x = inlet_x - search_radius
        max_x = inlet_x + search_radius
        min_y = inlet_y - search_radius
        max_y = inlet_y + search_radius
        
        # Convert to pixel coordinates (note: Y axis is flipped in raster space)
        row_max_temp, col_min = rowcol(self.dem_transform, min_x, min_y)
        row_min_temp, col_max = rowcol(self.dem_transform, max_x, max_y)
        
        # Clamp to DEM bounds and ensure correct ordering
        row_min = max(0, min(row_min_temp, row_max_temp))
        row_max = min(self.dem.height - 1, max(row_min_temp, row_max_temp))
        col_min = max(0, min(col_min, col_max))
        col_max = min(self.dem.width - 1, max(col_min, col_max))
        
        candidates = []
        
        # Sample pixels within search radius
        sampling_factor = self.config.pixel_sampling_factor
        
        logger.debug(f"Search area for inlet {inlet.get('inlet_site_id', 'unknown')}: "
                    f"rows {row_min}-{row_max} ({row_max - row_min + 1} rows), "
                    f"cols {col_min}-{col_max} ({col_max - col_min + 1} cols), "
                    f"sampling every {sampling_factor} pixels")
        
        for row in range(row_min, row_max + 1, sampling_factor):
            for col in range(col_min, col_max + 1, sampling_factor):
                # Get pixel coordinates
                x, y = self.dem_transform * (col, row)
                
                # Calculate distance from inlet
                distance = math.sqrt((x - inlet_x)**2 + (y - inlet_y)**2)
                
                # Check distance constraints
                if distance < self.config.min_distance_m or distance > search_radius:
                    continue
                
                # Get elevation
                elevation = float(self.dem_array[row, col])
                
                # Check for nodata
                if self.dem_nodata is not None and abs(elevation - self.dem_nodata) < 0.1:
                    continue
                if elevation < -9999:
                    continue
                
                # Check elevation tolerance
                elevation_diff = abs(elevation - inlet_elevation)
                if elevation_diff > self.config.elevation_tolerance_m:
                    continue
                
                # Check directional constraint (toward outlets)
                is_toward_outlets = self.check_directional_constraint(
                    x, y, inlet_x, inlet_y, inlet['outlets']
                )
                
                if not is_toward_outlets:
                    continue
                
                # Create candidate
                candidate = {
                    'x': x,
                    'y': y,
                    'elevation': elevation,
                    'distance_from_inlet': distance,
                    'elevation_difference': elevation - inlet_elevation,
                    'is_toward_outlets': is_toward_outlets,
                }
                
                candidates.append(candidate)
        
        logger.info(f"Found {len(candidates)} weir candidates for inlet {inlet['inlet_site_id']}")
        
        return candidates
    
    def check_directional_constraint(
        self,
        candidate_x: float,
        candidate_y: float,
        inlet_x: float,
        inlet_y: float,
        outlets: List[Dict]
    ) -> bool:
        """
        Check if candidate is in direction toward any outlet.
        
        Uses cone angle to define "toward outlet" region.
        
        Args:
            candidate_x: Candidate X coordinate
            candidate_y: Candidate Y coordinate
            inlet_x: Inlet X coordinate
            inlet_y: Inlet Y coordinate
            outlets: List of outlet dictionaries
        
        Returns:
            True if candidate is toward at least one outlet
        """
        # Vector from inlet to candidate
        vec_ic_x = candidate_x - inlet_x
        vec_ic_y = candidate_y - inlet_y
        len_ic = math.sqrt(vec_ic_x**2 + vec_ic_y**2)
        
        if len_ic < 1e-6:
            return True  # Candidate is at inlet (edge case)
        
        # Normalize
        vec_ic_x /= len_ic
        vec_ic_y /= len_ic
        
        cone_angle_rad = math.radians(self.config.cone_angle_deg)
        
        for outlet in outlets:
            outlet_x = outlet['geometry'].x
            outlet_y = outlet['geometry'].y
            
            # Vector from inlet to outlet
            vec_io_x = outlet_x - inlet_x
            vec_io_y = outlet_y - inlet_y
            len_io = math.sqrt(vec_io_x**2 + vec_io_y**2)
            
            if len_io < 1e-6:
                continue  # Skip if outlet is at inlet (shouldn't happen)
            
            # Normalize
            vec_io_x /= len_io
            vec_io_y /= len_io
            
            # Dot product to get angle
            dot_product = vec_ic_x * vec_io_x + vec_ic_y * vec_io_y
            dot_product = max(-1.0, min(1.0, dot_product))  # Clamp to [-1, 1]
            
            angle_rad = math.acos(dot_product)
            
            # Check if within cone angle
            if angle_rad <= cone_angle_rad:
                return True
        
        return False
    
    def rank_candidates(self, candidates: List[Dict], inlet: Dict) -> List[Dict]:
        """
        Rank weir candidates by suitability.
        
        Scoring factors:
        - Proximity to inlet (closer is better, within constraints)
        - Elevation match (closer to inlet elevation is better)
        - Direction toward outlets
        
        Args:
            candidates: List of candidate dictionaries
            inlet: Inlet dictionary
        
        Returns:
            Ranked list of candidates (sorted by score, descending)
        """
        for candidate in candidates:
            # Distance score (prefer closer, but not too close)
            distance = candidate['distance_from_inlet']
            if distance < 200:
                distance_score = (distance / 200.0) * 50
            elif distance <= 400:
                distance_score = 50 + ((400 - distance) / 200.0) * 50
            else:
                distance_score = max(0, 50 - ((distance - 400) / 100.0) * 10)
            
            # Elevation match score
            elev_diff = abs(candidate['elevation_difference'])
            elevation_score = max(0, 100 - (elev_diff / self.config.elevation_tolerance_m) * 100)
            
            # Total score
            candidate['suitability_score'] = round(
                distance_score * 0.4 + elevation_score * 0.6,
                2
            )
        
        # Sort by score (descending)
        sorted_candidates = sorted(candidates, key=lambda c: c['suitability_score'], reverse=True)
        
        # Assign ranks
        for rank, candidate in enumerate(sorted_candidates, start=1):
            candidate['rank_within_inlet'] = rank
        
        # Limit to top N
        top_candidates = sorted_candidates[:self.config.max_candidates_per_inlet]
        
        if top_candidates:
            logger.info(f"Top weir candidate score: {top_candidates[0]['suitability_score']:.1f}")
        
        return top_candidates
    
    def save_to_database(self, inlets_with_candidates: List[Dict], raster_layer_id: int):
        """
        Save weir candidates to database.
        
        Args:
            inlets_with_candidates: List of inlet dicts with 'candidates' key
            raster_layer_id: Database ID of RasterLayer
        """
        from hydropower.models import WeirCandidate, SitePoint, SitePair, RasterLayer
        
        try:
            raster_layer = RasterLayer.objects.get(id=raster_layer_id)
        except RasterLayer.DoesNotExist:
            raise ValueError(f"RasterLayer with ID {raster_layer_id} not found")
        
        # Delete existing weir candidates for this raster layer
        deleted_count, _ = WeirCandidate.objects.filter(raster_layer=raster_layer).delete()
        if deleted_count > 0:
            logger.info(f"Deleted {deleted_count} existing weir candidates")
        
        total_saved = 0
        
        with transaction.atomic():
            for inlet_data in inlets_with_candidates:
                inlet_site_id = inlet_data['inlet_site_id']
                
                # Get inlet SitePoint
                try:
                    inlet_point = SitePoint.objects.get(site_id=inlet_site_id, raster_layer=raster_layer)
                except SitePoint.DoesNotExist:
                    logger.warning(f"Inlet {inlet_site_id} not found, skipping")
                    continue
                
                # Get associated site pairs
                site_pairs = SitePair.objects.filter(inlet=inlet_point, raster_layer=raster_layer)
                
                # Get outlet node IDs
                outlet_node_ids = [pair.outlet.site_id for pair in site_pairs]
                
                # Save each candidate
                for candidate in inlet_data.get('candidates', []):
                    candidate_id = f"WEIR_{inlet_site_id}_{candidate['rank_within_inlet']:03d}"
                    
                    # Create GEOS Point
                    geom = GEOSPoint(candidate['x'], candidate['y'], srid=32651)
                    
                    weir_candidate = WeirCandidate.objects.create(
                        raster_layer=raster_layer,
                        inlet_point=inlet_point,
                        candidate_id=candidate_id,
                        geometry=geom,
                        elevation=candidate['elevation'],
                        distance_from_inlet=candidate['distance_from_inlet'],
                        elevation_difference=candidate['elevation_difference'],
                        inlet_elevation=inlet_data['elevation'],
                        inlet_node_id=inlet_site_id,
                        outlet_count=len(outlet_node_ids),
                        outlet_node_ids=','.join(outlet_node_ids),
                        is_toward_outlet=candidate['is_toward_outlets'],
                        pair_count=len(site_pairs),
                        pair_ids_list=','.join([pair.pair_id for pair in site_pairs]),
                        suitability_score=candidate['suitability_score'],
                        rank_within_inlet=candidate['rank_within_inlet'],
                        search_radius=self.config.search_radius_m,
                        elevation_tolerance=self.config.elevation_tolerance_m,
                        min_distance=self.config.min_distance_m,
                        cone_angle_deg=self.config.cone_angle_deg,
                    )
                    
                    # Associate with site pairs (many-to-many)
                    weir_candidate.site_pairs.set(site_pairs)
                    
                    total_saved += 1
        
        logger.info(f"Saved {total_saved} weir candidates to database")


def run_main_channel_weir_search(
    raster_layer_id: int,
    dem_path: str,
    top_n_pairs: int = 50,
    config: WeirSearchConfig = None,
    generate_infrastructure: bool = True
) -> Dict:
    """
    Main function to run weir search for main channel site pairs.
    
    Workflow:
    1. Load top N site pairs from main channel pairing
    2. Extract unique inlet points
    3. Search for optimal weir locations around each inlet
    4. Highlight the best weir for each inlet
    5. Optionally generate infrastructure layout for best weir candidates
    
    Args:
        raster_layer_id: Database ID of RasterLayer
        dem_path: Path to DEM GeoTIFF
        top_n_pairs: Number of top site pairs to consider (default: 50)
        config: Weir search configuration
        generate_infrastructure: Generate infrastructure layout for best weirs (default: True)
    
    Returns:
        Dictionary with summary statistics:
        - total_candidates: Total weir candidates found
        - inlets_processed: Number of inlet points processed
        - best_weirs: List of best weir candidates (one per inlet)
        - infrastructure_generated: Number of infrastructure layouts created
    """
    search = MainChannelWeirSearch(config=config)
    
    # Load DEM
    search.load_dem(dem_path)
    
    # Load top site pairs
    pairs = search.load_top_site_pairs(raster_layer_id, top_n=top_n_pairs)
    
    if not pairs:
        logger.error("No site pairs found. Run site pairing first.")
        return {
            'total_candidates': 0,
            'inlets_processed': 0,
            'best_weirs': [],
            'infrastructure_generated': 0
        }
    
    # Extract unique inlets
    inlets = search.extract_unique_inlets(pairs)
    
    # Search weir candidates for each inlet
    inlets_with_candidates = []
    total_candidates = 0
    best_weirs = []
    
    for inlet in inlets:
        candidates = search.search_weir_candidates(inlet)
        
        if candidates:
            ranked_candidates = search.rank_candidates(candidates, inlet)
            inlet['candidates'] = ranked_candidates
            inlets_with_candidates.append(inlet)
            total_candidates += len(ranked_candidates)
            
            # Store best weir (rank 1)
            if ranked_candidates:
                best_weir = ranked_candidates[0].copy()
                best_weir['inlet_site_id'] = inlet['inlet_site_id']
                best_weir['inlet_id'] = inlet['inlet_id']
                best_weirs.append(best_weir)
                logger.info(f"Best weir for {inlet['inlet_site_id']}: "
                           f"Score={best_weir['suitability_score']:.1f}, "
                           f"Distance={best_weir['distance_from_inlet']:.0f}m")
    
    # Save to database
    search.save_to_database(inlets_with_candidates, raster_layer_id)
    
    # Generate infrastructure for best weir locations
    infrastructure_count = 0
    if generate_infrastructure and best_weirs:
        infrastructure_count = generate_weir_infrastructure(
            raster_layer_id, 
            best_weirs,
            dem_path
        )
    
    logger.info(f"Weir search complete: {total_candidates} candidates for {len(inlets_with_candidates)} inlets")
    logger.info(f"Best weir candidates: {len(best_weirs)}")
    logger.info(f"Infrastructure layouts generated: {infrastructure_count}")
    
    return {
        'total_candidates': total_candidates,
        'inlets_processed': len(inlets_with_candidates),
        'best_weirs': best_weirs,
        'infrastructure_generated': infrastructure_count
    }


def generate_weir_infrastructure(
    raster_layer_id: int,
    best_weirs: List[Dict],
    dem_path: str,
    top_n: int = 3
) -> int:
    """
    Generate infrastructure layout for top N best weir candidates.
    
    Ranks all best weirs by combined score (weir suitability + penstock distance)
    and generates infrastructure for the top N weirs only.
    
    For each selected weir location:
    1. Update the associated site pairs to use the weir location as intake
    2. Calculate infrastructure components (channel, penstock, powerhouse)
    3. Save infrastructure geometries to database
    
    Args:
        raster_layer_id: Database ID of RasterLayer
        best_weirs: List of best weir candidate dictionaries
        dem_path: Path to DEM GeoTIFF
        top_n: Number of top weirs to generate infrastructure for (default: 3)
    
    Returns:
        Number of infrastructure layouts generated
    """
    from hydropower.models import WeirCandidate, SitePair, RasterLayer
    from hydropower.site_pairing import InletOutletPairing
    from django.db import transaction
    
    try:
        raster_layer = RasterLayer.objects.get(id=raster_layer_id)
    except RasterLayer.DoesNotExist:
        logger.error(f"RasterLayer {raster_layer_id} not found")
        return 0
    
    # Rank best weirs by combined score (suitability + penstock proximity)
    # Sort by: 1) Highest suitability score, 2) Shortest distance from inlet
    sorted_weirs = sorted(
        best_weirs,
        key=lambda w: (w.get('suitability_score', 0), -w.get('distance_from_inlet', 999999)),
        reverse=True
    )
    
    # Select top N weirs
    top_weirs = sorted_weirs[:top_n]
    
    logger.info(f"Generating infrastructure for top {len(top_weirs)} weirs (out of {len(best_weirs)} total)")
    for i, weir in enumerate(top_weirs, 1):
        logger.info(f"  Rank {i}: {weir['inlet_site_id']} - Score={weir.get('suitability_score', 0):.1f}, "
                   f"Distance={weir.get('distance_from_inlet', 0):.0f}m")
    
    infrastructure_count = 0
    
    for weir_data in top_weirs:  # Process only top N weirs
        inlet_id = weir_data['inlet_id']
        weir_x = weir_data['x']
        weir_y = weir_data['y']
        
        # Get best inlet-outlet pair for this inlet (top 1 pair with highest score)
        site_pairs = SitePair.objects.filter(
            inlet_id=inlet_id,
            raster_layer=raster_layer,
            is_feasible=True,
            pair_id__startswith='HP_'  # Main channel pairs only
        ).order_by('rank')[:1]  # Process best pair only (closest outlet via penstock)
        
        if not site_pairs:
            logger.warning(f"No site pairs found for inlet {weir_data['inlet_site_id']}")
            continue
        
        logger.info(f"Generating infrastructure for {len(site_pairs)} pairs using weir at "
                   f"({weir_x:.1f}, {weir_y:.1f})")
        
        # Generate infrastructure for each site pair
        for site_pair in site_pairs:
            try:
                with transaction.atomic():
                    # Calculate infrastructure with weir as intake point
                    infrastructure = InletOutletPairing.calculate_infrastructure_layout(site_pair)
                    
                    # Override intake basin with weir location
                    weir_point = GEOSPoint(weir_x, weir_y, srid=32651)
                    
                    # Update site pair with infrastructure
                    site_pair.intake_basin_geom = weir_point
                    site_pair.settling_basin_geom = infrastructure['settling_basin_geom']
                    site_pair.channel_geom = infrastructure['channel_geom']
                    site_pair.channel_length = infrastructure['channel_length']
                    site_pair.forebay_tank_geom = infrastructure['forebay_tank_geom']
                    site_pair.penstock_geom = infrastructure['penstock_geom']
                    site_pair.penstock_length = infrastructure['penstock_length']
                    site_pair.penstock_diameter = infrastructure['penstock_diameter']
                    site_pair.powerhouse_geom = infrastructure['powerhouse_geom']
                    
                    site_pair.save()
                    
                    infrastructure_count += 1
                    logger.info(f"  ✓ Infrastructure generated for {site_pair.pair_id}")
                    
            except Exception as e:
                logger.error(f"  ✗ Failed to generate infrastructure for {site_pair.pair_id}: {e}")
                continue
    
    return infrastructure_count
