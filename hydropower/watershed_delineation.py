"""
Watershed Delineation Module

This module handles watershed boundary delineation, stream network extraction,
and vectorization of hydrological features using WhiteboxTools.

Implements tasks from Phase 2: Watershed Delineation
- Extract stream network based on flow accumulation threshold
- Delineate watershed boundaries
- Vectorize stream network to PostGIS LineString
- Vectorize watershed boundaries to PostGIS Polygon
- Compute watershed statistics
"""

import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, List

import numpy as np
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape, LineString, Polygon, MultiPolygon
from shapely.ops import unary_union
from django.conf import settings
from django.contrib.gis.geos import GEOSGeometry, MultiLineString, MultiPolygon as GEOSMultiPolygon
from whitebox import WhiteboxTools

from .models import RasterLayer

logger = logging.getLogger(__name__)


class WatershedDelineator:
    """
    Handles watershed delineation and stream network extraction using WhiteboxTools.
    
    Attributes:
        wbt: WhiteboxTools instance
        raster_layer: RasterLayer model instance (must have preprocessed flow data)
        output_dir: Directory for watershed outputs
        stream_threshold: Flow accumulation threshold for stream extraction
    """
    
    def __init__(self, raster_layer: RasterLayer, stream_threshold: int = 1000, output_dir: str = None):
        """
        Initialize watershed delineator.
        
        Args:
            raster_layer: RasterLayer with preprocessed flow direction/accumulation
            stream_threshold: Minimum flow accumulation cells for stream extraction
            output_dir: Directory for output files (defaults to media/watersheds/)
        """
        if not raster_layer.is_preprocessed:
            raise ValueError("RasterLayer must be preprocessed (flow direction/accumulation required)")
        
        if not raster_layer.flow_direction_path or not raster_layer.flow_accumulation_path:
            raise ValueError("Flow direction and flow accumulation rasters required")
        
        self.wbt = WhiteboxTools()
        self.wbt.set_verbose_mode(False)
        
        self.raster_layer = raster_layer
        self.stream_threshold = stream_threshold
        
        # Set up output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(
                settings.MEDIA_ROOT, 
                'watersheds',
                f'watershed_{raster_layer.id}_{timestamp}'
            )
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set WhiteboxTools working directory to output directory
        self.wbt.work_dir = str(self.output_dir)
        
        logger.info(f"Initialized WatershedDelineator for RasterLayer {raster_layer.id}")
        logger.info(f"Stream threshold: {stream_threshold} cells")
        logger.info(f"Output directory: {self.output_dir}")
    
    def extract_stream_network(self) -> str:
        """
        Extract stream network based on flow accumulation threshold.
        
        Uses WhiteboxTools to:
        1. Threshold flow accumulation raster
        2. Extract stream grid cells
        
        Returns:
            Path to stream raster file
        """
        logger.info(f"Extracting stream network with threshold {self.stream_threshold}")
        
        # Get absolute path to flow accumulation raster
        flow_accum_path = os.path.join(settings.MEDIA_ROOT, self.raster_layer.flow_accumulation_path)
        stream_raster_path = str(self.output_dir / 'streams.tif')
        
        logger.info(f"Flow accumulation path: {flow_accum_path}")
        logger.info(f"Output stream path: {stream_raster_path}")
        
        # Extract streams by thresholding flow accumulation
        self.wbt.extract_streams(
            flow_accum=flow_accum_path,
            output=stream_raster_path,
            threshold=self.stream_threshold
        )
        
        # Validate that streams were extracted
        with rasterio.open(stream_raster_path) as src:
            stream_array = src.read(1)
            nodata = src.nodata
            
            # Count valid stream pixels
            if nodata is not None:
                valid_pixels = np.count_nonzero((stream_array != nodata) & (stream_array > 0))
            else:
                valid_pixels = np.count_nonzero(stream_array > 0)
            
            if valid_pixels == 0:
                # Check flow accumulation statistics to suggest better threshold
                with rasterio.open(flow_accum_path) as flow_src:
                    flow_array = flow_src.read(1)
                    flow_nodata = flow_src.nodata
                    if flow_nodata is not None:
                        valid_flow = flow_array[flow_array != flow_nodata]
                    else:
                        valid_flow = flow_array[flow_array > 0]
                    
                    max_flow = float(np.max(valid_flow)) if len(valid_flow) > 0 else 0
                    p95_flow = float(np.percentile(valid_flow, 95)) if len(valid_flow) > 0 else 0
                
                raise ValueError(
                    f"No streams extracted with threshold {self.stream_threshold}. "
                    f"Flow accumulation max: {max_flow:.0f}, 95th percentile: {p95_flow:.0f}. "
                    f"Try a lower threshold (e.g., {int(p95_flow * 0.5)} to {int(p95_flow)})."
                )
        
        logger.info(f"Stream raster created: {stream_raster_path} ({valid_pixels} stream pixels)")
        return stream_raster_path
    
    def delineate_watersheds(self, pour_points: Optional[str] = None) -> str:
        """
        Delineate watershed boundaries using flow direction.
        
        Args:
            pour_points: Optional shapefile path with pour points (outlet locations)
                        If None, extracts watersheds for all stream outlets
        
        Returns:
            Path to watershed raster file
        """
        logger.info("Delineating watershed boundaries")
        
        # Get absolute path to flow direction raster
        flow_dir_path = os.path.join(settings.MEDIA_ROOT, self.raster_layer.flow_direction_path)
        watershed_raster_path = str(self.output_dir / 'watersheds.tif')
        
        logger.info(f"Flow direction path: {flow_dir_path}")
        logger.info(f"Output watershed path: {watershed_raster_path}")
        
        if pour_points:
            # Delineate watersheds from specific pour points
            logger.info(f"Using pour points: {pour_points}")
            self.wbt.watershed(
                d8_pntr=flow_dir_path,
                pour_pts=pour_points,
                output=watershed_raster_path
            )
        else:
            # Use basins tool to extract all watersheds
            logger.info("Extracting all watershed basins")
            self.wbt.basins(
                d8_pntr=flow_dir_path,
                output=watershed_raster_path
            )
        
        logger.info(f"Watershed raster created: {watershed_raster_path}")
        return watershed_raster_path
    
    def vectorize_stream_network(self, stream_raster_path: str) -> gpd.GeoDataFrame:
        """
        Vectorize stream network raster to LineString geometries using WhiteboxTools.
        
        Args:
            stream_raster_path: Path to stream raster file
        
        Returns:
            GeoDataFrame with stream LineString geometries
        """
        logger.info("Vectorizing stream network to LineStrings with WhiteboxTools")
        
        stream_vector_path = str(self.output_dir / 'streams_vector.shp')
        flow_dir_path = os.path.join(settings.MEDIA_ROOT, self.raster_layer.flow_direction_path)
        
        # Set WhiteboxTools working directory
        original_wd = self.wbt.get_working_dir()
        self.wbt.set_working_dir(str(self.output_dir))
        
        logger.info(f"Stream raster: {stream_raster_path}")
        logger.info(f"Flow direction: {flow_dir_path}")
        logger.info(f"Output vector: {stream_vector_path}")
        logger.info(f"WBT working dir: {self.wbt.get_working_dir()}")
        
        try:
            # Use WhiteboxTools to vectorize streams
            self.wbt.raster_streams_to_vector(
                streams=stream_raster_path,
                d8_pntr=flow_dir_path,
                output=stream_vector_path
            )
            
            # Check if output was created
            if not os.path.exists(stream_vector_path):
                # List files to debug
                output_files = list(self.output_dir.glob('*'))
                logger.error(f"Output not created. Files in dir: {[f.name for f in output_files]}")
                raise FileNotFoundError(f"WhiteboxTools did not create: {stream_vector_path}")
            
            # Read vectorized streams
            streams_gdf = gpd.read_file(stream_vector_path)
            logger.info(f"WhiteboxTools vectorization successful: {len(streams_gdf)} features")
            
        except Exception as e:
            logger.error(f"WhiteboxTools vectorization failed: {e}")
            raise ValueError(
                f"Stream vectorization failed: {e}. "
                f"Try lowering threshold (current: {self.stream_threshold})."
            )
        finally:
            # Restore working directory
            self.wbt.set_working_dir(original_wd)
        
        # Ensure CRS is set
        if streams_gdf.crs is None:
            with rasterio.open(stream_raster_path) as src:
                streams_gdf.crs = src.crs
        
        # Clean up geometry
        streams_gdf = streams_gdf[streams_gdf.geometry.notnull()]
        streams_gdf = streams_gdf[streams_gdf.geometry.is_valid]
        
        if len(streams_gdf) == 0:
            raise ValueError("No valid stream features after cleanup")
        
        # Compute stream lengths
        streams_gdf['length_m'] = streams_gdf.geometry.length
        
        # Calculate stream order (with limits for performance)
        if len(streams_gdf) > 500:
            logger.warning(f"Large network ({len(streams_gdf)} segments). Skipping topology.")
            streams_gdf['stream_order'] = 1
        else:
            try:
                streams_gdf = self._calculate_stream_order(streams_gdf)
            except Exception as e:
                logger.warning(f"Stream order calc failed: {e}. Using default.")
                streams_gdf['stream_order'] = 1
        
        # Extract nodes for smaller networks
        if len(streams_gdf) <= 200:
            try:
                streams_gdf = self._extract_stream_nodes(streams_gdf)
            except Exception as e:
                logger.warning(f"Node extraction failed: {e}")
        
        logger.info(f"Vectorized {len(streams_gdf)} stream segments")
        logger.info(f"Total length: {streams_gdf['length_m'].sum():.2f} m")
        if 'stream_order' in streams_gdf.columns and streams_gdf['stream_order'].nunique() > 1:
            logger.info(f"Stream orders: {sorted(streams_gdf['stream_order'].unique())}")
        
        return streams_gdf
    
    def _calculate_stream_order(self, streams_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Calculate Strahler stream order for stream network.
        
        Uses topology analysis to determine stream hierarchy:
        - Order 1: Headwater streams (no upstream tributaries)
        - Order n+1: Junction of two order n streams
        - Order n: Junction of order n and lower order stream
        
        Args:
            streams_gdf: GeoDataFrame with stream LineStrings
        
        Returns:
            GeoDataFrame with stream_order column
        """
        logger.info("Calculating Strahler stream orders...")
        
        from shapely.geometry import Point, LineString
        from collections import defaultdict
        
        # Initialize stream order to 1 (headwater streams)
        streams_gdf['stream_order'] = 1
        streams_gdf['upstream_count'] = 0
        streams_gdf['is_confluence'] = False
        streams_gdf['is_outlet'] = False
        
        # Build topology: map endpoints to stream segments
        endpoint_map = defaultdict(list)  # Maps (x, y) -> [stream_indices]
        
        for idx, row in streams_gdf.iterrows():
            geom = row.geometry
            if isinstance(geom, LineString):
                start_pt = (round(geom.coords[0][0], 2), round(geom.coords[0][1], 2))
                end_pt = (round(geom.coords[-1][0], 2), round(geom.coords[-1][1], 2))
                
                # Start point: this stream flows FROM here
                # End point: this stream flows TO here
                endpoint_map[end_pt].append(('end', idx))
                endpoint_map[start_pt].append(('start', idx))
        
        # Identify confluences (where multiple streams meet)
        confluences = []
        outlets = []
        
        for point, connections in endpoint_map.items():
            # Count incoming streams (streams ending at this point)
            incoming = [conn for conn in connections if conn[0] == 'end']
            # Count outgoing streams (streams starting at this point)
            outgoing = [conn for conn in connections if conn[0] == 'start']
            
            if len(incoming) >= 2:
                # Confluence: multiple streams join
                confluences.append((point, incoming, outgoing))
            elif len(incoming) == 1 and len(outgoing) == 0:
                # Outlet: stream ends here
                outlets.append((point, incoming[0][1]))
        
        logger.info(f"Found {len(confluences)} confluences and {len(outlets)} outlets")
        
        # Mark outlets
        for point, stream_idx in outlets:
            streams_gdf.loc[stream_idx, 'is_outlet'] = True
        
        # Mark confluences and count upstream tributaries
        for point, incoming, outgoing in confluences:
            for _, stream_idx in incoming:
                streams_gdf.loc[stream_idx, 'is_confluence'] = True
            
            # If there's an outgoing stream, set its upstream count
            if outgoing:
                for _, out_idx in outgoing:
                    streams_gdf.loc[out_idx, 'upstream_count'] = len(incoming)
        
        # Calculate Strahler order iteratively
        # Start from headwaters (upstream_count = 0) and work downstream
        max_iterations = 100
        iteration = 0
        
        while iteration < max_iterations:
            updated = False
            
            for idx, row in streams_gdf.iterrows():
                if row['upstream_count'] == 0:
                    # Headwater stream - already order 1
                    continue
                
                # Find upstream streams
                geom = row.geometry
                start_pt = (round(geom.coords[0][0], 2), round(geom.coords[0][1], 2))
                
                # Get all streams that flow into this stream's start point
                upstream_streams = []
                for conn_type, conn_idx in endpoint_map[start_pt]:
                    if conn_type == 'end' and conn_idx != idx:
                        upstream_streams.append(conn_idx)
                
                if upstream_streams:
                    upstream_orders = [streams_gdf.loc[u, 'stream_order'] for u in upstream_streams]
                    max_order = max(upstream_orders)
                    count_max = upstream_orders.count(max_order)
                    
                    # Strahler rule: if two or more streams of same max order join, increase order
                    if count_max >= 2:
                        new_order = max_order + 1
                    else:
                        new_order = max_order
                    
                    if streams_gdf.loc[idx, 'stream_order'] != new_order:
                        streams_gdf.loc[idx, 'stream_order'] = new_order
                        updated = True
            
            if not updated:
                break
            iteration += 1
        
        logger.info(f"Stream order calculation completed in {iteration} iterations")
        
        return streams_gdf
    
    def _extract_stream_nodes(self, streams_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Extract stream nodes (confluences and outlets) and assign node IDs.
        
        Args:
            streams_gdf: GeoDataFrame with stream segments
        
        Returns:
            Enhanced GeoDataFrame with node information
        """
        logger.info("Extracting stream nodes...")
        
        from shapely.geometry import Point
        from collections import defaultdict
        
        # Collect all endpoints
        nodes = []
        node_map = {}  # Maps (x, y) -> node_id
        node_id = 1
        
        for idx, row in streams_gdf.iterrows():
            geom = row.geometry
            if isinstance(geom, LineString):
                start_pt = (round(geom.coords[0][0], 2), round(geom.coords[0][1], 2))
                end_pt = (round(geom.coords[-1][0], 2), round(geom.coords[-1][1], 2))
                
                # Assign node IDs
                if start_pt not in node_map:
                    node_map[start_pt] = node_id
                    nodes.append({
                        'node_id': node_id,
                        'x': start_pt[0],
                        'y': start_pt[1],
                        'node_type': 'unknown'
                    })
                    node_id += 1
                
                if end_pt not in node_map:
                    node_map[end_pt] = node_id
                    nodes.append({
                        'node_id': node_id,
                        'x': end_pt[0],
                        'y': end_pt[1],
                        'node_type': 'unknown'
                    })
                    node_id += 1
        
        # Classify node types based on stream connections
        endpoint_connections = defaultdict(list)
        for idx, row in streams_gdf.iterrows():
            geom = row.geometry
            if isinstance(geom, LineString):
                start_pt = (round(geom.coords[0][0], 2), round(geom.coords[0][1], 2))
                end_pt = (round(geom.coords[-1][0], 2), round(geom.coords[-1][1], 2))
                
                endpoint_connections[start_pt].append(('start', idx))
                endpoint_connections[end_pt].append(('end', idx))
        
        # Update node types
        for i, node in enumerate(nodes):
            pt = (node['x'], node['y'])
            connections = endpoint_connections[pt]
            
            incoming = sum(1 for conn in connections if conn[0] == 'end')
            outgoing = sum(1 for conn in connections if conn[0] == 'start')
            
            if incoming == 0 and outgoing == 1:
                node['node_type'] = 'source'
            elif incoming == 1 and outgoing == 0:
                node['node_type'] = 'outlet'
            elif incoming >= 2:
                node['node_type'] = 'confluence'
            elif outgoing >= 2:
                node['node_type'] = 'divergence'
            else:
                node['node_type'] = 'junction'
        
        # Add node IDs to stream segments
        streams_gdf['from_node'] = 0
        streams_gdf['to_node'] = 0
        
        for idx, row in streams_gdf.iterrows():
            geom = row.geometry
            if isinstance(geom, LineString):
                start_pt = (round(geom.coords[0][0], 2), round(geom.coords[0][1], 2))
                end_pt = (round(geom.coords[-1][0], 2), round(geom.coords[-1][1], 2))
                
                streams_gdf.loc[idx, 'from_node'] = node_map[start_pt]
                streams_gdf.loc[idx, 'to_node'] = node_map[end_pt]
        
        # Save nodes to separate file
        nodes_gdf = gpd.GeoDataFrame(
            nodes,
            geometry=[Point(n['x'], n['y']) for n in nodes],
            crs=streams_gdf.crs
        )
        nodes_path = str(self.output_dir / 'stream_nodes.shp')
        nodes_gdf.to_file(nodes_path)
        
        logger.info(f"Extracted {len(nodes)} stream nodes")
        logger.info(f"Node types: {nodes_gdf['node_type'].value_counts().to_dict()}")
        
        return streams_gdf
    
    def vectorize_watersheds(self, watershed_raster_path: str) -> gpd.GeoDataFrame:
        """
        Vectorize watershed raster to Polygon geometries.
        
        Args:
            watershed_raster_path: Path to watershed raster file
        
        Returns:
            GeoDataFrame with watershed Polygon geometries
        """
        logger.info("Vectorizing watershed boundaries to Polygons")
        
        # Read watershed raster
        with rasterio.open(watershed_raster_path) as src:
            watershed_array = src.read(1)
            transform = src.transform
            crs = src.crs
            nodata = src.nodata
        
        # Handle nodata
        if nodata is not None:
            mask = watershed_array != nodata
        else:
            mask = np.ones_like(watershed_array, dtype=bool)
        
        # Extract shapes (polygons) from raster
        logger.info("Extracting polygon shapes from raster...")
        polygons = []
        watershed_ids = []
        
        for geom, value in shapes(watershed_array, mask=mask, transform=transform):
            if value != 0:  # Skip background
                poly = shape(geom)
                if poly.is_valid:
                    polygons.append(poly)
                    watershed_ids.append(int(value))
        
        # Create GeoDataFrame
        watersheds_gdf = gpd.GeoDataFrame({
            'watershed_id': watershed_ids,
            'geometry': polygons
        }, crs=crs)
        
        # Compute watershed statistics
        watersheds_gdf['area_m2'] = watersheds_gdf.geometry.area
        watersheds_gdf['area_km2'] = watersheds_gdf['area_m2'] / 1_000_000
        watersheds_gdf['perimeter_m'] = watersheds_gdf.geometry.length
        
        logger.info(f"Vectorized {len(watersheds_gdf)} watershed polygons")
        logger.info(f"Total watershed area: {watersheds_gdf['area_km2'].sum():.2f} km²")
        
        return watersheds_gdf
    
    def compute_watershed_statistics(
        self, 
        watersheds_gdf: gpd.GeoDataFrame, 
        streams_gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Compute additional watershed statistics including stream statistics.
        
        Args:
            watersheds_gdf: GeoDataFrame with watershed polygons
            streams_gdf: GeoDataFrame with stream LineStrings
        
        Returns:
            Enhanced watersheds_gdf with additional statistics
        """
        logger.info("Computing watershed statistics")
        
        # Spatial join streams to watersheds
        streams_in_watersheds = gpd.sjoin(streams_gdf, watersheds_gdf, how='left', predicate='within')
        
        # Compute stream length per watershed
        stream_lengths = streams_in_watersheds.groupby('watershed_id')['length_m'].sum()
        watersheds_gdf['stream_length_m'] = watersheds_gdf['watershed_id'].map(stream_lengths).fillna(0)
        watersheds_gdf['stream_length_km'] = watersheds_gdf['stream_length_m'] / 1000
        
        # Compute drainage density (stream length / area)
        watersheds_gdf['drainage_density'] = (
            watersheds_gdf['stream_length_km'] / watersheds_gdf['area_km2']
        )
        
        # Count stream segments per watershed
        stream_counts = streams_in_watersheds.groupby('watershed_id').size()
        watersheds_gdf['stream_count'] = watersheds_gdf['watershed_id'].map(stream_counts).fillna(0).astype(int)
        
        # Compute compactness coefficient (Gravelius coefficient)
        # CC = perimeter / (2 * sqrt(π * area))
        watersheds_gdf['compactness'] = (
            watersheds_gdf['perimeter_m'] / 
            (2 * np.sqrt(np.pi * watersheds_gdf['area_m2']))
        )
        
        logger.info("Watershed statistics computed successfully")
        
        return watersheds_gdf
    
    def validate_against_reference(
        self, 
        watersheds_gdf: gpd.GeoDataFrame, 
        reference_shapefile: str,
        tolerance_pct: float = 10.0
    ) -> Dict[str, any]:
        """
        Validate delineated watersheds against reference shapefile.
        
        Args:
            watersheds_gdf: Delineated watershed GeoDataFrame
            reference_shapefile: Path to reference watershed shapefile
            tolerance_pct: Acceptable area difference percentage
        
        Returns:
            Validation report dictionary
        """
        logger.info(f"Validating watersheds against reference: {reference_shapefile}")
        
        # Read reference shapefile
        reference_gdf = gpd.read_file(reference_shapefile)
        
        # Ensure same CRS
        if reference_gdf.crs != watersheds_gdf.crs:
            reference_gdf = reference_gdf.to_crs(watersheds_gdf.crs)
        
        # Compute total areas
        delineated_area = watersheds_gdf['area_km2'].sum()
        reference_area = reference_gdf.geometry.area.sum() / 1_000_000  # Convert to km²
        
        # Compute area difference
        area_diff_pct = abs(delineated_area - reference_area) / reference_area * 100
        
        # Compute spatial overlap
        delineated_union = unary_union(watersheds_gdf.geometry)
        reference_union = unary_union(reference_gdf.geometry)
        
        intersection = delineated_union.intersection(reference_union)
        union = delineated_union.union(reference_union)
        
        jaccard_index = intersection.area / union.area if union.area > 0 else 0
        overlap_pct = (intersection.area / reference_union.area * 100) if reference_union.area > 0 else 0
        
        # Validation results
        validation_report = {
            'delineated_count': len(watersheds_gdf),
            'reference_count': len(reference_gdf),
            'delineated_area_km2': float(delineated_area),
            'reference_area_km2': float(reference_area),
            'area_difference_pct': float(area_diff_pct),
            'jaccard_index': float(jaccard_index),
            'overlap_pct': float(overlap_pct),
            'validation_passed': area_diff_pct <= tolerance_pct and jaccard_index >= 0.7,
            'tolerance_pct': tolerance_pct
        }
        
        logger.info(f"Validation complete: {validation_report}")
        
        return validation_report
    
    def save_to_postgis(
        self, 
        watersheds_gdf: gpd.GeoDataFrame, 
        streams_gdf: gpd.GeoDataFrame
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Save vectorized watersheds, streams, and nodes to PostGIS database.
        
        Args:
            watersheds_gdf: Watershed polygons GeoDataFrame
            streams_gdf: Stream LineStrings GeoDataFrame
        
        Returns:
            Tuple of (watershed_ids, stream_ids, node_ids) created in database
        """
        from .models import WatershedPolygon, StreamNetwork, StreamNode
        
        logger.info("Saving watersheds, streams, and nodes to PostGIS")
        
        # Save watersheds
        watershed_ids = []
        for idx, row in watersheds_gdf.iterrows():
            # Convert shapely geometry to GEOS
            geom = GEOSGeometry(row.geometry.wkt, srid=32651)
            
            watershed = WatershedPolygon.objects.create(
                raster_layer=self.raster_layer,
                watershed_id=row['watershed_id'],
                geometry=geom,
                area_m2=row['area_m2'],
                area_km2=row['area_km2'],
                perimeter_m=row['perimeter_m'],
                stream_length_m=row.get('stream_length_m', 0),
                stream_length_km=row.get('stream_length_km', 0),
                drainage_density=row.get('drainage_density', 0),
                stream_count=row.get('stream_count', 0),
                compactness=row.get('compactness', 0),
                stream_threshold=self.stream_threshold
            )
            watershed_ids.append(watershed.id)
        
        logger.info(f"Saved {len(watershed_ids)} watersheds to PostGIS")
        
        # Save stream nodes first (if nodes file exists)
        node_ids = []
        nodes_path = str(self.output_dir / 'stream_nodes.shp')
        if os.path.exists(nodes_path):
            nodes_gdf = gpd.read_file(nodes_path)
            
            for idx, row in nodes_gdf.iterrows():
                geom = GEOSGeometry(row.geometry.wkt, srid=32651)
                
                # Count incoming/outgoing streams
                incoming = len(streams_gdf[streams_gdf['to_node'] == row['node_id']])
                outgoing = len(streams_gdf[streams_gdf['from_node'] == row['node_id']])
                
                node = StreamNode.objects.create(
                    raster_layer=self.raster_layer,
                    node_id=row['node_id'],
                    geometry=geom,
                    node_type=row['node_type'],
                    incoming_streams=incoming,
                    outgoing_streams=outgoing,
                    stream_threshold=self.stream_threshold
                )
                node_ids.append(node.id)
            
            logger.info(f"Saved {len(node_ids)} stream nodes to PostGIS")
        
        # Save streams with topology
        stream_ids = []
        for idx, row in streams_gdf.iterrows():
            # Convert shapely geometry to GEOS
            geom = GEOSGeometry(row.geometry.wkt, srid=32651)
            
            stream = StreamNetwork.objects.create(
                raster_layer=self.raster_layer,
                geometry=geom,
                length_m=row['length_m'],
                stream_order=row.get('stream_order', 1),
                from_node=row.get('from_node', 0),
                to_node=row.get('to_node', 0),
                upstream_count=row.get('upstream_count', 0),
                is_confluence=row.get('is_confluence', False),
                is_outlet=row.get('is_outlet', False),
                stream_threshold=self.stream_threshold
            )
            stream_ids.append(stream.id)
        
        logger.info(f"Saved {len(stream_ids)} stream segments to PostGIS")
        
        return watershed_ids, stream_ids, node_ids
    
    def full_delineation_pipeline(
        self, 
        pour_points: Optional[str] = None,
        reference_shapefile: Optional[str] = None,
        save_to_db: bool = True
    ) -> Dict[str, any]:
        """
        Execute full watershed delineation pipeline.
        
        Args:
            pour_points: Optional pour points shapefile
            reference_shapefile: Optional reference watershed for validation
            save_to_db: Whether to save results to PostGIS
        
        Returns:
            Dictionary with results and file paths
        """
        logger.info("Starting full watershed delineation pipeline")
        start_time = datetime.now()
        
        results = {
            'raster_layer_id': self.raster_layer.id,
            'stream_threshold': self.stream_threshold,
            'start_time': start_time.isoformat(),
        }
        
        try:
            # Step 1: Extract stream network
            stream_raster_path = self.extract_stream_network()
            results['stream_raster_path'] = stream_raster_path
            
            # Step 2: Delineate watersheds
            watershed_raster_path = self.delineate_watersheds(pour_points)
            results['watershed_raster_path'] = watershed_raster_path
            
            # Step 3: Vectorize streams
            streams_gdf = self.vectorize_stream_network(stream_raster_path)
            stream_vector_path = str(self.output_dir / 'streams.shp')
            streams_gdf.to_file(stream_vector_path)
            results['stream_vector_path'] = stream_vector_path
            results['stream_count'] = len(streams_gdf)
            results['total_stream_length_km'] = float(streams_gdf['length_m'].sum() / 1000)
            
            # Step 4: Vectorize watersheds
            watersheds_gdf = self.vectorize_watersheds(watershed_raster_path)
            
            # Step 5: Compute statistics
            watersheds_gdf = self.compute_watershed_statistics(watersheds_gdf, streams_gdf)
            watershed_vector_path = str(self.output_dir / 'watersheds.shp')
            watersheds_gdf.to_file(watershed_vector_path)
            results['watershed_vector_path'] = watershed_vector_path
            results['watershed_count'] = len(watersheds_gdf)
            results['total_watershed_area_km2'] = float(watersheds_gdf['area_km2'].sum())
            
            # Step 6: Validation (if reference provided)
            if reference_shapefile:
                validation_report = self.validate_against_reference(
                    watersheds_gdf, reference_shapefile
                )
                results['validation'] = validation_report
            
            # Step 7: Save to PostGIS (if requested)
            if save_to_db:
                watershed_ids, stream_ids, node_ids = self.save_to_postgis(watersheds_gdf, streams_gdf)
                results['watershed_db_ids'] = watershed_ids
                results['stream_db_ids'] = stream_ids
                results['node_db_ids'] = node_ids
            
            # Compute elapsed time
            end_time = datetime.now()
            elapsed = (end_time - start_time).total_seconds()
            results['end_time'] = end_time.isoformat()
            results['elapsed_seconds'] = elapsed
            results['status'] = 'success'
            
            logger.info(f"Watershed delineation pipeline completed in {elapsed:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Watershed delineation pipeline failed: {str(e)}", exc_info=True)
            results['status'] = 'failed'
            results['error'] = str(e)
            raise
        
        return results
