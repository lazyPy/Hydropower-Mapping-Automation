"""
HP Node Generation from Main Channel Vector

This module generates hydropower candidate nodes by sampling points along
the main river channel at regular intervals. These nodes serve as inlet and
outlet candidates for systematic site pairing.

Workflow:
1. Load main channel vector layer (e.g., Nominal-Channel.gpkg)
2. Sample points at regular intervals (e.g., every 500m)
3. Extract elevation from DEM at each point
4. Store as HPNode objects in database

Input: Main channel vector (LineString/MultiLineString)
Output: HP nodes (Point features along channel with elevation)

Based on systematic hydropower site assessment methodology.
"""

import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.ops import linemerge
import numpy as np
import rasterio
from rasterio.transform import rowcol
from typing import List, Tuple, Optional
import logging
from pathlib import Path
from django.contrib.gis.geos import Point as GEOSPoint
from django.db import transaction

logger = logging.getLogger(__name__)


class HPNodeGenerator:
    """
    Generator for HP nodes along main channel.
    
    Samples points at regular intervals along the main river channel,
    extracts elevation from DEM, and stores as candidate inlet/outlet nodes.
    """
    
    def __init__(self, sampling_interval_m: float = 500.0):
        """
        Initialize HP node generator.
        
        Args:
            sampling_interval_m: Distance between sampled nodes (meters)
        """
        self.sampling_interval = sampling_interval_m
        self.dem = None
        self.dem_transform = None
        self.dem_array = None
        self.dem_nodata = None
        
    def load_dem(self, dem_path: str):
        """
        Load DEM raster for elevation extraction.
        
        Args:
            dem_path: Path to DEM GeoTIFF file
        """
        try:
            self.dem = rasterio.open(dem_path)
            self.dem_array = self.dem.read(1)
            self.dem_transform = self.dem.transform
            self.dem_nodata = self.dem.nodata
            
            logger.info(f"Loaded DEM for HP node generation: {self.dem.width}x{self.dem.height}, "
                       f"CRS={self.dem.crs}")
        except Exception as e:
            logger.error(f"Failed to load DEM: {e}")
            raise
    
    def load_main_channel(self, vector_path: str, layer_name: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Load main channel vector layer.
        
        Args:
            vector_path: Path to vector file (e.g., .gpkg, .shp)
            layer_name: Layer name (for multi-layer files like GeoPackage)
        
        Returns:
            GeoDataFrame with main channel geometry
        """
        try:
            if layer_name:
                gdf = gpd.read_file(vector_path, layer=layer_name)
            else:
                gdf = gpd.read_file(vector_path)
            
            logger.info(f"Loaded main channel: {len(gdf)} features, CRS={gdf.crs}")
            
            # Ensure CRS matches DEM (EPSG:32651)
            if gdf.crs and gdf.crs != 'EPSG:32651':
                logger.info(f"Reprojecting main channel from {gdf.crs} to EPSG:32651")
                gdf = gdf.to_crs('EPSG:32651')
            
            return gdf
        except Exception as e:
            logger.error(f"Failed to load main channel: {e}")
            raise
    
    def merge_channel_lines(self, gdf: gpd.GeoDataFrame):
        """
        Merge multiple LineString features into single continuous line.
        Returns all significant segments if full merge fails.
        
        Args:
            gdf: GeoDataFrame with LineString geometries
        
        Returns:
            Single LineString or MultiLineString with all segments
        """
        # Collect all geometries
        geoms = []
        for geom in gdf.geometry:
            if geom.geom_type == 'LineString':
                geoms.append(geom)
            elif geom.geom_type == 'MultiLineString':
                # Split MultiLineString into individual LineStrings
                geoms.extend(list(geom.geoms))
        
        # Merge lines
        merged = linemerge(geoms)
        
        if merged.geom_type == 'LineString':
            logger.info(f"Merged channel into single LineString: {merged.length:.2f}m")
            return merged
        elif merged.geom_type == 'MultiLineString':
            # Return all segments for sampling (don't discard upper reaches!)
            total_length = sum(seg.length for seg in merged.geoms)
            logger.warning(f"Channel has {len(merged.geoms)} disconnected segments (total: {total_length:.2f}m)")
            logger.info(f"Will sample ALL segments to cover entire channel network")
            return merged
        else:
            raise ValueError(f"Unexpected geometry type after merge: {merged.geom_type}")
    
    def sample_points_along_line(self, line: LineString) -> List[Tuple[Point, float]]:
        """
        Sample points at regular intervals along a LineString.
        
        Args:
            line: LineString geometry (main channel)
        
        Returns:
            List of (Point, distance_along_line) tuples
        """
        total_length = line.length
        num_points = int(total_length / self.sampling_interval) + 1
        
        sampled_points = []
        
        for i in range(num_points):
            distance = i * self.sampling_interval
            
            # Clamp to line length
            if distance > total_length:
                distance = total_length
            
            # Interpolate point at distance
            point = line.interpolate(distance)
            sampled_points.append((point, distance))
        
        logger.info(f"Sampled {len(sampled_points)} points along {total_length:.2f}m channel "
                   f"(interval: {self.sampling_interval:.0f}m)")
        
        return sampled_points
    
    def extract_elevation(self, x: float, y: float) -> Optional[float]:
        """
        Extract elevation from DEM at given coordinates.
        
        Args:
            x: X coordinate (UTM)
            y: Y coordinate (UTM)
        
        Returns:
            Elevation in meters, or None if outside DEM or nodata
        """
        if self.dem is None:
            raise RuntimeError("DEM not loaded. Call load_dem() first.")
        
        try:
            row, col = rowcol(self.dem_transform, x, y)
            
            # Check if within bounds
            if 0 <= row < self.dem.height and 0 <= col < self.dem.width:
                elevation = float(self.dem_array[row, col])
                
                # Check for nodata
                if self.dem_nodata is not None and abs(elevation - self.dem_nodata) < 0.1:
                    return None
                
                if elevation < -9999:  # Additional nodata check
                    return None
                
                return elevation
            else:
                return None
        except Exception as e:
            logger.warning(f"Error extracting elevation at ({x}, {y}): {e}")
            return None
    
    def generate_hp_nodes(
        self,
        main_channel_path: str,
        dem_path: str,
        raster_layer_id: int,
        vector_layer_id: Optional[int] = None,
        layer_name: Optional[str] = None
    ) -> List[dict]:
        """
        Generate HP nodes from main channel vector.
        
        Args:
            main_channel_path: Path to main channel vector file
            dem_path: Path to DEM GeoTIFF
            raster_layer_id: Database ID of RasterLayer
            vector_layer_id: Database ID of VectorLayer (optional)
            layer_name: Layer name for multi-layer files
        
        Returns:
            List of HP node dictionaries with geometry and attributes
        """
        # Load data
        self.load_dem(dem_path)
        gdf = self.load_main_channel(main_channel_path, layer_name)
        
        # Merge channel into line(s)
        merged_geom = self.merge_channel_lines(gdf)
        
        # Sample points along channel (handle both LineString and MultiLineString)
        all_sampled_points = []
        cumulative_distance = 0.0
        
        if merged_geom.geom_type == 'LineString':
            # Single line - sample normally
            all_sampled_points = self.sample_points_along_line(merged_geom)
        elif merged_geom.geom_type == 'MultiLineString':
            # Multiple segments - sample each and track cumulative distance
            logger.info(f"Sampling {len(merged_geom.geoms)} channel segments separately")
            for seg_idx, segment in enumerate(merged_geom.geoms):
                seg_points = self.sample_points_along_line(segment)
                # Adjust distance to be cumulative across all segments
                adjusted_points = [(pt, dist + cumulative_distance) for pt, dist in seg_points]
                all_sampled_points.extend(adjusted_points)
                cumulative_distance += segment.length
                logger.info(f"  Segment {seg_idx+1}: {len(seg_points)} points, length {segment.length:.1f}m")
        
        # Extract elevation for each point
        hp_nodes = []
        skipped_count = 0
        
        for idx, (point, distance) in enumerate(all_sampled_points):
            elevation = self.extract_elevation(point.x, point.y)
            
            if elevation is None:
                skipped_count += 1
                logger.warning(f"Skipping node at distance {distance:.0f}m (no valid elevation)")
                continue
            
            # Create HP node dict
            node_id = f"HP_{idx+1:03d}"
            chainage_km = distance / 1000.0
            
            hp_node = {
                'node_id': node_id,
                'geometry': point,  # Shapely Point
                'elevation': elevation,
                'distance_along_channel': distance,
                'chainage': chainage_km,
                'raster_layer_id': raster_layer_id,
                'vector_layer_id': vector_layer_id,
                'source_vector_name': Path(main_channel_path).name,
                'sampling_interval': self.sampling_interval,
            }
            
            hp_nodes.append(hp_node)
        
        logger.info(f"Generated {len(hp_nodes)} HP nodes along main channel "
                   f"(skipped {skipped_count} due to invalid elevation)")
        
        return hp_nodes
    
    def save_to_database(self, hp_nodes: List[dict]):
        """
        Save HP nodes to database.
        
        Args:
            hp_nodes: List of HP node dictionaries from generate_hp_nodes()
        """
        from hydropower.models import HPNode, RasterLayer, VectorLayer
        
        if not hp_nodes:
            logger.warning("No HP nodes to save")
            return
        
        raster_layer_id = hp_nodes[0]['raster_layer_id']
        
        try:
            raster_layer = RasterLayer.objects.get(id=raster_layer_id)
        except RasterLayer.DoesNotExist:
            raise ValueError(f"RasterLayer with ID {raster_layer_id} not found")
        
        # Get vector layer if provided
        vector_layer = None
        vector_layer_id = hp_nodes[0].get('vector_layer_id')
        if vector_layer_id:
            try:
                vector_layer = VectorLayer.objects.get(id=vector_layer_id)
            except VectorLayer.DoesNotExist:
                logger.warning(f"VectorLayer with ID {vector_layer_id} not found")
        
        # Delete existing HP nodes for this raster layer
        deleted_count, _ = HPNode.objects.filter(raster_layer=raster_layer).delete()
        if deleted_count > 0:
            logger.info(f"Deleted {deleted_count} existing HP nodes for RasterLayer {raster_layer_id}")
        
        # Bulk create new HP nodes
        hp_node_objects = []
        
        with transaction.atomic():
            for node_data in hp_nodes:
                # Convert Shapely Point to Django GEOSPoint
                shapely_point = node_data['geometry']
                geos_point = GEOSPoint(shapely_point.x, shapely_point.y, srid=32651)
                
                hp_node_obj = HPNode(
                    node_id=node_data['node_id'],
                    geometry=geos_point,
                    elevation=node_data['elevation'],
                    distance_along_channel=node_data['distance_along_channel'],
                    chainage=node_data['chainage'],
                    raster_layer=raster_layer,
                    source_vector_layer=vector_layer,
                    source_vector_name=node_data['source_vector_name'],
                    sampling_interval=node_data['sampling_interval'],
                    can_be_inlet=True,
                    can_be_outlet=True,
                )
                
                hp_node_objects.append(hp_node_obj)
            
            # Bulk create
            HPNode.objects.bulk_create(hp_node_objects)
            logger.info(f"Saved {len(hp_node_objects)} HP nodes to database")
        
        return hp_node_objects


def generate_hp_nodes_from_main_channel(
    main_channel_path: str,
    dem_path: str,
    raster_layer_id: int,
    vector_layer_id: Optional[int] = None,
    layer_name: Optional[str] = None,
    sampling_interval_m: float = 500.0
) -> int:
    """
    Main function to generate HP nodes from main channel vector.
    
    Args:
        main_channel_path: Path to main channel vector file (e.g., Nominal-Channel.gpkg)
        dem_path: Path to DEM GeoTIFF
        raster_layer_id: Database ID of RasterLayer
        vector_layer_id: Database ID of VectorLayer (optional)
        layer_name: Layer name for multi-layer files (e.g., 'main_channel')
        sampling_interval_m: Distance between sampled nodes (meters)
    
    Returns:
        Number of HP nodes generated
    """
    generator = HPNodeGenerator(sampling_interval_m=sampling_interval_m)
    
    # Generate nodes
    hp_nodes = generator.generate_hp_nodes(
        main_channel_path=main_channel_path,
        dem_path=dem_path,
        raster_layer_id=raster_layer_id,
        vector_layer_id=vector_layer_id,
        layer_name=layer_name
    )
    
    # Save to database
    generator.save_to_database(hp_nodes)
    
    return len(hp_nodes)
