"""
Watershed-Based HP Node Generation

This module generates HP nodes along ALL river segments WITHIN each watershed/subbasin.
Unlike main_channel-only generation, this creates comprehensive coverage of all streams
in the watershed for systematic hydropower assessment.

Workflow:
1. Load watersheds (WatershedPolygon) from database
2. For each watershed:
   a. Get all stream segments that intersect the watershed
   b. Sample points along each stream at regular intervals
   c. Extract elevation from DEM
   d. Associate nodes with watershed and stream segment
3. Store as HPNode objects with watershed/stream associations

Based on PRD requirement: "HP nodes should be generated along ALL rivers inside subbasins"
"""

import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import linemerge
import numpy as np
import rasterio
from rasterio.transform import rowcol
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
from django.contrib.gis.geos import Point as GEOSPoint
from django.db import transaction
from django.contrib.gis.geos import GEOSGeometry

logger = logging.getLogger(__name__)


class WatershedHPNodeGenerator:
    """
    Generator for HP nodes along all rivers within watersheds.
    
    Samples points at regular intervals along ALL stream segments within
    each watershed, ensuring comprehensive coverage for site pairing.
    """
    
    def __init__(self, sampling_interval_m: float = 200.0):
        """
        Initialize watershed-based HP node generator.
        
        Args:
            sampling_interval_m: Distance between sampled nodes (meters)
                                 Default 200m for dense coverage within watersheds
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
    
    def sample_points_along_line(self, line: LineString, start_distance: float = 0.0) -> List[Tuple[Point, float]]:
        """
        Sample points at regular intervals along a LineString.
        
        Args:
            line: LineString geometry (stream segment)
            start_distance: Starting distance offset for cumulative distance tracking
        
        Returns:
            List of (Point, cumulative_distance) tuples
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
            cumulative_distance = start_distance + distance
            sampled_points.append((point, cumulative_distance))
        
        return sampled_points
    
    def clip_stream_to_watershed(self, stream_geom, watershed_geom):
        """
        Clip a stream geometry to watershed boundary.
        
        Args:
            stream_geom: Stream LineString geometry (Shapely)
            watershed_geom: Watershed Polygon geometry (Shapely)
        
        Returns:
            Clipped stream geometry (LineString or MultiLineString)
        """
        try:
            clipped = stream_geom.intersection(watershed_geom)
            
            # Filter out very small segments (< 50m)
            if clipped.geom_type == 'MultiLineString':
                valid_segments = [seg for seg in clipped.geoms if seg.length >= 50.0]
                if len(valid_segments) == 0:
                    return None
                elif len(valid_segments) == 1:
                    return valid_segments[0]
                else:
                    return MultiLineString(valid_segments)
            elif clipped.geom_type == 'LineString':
                return clipped if clipped.length >= 50.0 else None
            else:
                return None
        except Exception as e:
            logger.warning(f"Error clipping stream to watershed: {e}")
            return None
    
    def generate_nodes_for_watershed(
        self,
        watershed_id: int,
        watershed_geom,
        stream_segments: List[Dict],
        raster_layer_id: int
    ) -> List[Dict]:
        """
        Generate HP nodes for all streams within a single watershed.
        
        Args:
            watershed_id: Database ID of WatershedPolygon
            watershed_geom: Shapely geometry of watershed boundary
            stream_segments: List of stream segment dicts with geometry and ID
            raster_layer_id: Database ID of RasterLayer
        
        Returns:
            List of HP node dictionaries
        """
        hp_nodes = []
        skipped_count = 0
        cumulative_distance = 0.0
        
        logger.info(f"Generating HP nodes for watershed {watershed_id}: "
                   f"{len(stream_segments)} stream segments")
        
        for seg_idx, stream_seg in enumerate(stream_segments):
            stream_geom = stream_seg['geometry']
            stream_id = stream_seg['id']
            
            # Clip stream to watershed boundary
            clipped_stream = self.clip_stream_to_watershed(stream_geom, watershed_geom)
            
            if clipped_stream is None:
                continue
            
            # Handle both LineString and MultiLineString
            segments_to_process = []
            if clipped_stream.geom_type == 'LineString':
                segments_to_process = [clipped_stream]
            elif clipped_stream.geom_type == 'MultiLineString':
                segments_to_process = list(clipped_stream.geoms)
            
            # Sample points along each segment
            for segment in segments_to_process:
                sampled_points = self.sample_points_along_line(segment, cumulative_distance)
                
                for point, distance in sampled_points:
                    elevation = self.extract_elevation(point.x, point.y)
                    
                    if elevation is None:
                        skipped_count += 1
                        continue
                    
                    # Create HP node dict
                    node_id = f"WS{watershed_id}_HP_{len(hp_nodes)+1:04d}"
                    chainage_km = distance / 1000.0
                    
                    hp_node = {
                        'node_id': node_id,
                        'geometry': point,  # Shapely Point
                        'elevation': elevation,
                        'distance_along_channel': distance,
                        'chainage': chainage_km,
                        'watershed_id': watershed_id,
                        'stream_segment_id': stream_id,
                        'raster_layer_id': raster_layer_id,
                        'sampling_interval': self.sampling_interval,
                    }
                    
                    hp_nodes.append(hp_node)
                
                cumulative_distance += segment.length
        
        logger.info(f"  Generated {len(hp_nodes)} HP nodes for watershed {watershed_id} "
                   f"(skipped {skipped_count} due to invalid elevation)")
        
        return hp_nodes
    
    def generate_nodes_for_all_watersheds(
        self,
        dem_path: str,
        raster_layer_id: int,
        min_watershed_area_km2: float = 1.0,
        max_watersheds: Optional[int] = None
    ) -> Dict[int, List[Dict]]:
        """
        Generate HP nodes for all watersheds in the database.
        
        Args:
            dem_path: Path to DEM GeoTIFF
            raster_layer_id: Database ID of RasterLayer
            min_watershed_area_km2: Minimum watershed area to process (default 1.0 km²)
            max_watersheds: Maximum number of watersheds to process (None = all)
        
        Returns:
            Dictionary mapping watershed_id to list of HP node dicts
        """
        from hydropower.models import WatershedPolygon, StreamNetwork, RasterLayer
        
        # Load DEM
        self.load_dem(dem_path)
        
        # Get raster layer
        try:
            raster_layer = RasterLayer.objects.get(id=raster_layer_id)
        except RasterLayer.DoesNotExist:
            raise ValueError(f"RasterLayer with ID {raster_layer_id} not found")
        
        # Get watersheds (filtered by area, ordered by size descending)
        watersheds = WatershedPolygon.objects.filter(
            raster_layer=raster_layer,
            area_km2__gte=min_watershed_area_km2
        ).order_by('-area_km2')
        
        if max_watersheds:
            watersheds = watersheds[:max_watersheds]
        
        total_watersheds = watersheds.count()
        logger.info(f"Processing {total_watersheds} watersheds (min area: {min_watershed_area_km2} km²)")
        
        all_hp_nodes = {}
        
        for idx, watershed in enumerate(watersheds, 1):
            logger.info(f"\n[{idx}/{total_watersheds}] Processing Watershed {watershed.watershed_id} "
                       f"({watershed.area_km2:.2f} km²)")
            
            # Convert Django geometry to Shapely
            watershed_geom = GEOSGeometry(watershed.geometry.wkt)
            from shapely import wkt as shapely_wkt
            watershed_shapely = shapely_wkt.loads(watershed_geom.wkt)
            
            # Get all stream segments that intersect this watershed
            stream_segments = StreamNetwork.objects.filter(
                raster_layer=raster_layer,
                geometry__intersects=watershed.geometry
            )
            
            if stream_segments.count() == 0:
                logger.warning(f"  No streams found for watershed {watershed.watershed_id}")
                continue
            
            # Convert stream geometries to Shapely
            stream_dicts = []
            for stream in stream_segments:
                stream_geom = GEOSGeometry(stream.geometry.wkt)
                stream_shapely = shapely_wkt.loads(stream_geom.wkt)
                stream_dicts.append({
                    'id': stream.id,
                    'geometry': stream_shapely,
                    'stream_order': stream.stream_order,
                })
            
            # Generate HP nodes for this watershed
            hp_nodes = self.generate_nodes_for_watershed(
                watershed_id=watershed.id,
                watershed_geom=watershed_shapely,
                stream_segments=stream_dicts,
                raster_layer_id=raster_layer_id
            )
            
            all_hp_nodes[watershed.id] = hp_nodes
        
        total_nodes = sum(len(nodes) for nodes in all_hp_nodes.values())
        logger.info(f"\n=== SUMMARY ===")
        logger.info(f"Total watersheds processed: {len(all_hp_nodes)}")
        logger.info(f"Total HP nodes generated: {total_nodes}")
        if len(all_hp_nodes) > 0:
            logger.info(f"Average nodes per watershed: {total_nodes / len(all_hp_nodes):.1f}")
        else:
            logger.warning(f"No watersheds with streams found!")
        
        return all_hp_nodes
    
    def save_to_database(self, all_hp_nodes: Dict[int, List[Dict]]):
        """
        Save HP nodes to database (bulk operation).
        
        Args:
            all_hp_nodes: Dictionary mapping watershed_id to list of HP node dicts
        """
        from hydropower.models import HPNode, RasterLayer, WatershedPolygon, StreamNetwork
        
        if not all_hp_nodes:
            logger.warning("No HP nodes to save")
            return
        
        # Get raster layer ID from first node
        first_watershed_nodes = next(iter(all_hp_nodes.values()))
        if not first_watershed_nodes:
            logger.warning("No HP nodes to save")
            return
        
        raster_layer_id = first_watershed_nodes[0]['raster_layer_id']
        
        try:
            raster_layer = RasterLayer.objects.get(id=raster_layer_id)
        except RasterLayer.DoesNotExist:
            raise ValueError(f"RasterLayer with ID {raster_layer_id} not found")
        
        # Delete existing HP nodes for this raster layer
        deleted_count, _ = HPNode.objects.filter(raster_layer=raster_layer).delete()
        if deleted_count > 0:
            logger.info(f"Deleted {deleted_count} existing HP nodes for RasterLayer {raster_layer_id}")
        
        # Bulk create new HP nodes
        hp_node_objects = []
        
        with transaction.atomic():
            for watershed_id, hp_nodes in all_hp_nodes.items():
                # Get watershed and stream objects
                try:
                    watershed = WatershedPolygon.objects.get(id=watershed_id)
                except WatershedPolygon.DoesNotExist:
                    logger.warning(f"Watershed {watershed_id} not found, skipping nodes")
                    continue
                
                for node_data in hp_nodes:
                    # Convert Shapely Point to Django GEOSPoint
                    shapely_point = node_data['geometry']
                    geos_point = GEOSPoint(shapely_point.x, shapely_point.y, srid=32651)
                    
                    # Get stream segment if available
                    stream_segment = None
                    if node_data.get('stream_segment_id'):
                        try:
                            stream_segment = StreamNetwork.objects.get(id=node_data['stream_segment_id'])
                        except StreamNetwork.DoesNotExist:
                            pass
                    
                    hp_node_obj = HPNode(
                        node_id=node_data['node_id'],
                        geometry=geos_point,
                        elevation=node_data['elevation'],
                        distance_along_channel=node_data['distance_along_channel'],
                        chainage=node_data['chainage'],
                        raster_layer=raster_layer,
                        watershed=watershed,
                        stream_segment=stream_segment,
                        sampling_interval=node_data['sampling_interval'],
                        can_be_inlet=True,
                        can_be_outlet=True,
                    )
                    
                    hp_node_objects.append(hp_node_obj)
            
            # Bulk create in batches of 1000
            batch_size = 1000
            for i in range(0, len(hp_node_objects), batch_size):
                batch = hp_node_objects[i:i+batch_size]
                HPNode.objects.bulk_create(batch)
                logger.info(f"  Saved batch {i//batch_size + 1}: {len(batch)} HP nodes")
            
            logger.info(f"=== Saved {len(hp_node_objects)} HP nodes to database ===")
        
        return hp_node_objects


def generate_watershed_hp_nodes(
    dem_path: str,
    raster_layer_id: int,
    sampling_interval_m: float = 200.0,
    min_watershed_area_km2: float = 1.0,
    max_watersheds: Optional[int] = None
) -> int:
    """
    Main function to generate HP nodes for all rivers within watersheds.
    
    Args:
        dem_path: Path to DEM GeoTIFF
        raster_layer_id: Database ID of RasterLayer
        sampling_interval_m: Distance between sampled nodes (meters)
        min_watershed_area_km2: Minimum watershed area to process (km²)
        max_watersheds: Maximum number of watersheds to process (None = all)
    
    Returns:
        Total number of HP nodes generated
    """
    generator = WatershedHPNodeGenerator(sampling_interval_m=sampling_interval_m)
    
    # Generate nodes for all watersheds
    all_hp_nodes = generator.generate_nodes_for_all_watersheds(
        dem_path=dem_path,
        raster_layer_id=raster_layer_id,
        min_watershed_area_km2=min_watershed_area_km2,
        max_watersheds=max_watersheds
    )
    
    # Save to database
    generator.save_to_database(all_hp_nodes)
    
    total_nodes = sum(len(nodes) for nodes in all_hp_nodes.values())
    return total_nodes
