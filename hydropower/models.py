from django.db import models
from django.contrib.gis.db import models as gis_models
from django.utils import timezone
import hashlib
import os


# Create your models here.

class Dataset(models.Model):
    """Model to track uploaded datasets"""
    
    DATASET_TYPES = [
        ('DEM', 'Digital Elevation Model'),
        ('SHAPEFILE', 'Shapefile'),
        ('TIMESERIES', 'Time Series (CSV/Excel)'),
        ('HMS', 'HEC-HMS Output'),
    ]
    
    VALIDATION_STATUS = [
        ('PENDING', 'Pending Validation'),
        ('VALID', 'Valid'),
        ('INVALID', 'Invalid'),
        ('ERROR', 'Validation Error'),
    ]
    
    name = models.CharField(max_length=255, help_text="Dataset name")
    dataset_type = models.CharField(max_length=20, choices=DATASET_TYPES)
    file = models.FileField(upload_to='uploads/%Y/%m/%d/')
    file_size = models.BigIntegerField(help_text="File size in bytes")
    file_checksum = models.CharField(max_length=64, blank=True, help_text="SHA256 checksum")
    
    # Metadata
    original_filename = models.CharField(max_length=255)
    mime_type = models.CharField(max_length=100, blank=True)
    
    # Validation
    validation_status = models.CharField(max_length=20, choices=VALIDATION_STATUS, default='PENDING')
    validation_message = models.TextField(blank=True)
    
    # CRS information
    crs = models.CharField(max_length=50, blank=True, help_text="Coordinate Reference System (EPSG code)")
    
    # Timestamps
    uploaded_at = models.DateTimeField(default=timezone.now)
    validated_at = models.DateTimeField(null=True, blank=True)
    
    # Tags and description
    tags = models.CharField(max_length=255, blank=True, help_text="Comma-separated tags")
    description = models.TextField(blank=True)
    
    class Meta:
        db_table = "datasets"
        verbose_name = "Dataset"
        verbose_name_plural = "Datasets"
        ordering = ['-uploaded_at']
    
    def __str__(self):
        return f"{self.name} ({self.get_dataset_type_display()})"
    
    def calculate_checksum(self):
        """Calculate SHA256 checksum of the uploaded file"""
        sha256 = hashlib.sha256()
        with self.file.open('rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        self.file_checksum = sha256.hexdigest()
        return self.file_checksum
    
    def save(self, *args, **kwargs):
        # Calculate checksum on first save
        if not self.file_checksum and self.file:
            self.calculate_checksum()
        super().save(*args, **kwargs)


class RasterLayer(models.Model):
    """Model to store DEM/raster metadata"""
    
    dataset = models.OneToOneField(Dataset, on_delete=models.CASCADE, related_name='raster_metadata')
    
    # Raster properties
    width = models.IntegerField(help_text="Raster width in pixels")
    height = models.IntegerField(help_text="Raster height in pixels")
    bands = models.IntegerField(default=1, help_text="Number of bands")
    
    # Geospatial properties
    bounds_minx = models.FloatField(help_text="Minimum X coordinate")
    bounds_miny = models.FloatField(help_text="Minimum Y coordinate")
    bounds_maxx = models.FloatField(help_text="Maximum X coordinate")
    bounds_maxy = models.FloatField(help_text="Maximum Y coordinate")
    
    pixel_size_x = models.FloatField(help_text="Pixel size in X direction")
    pixel_size_y = models.FloatField(help_text="Pixel size in Y direction")
    
    # Statistics
    nodata_value = models.FloatField(null=True, blank=True)
    min_value = models.FloatField(null=True, blank=True, help_text="Minimum elevation")
    max_value = models.FloatField(null=True, blank=True, help_text="Maximum elevation")
    mean_value = models.FloatField(null=True, blank=True, help_text="Mean elevation")
    std_value = models.FloatField(null=True, blank=True, help_text="Standard deviation")
    
    # Preview/thumbnail
    thumbnail = models.ImageField(upload_to='thumbnails/%Y/%m/%d/', null=True, blank=True, help_text="DEM preview image")
    
    # Preprocessing fields
    is_preprocessed = models.BooleanField(default=False, help_text="Whether DEM has been preprocessed")
    preprocessing_method = models.CharField(max_length=20, blank=True, 
                                           choices=[('FILLED', 'Depression Filling'), 
                                                   ('BREACHED', 'Depression Breaching')],
                                           help_text="Method used for depression removal")
    preprocessing_date = models.DateTimeField(null=True, blank=True, help_text="When preprocessing was completed")
    
    # Preprocessed raster file paths (relative to MEDIA_ROOT)
    filled_dem_path = models.CharField(max_length=500, blank=True, help_text="Path to filled/breached DEM")
    flow_direction_path = models.CharField(max_length=500, blank=True, help_text="Path to D8 flow direction raster")
    flow_accumulation_path = models.CharField(max_length=500, blank=True, help_text="Path to flow accumulation raster")
    smoothed_dem_path = models.CharField(max_length=500, blank=True, help_text="Path to smoothed DEM (if applied)")
    
    # Preprocessing metadata (JSON)
    preprocessing_params = models.JSONField(default=dict, blank=True, 
                                           help_text="Parameters used for preprocessing")
    preprocessing_stats = models.JSONField(default=dict, blank=True,
                                          help_text="Statistics from preprocessing outputs")
    
    # Watershed delineation fields
    watershed_delineated = models.BooleanField(default=False, help_text="Whether watershed delineation has been performed")
    watershed_count = models.IntegerField(default=0, help_text="Number of watersheds delineated")
    stream_count = models.IntegerField(default=0, help_text="Number of stream segments extracted")
    stream_threshold = models.IntegerField(null=True, blank=True, help_text="Flow accumulation threshold used for streams")
    watershed_delineation_date = models.DateTimeField(null=True, blank=True, help_text="When watershed delineation was completed")
    
    # Site pairing fields
    site_pairing_completed = models.BooleanField(default=False, help_text="Whether site pairing has been performed")
    site_pair_count = models.IntegerField(default=0, help_text="Number of site pairs generated")
    site_pairing_date = models.DateTimeField(null=True, blank=True, help_text="When site pairing was completed")
    
    # Discharge raster (spatially-varying discharge)
    discharge_raster_path = models.CharField(max_length=500, blank=True, help_text="Path to discharge raster (m³/s per cell)")
    discharge_computed = models.BooleanField(default=False, help_text="Whether spatially-varying discharge has been computed")
    discharge_q_outlet = models.FloatField(null=True, blank=True, help_text="Reference Q at outlet used for discharge calculation (m³/s)")
    discharge_computation_date = models.DateTimeField(null=True, blank=True, help_text="When discharge was computed")
    
    class Meta:
        db_table = "raster_layers"
        verbose_name = "Raster Layer"
        verbose_name_plural = "Raster Layers"
    
    def __str__(self):
        return f"Raster: {self.dataset.name}"


class VectorLayer(models.Model):
    """Model to store shapefile/vector metadata"""
    
    GEOMETRY_TYPES = [
        ('POINT', 'Point'),
        ('LINESTRING', 'LineString'),
        ('POLYGON', 'Polygon'),
        ('MULTIPOINT', 'MultiPoint'),
        ('MULTILINESTRING', 'MultiLineString'),
        ('MULTIPOLYGON', 'MultiPolygon'),
        ('GEOMETRYCOLLECTION', 'GeometryCollection'),
    ]
    
    dataset = models.OneToOneField(Dataset, on_delete=models.CASCADE, related_name='vector_metadata')
    
    # Vector properties
    geometry_type = models.CharField(max_length=20, choices=GEOMETRY_TYPES)
    feature_count = models.IntegerField(help_text="Number of features")
    
    # Geospatial properties
    bounds_minx = models.FloatField(help_text="Minimum X coordinate")
    bounds_miny = models.FloatField(help_text="Minimum Y coordinate")
    bounds_maxx = models.FloatField(help_text="Maximum X coordinate")
    bounds_maxy = models.FloatField(help_text="Maximum Y coordinate")
    
    # Attributes
    attributes = models.JSONField(default=dict, help_text="Field names and types")
    
    # Preview/thumbnail (optional for vector data)
    thumbnail = models.ImageField(upload_to='thumbnails/%Y/%m/%d/', null=True, blank=True, help_text="Shapefile preview image")
    
    # Validation flags
    has_invalid_geometries = models.BooleanField(default=False, help_text="Whether shapefile contains invalid geometries")
    invalid_geometry_count = models.IntegerField(default=0, help_text="Number of invalid/null geometries")
    
    class Meta:
        db_table = "vector_layers"
        verbose_name = "Vector Layer"
        verbose_name_plural = "Vector Layers"
    
    def __str__(self):
        return f"Vector: {self.dataset.name}"


class TimeSeries(models.Model):
    """Model to store time series data (rainfall, discharge)"""
    
    DATA_TYPES = [
        ('RAINFALL', 'Rainfall'),
        ('DISCHARGE', 'Discharge'),
        ('OTHER', 'Other'),
    ]
    
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='timeseries')
    
    data_type = models.CharField(max_length=20, choices=DATA_TYPES)
    station_id = models.CharField(max_length=100, blank=True, help_text="Station or source ID")
    
    datetime = models.DateTimeField(help_text="Observation datetime")
    value = models.FloatField(help_text="Measured value")
    units = models.CharField(max_length=50, blank=True, help_text="Units (m³/s, mm, etc.)")
    
    class Meta:
        db_table = "timeseries"
        verbose_name = "Time Series"
        verbose_name_plural = "Time Series"
        ordering = ['datetime']
        indexes = [
            models.Index(fields=['dataset', 'datetime']),
            models.Index(fields=['station_id', 'datetime']),
        ]
    
    def __str__(self):
        return f"{self.station_id}: {self.value} {self.units} at {self.datetime}"


class HMSRun(models.Model):
    """Model to store HEC-HMS simulation run metadata"""
    
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='hms_runs')
    
    event_name = models.CharField(max_length=200, help_text="Storm event name or simulation scenario")
    return_period = models.CharField(max_length=50, blank=True, help_text="Return period (e.g., '25-year', '100-year')")
    
    # Metadata
    num_elements = models.IntegerField(default=0, help_text="Number of HMS elements (subbasins, junctions, etc.)")
    num_timesteps = models.IntegerField(default=0, help_text="Number of timesteps in simulation")
    start_time = models.DateTimeField(null=True, blank=True, help_text="Simulation start time")
    end_time = models.DateTimeField(null=True, blank=True, help_text="Simulation end time")
    
    # Statistics
    peak_discharge = models.FloatField(null=True, blank=True, help_text="Peak discharge across all elements (m³/s)")
    peak_element = models.CharField(max_length=200, blank=True, help_text="Element ID with peak discharge")
    
    created_at = models.DateTimeField(default=timezone.now)
    
    class Meta:
        db_table = "hms_runs"
        verbose_name = "HEC-HMS Run"
        verbose_name_plural = "HEC-HMS Runs"
        ordering = ['-created_at']
    
    def __str__(self):
        return f"HMS: {self.event_name} ({self.return_period})"


class WatershedPolygon(models.Model):
    """Model to store delineated watershed polygons with statistics"""
    
    raster_layer = models.ForeignKey(
        RasterLayer, 
        on_delete=models.CASCADE, 
        related_name='watersheds',
        help_text="Source DEM used for delineation"
    )
    
    watershed_id = models.IntegerField(help_text="Watershed identifier from raster")
    geometry = gis_models.PolygonField(srid=32651, help_text="Watershed boundary polygon (UTM Zone 51N)")
    
    # Area statistics
    area_m2 = models.FloatField(help_text="Watershed area in square meters")
    area_km2 = models.FloatField(help_text="Watershed area in square kilometers")
    perimeter_m = models.FloatField(help_text="Watershed perimeter in meters")
    
    # Stream statistics
    stream_length_m = models.FloatField(default=0, help_text="Total stream length within watershed (m)")
    stream_length_km = models.FloatField(default=0, help_text="Total stream length within watershed (km)")
    drainage_density = models.FloatField(default=0, help_text="Stream length per unit area (km/km²)")
    stream_count = models.IntegerField(default=0, help_text="Number of stream segments in watershed")
    
    # Morphometric indices
    compactness = models.FloatField(default=0, help_text="Gravelius compactness coefficient")
    
    # Processing metadata
    stream_threshold = models.IntegerField(help_text="Flow accumulation threshold used for stream extraction")
    delineated_at = models.DateTimeField(default=timezone.now)
    
    class Meta:
        db_table = "watershed_polygons"
        verbose_name = "Watershed Polygon"
        verbose_name_plural = "Watershed Polygons"
        ordering = ['-area_km2']
        indexes = [
            models.Index(fields=['raster_layer', 'watershed_id']),
            models.Index(fields=['area_km2']),
        ]
    
    def __str__(self):
        return f"Watershed {self.watershed_id} ({self.area_km2:.2f} km²)"


class StreamNetwork(models.Model):
    """Model to store extracted stream network as LineStrings"""
    
    raster_layer = models.ForeignKey(
        RasterLayer, 
        on_delete=models.CASCADE, 
        related_name='streams',
        help_text="Source DEM used for stream extraction"
    )
    
    geometry = gis_models.LineStringField(srid=32651, help_text="Stream segment LineString (UTM Zone 51N)")
    
    # Stream attributes
    length_m = models.FloatField(help_text="Stream segment length in meters")
    stream_order = models.IntegerField(default=1, help_text="Strahler or Shreve stream order")
    is_main_river = models.BooleanField(default=False, help_text="Part of main river stem or major reach (for practical site selection)")
    
    # Topology fields
    from_node = models.IntegerField(default=0, help_text="Upstream node ID")
    to_node = models.IntegerField(default=0, help_text="Downstream node ID")
    upstream_count = models.IntegerField(default=0, help_text="Number of upstream tributaries")
    is_confluence = models.BooleanField(default=False, help_text="Stream ends at confluence")
    is_outlet = models.BooleanField(default=False, help_text="Stream is an outlet (terminal)")
    
    # Processing metadata
    stream_threshold = models.IntegerField(help_text="Flow accumulation threshold used for extraction")
    extracted_at = models.DateTimeField(default=timezone.now)
    
    class Meta:
        db_table = "stream_network"
        verbose_name = "Stream Network"
        verbose_name_plural = "Stream Networks"
        ordering = ['-stream_order', '-length_m']
        indexes = [
            models.Index(fields=['raster_layer', 'stream_order']),
            models.Index(fields=['from_node', 'to_node']),
            models.Index(fields=['stream_order', 'is_outlet']),
            models.Index(fields=['is_main_river']),
        ]
    
    def __str__(self):
        return f"Stream (Order {self.stream_order}, {self.length_m:.2f} m)"


class StreamNode(models.Model):
    """Model to store stream network nodes (confluences, outlets, sources)"""
    
    NODE_TYPES = [
        ('source', 'Source (headwater)'),
        ('confluence', 'Confluence (junction)'),
        ('outlet', 'Outlet (terminal)'),
        ('junction', 'Junction'),
        ('divergence', 'Divergence'),
    ]
    
    raster_layer = models.ForeignKey(
        RasterLayer,
        on_delete=models.CASCADE,
        related_name='stream_nodes',
        help_text="Source DEM used for stream extraction"
    )
    
    node_id = models.IntegerField(help_text="Unique node identifier within stream network")
    geometry = gis_models.PointField(srid=32651, help_text="Node location (UTM Zone 51N)")
    node_type = models.CharField(max_length=20, choices=NODE_TYPES, help_text="Type of stream node")
    
    # Connectivity
    incoming_streams = models.IntegerField(default=0, help_text="Number of incoming stream segments")
    outgoing_streams = models.IntegerField(default=0, help_text="Number of outgoing stream segments")
    
    # Processing metadata
    stream_threshold = models.IntegerField(help_text="Flow accumulation threshold used for extraction")
    extracted_at = models.DateTimeField(default=timezone.now)
    
    class Meta:
        db_table = "stream_nodes"
        verbose_name = "Stream Node"
        verbose_name_plural = "Stream Nodes"
        ordering = ['node_id']
        unique_together = [['raster_layer', 'node_id']]
        indexes = [
            models.Index(fields=['raster_layer', 'node_type']),
            models.Index(fields=['node_id']),
        ]
    
    def __str__(self):
        return f"Node {self.node_id} ({self.node_type})"


class SitePoint(gis_models.Model):
    """Model for hydropower site points (inlets or outlets)"""
    
    SITE_TYPES = [
        ('INLET', 'Inlet'),
        ('OUTLET', 'Outlet'),
    ]
    
    # Foreign keys
    raster_layer = models.ForeignKey(
        'RasterLayer',
        on_delete=models.CASCADE,
        related_name='site_points',
        help_text="Source DEM used for elevation extraction"
    )
    stream_network = models.ForeignKey(
        'StreamNetwork',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='site_points',
        help_text="Associated stream segment (if on stream network)"
    )
    stream_node = models.ForeignKey(
        'StreamNode',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='site_points',
        help_text="Associated stream node (if at confluence/outlet)"
    )
    
    # Site identification
    site_id = models.CharField(max_length=50, unique=True, help_text="Unique site identifier")
    site_type = models.CharField(max_length=10, choices=SITE_TYPES, help_text="Inlet or Outlet")
    
    # Spatial data
    geometry = gis_models.PointField(srid=32651, help_text="Site location (UTM Zone 51N)")
    elevation = models.FloatField(help_text="Elevation at site (meters above sea level)")
    
    # Stream network attributes
    stream_order = models.IntegerField(null=True, blank=True, help_text="Strahler stream order at site")
    distance_to_stream = models.FloatField(default=0.0, help_text="Distance to nearest stream (meters)")
    
    # Processing metadata
    extracted_at = models.DateTimeField(default=timezone.now)
    
    class Meta:
        db_table = "site_points"
        verbose_name = "Site Point"
        verbose_name_plural = "Site Points"
        ordering = ['site_id']
        indexes = [
            models.Index(fields=['raster_layer', 'site_type']),
            models.Index(fields=['site_type', 'elevation']),
            models.Index(fields=['stream_order']),
        ]
    
    def __str__(self):
        return f"{self.site_id} ({self.site_type})"


class SitePair(gis_models.Model):
    """Model for inlet-outlet site pairs with hydropower potential"""
    
    # Foreign keys
    raster_layer = models.ForeignKey(
        'RasterLayer',
        on_delete=models.CASCADE,
        related_name='site_pairs',
        help_text="Source DEM used for calculations"
    )
    inlet = models.ForeignKey(
        'SitePoint',
        on_delete=models.CASCADE,
        related_name='pairs_as_inlet',
        help_text="Upstream inlet point"
    )
    outlet = models.ForeignKey(
        'SitePoint',
        on_delete=models.CASCADE,
        related_name='pairs_as_outlet',
        help_text="Downstream outlet point"
    )
    hms_run = models.ForeignKey(
        'HMSRun',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='site_pairs',
        help_text="Associated HEC-HMS simulation (discharge source)"
    )
    
    # Site pair identification
    pair_id = models.CharField(max_length=100, unique=True, help_text="Unique pair identifier (inlet_id-outlet_id)")
    
    # Spatial data
    geometry = gis_models.LineStringField(srid=32651, help_text="Line connecting inlet to outlet (UTM Zone 51N)")
    river_distance = models.FloatField(help_text="Distance along river between inlet and outlet (meters)")
    euclidean_distance = models.FloatField(help_text="Straight-line distance between inlet and outlet (meters)")
    
    # Hydropower calculations
    head = models.FloatField(help_text="Hydraulic head: H = z_inlet - z_outlet (meters)")
    discharge = models.FloatField(null=True, blank=True, help_text="Design discharge Q (m³/s)")
    efficiency = models.FloatField(default=0.7, help_text="Turbine efficiency factor η (0.6–0.85)")
    power = models.FloatField(null=True, blank=True, help_text="Power output P = ρ × g × Q × H × η (kW)")
    
    # HMS discharge metadata
    hms_element_id = models.CharField(max_length=100, blank=True, help_text="HEC-HMS element ID (junction/reach)")
    discharge_timestamp = models.DateTimeField(null=True, blank=True, help_text="Timestamp of discharge value")
    discharge_type = models.CharField(max_length=50, blank=True, help_text="Discharge type (peak/average/etc.)")
    return_period = models.CharField(max_length=50, blank=True, help_text="Return period or scenario (e.g., 100-year)")
    
    # Scoring and ranking
    score = models.FloatField(null=True, blank=True, help_text="Multi-criteria score for site ranking")
    rank = models.IntegerField(null=True, blank=True, help_text="Rank among all site pairs (1=best)")
    
    # Run-of-River Hydropower Infrastructure Components
    # These geometries represent the physical infrastructure layout
    intake_basin_geom = gis_models.PointField(srid=32651, null=True, blank=True, help_text="Water intake/weir location (at inlet)")
    settling_basin_geom = gis_models.PointField(srid=32651, null=True, blank=True, help_text="Settling basin location (near intake)")
    channel_geom = gis_models.LineStringField(srid=32651, null=True, blank=True, help_text="Water conveyance channel path")
    forebay_tank_geom = gis_models.PointField(srid=32651, null=True, blank=True, help_text="Forebay tank/surge tank location")
    penstock_geom = gis_models.LineStringField(srid=32651, null=True, blank=True, help_text="Penstock pipeline (pressure pipe)")
    powerhouse_geom = gis_models.PointField(srid=32651, null=True, blank=True, help_text="Powerhouse location (turbine & generator)")
    
    # Infrastructure dimensions (for engineering design)
    channel_length = models.FloatField(null=True, blank=True, help_text="Channel length in meters")
    penstock_length = models.FloatField(null=True, blank=True, help_text="Penstock length in meters")
    penstock_diameter = models.FloatField(null=True, blank=True, help_text="Penstock diameter in meters")
    
    # Validation flags
    meets_head_constraint = models.BooleanField(default=True, help_text="Passes minimum head constraint")
    meets_distance_constraint = models.BooleanField(default=True, help_text="Passes minimum river distance constraint")
    meets_watershed_constraint = models.BooleanField(default=True, help_text="Both inlet and outlet are within watershed boundary")
    meets_land_constraint = models.BooleanField(default=True, help_text="Passes land proximity constraint")
    is_feasible = models.BooleanField(default=True, help_text="Passes all constraints")
    
    # Processing metadata
    created_at = models.DateTimeField(default=timezone.now)
    
    # Diversion Zone reference (Step 2 of objective - similar elevation search)
    diversion_zone = models.OneToOneField(
        'DiversionZone',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='site_pair',
        help_text="Associated diversion zone (similar elevation area around inlet)"
    )
    
    class Meta:
        db_table = "site_pairs"
        verbose_name = "Site Pair"
        verbose_name_plural = "Site Pairs"
        ordering = ['-power', '-head']
        indexes = [
            models.Index(fields=['raster_layer', 'is_feasible']),
            models.Index(fields=['power', 'head']),
            models.Index(fields=['is_feasible', '-power']),
            models.Index(fields=['rank']),
        ]
    
    def __str__(self):
        return f"{self.pair_id} (H={self.head:.1f}m, P={self.power:.1f}kW)" if self.power else f"{self.pair_id} (H={self.head:.1f}m)"
    
    def calculate_power(self, rho=1000, g=9.81):
        """
        Calculate hydropower potential using the equation:
        P = ρ × g × Q × H × η
        
        Args:
            rho: Water density (kg/m³), default 1000
            g: Gravitational acceleration (m/s²), default 9.81
        
        Returns:
            Power in kW
        """
        if self.discharge and self.head and self.efficiency:
            # P = ρ × g × Q × H × η (Watts)
            power_watts = rho * g * self.discharge * self.head * self.efficiency
            # Convert to kW
            self.power = power_watts / 1000.0
            return self.power
        return None


class DiversionZone(gis_models.Model):
    """
    Model for water diversion/storage zones around inlet points.
    
    This implements Step 2 of the hydropower site identification workflow:
    - For each inlet point, search the DEM in a local neighborhood
    - Find DEM cells with elevation similar to the inlet's elevation
    - These zones represent areas where water could be diverted/stored at that head
    
    The result is a polygon (or multipolygon) representing the "similar elevation"
    area around each inlet, useful for identifying:
    - Potential intake/weir locations
    - Natural storage/reservoir areas
    - Water diversion corridors at constant head
    """
    
    # Link to source raster layer (DEM)
    raster_layer = models.ForeignKey(
        'RasterLayer',
        on_delete=models.CASCADE,
        related_name='diversion_zones',
        help_text="Source DEM used for elevation analysis"
    )
    
    # Link to inlet point
    inlet_point = models.ForeignKey(
        'SitePoint',
        on_delete=models.CASCADE,
        related_name='diversion_zones',
        help_text="Inlet point around which the zone was computed"
    )
    
    # Zone identification
    zone_id = models.CharField(max_length=100, unique=True, help_text="Unique zone identifier (e.g., 'DZ_INLET_001')")
    
    # Geometry - the polygon representing similar elevation area
    geometry = gis_models.MultiPolygonField(srid=32651, help_text="Diversion zone polygon(s) (UTM Zone 51N)")
    centroid = gis_models.PointField(srid=32651, null=True, blank=True, help_text="Zone centroid")
    
    # Elevation parameters
    reference_elevation = models.FloatField(help_text="Reference elevation of inlet point (m)")
    elevation_tolerance = models.FloatField(default=2.0, help_text="Elevation tolerance used for search (m)")
    min_elevation = models.FloatField(help_text="Minimum elevation within zone (m)")
    max_elevation = models.FloatField(help_text="Maximum elevation within zone (m)")
    mean_elevation = models.FloatField(help_text="Mean elevation within zone (m)")
    
    # Search parameters
    search_radius = models.FloatField(default=500.0, help_text="Search radius around inlet point (m)")
    
    # Zone statistics
    area_m2 = models.FloatField(help_text="Zone area in square meters")
    area_hectares = models.FloatField(help_text="Zone area in hectares")
    perimeter_m = models.FloatField(help_text="Zone perimeter in meters")
    pixel_count = models.IntegerField(help_text="Number of DEM pixels in zone")
    
    # Slope statistics within zone
    mean_slope = models.FloatField(null=True, blank=True, help_text="Mean slope within zone (degrees)")
    max_slope = models.FloatField(null=True, blank=True, help_text="Maximum slope within zone (degrees)")
    
    # Suitability assessment
    is_contiguous = models.BooleanField(default=True, help_text="Whether zone is a single contiguous area")
    fragment_count = models.IntegerField(default=1, help_text="Number of separate fragments")
    largest_fragment_area_m2 = models.FloatField(null=True, blank=True, help_text="Area of largest contiguous fragment")
    
    # Proximity to stream
    distance_to_stream = models.FloatField(null=True, blank=True, help_text="Distance from zone centroid to nearest stream (m)")
    
    # Classification
    suitability_score = models.FloatField(null=True, blank=True, help_text="Suitability score for water diversion (0-100)")
    
    # Processing metadata
    created_at = models.DateTimeField(default=timezone.now)
    processing_time_seconds = models.FloatField(null=True, blank=True, help_text="Time to compute this zone")
    
    class Meta:
        db_table = "diversion_zones"
        verbose_name = "Diversion Zone"
        verbose_name_plural = "Diversion Zones"
        ordering = ['-area_m2']
        indexes = [
            models.Index(fields=['raster_layer', 'inlet_point']),
            models.Index(fields=['reference_elevation']),
            models.Index(fields=['area_m2']),
            models.Index(fields=['suitability_score']),
        ]
    
    def __str__(self):
        return f"{self.zone_id} (Elev: {self.reference_elevation:.1f}m, Area: {self.area_hectares:.2f} ha)"


class WeirCandidate(gis_models.Model):
    """
    Model for candidate weir/diversion locations near inlet points.
    
    This implements Objective 3 (Weir Identification):
    - For each high-ranked inlet point, search the DEM for candidate weir locations
    - Candidates must be within a search radius and elevation tolerance
    - Directional constraint: candidates must be toward the outlet (hydraulically feasible)
    - Each candidate represents a potential water intake/weir structure location
    
    The weir search refines IO pairs into concrete infrastructure layouts:
    - Water abstracted at weir → routed through headrace/penstock → discharged at outlet
    - Ensures hydraulic compatibility with the original IO pair's Q, H, and P calculations
    """
    
    # Link to source raster layer (DEM)
    raster_layer = models.ForeignKey(
        'RasterLayer',
        on_delete=models.CASCADE,
        related_name='weir_candidates',
        help_text="Source DEM used for weir search"
    )
    
    # Link to inlet point (from top-ranked IO pairs)
    inlet_point = models.ForeignKey(
        'SitePoint',
        on_delete=models.CASCADE,
        related_name='weir_candidates',
        help_text="Inlet point around which weir candidates were searched"
    )
    
    # Link to associated site pairs (an inlet may belong to multiple high-ranked pairs)
    site_pairs = models.ManyToManyField(
        'SitePair',
        related_name='weir_candidates',
        help_text="High-ranked IO pairs that use this inlet"
    )
    
    # Weir candidate identification
    candidate_id = models.CharField(max_length=100, help_text="Unique candidate identifier (e.g., 'WEIR_IN_97_79_001')")
    
    # Geometry - the point representing the candidate weir location
    geometry = gis_models.PointField(srid=32651, help_text="Weir candidate location (UTM Zone 51N)")
    
    # Elevation at weir candidate
    elevation = models.FloatField(help_text="DEM elevation at weir candidate (m)")
    
    # Relationship to inlet
    distance_from_inlet = models.FloatField(help_text="Distance from inlet point to weir candidate (m)")
    elevation_difference = models.FloatField(help_text="Elevation difference: candidate - inlet (m)")
    
    # Inlet metadata (cached for performance)
    inlet_elevation = models.FloatField(help_text="Inlet elevation (m)")
    inlet_node_id = models.CharField(max_length=100, help_text="Inlet node ID")
    
    # Associated outlets metadata (for directional constraint validation)
    outlet_count = models.IntegerField(default=0, help_text="Number of outlets associated with this inlet")
    outlet_node_ids = models.TextField(blank=True, help_text="Comma-separated outlet node IDs")
    
    # Directional constraint metadata
    is_toward_outlet = models.BooleanField(default=True, help_text="Whether candidate is in direction toward outlet(s)")
    angle_to_outlet_deg = models.FloatField(null=True, blank=True, help_text="Angle from inlet to candidate toward outlet (degrees)")
    
    # Associated pairs metadata
    pair_count = models.IntegerField(default=0, help_text="Number of high-ranked pairs using this inlet")
    pair_ids_list = models.TextField(blank=True, help_text="Comma-separated pair IDs")
    
    # Ranking/scoring
    suitability_score = models.FloatField(null=True, blank=True, help_text="Suitability score for weir location (0-100)")
    rank_within_inlet = models.IntegerField(null=True, blank=True, help_text="Rank among candidates for this inlet (1=best)")
    
    # Search parameters used
    search_radius = models.FloatField(default=500.0, help_text="Search radius used (m)")
    elevation_tolerance = models.FloatField(default=20.0, help_text="Elevation tolerance used (m)")
    min_distance = models.FloatField(default=100.0, help_text="Minimum distance from inlet (m)")
    cone_angle_deg = models.FloatField(default=90.0, help_text="Directional cone half-angle (degrees)")
    
    # Processing metadata
    created_at = models.DateTimeField(default=timezone.now)
    
    class Meta:
        db_table = "weir_candidates"
        verbose_name = "Weir Candidate"
        verbose_name_plural = "Weir Candidates"
        ordering = ['inlet_point', 'rank_within_inlet']
        indexes = [
            models.Index(fields=['raster_layer', 'inlet_point']),
            models.Index(fields=['inlet_node_id']),
            models.Index(fields=['suitability_score']),
            models.Index(fields=['distance_from_inlet']),
            models.Index(fields=['rank_within_inlet']),
        ]
    
    def __str__(self):
        return f"{self.candidate_id} (Elev: {self.elevation:.1f}m, Dist: {self.distance_from_inlet:.0f}m)"


class ProcessingRun(models.Model):
    """Model to track processing job execution and status"""
    
    JOB_STATUS = [
        ('QUEUED', 'Queued'),
        ('RUNNING', 'Running'),
        ('SUCCEEDED', 'Succeeded'),
        ('FAILED', 'Failed'),
        ('CANCELLED', 'Cancelled'),
    ]
    
    JOB_TYPES = [
        ('DEM_PREPROCESSING', 'DEM Preprocessing'),
        ('WATERSHED_DELINEATION', 'Watershed Delineation'),
        ('STREAM_EXTRACTION', 'Stream Network Extraction'),
        ('SITE_PAIRING', 'Inlet-Outlet Site Pairing'),
        ('DISCHARGE_ASSOCIATION', 'Discharge Association'),
        ('POWER_CALCULATION', 'Power Calculation'),
        ('EXPORT_CSV', 'CSV Export'),
        ('EXPORT_GEOJSON', 'GeoJSON Export'),
        ('EXPORT_PDF', 'PDF Export'),
    ]
    
    # Job identification
    job_id = models.CharField(max_length=100, unique=True, help_text="Unique job identifier (UUID or custom)")
    job_type = models.CharField(max_length=50, choices=JOB_TYPES, help_text="Type of processing job")
    status = models.CharField(max_length=20, choices=JOB_STATUS, default='QUEUED', help_text="Current job status")
    
    # Input datasets
    input_datasets = models.ManyToManyField(
        'Dataset',
        related_name='processing_runs',
        blank=True,
        help_text="Input datasets used for this processing run"
    )
    
    # Job parameters (stored as JSON)
    parameters = models.JSONField(
        default=dict,
        blank=True,
        help_text="Processing parameters and configuration (JSON)"
    )
    
    # Output references
    output_raster_layers = models.ManyToManyField(
        'RasterLayer',
        related_name='processing_runs',
        blank=True,
        help_text="Generated raster layers"
    )
    output_watersheds = models.ManyToManyField(
        'WatershedPolygon',
        related_name='processing_runs',
        blank=True,
        help_text="Generated watershed polygons"
    )
    output_streams = models.ManyToManyField(
        'StreamNetwork',
        related_name='processing_runs',
        blank=True,
        help_text="Generated stream segments"
    )
    output_site_pairs = models.ManyToManyField(
        'SitePair',
        related_name='processing_runs',
        blank=True,
        help_text="Generated site pairs"
    )
    
    # Execution tracking
    queued_at = models.DateTimeField(default=timezone.now, help_text="Time job was queued")
    started_at = models.DateTimeField(null=True, blank=True, help_text="Time job started execution")
    completed_at = models.DateTimeField(null=True, blank=True, help_text="Time job completed")
    duration_seconds = models.FloatField(null=True, blank=True, help_text="Job execution duration in seconds")
    
    # Progress tracking
    progress_percent = models.IntegerField(default=0, help_text="Job progress (0-100%)")
    progress_message = models.CharField(max_length=500, blank=True, help_text="Current progress message")
    
    # Error handling
    error_message = models.TextField(blank=True, help_text="Error message if job failed")
    error_traceback = models.TextField(blank=True, help_text="Full error traceback for debugging")
    warning_messages = models.JSONField(default=list, blank=True, help_text="List of warning messages")
    
    # Results summary
    results_summary = models.JSONField(
        default=dict,
        blank=True,
        help_text="Summary of processing results (counts, statistics, etc.)"
    )
    
    # Celery task ID (if using Celery)
    celery_task_id = models.CharField(max_length=100, blank=True, help_text="Celery task ID for async execution")
    
    # Data lineage
    parent_job = models.ForeignKey(
        'self',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='child_jobs',
        help_text="Parent job if this is part of a processing pipeline"
    )
    
    class Meta:
        db_table = "processing_runs"
        verbose_name = "Processing Run"
        verbose_name_plural = "Processing Runs"
        ordering = ['-queued_at']
        indexes = [
            models.Index(fields=['job_id']),
            models.Index(fields=['status', '-queued_at']),
            models.Index(fields=['job_type', 'status']),
            models.Index(fields=['-queued_at']),
        ]
    
    def __str__(self):
        return f"{self.job_type} ({self.job_id}) - {self.status}"
    
    def start(self):
        """Mark job as started"""
        self.status = 'RUNNING'
        self.started_at = timezone.now()
        self.save()
    
    def succeed(self, results_summary: dict = None):
        """Mark job as succeeded"""
        self.status = 'SUCCEEDED'
        self.completed_at = timezone.now()
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()
        self.progress_percent = 100
        if results_summary:
            self.results_summary = results_summary
        self.save()
    
    def fail(self, error_message: str, error_traceback: str = ''):
        """Mark job as failed"""
        self.status = 'FAILED'
        self.completed_at = timezone.now()
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()
        self.error_message = error_message
        self.error_traceback = error_traceback
        self.save()
    
    def cancel(self):
        """Mark job as cancelled"""
        self.status = 'CANCELLED'
        self.completed_at = timezone.now()
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()
        self.save()
    
    def update_progress(self, percent: int, message: str = ''):
        """Update job progress"""
        self.progress_percent = min(100, max(0, percent))
        self.progress_message = message
        self.save()
    
    def add_warning(self, warning_message: str):
        """Add a warning message"""
        if not isinstance(self.warning_messages, list):
            self.warning_messages = []
        self.warning_messages.append({
            'timestamp': timezone.now().isoformat(),
            'message': warning_message
        })
        self.save()
    
    def is_finished(self) -> bool:
        """Check if job has finished (succeeded, failed, or cancelled)"""
        return self.status in ['SUCCEEDED', 'FAILED', 'CANCELLED']
    
    def is_running(self) -> bool:
        """Check if job is currently running"""
        return self.status == 'RUNNING'
    
    def can_cancel(self) -> bool:
        """Check if job can be cancelled"""
        return self.status in ['QUEUED', 'RUNNING']


class ProcessingLayer(models.Model):
    """
    Model to track intermediate processing outputs for map visualization.
    
    This enables HEC-HMS-style layer visualization where users can see outputs
    from different processing stages (DEM, Flow Direction, Flow Accumulation,
    Streams, Watersheds, Discharge, etc.)
    """
    
    LAYER_TYPES = [
        ('DEM', 'Digital Elevation Model'),
        ('FILLED_DEM', 'Filled DEM (Depressions Removed)'),
        ('FLOW_DIRECTION', 'Flow Direction (D8)'),
        ('FLOW_ACCUMULATION', 'Flow Accumulation'),
        ('STREAM_RASTER', 'Stream Raster'),
        ('STREAM_VECTOR', 'Stream Network (Vector)'),
        ('WATERSHED_RASTER', 'Watershed Raster'),
        ('WATERSHED_VECTOR', 'Watershed Boundaries (Vector)'),
        ('DISCHARGE_RASTER', 'Discharge (m³/s)'),
        ('SLOPE', 'Slope (Degrees)'),
        ('ASPECT', 'Aspect'),
        ('SUBBASIN', 'Subbasins'),
        ('SITE_POINTS', 'Site Points (Inlets/Outlets)'),
        ('SITE_PAIRS', 'Site Pairs'),
        ('INFRASTRUCTURE', 'Infrastructure Components'),
        ('CUSTOM', 'Custom Layer'),
    ]
    
    DATA_FORMATS = [
        ('RASTER', 'Raster (GeoTIFF)'),
        ('VECTOR', 'Vector (GeoJSON/Shapefile)'),
        ('POINT', 'Point Features'),
        ('LINE', 'Line Features'),
        ('POLYGON', 'Polygon Features'),
    ]
    
    # Link to source raster layer (DEM)
    raster_layer = models.ForeignKey(
        RasterLayer,
        on_delete=models.CASCADE,
        related_name='processing_layers',
        help_text="Source DEM/RasterLayer"
    )
    
    # Layer identification
    name = models.CharField(max_length=200, help_text="Layer display name")
    layer_type = models.CharField(max_length=30, choices=LAYER_TYPES, help_text="Type of processing layer")
    data_format = models.CharField(max_length=20, choices=DATA_FORMATS, default='RASTER', help_text="Data format")
    
    # File path (relative to MEDIA_ROOT)
    file_path = models.CharField(max_length=500, help_text="Path to layer file")
    
    # Visualization settings (JSON)
    style = models.JSONField(default=dict, blank=True, help_text="Layer style configuration (colors, opacity, etc.)")
    
    # Metadata
    description = models.TextField(blank=True, help_text="Layer description")
    processing_step = models.IntegerField(default=0, help_text="Processing step number (for ordering)")
    is_visible = models.BooleanField(default=True, help_text="Whether layer is visible by default")
    is_base_layer = models.BooleanField(default=False, help_text="Whether this is a base/background layer")
    
    # Statistics (JSON) - for rasters: min, max, mean, etc.
    statistics = models.JSONField(default=dict, blank=True, help_text="Layer statistics")
    
    # Bounds
    bounds_minx = models.FloatField(null=True, blank=True)
    bounds_miny = models.FloatField(null=True, blank=True)
    bounds_maxx = models.FloatField(null=True, blank=True)
    bounds_maxy = models.FloatField(null=True, blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = "processing_layers"
        verbose_name = "Processing Layer"
        verbose_name_plural = "Processing Layers"
        ordering = ['raster_layer', 'processing_step', 'name']
        unique_together = [['raster_layer', 'layer_type']]
    
    def __str__(self):
        return f"{self.name} ({self.get_layer_type_display()})"
    
    def get_absolute_path(self):
        """Get the absolute file path"""
        from django.conf import settings
        from pathlib import Path
        return Path(settings.MEDIA_ROOT) / self.file_path


# Uncomment this model when PostGIS is set up:
"""
class TestLocation(models.Model):
    '''Test model to verify PostGIS functionality'''
    name = models.CharField(max_length=200)
    location = gis_models.PointField(srid=32651)  # UTM Zone 51N
    created_at = models.DateTimeField(default=timezone.now)
    
    class Meta:
        db_table = "test_locations"
        verbose_name = "Test Location"
        verbose_name_plural = "Test Locations"
    
    def __str__(self):
        return self.name
"""

