from django.contrib import admin
from .models import Dataset, RasterLayer, VectorLayer, TimeSeries, WatershedPolygon, StreamNetwork, StreamNode, SitePoint, SitePair, ProcessingRun


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ['name', 'dataset_type', 'validation_status', 'file_size', 'crs', 'uploaded_at']
    list_filter = ['dataset_type', 'validation_status', 'uploaded_at']
    search_fields = ['name', 'original_filename', 'tags', 'description']
    readonly_fields = ['file_checksum', 'file_size', 'uploaded_at', 'validated_at']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'dataset_type', 'file', 'description', 'tags')
        }),
        ('File Metadata', {
            'fields': ('original_filename', 'file_size', 'file_checksum', 'mime_type')
        }),
        ('Validation', {
            'fields': ('validation_status', 'validation_message', 'validated_at')
        }),
        ('Geospatial', {
            'fields': ('crs',)
        }),
        ('Timestamps', {
            'fields': ('uploaded_at',)
        }),
    )


@admin.register(RasterLayer)
class RasterLayerAdmin(admin.ModelAdmin):
    list_display = ['dataset', 'width', 'height', 'bands', 'min_value', 'max_value', 'is_preprocessed', 'preprocessing_method']
    list_filter = ['is_preprocessed', 'preprocessing_method']
    search_fields = ['dataset__name']
    readonly_fields = ['dataset', 'preprocessing_date']
    
    fieldsets = (
        ('Dataset', {
            'fields': ('dataset',)
        }),
        ('Raster Properties', {
            'fields': ('width', 'height', 'bands', 'pixel_size_x', 'pixel_size_y')
        }),
        ('Bounds', {
            'fields': ('bounds_minx', 'bounds_miny', 'bounds_maxx', 'bounds_maxy')
        }),
        ('Statistics', {
            'fields': ('nodata_value', 'min_value', 'max_value', 'mean_value', 'std_value')
        }),
        ('Preprocessing', {
            'fields': ('is_preprocessed', 'preprocessing_method', 'preprocessing_date',
                      'filled_dem_path', 'flow_direction_path', 'flow_accumulation_path', 
                      'smoothed_dem_path', 'preprocessing_params', 'preprocessing_stats'),
            'classes': ('collapse',)
        }),
        ('Watershed Delineation', {
            'fields': ('watershed_delineated', 'watershed_count', 'stream_count', 
                      'stream_threshold', 'watershed_delineation_date'),
            'classes': ('collapse',)
        }),
        ('Thumbnail', {
            'fields': ('thumbnail',)
        }),
    )


@admin.register(VectorLayer)
class VectorLayerAdmin(admin.ModelAdmin):
    list_display = ['dataset', 'geometry_type', 'feature_count']
    search_fields = ['dataset__name']
    readonly_fields = ['dataset']


@admin.register(TimeSeries)
class TimeSeriesAdmin(admin.ModelAdmin):
    list_display = ['dataset', 'data_type', 'station_id', 'datetime', 'value', 'units']
    list_filter = ['data_type', 'datetime']
    search_fields = ['dataset__name', 'station_id']
    date_hierarchy = 'datetime'


@admin.register(WatershedPolygon)
class WatershedPolygonAdmin(admin.ModelAdmin):
    list_display = ['watershed_id', 'raster_layer', 'area_km2', 'perimeter_m', 'stream_count', 'drainage_density', 'delineated_at']
    list_filter = ['delineated_at', 'stream_threshold']
    search_fields = ['watershed_id', 'raster_layer__dataset__name']
    readonly_fields = ['raster_layer', 'watershed_id', 'geometry', 'area_m2', 'area_km2', 
                      'perimeter_m', 'stream_length_m', 'stream_length_km', 'drainage_density',
                      'stream_count', 'compactness', 'delineated_at']
    
    fieldsets = (
        ('Watershed Information', {
            'fields': ('raster_layer', 'watershed_id', 'geometry')
        }),
        ('Area Statistics', {
            'fields': ('area_m2', 'area_km2', 'perimeter_m')
        }),
        ('Stream Statistics', {
            'fields': ('stream_length_m', 'stream_length_km', 'drainage_density', 'stream_count')
        }),
        ('Morphometric Indices', {
            'fields': ('compactness',)
        }),
        ('Processing Metadata', {
            'fields': ('stream_threshold', 'delineated_at')
        }),
    )


@admin.register(StreamNetwork)
class StreamNetworkAdmin(admin.ModelAdmin):
    list_display = ['id', 'raster_layer', 'stream_order', 'length_m', 'from_node', 'to_node', 'is_outlet', 'extracted_at']
    list_filter = ['stream_order', 'is_outlet', 'is_confluence', 'extracted_at', 'stream_threshold']
    search_fields = ['raster_layer__dataset__name', 'from_node', 'to_node']
    readonly_fields = ['raster_layer', 'geometry', 'length_m', 'stream_order', 'from_node', 'to_node', 
                      'upstream_count', 'is_confluence', 'is_outlet', 'extracted_at']
    
    fieldsets = (
        ('Stream Information', {
            'fields': ('raster_layer', 'geometry', 'stream_order', 'length_m')
        }),
        ('Topology', {
            'fields': ('from_node', 'to_node', 'upstream_count', 'is_confluence', 'is_outlet')
        }),
        ('Processing Metadata', {
            'fields': ('stream_threshold', 'extracted_at')
        }),
    )


@admin.register(StreamNode)
class StreamNodeAdmin(admin.ModelAdmin):
    list_display = ['node_id', 'raster_layer', 'node_type', 'incoming_streams', 'outgoing_streams', 'extracted_at']
    list_filter = ['node_type', 'extracted_at', 'stream_threshold']
    search_fields = ['node_id', 'raster_layer__dataset__name']
    readonly_fields = ['raster_layer', 'node_id', 'geometry', 'incoming_streams', 'outgoing_streams', 'extracted_at']
    
    fieldsets = (
        ('Node Information', {
            'fields': ('raster_layer', 'node_id', 'geometry', 'node_type')
        }),
        ('Connectivity', {
            'fields': ('incoming_streams', 'outgoing_streams')
        }),
        ('Processing Metadata', {
            'fields': ('stream_threshold', 'extracted_at')
        }),
    )


@admin.register(SitePoint)
class SitePointAdmin(admin.ModelAdmin):
    list_display = ['site_id', 'site_type', 'elevation', 'stream_order', 'raster_layer', 'extracted_at']
    list_filter = ['site_type', 'stream_order', 'extracted_at']
    search_fields = ['site_id', 'raster_layer__dataset__name']
    readonly_fields = ['site_id', 'geometry', 'elevation', 'stream_order', 'distance_to_stream', 'extracted_at']
    
    fieldsets = (
        ('Site Information', {
            'fields': ('site_id', 'site_type', 'geometry', 'elevation')
        }),
        ('Stream Network', {
            'fields': ('stream_network', 'stream_node', 'stream_order', 'distance_to_stream')
        }),
        ('Source Data', {
            'fields': ('raster_layer',)
        }),
        ('Processing Metadata', {
            'fields': ('extracted_at',)
        }),
    )


@admin.register(SitePair)
class SitePairAdmin(admin.ModelAdmin):
    list_display = ['pair_id', 'head', 'discharge', 'power', 'score', 'rank', 'is_feasible', 'created_at']
    list_filter = ['is_feasible', 'meets_head_constraint', 'meets_distance_constraint', 
                   'meets_land_constraint', 'return_period', 'created_at']
    search_fields = ['pair_id', 'hms_element_id', 'raster_layer__dataset__name']
    readonly_fields = ['pair_id', 'geometry', 'river_distance', 'euclidean_distance', 
                      'head', 'power', 'score', 'rank', 'created_at']
    
    fieldsets = (
        ('Site Pair Information', {
            'fields': ('pair_id', 'geometry', 'inlet', 'outlet', 'raster_layer')
        }),
        ('Distances', {
            'fields': ('river_distance', 'euclidean_distance')
        }),
        ('Hydropower Calculations', {
            'fields': ('head', 'discharge', 'efficiency', 'power')
        }),
        ('HEC-HMS Data', {
            'fields': ('hms_run', 'hms_element_id', 'discharge_timestamp', 
                      'discharge_type', 'return_period'),
            'classes': ('collapse',)
        }),
        ('Scoring and Ranking', {
            'fields': ('score', 'rank')
        }),
        ('Constraints', {
            'fields': ('meets_head_constraint', 'meets_distance_constraint', 
                      'meets_land_constraint', 'is_feasible')
        }),
        ('Processing Metadata', {
            'fields': ('created_at',)
        }),
    )
    
    def get_readonly_fields(self, request, obj=None):
        """Make most fields readonly after creation"""
        if obj:  # Editing existing object
            return self.readonly_fields + ['inlet', 'outlet', 'raster_layer', 'hms_run']
        return self.readonly_fields


@admin.register(ProcessingRun)
class ProcessingRunAdmin(admin.ModelAdmin):
    list_display = ['job_id', 'job_type', 'status', 'progress_percent', 'queued_at', 'duration_seconds']
    list_filter = ['status', 'job_type', 'queued_at']
    search_fields = ['job_id', 'error_message', 'progress_message']
    readonly_fields = ['job_id', 'queued_at', 'started_at', 'completed_at', 'duration_seconds',
                      'error_traceback', 'celery_task_id']
    
    fieldsets = (
        ('Job Information', {
            'fields': ('job_id', 'job_type', 'status', 'celery_task_id')
        }),
        ('Progress', {
            'fields': ('progress_percent', 'progress_message')
        }),
        ('Timing', {
            'fields': ('queued_at', 'started_at', 'completed_at', 'duration_seconds')
        }),
        ('Input Datasets', {
            'fields': ('input_datasets',),
            'classes': ('collapse',)
        }),
        ('Parameters', {
            'fields': ('parameters',),
            'classes': ('collapse',)
        }),
        ('Outputs', {
            'fields': ('output_raster_layers', 'output_watersheds', 'output_streams', 'output_site_pairs'),
            'classes': ('collapse',)
        }),
        ('Results', {
            'fields': ('results_summary',),
            'classes': ('collapse',)
        }),
        ('Errors and Warnings', {
            'fields': ('error_message', 'error_traceback', 'warning_messages'),
            'classes': ('collapse',)
        }),
        ('Lineage', {
            'fields': ('parent_job',),
            'classes': ('collapse',)
        }),
    )
    
    def get_readonly_fields(self, request, obj=None):
        """Make most fields readonly after creation"""
        if obj:  # Editing existing object
            return self.readonly_fields + ['job_type', 'input_datasets', 'parent_job']
        return self.readonly_fields
    
    actions = ['cancel_jobs']
    
    def cancel_jobs(self, request, queryset):
        """Admin action to cancel selected jobs"""
        cancelled_count = 0
        for job in queryset:
            if job.can_cancel():
                job.cancel()
                cancelled_count += 1
        
        self.message_user(request, f"Cancelled {cancelled_count} job(s).")
    cancel_jobs.short_description = "Cancel selected jobs"


