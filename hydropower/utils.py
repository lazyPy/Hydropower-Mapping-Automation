"""
Utilities for file upload handling, validation, and sanitization
"""
import os
import re
import uuid
import mimetypes
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
from django.core.files.uploadedfile import UploadedFile
from django.core.files.base import ContentFile
from django.conf import settings
from PIL import Image
import io

# Import rasterio for DEM processing
try:
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.plot import show
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    print("Warning: rasterio not available. DEM processing will be disabled.")


# Allowed MIME types for each dataset type
ALLOWED_MIME_TYPES = {
    'DEM': [
        'image/tiff',
        'image/tif',
        'application/octet-stream',  # Some TIFFs report as binary
    ],
    'SHAPEFILE': [
        'application/x-qgis',
        'application/octet-stream',
        'application/x-esri-shape',
        'application/zip',  # For zipped shapefiles
        'application/x-zip-compressed',
    ],
    'TIMESERIES': [
        'text/csv',
        'application/csv',
        'text/plain',
        'application/vnd.ms-excel',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    ],
    'HMS': [
        'text/csv',
        'application/csv',
        'text/plain',
    ],
}


# Allowed file extensions
ALLOWED_EXTENSIONS = {
    'DEM': ['.tif', '.tiff', '.img'],
    'SHAPEFILE': ['.shp', '.shx', '.dbf', '.prj', '.zip'],
    'TIMESERIES': ['.csv', '.xlsx', '.xls'],
    'HMS': ['.csv', '.txt'],
}


def sanitize_filename(filename: str) -> str:
    """
    Sanitize uploaded filename to prevent directory traversal and other security issues.
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename with special characters removed
    """
    # Get file extension
    name, ext = os.path.splitext(filename)
    
    # Remove directory components
    name = os.path.basename(name)
    
    # Replace spaces with underscores
    name = name.replace(' ', '_')
    
    # Remove all non-alphanumeric characters except underscores and hyphens
    name = re.sub(r'[^\w\-]', '', name)
    
    # Limit length
    max_length = 100
    if len(name) > max_length:
        name = name[:max_length]
    
    # Ensure name is not empty
    if not name:
        name = 'file'
    
    return f"{name}{ext.lower()}"


def generate_unique_filename(original_filename: str, dataset_type: str) -> str:
    """
    Generate unique filename with UUID prefix to prevent collisions.
    
    Args:
        original_filename: Original uploaded filename
        dataset_type: Type of dataset (DEM, SHAPEFILE, etc.)
        
    Returns:
        Unique filename with UUID prefix
    """
    # Sanitize the original filename
    safe_name = sanitize_filename(original_filename)
    
    # Extract extension
    name, ext = os.path.splitext(safe_name)
    
    # Generate UUID
    unique_id = uuid.uuid4().hex[:12]
    
    # Construct unique filename
    return f"{dataset_type.lower()}_{unique_id}_{name}{ext}"


def validate_file_type(uploaded_file: UploadedFile, dataset_type: str) -> Tuple[bool, str]:
    """
    Validate uploaded file type against allowed MIME types and extensions.
    
    Args:
        uploaded_file: Django UploadedFile instance
        dataset_type: Type of dataset
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Get file extension
    filename = uploaded_file.name
    _, ext = os.path.splitext(filename.lower())
    
    # Check extension
    allowed_exts = ALLOWED_EXTENSIONS.get(dataset_type, [])
    if ext not in allowed_exts:
        return False, f"Invalid file extension '{ext}'. Allowed: {', '.join(allowed_exts)}"
    
    # Check MIME type
    allowed_mimes = ALLOWED_MIME_TYPES.get(dataset_type, [])
    content_type = uploaded_file.content_type
    
    if content_type not in allowed_mimes:
        # Try to guess MIME type from extension as fallback
        guessed_type, _ = mimetypes.guess_type(filename)
        if guessed_type and guessed_type in allowed_mimes:
            return True, ""
        
        return False, f"Invalid file type '{content_type}'. Allowed: {', '.join(allowed_mimes)}"
    
    return True, ""


def validate_file_size(uploaded_file: UploadedFile, max_size_gb: float = 2.0) -> Tuple[bool, str]:
    """
    Validate file size.
    
    Args:
        uploaded_file: Django UploadedFile instance
        max_size_gb: Maximum allowed size in GB
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
    
    if uploaded_file.size > max_size_bytes:
        size_gb = uploaded_file.size / (1024 ** 3)
        return False, f"File size ({size_gb:.2f} GB) exceeds maximum allowed size ({max_size_gb} GB)"
    
    return True, ""


def validate_shapefile_bundle(files: list) -> Tuple[bool, str]:
    """
    Validate that a shapefile bundle contains all required components.
    
    Args:
        files: List of uploaded files
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_extensions = {'.shp', '.shx', '.dbf'}
    optional_extensions = {'.prj', '.sbn', '.sbx', '.cpg', '.qpj'}
    
    # Get extensions from uploaded files
    uploaded_extensions = {os.path.splitext(f.name.lower())[1] for f in files}
    
    # Check required files
    missing = required_extensions - uploaded_extensions
    if missing:
        return False, f"Missing required shapefile components: {', '.join(missing)}"
    
    # Check for unexpected files
    valid_extensions = required_extensions | optional_extensions
    unexpected = uploaded_extensions - valid_extensions
    if unexpected:
        return False, f"Unexpected files in shapefile bundle: {', '.join(unexpected)}"
    
    return True, ""


def create_upload_directory(dataset_type: str) -> Path:
    """
    Create organized directory structure for uploaded files.
    
    Args:
        dataset_type: Type of dataset
        
    Returns:
        Path to the created directory
    """
    from datetime import datetime
    
    # Create directory structure: media/uploads/YYYY/MM/DD/dataset_type/
    today = datetime.now()
    upload_path = Path(settings.MEDIA_ROOT) / 'uploads' / str(today.year) / \
                  f"{today.month:02d}" / f"{today.day:02d}" / dataset_type.lower()
    
    # Create directories if they don't exist
    upload_path.mkdir(parents=True, exist_ok=True)
    
    return upload_path


def scan_file_for_viruses(file_path: str) -> Tuple[bool, str]:
    """
    Scan uploaded file for viruses using ClamAV (if available).
    
    Args:
        file_path: Path to uploaded file
        
    Returns:
        Tuple of (is_clean, message)
    
    Note:
        This is a placeholder. Actual implementation requires ClamAV installation.
    """
    # TODO: Implement virus scanning with ClamAV
    # For now, return True (no scan performed)
    return True, "Virus scan not implemented (ClamAV not configured)"


def get_file_metadata(uploaded_file: UploadedFile) -> dict:
    """
    Extract metadata from uploaded file.
    
    Args:
        uploaded_file: Django UploadedFile instance
        
    Returns:
        Dictionary with file metadata
    """
    return {
        'original_filename': uploaded_file.name,
        'safe_filename': sanitize_filename(uploaded_file.name),
        'size': uploaded_file.size,
        'size_mb': round(uploaded_file.size / (1024 * 1024), 2),
        'content_type': uploaded_file.content_type,
        'charset': uploaded_file.charset,
    }


def validate_dataset_upload(
    uploaded_file: UploadedFile, 
    dataset_type: str,
    max_size_gb: float = 2.0
) -> Tuple[bool, str, Optional[dict]]:
    """
    Comprehensive validation for uploaded dataset.
    
    Args:
        uploaded_file: Django UploadedFile instance
        dataset_type: Type of dataset
        max_size_gb: Maximum allowed size in GB
        
    Returns:
        Tuple of (is_valid, error_message, metadata)
    """
    # Validate file size
    is_valid, error = validate_file_size(uploaded_file, max_size_gb)
    if not is_valid:
        return False, error, None
    
    # Validate file type
    is_valid, error = validate_file_type(uploaded_file, dataset_type)
    if not is_valid:
        return False, error, None
    
    # Get file metadata
    metadata = get_file_metadata(uploaded_file)
    
    return True, "", metadata


# ========================================
# DEM-Specific Validation and Processing
# ========================================

def validate_geotiff(file_path: str) -> Tuple[bool, str]:
    """
    Validate that a file is a valid GeoTIFF using rasterio.
    
    Args:
        file_path: Path to the GeoTIFF file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not RASTERIO_AVAILABLE:
        return False, "Rasterio library not available for GeoTIFF validation"
    
    try:
        with rasterio.open(file_path) as src:
            # Basic validation - check if we can read the file
            _ = src.read(1, masked=True)
            return True, ""
    except rasterio.errors.RasterioIOError as e:
        return False, f"Invalid GeoTIFF file: {str(e)}"
    except Exception as e:
        return False, f"Error reading GeoTIFF: {str(e)}"


def extract_dem_metadata(file_path: str) -> Dict[str, Any]:
    """
    Extract comprehensive metadata from a DEM GeoTIFF file.
    
    Args:
        file_path: Path to the GeoTIFF file
        
    Returns:
        Dictionary with DEM metadata
        
    Raises:
        RuntimeError: If rasterio is not available or file cannot be read
    """
    if not RASTERIO_AVAILABLE:
        raise RuntimeError("Rasterio library not available")
    
    with rasterio.open(file_path) as src:
        bounds = src.bounds
        transform = src.transform
        
        metadata = {
            # Raster dimensions
            'width': src.width,
            'height': src.height,
            'bands': src.count,
            
            # Spatial properties
            'bounds': {
                'minx': bounds.left,
                'miny': bounds.bottom,
                'maxx': bounds.right,
                'maxy': bounds.top,
            },
            'pixel_size_x': abs(transform.a),
            'pixel_size_y': abs(transform.e),
            
            # CRS information
            'crs': src.crs.to_string() if src.crs else None,
            'epsg': src.crs.to_epsg() if src.crs else None,
            
            # Data properties
            'dtype': str(src.dtypes[0]),
            'nodata': src.nodata,
            
            # Additional metadata
            'driver': src.driver,
            'compress': src.compression,
        }
        
        return metadata


def compute_dem_statistics(file_path: str, sample_factor: int = 10) -> Dict[str, float]:
    """
    Compute elevation statistics from DEM (min, max, mean, std dev).
    Uses sampling for large DEMs to improve performance.
    
    Args:
        file_path: Path to the GeoTIFF file
        sample_factor: Read every Nth pixel (default: 10 for ~1% sampling)
        
    Returns:
        Dictionary with statistics: min, max, mean, std
        
    Raises:
        RuntimeError: If rasterio is not available or file cannot be read
    """
    if not RASTERIO_AVAILABLE:
        raise RuntimeError("Rasterio library not available")
    
    with rasterio.open(file_path) as src:
        # Read first band with sampling
        data = src.read(
            1,
            out_shape=(
                src.count,
                int(src.height / sample_factor),
                int(src.width / sample_factor)
            ),
            resampling=Resampling.average,
            masked=True  # Mask nodata values
        )
        
        # Compute statistics (masked array handles nodata automatically)
        stats = {
            'min_value': float(np.min(data)) if data.size > 0 else None,
            'max_value': float(np.max(data)) if data.size > 0 else None,
            'mean_value': float(np.mean(data)) if data.size > 0 else None,
            'std_value': float(np.std(data)) if data.size > 0 else None,
        }
        
        return stats


def detect_dem_crs(file_path: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Detect CRS from DEM file metadata.
    
    Args:
        file_path: Path to the GeoTIFF file
        
    Returns:
        Tuple of (crs_string, epsg_code)
        Returns (None, None) if CRS is not defined
        
    Raises:
        RuntimeError: If rasterio is not available
    """
    if not RASTERIO_AVAILABLE:
        raise RuntimeError("Rasterio library not available")
    
    with rasterio.open(file_path) as src:
        if src.crs is None:
            return None, None
        
        crs_string = src.crs.to_string()
        epsg_code = src.crs.to_epsg()
        
        return crs_string, epsg_code


def generate_dem_thumbnail(
    file_path: str,
    output_path: Optional[str] = None,
    size: Tuple[int, int] = (400, 400),
    colormap: str = 'terrain'
) -> ContentFile:
    """
    Generate a thumbnail preview image from DEM.
    
    Args:
        file_path: Path to the GeoTIFF file
        output_path: Optional path to save thumbnail (if None, returns ContentFile)
        size: Thumbnail size (width, height)
        colormap: Matplotlib colormap name
        
    Returns:
        Django ContentFile containing the thumbnail image
        
    Raises:
        RuntimeError: If required libraries are not available
    """
    if not RASTERIO_AVAILABLE:
        raise RuntimeError("Rasterio library not available")
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError("Matplotlib library not available")
    
    with rasterio.open(file_path) as src:
        # Read downsampled data for thumbnail
        downsample = max(src.width // size[0], src.height // size[1], 1)
        
        data = src.read(
            1,
            out_shape=(
                src.count,
                int(src.height / downsample),
                int(src.width / downsample)
            ),
            resampling=Resampling.average,
            masked=True
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=(size[0]/100, size[1]/100), dpi=100)
        ax.axis('off')
        
        # Plot data
        im = ax.imshow(data, cmap=colormap, interpolation='bilinear')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Elevation (m)')
        
        plt.tight_layout(pad=0)
        
        # Save to BytesIO
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        plt.close(fig)
        
        buffer.seek(0)
        
        # Save to file if path provided
        if output_path:
            with open(output_path, 'wb') as f:
                f.write(buffer.getvalue())
            buffer.seek(0)
        
        # Return as ContentFile for Django
        return ContentFile(buffer.getvalue(), name='thumbnail.png')


def validate_and_process_dem(
    file_path: str,
    user_provided_epsg: Optional[int] = None
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Comprehensive DEM validation and metadata extraction.
    
    This is the main function to use for DEM upload processing.
    
    Args:
        file_path: Path to the uploaded GeoTIFF file
        user_provided_epsg: Optional EPSG code if CRS is missing
        
    Returns:
        Tuple of (is_valid, error_message, dem_data)
        dem_data contains: metadata, statistics, thumbnail
    """
    if not RASTERIO_AVAILABLE:
        return False, "Rasterio library not available. Cannot process DEM files.", None
    
    # Step 1: Validate GeoTIFF format
    is_valid, error = validate_geotiff(file_path)
    if not is_valid:
        return False, error, None
    
    try:
        # Step 2: Extract metadata
        metadata = extract_dem_metadata(file_path)
        
        # Step 3: Check CRS
        crs_string, epsg_code = detect_dem_crs(file_path)
        
        if epsg_code is None and user_provided_epsg is None:
            return False, "CRS not found in DEM file. Please provide an EPSG code.", {
                'metadata': metadata,
                'requires_crs': True
            }
        
        # Use user-provided EPSG if available
        if user_provided_epsg:
            epsg_code = user_provided_epsg
            crs_string = f"EPSG:{user_provided_epsg}"
        
        metadata['crs'] = crs_string
        metadata['epsg'] = epsg_code
        
        # Step 4: Compute statistics
        statistics = compute_dem_statistics(file_path)
        
        # Step 5: Generate thumbnail
        thumbnail = generate_dem_thumbnail(file_path)
        
        # Combine all data
        dem_data = {
            'metadata': metadata,
            'statistics': statistics,
            'thumbnail': thumbnail,
            'crs_string': crs_string,
            'epsg_code': epsg_code,
        }
        
        return True, "", dem_data
        
    except Exception as e:
        return False, f"Error processing DEM: {str(e)}", None


# ===========================
# Shapefile Processing Functions
# ===========================

try:
    import geopandas as gpd
    import fiona
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    print("Warning: geopandas/fiona not available. Shapefile processing will be disabled.")


def validate_shapefile_completeness(file_paths: Dict[str, str]) -> Tuple[bool, str]:
    """
    Validate that all required shapefile components are present.
    
    Args:
        file_paths: Dict with keys 'shp', 'shx', 'dbf', 'prj' (optional)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_extensions = ['shp', 'shx', 'dbf']
    
    # Check required files
    missing = [ext for ext in required_extensions if not file_paths.get(ext)]
    
    if missing:
        return False, f"Missing required shapefile components: {', '.join(missing)}"
    
    # Check if files exist
    for ext, path in file_paths.items():
        if ext in required_extensions and not os.path.exists(path):
            return False, f"File not found: {path}"
    
    return True, ""


def extract_shapefile_crs(prj_file_path: Optional[str] = None, shp_file_path: Optional[str] = None) -> Tuple[Optional[str], Optional[int]]:
    """
    Extract CRS from .prj file or shapefile metadata.
    
    Args:
        prj_file_path: Path to .prj file (optional)
        shp_file_path: Path to .shp file (optional, used if no .prj)
        
    Returns:
        Tuple of (crs_string, epsg_code) or (None, None) if not found
    """
    if not GEOPANDAS_AVAILABLE:
        return None, None
    
    try:
        # Try .prj file first
        if prj_file_path and os.path.exists(prj_file_path):
            with open(prj_file_path, 'r') as f:
                prj_content = f.read().strip()
                
            # Try to parse with geopandas
            import pyproj
            crs = pyproj.CRS.from_wkt(prj_content)
            
            if crs.to_epsg():
                epsg_code = crs.to_epsg()
                return f"EPSG:{epsg_code}", epsg_code
            else:
                return crs.to_wkt(), None
        
        # Fallback: read CRS from shapefile
        if shp_file_path and os.path.exists(shp_file_path):
            gdf = gpd.read_file(shp_file_path)
            if gdf.crs:
                epsg_code = gdf.crs.to_epsg()
                if epsg_code:
                    return f"EPSG:{epsg_code}", epsg_code
                else:
                    return gdf.crs.to_wkt(), None
        
        return None, None
        
    except Exception as e:
        print(f"Error extracting CRS: {e}")
        return None, None


def validate_shapefile_geometries(gdf: 'gpd.GeoDataFrame') -> Dict[str, Any]:
    """
    Validate geometries in a GeoDataFrame.
    
    Args:
        gdf: GeoDataFrame to validate
        
    Returns:
        Dict with validation results:
            - has_invalid: bool
            - invalid_count: int
            - null_count: int
            - geometry_types: list of unique geometry types
    """
    if not GEOPANDAS_AVAILABLE:
        return {
            'has_invalid': False,
            'invalid_count': 0,
            'null_count': 0,
            'geometry_types': []
        }
    
    # Check for null geometries
    null_geoms = gdf.geometry.isna()
    null_count = null_geoms.sum()
    
    # Check for invalid geometries (excluding nulls)
    valid_geoms = gdf[~null_geoms].geometry.is_valid
    invalid_count = (~valid_geoms).sum()
    
    # Get unique geometry types
    geom_types = gdf[~null_geoms].geometry.geom_type.unique().tolist()
    
    return {
        'has_invalid': invalid_count > 0 or null_count > 0,
        'invalid_count': int(invalid_count),
        'null_count': int(null_count),
        'geometry_types': geom_types,
    }


def parse_shapefile(shp_file_path: str) -> Tuple[bool, str, Optional['gpd.GeoDataFrame']]:
    """
    Parse shapefile using geopandas.
    
    Args:
        shp_file_path: Path to .shp file
        
    Returns:
        Tuple of (success, error_message, GeoDataFrame)
    """
    if not GEOPANDAS_AVAILABLE:
        return False, "geopandas not available", None
    
    try:
        gdf = gpd.read_file(shp_file_path)
        
        if len(gdf) == 0:
            return False, "Shapefile contains no features", None
        
        return True, "", gdf
        
    except Exception as e:
        return False, f"Error parsing shapefile: {str(e)}", None


def reproject_shapefile(gdf: 'gpd.GeoDataFrame', target_epsg: int) -> Tuple[bool, str, Optional['gpd.GeoDataFrame']]:
    """
    Reproject GeoDataFrame to target EPSG code.
    
    Args:
        gdf: GeoDataFrame to reproject
        target_epsg: Target EPSG code (e.g., 32651)
        
    Returns:
        Tuple of (success, error_message, reprojected_gdf)
    """
    import logging
    logger = logging.getLogger(__name__)
    
    if not GEOPANDAS_AVAILABLE:
        return False, "geopandas not available", None
    
    try:
        # Check if already in target CRS
        current_epsg = gdf.crs.to_epsg() if gdf.crs else None
        
        if current_epsg == target_epsg:
            return True, "", gdf
        
        # Log CRS transformation for audit trail
        logger.info(f"CRS Transformation: Reprojecting from EPSG:{current_epsg} to EPSG:{target_epsg}")
        
        # Reproject
        gdf_reprojected = gdf.to_crs(epsg=target_epsg)
        
        # Validate reprojected geometries
        invalid_after = gdf_reprojected.geometry.is_valid.sum() < len(gdf_reprojected)
        if invalid_after:
            logger.warning(f"CRS Transformation Warning: Some geometries became invalid after reprojection")
        else:
            logger.info(f"CRS Transformation Success: All geometries valid after reprojection")
        
        return True, "", gdf_reprojected
        
    except Exception as e:
        logger.error(f"CRS Transformation Error: {str(e)}")
        return False, f"Error reprojecting shapefile: {str(e)}", None


def extract_shapefile_metadata(gdf: 'gpd.GeoDataFrame') -> Dict[str, Any]:
    """
    Extract metadata from a GeoDataFrame.
    
    Args:
        gdf: GeoDataFrame to extract metadata from
        
    Returns:
        Dict with metadata:
            - feature_count: int
            - geometry_type: str (primary geometry type)
            - bounds: dict with minx, miny, maxx, maxy
            - attributes: dict of column names and types
            - crs: str
            - epsg: int or None
    """
    if not GEOPANDAS_AVAILABLE:
        return {}
    
    try:
        # Get bounds
        bounds = gdf.total_bounds
        
        # Get primary geometry type
        geom_types = gdf.geometry.geom_type.value_counts()
        primary_geom_type = geom_types.index[0] if len(geom_types) > 0 else 'Unknown'
        
        # Map geometry types to model choices
        geom_type_map = {
            'Point': 'POINT',
            'LineString': 'LINESTRING',
            'Polygon': 'POLYGON',
            'MultiPoint': 'MULTIPOINT',
            'MultiLineString': 'MULTILINESTRING',
            'MultiPolygon': 'MULTIPOLYGON',
            'GeometryCollection': 'GEOMETRYCOLLECTION',
        }
        
        primary_geom_type = geom_type_map.get(primary_geom_type, 'POLYGON')
        
        # Get attribute schema
        attributes = {}
        for col in gdf.columns:
            if col != 'geometry':
                attributes[col] = str(gdf[col].dtype)
        
        # Get CRS info
        crs_string = None
        epsg_code = None
        if gdf.crs:
            epsg_code = gdf.crs.to_epsg()
            crs_string = f"EPSG:{epsg_code}" if epsg_code else gdf.crs.to_wkt()
        
        return {
            'feature_count': len(gdf),
            'geometry_type': primary_geom_type,
            'bounds': {
                'minx': float(bounds[0]),
                'miny': float(bounds[1]),
                'maxx': float(bounds[2]),
                'maxy': float(bounds[3]),
            },
            'attributes': attributes,
            'crs': crs_string,
            'epsg': epsg_code,
        }
        
    except Exception as e:
        print(f"Error extracting shapefile metadata: {e}")
        return {}


def generate_shapefile_thumbnail(gdf: 'gpd.GeoDataFrame', output_size: Tuple[int, int] = (400, 400)) -> Optional[ContentFile]:
    """
    Generate a thumbnail image for a shapefile.
    
    Args:
        gdf: GeoDataFrame to visualize
        output_size: Tuple of (width, height) for output image
        
    Returns:
        ContentFile with PNG image, or None if failed
    """
    if not GEOPANDAS_AVAILABLE:
        return None
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-GUI backend
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(output_size[0]/100, output_size[1]/100), dpi=100)
        
        # Plot shapefile
        gdf.plot(ax=ax, color='steelblue', edgecolor='black', linewidth=0.5, alpha=0.7)
        
        ax.set_aspect('equal')
        ax.axis('off')
        plt.tight_layout(pad=0)
        
        # Save to buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close(fig)
        
        buffer.seek(0)
        return ContentFile(buffer.read(), name='shapefile_thumbnail.png')
        
    except Exception as e:
        print(f"Error generating shapefile thumbnail: {e}")
        return None


def validate_and_process_shapefile(
    file_paths: Dict[str, str],
    user_provided_epsg: Optional[int] = None,
    target_epsg: int = 32651
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Main function to validate and process a shapefile upload.
    
    This orchestrates all shapefile validation and processing steps:
    1. Validate file completeness
    2. Parse shapefile with geopandas
    3. Extract CRS
    4. Validate geometries
    5. Reproject if necessary
    6. Extract metadata
    7. Generate thumbnail
    
    Args:
        file_paths: Dict with keys 'shp', 'shx', 'dbf', 'prj' (optional)
        user_provided_epsg: EPSG code provided by user (if .prj missing)
        target_epsg: Target EPSG code for reprojection (default: 32651)
        
    Returns:
        Tuple of (success, error_message, shapefile_data)
        shapefile_data contains: metadata, validation, thumbnail, gdf
    """
    if not GEOPANDAS_AVAILABLE:
        return False, "geopandas not installed. Cannot process shapefiles.", None
    
    try:
        # Step 1: Validate completeness
        is_complete, error = validate_shapefile_completeness(file_paths)
        if not is_complete:
            return False, error, None
        
        # Step 2: Parse shapefile
        success, error, gdf = parse_shapefile(file_paths['shp'])
        if not success:
            return False, error, None
        
        # Step 3: Extract CRS
        crs_string, epsg_code = extract_shapefile_crs(
            prj_file_path=file_paths.get('prj'),
            shp_file_path=file_paths['shp']
        )
        
        if epsg_code is None and user_provided_epsg is None:
            # Extract metadata before returning
            metadata = extract_shapefile_metadata(gdf)
            return False, "CRS not found in shapefile. Please provide an EPSG code.", {
                'metadata': metadata,
                'requires_crs': True,
                'gdf': gdf
            }
        
        # Use user-provided EPSG if available
        if user_provided_epsg:
            epsg_code = user_provided_epsg
            crs_string = f"EPSG:{user_provided_epsg}"
            # Set CRS on GeoDataFrame
            gdf = gdf.set_crs(epsg=user_provided_epsg, allow_override=True)
        
        # Step 4: Validate geometries
        validation = validate_shapefile_geometries(gdf)
        
        # Step 5: Reproject if necessary
        if epsg_code != target_epsg:
            success, error, gdf = reproject_shapefile(gdf, target_epsg)
            if not success:
                return False, error, None
            epsg_code = target_epsg
            crs_string = f"EPSG:{target_epsg}"
        
        # Step 6: Extract metadata (after reprojection)
        metadata = extract_shapefile_metadata(gdf)
        
        # Step 7: Generate thumbnail
        thumbnail = generate_shapefile_thumbnail(gdf)
        
        # Combine all data
        shapefile_data = {
            'metadata': metadata,
            'validation': validation,
            'thumbnail': thumbnail,
            'crs_string': crs_string,
            'epsg_code': epsg_code,
            'gdf': gdf,  # Return GeoDataFrame for saving
        }
        
        return True, "", shapefile_data
        
    except Exception as e:
        return False, f"Error processing shapefile: {str(e)}", None


# ===========================
# Time Series Processing Functions
# ===========================

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not available. Time series processing will be disabled.")


def parse_csv_timeseries(file_path: str) -> Tuple[bool, str, Optional['pd.DataFrame']]:
    """
    Parse CSV time series file using pandas.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        Tuple of (success, error_message, dataframe)
    """
    if not PANDAS_AVAILABLE:
        return False, "pandas not installed. Cannot parse CSV files.", None
    
    try:
        # Try different encodings and delimiters
        for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                
                if len(df) == 0:
                    return False, "CSV file is empty", None
                
                return True, "", df
                
            except UnicodeDecodeError:
                continue
        
        return False, "Could not decode CSV file with supported encodings", None
        
    except Exception as e:
        return False, f"Error parsing CSV: {str(e)}", None


def parse_excel_timeseries(file_path: str) -> Tuple[bool, str, Optional['pd.DataFrame']]:
    """
    Parse Excel time series file using pandas.
    
    Args:
        file_path: Path to Excel file (.xlsx or .xls)
        
    Returns:
        Tuple of (success, error_message, dataframe)
    """
    if not PANDAS_AVAILABLE:
        return False, "pandas not installed. Cannot parse Excel files.", None
    
    try:
        # First, try reading normally
        df = pd.read_excel(file_path, sheet_name=0)
        
        # Check if first row contains header-like text (common Excel format issue)
        # If the first row has all string values that look like headers, skip it
        if len(df) > 0:
            first_row = df.iloc[0]
            # Check if first row looks like a header (all strings, no numeric data)
            if all(isinstance(val, str) for val in first_row.values):
                # Re-read with skiprows=1 to skip the text header
                df = pd.read_excel(file_path, sheet_name=0, skiprows=1)
        
        if len(df) == 0:
            return False, "Excel file is empty", None
        
        return True, "", df
        
    except Exception as e:
        return False, f"Error parsing Excel: {str(e)}", None


def detect_datetime_column(df: 'pd.DataFrame') -> Optional[str]:
    """
    Auto-detect datetime column in dataframe.
    
    Args:
        df: Pandas DataFrame
        
    Returns:
        Name of datetime column, or None if not found
    """
    if not PANDAS_AVAILABLE:
        return None
    
    # Common datetime column names
    datetime_names = ['datetime', 'date', 'time', 'timestamp', 'date_time', 'Date', 'DateTime', 'Timestamp']
    
    # Check for exact matches first
    for col in df.columns:
        if col in datetime_names:
            return col
    
    # Check for partial matches
    for col in df.columns:
        col_lower = str(col).lower()
        if any(name.lower() in col_lower for name in datetime_names):
            return col
    
    # Try to detect by data type
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    
    return None


def detect_value_column(df: 'pd.DataFrame', data_type: str = None) -> Optional[str]:
    """
    Auto-detect value column in dataframe.
    
    Args:
        df: Pandas DataFrame
        data_type: Type of data ('RAINFALL', 'DISCHARGE', 'OTHER')
        
    Returns:
        Name of value column, or None if not found
    """
    if not PANDAS_AVAILABLE:
        return None
    
    # Common value column names based on data type
    if data_type == 'RAINFALL':
        value_names = ['rainfall', 'rain', 'precipitation', 'precip', 'mm', 'Rainfall', 'Rain']
    elif data_type == 'DISCHARGE':
        value_names = ['discharge', 'flow', 'streamflow', 'q', 'Discharge', 'Flow', 'Q']
    else:
        value_names = ['value', 'data', 'measurement', 'Value', 'Data']
    
    # Check for exact matches
    for col in df.columns:
        if col in value_names:
            return col
    
    # Check for partial matches
    for col in df.columns:
        col_lower = str(col).lower()
        if any(name.lower() in col_lower for name in value_names):
            return col
    
    # Find first numeric column (excluding datetime)
    datetime_col = detect_datetime_column(df)
    for col in df.columns:
        if col != datetime_col and pd.api.types.is_numeric_dtype(df[col]):
            return col
    
    return None


def validate_datetime_column(df: 'pd.DataFrame', column_name: str) -> Tuple[bool, str, Optional['pd.DataFrame']]:
    """
    Validate and parse datetime column.
    
    Args:
        df: Pandas DataFrame
        column_name: Name of datetime column
        
    Returns:
        Tuple of (success, error_message, dataframe_with_parsed_datetime)
    """
    if not PANDAS_AVAILABLE:
        return False, "pandas not installed", None
    
    try:
        if column_name not in df.columns:
            return False, f"Column '{column_name}' not found in data", None
        
        # Try to parse datetime
        df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
        
        # Check for failed parsing
        null_count = df[column_name].isna().sum()
        total_count = len(df)
        
        if null_count == total_count:
            return False, f"Could not parse any datetime values in column '{column_name}'", None
        
        if null_count > 0:
            # Remove rows with invalid dates
            df_clean = df[df[column_name].notna()].copy()
            return True, f"Warning: {null_count} rows with invalid dates were removed", df_clean
        
        return True, "", df
        
    except Exception as e:
        return False, f"Error validating datetime column: {str(e)}", None


def validate_numeric_column(df: 'pd.DataFrame', column_name: str, allow_negative: bool = False) -> Tuple[bool, str, Optional['pd.DataFrame']]:
    """
    Validate numeric value column.
    
    Args:
        df: Pandas DataFrame
        column_name: Name of value column
        allow_negative: Whether negative values are allowed
        
    Returns:
        Tuple of (success, error_message, dataframe_with_cleaned_data)
    """
    if not PANDAS_AVAILABLE:
        return False, "pandas not installed", None
    
    try:
        if column_name not in df.columns:
            return False, f"Column '{column_name}' not found in data", None
        
        # Convert to numeric
        df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
        
        # Check for failed parsing
        null_count = df[column_name].isna().sum()
        total_count = len(df)
        
        if null_count == total_count:
            return False, f"Column '{column_name}' contains no valid numeric values", None
        
        # Check for negative values if not allowed
        if not allow_negative:
            negative_count = (df[column_name] < 0).sum()
            if negative_count > 0:
                df_clean = df[df[column_name] >= 0].copy()
                return True, f"Warning: {negative_count} negative values were removed", df_clean
        
        if null_count > 0:
            df_clean = df[df[column_name].notna()].copy()
            return True, f"Warning: {null_count} rows with missing/invalid values were removed", df_clean
        
        return True, "", df
        
    except Exception as e:
        return False, f"Error validating numeric column: {str(e)}", None


def detect_units(column_name: str, data_type: str = None) -> str:
    """
    Auto-detect units from column name or data type.
    
    Args:
        column_name: Name of the value column
        data_type: Type of data ('RAINFALL', 'DISCHARGE', 'OTHER')
        
    Returns:
        Detected units string
    """
    column_lower = column_name.lower()
    
    # Check column name for unit indicators
    if 'm³/s' in column_lower or 'm3/s' in column_lower or 'cms' in column_lower:
        return 'm³/s'
    elif 'mm' in column_lower:
        return 'mm'
    elif 'mm/hr' in column_lower or 'mm/h' in column_lower:
        return 'mm/hr'
    elif 'm/s' in column_lower:
        return 'm/s'
    
    # Fallback to data type defaults
    if data_type == 'RAINFALL':
        return 'mm'
    elif data_type == 'DISCHARGE':
        return 'm³/s'
    else:
        return 'unitless'


def validate_and_process_timeseries(
    file_path: str,
    data_type: str,
    datetime_column: Optional[str] = None,
    value_column: Optional[str] = None,
    units: Optional[str] = None,
    station_id: Optional[str] = None
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Main function to validate and process time series upload.
    
    This orchestrates all time series validation and processing steps:
    1. Parse CSV or Excel file
    2. Auto-detect or validate datetime column
    3. Auto-detect or validate value column
    4. Validate datetime format
    5. Validate numeric values
    6. Auto-detect units if not provided
    7. Handle missing values
    
    Args:
        file_path: Path to uploaded file
        data_type: Type of data ('RAINFALL', 'DISCHARGE', 'OTHER')
        datetime_column: Name of datetime column (auto-detect if None)
        value_column: Name of value column (auto-detect if None)
        units: Units of measurement (auto-detect if None)
        station_id: Station identifier (optional)
        
    Returns:
        Tuple of (success, error_message, timeseries_data)
        timeseries_data contains: df, datetime_col, value_col, units, stats
    """
    if not PANDAS_AVAILABLE:
        return False, "pandas not installed. Cannot process time series.", None
    
    try:
        # Step 1: Parse file
        file_ext = file_path.lower().split('.')[-1]
        
        if file_ext == 'csv':
            success, error, df = parse_csv_timeseries(file_path)
        elif file_ext in ['xlsx', 'xls']:
            success, error, df = parse_excel_timeseries(file_path)
        else:
            return False, f"Unsupported file format: .{file_ext}", None
        
        if not success:
            return False, error, None
        
        # Step 2: Detect/validate datetime column
        if not datetime_column:
            datetime_column = detect_datetime_column(df)
            if not datetime_column:
                return False, "Could not auto-detect datetime column. Please specify column name.", {
                    'columns': list(df.columns),
                    'requires_column_mapping': True
                }
        
        success, warning, df = validate_datetime_column(df, datetime_column)
        if not success:
            return False, warning, None
        
        # Step 3: Detect/validate value column
        if not value_column:
            value_column = detect_value_column(df, data_type)
            if not value_column:
                return False, "Could not auto-detect value column. Please specify column name.", {
                    'columns': list(df.columns),
                    'datetime_col': datetime_column,
                    'requires_column_mapping': True
                }
        
        allow_negative = (data_type != 'RAINFALL')  # Rainfall can't be negative
        success, warning2, df = validate_numeric_column(df, value_column, allow_negative=allow_negative)
        if not success:
            return False, warning2, None
        
        # Step 4: Auto-detect units if not provided
        if not units:
            units = detect_units(value_column, data_type)
        
        # Step 5: Compute statistics
        stats = {
            'count': len(df),
            'min': float(df[value_column].min()),
            'max': float(df[value_column].max()),
            'mean': float(df[value_column].mean()),
            'std': float(df[value_column].std()),
            'missing_values': int(df[value_column].isna().sum()),
            'date_range': {
                'start': df[datetime_column].min().isoformat(),
                'end': df[datetime_column].max().isoformat(),
            }
        }
        
        # Combine warnings
        warnings = []
        if warning:
            warnings.append(warning)
        if warning2:
            warnings.append(warning2)
        
        warning_message = "; ".join(warnings) if warnings else ""
        
        # Combine all data
        timeseries_data = {
            'df': df,
            'datetime_column': datetime_column,
            'value_column': value_column,
            'units': units,
            'station_id': station_id or 'UNKNOWN',
            'statistics': stats,
            'warnings': warning_message,
        }
        
        return True, warning_message, timeseries_data
        
    except Exception as e:
        return False, f"Error processing time series: {str(e)}", None


# ============================================================================
# HEC-HMS CSV PROCESSING FUNCTIONS
# ============================================================================

def parse_hms_csv(file_path):
    """
    Parse HMS CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        pandas.DataFrame with HMS data
        
    Raises:
        ValueError if file cannot be parsed
    """
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                
                # Check if dataframe is not empty
                if df.empty:
                    raise ValueError("CSV file is empty")
                
                return df
                
            except UnicodeDecodeError:
                continue
        
        raise ValueError(f"Could not parse CSV with any of these encodings: {encodings}")
        
    except Exception as e:
        raise ValueError(f"Error parsing HMS CSV: {str(e)}")


def parse_hms_excel(file_path):
    """
    Parse HMS Excel file (.xlsx or .xls).
    
    Args:
        file_path: Path to the Excel file
        
    Returns:
        pandas.DataFrame with HMS data from first sheet
        
    Raises:
        ValueError if file cannot be parsed
    """
    try:
        df = pd.read_excel(file_path, sheet_name=0)
        
        if df.empty:
            raise ValueError("Excel file is empty")
        
        return df
        
    except Exception as e:
        raise ValueError(f"Error parsing HMS Excel: {str(e)}")


def detect_hms_element_column(df):
    """
    Auto-detect element ID column in HMS data.
    
    Looks for common HMS element column names:
    - element, Element, ELEMENT
    - name, Name, NAME
    - id, ID, Id
    - subbasin, Subbasin, SUBBASIN
    - junction, Junction, JUNCTION
    - reach, Reach, REACH
    
    Args:
        df: pandas.DataFrame
        
    Returns:
        str: Column name if found, None otherwise
    """
    # Common element column names
    element_names = [
        'element', 'Element', 'ELEMENT',
        'name', 'Name', 'NAME',
        'id', 'ID', 'Id',
        'subbasin', 'Subbasin', 'SUBBASIN',
        'junction', 'Junction', 'JUNCTION',
        'reach', 'Reach', 'REACH',
        'element_name', 'Element Name', 'ELEMENT NAME',
        'element_id', 'Element ID', 'ELEMENT ID',
    ]
    
    # Check for exact matches first
    for name in element_names:
        if name in df.columns:
            return name
    
    # Check for partial matches (case-insensitive)
    for col in df.columns:
        col_lower = str(col).lower()
        if any(name.lower() in col_lower for name in ['element', 'name', 'subbasin', 'junction']):
            return col
    
    return None


def detect_hms_discharge_column(df):
    """
    Auto-detect discharge column in HMS data.
    
    Looks for common discharge column names:
    - discharge, Discharge, DISCHARGE
    - flow, Flow, FLOW
    - outflow, Outflow, OUTFLOW
    - Q, q
    
    Args:
        df: pandas.DataFrame
        
    Returns:
        str: Column name if found, None otherwise
    """
    # Common discharge column names
    discharge_names = [
        'discharge', 'Discharge', 'DISCHARGE',
        'flow', 'Flow', 'FLOW',
        'outflow', 'Outflow', 'OUTFLOW',
        'Q', 'q',
        'discharge_cms', 'Discharge (m3/s)', 'Discharge (m³/s)',
        'flow_cms', 'Flow (m3/s)', 'Flow (m³/s)',
    ]
    
    # Check for exact matches first
    for name in discharge_names:
        if name in df.columns:
            return name
    
    # Check for partial matches (case-insensitive)
    for col in df.columns:
        col_lower = str(col).lower()
        if any(name.lower() in col_lower for name in ['discharge', 'flow', 'outflow']):
            # Check if it's likely numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                return col
    
    return None


def validate_hms_element_column(df, column_name):
    """
    Validate element ID column.
    
    Args:
        df: pandas.DataFrame
        column_name: Name of the element column
        
    Returns:
        tuple: (bool success, str message, list unique_elements)
    """
    if column_name not in df.columns:
        return False, f"Column '{column_name}' not found in data", []
    
    # Check for null/empty values
    null_count = df[column_name].isnull().sum()
    if null_count > 0:
        return False, f"Element column contains {null_count} null/empty values", []
    
    # Get unique elements
    unique_elements = df[column_name].unique().tolist()
    
    if len(unique_elements) == 0:
        return False, "No valid element IDs found", []
    
    return True, f"Found {len(unique_elements)} unique elements", unique_elements


def validate_hms_discharge_column(df, column_name):
    """
    Validate discharge column and convert to numeric.
    
    Args:
        df: pandas.DataFrame
        column_name: Name of the discharge column
        
    Returns:
        tuple: (pandas.DataFrame cleaned_df, int invalid_count, dict stats)
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in data")
    
    # Convert to numeric
    original_count = len(df)
    df_clean = df.copy()
    df_clean[column_name] = pd.to_numeric(df_clean[column_name], errors='coerce')
    
    # Remove rows with null discharge
    df_clean = df_clean[df_clean[column_name].notna()]
    invalid_count = original_count - len(df_clean)
    
    # Calculate statistics
    stats = {
        'count': len(df_clean),
        'min': float(df_clean[column_name].min()),
        'max': float(df_clean[column_name].max()),
        'mean': float(df_clean[column_name].mean()),
        'std': float(df_clean[column_name].std()),
    }
    
    return df_clean, invalid_count, stats


def validate_hms_discharge_units(df, discharge_column):
    """
    Validate that discharge values are in m³/s (reasonable range check).
    
    Args:
        df: pandas.DataFrame
        discharge_column: Name of the discharge column
        
    Returns:
        tuple: (bool is_valid, str message)
    """
    discharge_values = df[discharge_column]
    
    # Check for reasonable discharge range (0 to 10,000 m³/s for typical watersheds)
    # Adjust this range based on your watershed size
    min_val = discharge_values.min()
    max_val = discharge_values.max()
    
    if min_val < 0:
        return False, f"Negative discharge values found (min: {min_val})"
    
    if max_val > 100000:
        return False, f"Unreasonably high discharge values found (max: {max_val} m³/s). Please verify units."
    
    # Check if all values are zero
    if max_val == 0:
        return False, "All discharge values are zero"
    
    return True, f"Discharge range: {min_val:.2f} - {max_val:.2f} m³/s"


def map_hms_elements_to_features(element_ids):
    """
    Map HMS element IDs to spatial features (subbasins, junctions, etc.).
    
    This is a placeholder for future spatial mapping functionality.
    In the future, this will query VectorLayer model for matching features.
    
    Args:
        element_ids: List of element ID strings
        
    Returns:
        dict: Mapping of element_id -> feature_id (currently returns empty dict)
    """
    # TODO: Implement spatial feature mapping when VectorLayer has element IDs
    # This will involve:
    # 1. Query VectorLayer for features with matching names/IDs
    # 2. Create spatial mapping based on name matching or attribute tables
    # 3. Return dict like {'Subbasin1': vector_layer_id, 'Junction2': vector_layer_id}
    
    mapping = {}
    for element_id in element_ids:
        # Placeholder - in future, query database for matching feature
        mapping[element_id] = None
    
    return mapping


def validate_and_process_hms(file_path, event_name, return_period=None,
                              element_column=None, datetime_column=None, 
                              discharge_column=None):
    """
    Main HMS data validation and processing function.
    
    Args:
        file_path: Path to HMS CSV/Excel file
        event_name: Storm event or simulation name
        return_period: Return period string (optional)
        element_column: Name of element ID column (auto-detect if None)
        datetime_column: Name of datetime column (auto-detect if None)
        discharge_column: Name of discharge column (auto-detect if None)
        
    Returns:
        dict with keys:
            - df: Cleaned pandas.DataFrame
            - element_column: Detected/validated element column name
            - datetime_column: Detected/validated datetime column name
            - discharge_column: Detected/validated discharge column name
            - unique_elements: List of unique element IDs
            - element_mapping: Dict mapping elements to features (placeholder)
            - statistics: Dict with discharge stats (count, min, max, mean, std)
            - time_range: Dict with start and end times
            - peak_discharge: Float peak discharge value
            - peak_element: String element ID with peak discharge
            - warnings: List of warning messages
            
    Raises:
        ValueError if validation fails
    """
    warnings = []
    
    try:
        # Step 1: Parse file
        file_ext = file_path.split('.')[-1].lower()
        if file_ext == 'csv':
            df = parse_hms_csv(file_path)
        elif file_ext in ['xlsx', 'xls']:
            df = parse_hms_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: .{file_ext}")
        
        warnings.append(f"Loaded {len(df)} rows from HMS file")
        
        # Step 2: Detect element column
        if not element_column or element_column not in df.columns:
            detected = detect_hms_element_column(df)
            if detected:
                element_column = detected
                warnings.append(f"Auto-detected element column: '{element_column}'")
            else:
                available = ', '.join(df.columns.tolist()[:10])
                raise ValueError(
                    f"Could not auto-detect element column. "
                    f"Please specify manually. Available columns: {available}..."
                )
        
        # Step 3: Validate element column
        success, message, unique_elements = validate_hms_element_column(df, element_column)
        if not success:
            raise ValueError(f"Element column validation failed: {message}")
        warnings.append(message)
        
        # Step 4: Detect datetime column (use time series function)
        if not datetime_column or datetime_column not in df.columns:
            detected = detect_datetime_column(df)
            if detected:
                datetime_column = detected
                warnings.append(f"Auto-detected datetime column: '{datetime_column}'")
            else:
                available = ', '.join(df.columns.tolist()[:10])
                raise ValueError(
                    f"Could not auto-detect datetime column. "
                    f"Please specify manually. Available columns: {available}..."
                )
        
        # Step 5: Validate datetime column
        success, message, df_clean = validate_datetime_column(df, datetime_column)
        if not success:
            raise ValueError(f"Datetime validation failed: {message}")
        if message:  # Warning message about removed rows
            warnings.append(message)
        if df_clean is None:
            df_clean = df  # Use original if no cleaning was done
        
        # Step 6: Detect discharge column
        if not discharge_column or discharge_column not in df_clean.columns:
            detected = detect_hms_discharge_column(df_clean)
            if detected:
                discharge_column = detected
                warnings.append(f"Auto-detected discharge column: '{discharge_column}'")
            else:
                available = ', '.join(df_clean.columns.tolist()[:10])
                raise ValueError(
                    f"Could not auto-detect discharge column. "
                    f"Please specify manually. Available columns: {available}..."
                )
        
        # Step 7: Validate discharge column
        df_clean, invalid_discharge_count, discharge_stats = validate_hms_discharge_column(
            df_clean, discharge_column
        )
        if invalid_discharge_count > 0:
            warnings.append(f"Removed {invalid_discharge_count} rows with invalid discharge values")
        
        # Step 8: Validate discharge units
        valid_units, units_message = validate_hms_discharge_units(df_clean, discharge_column)
        if not valid_units:
            raise ValueError(f"Discharge units validation failed: {units_message}")
        warnings.append(units_message)
        
        # Step 9: Calculate time range
        time_range = {
            'start': df_clean[datetime_column].min(),
            'end': df_clean[datetime_column].max(),
        }
        warnings.append(f"Time range: {time_range['start']} to {time_range['end']}")
        
        # Step 10: Find peak discharge
        peak_idx = df_clean[discharge_column].idxmax()
        peak_discharge = float(df_clean.loc[peak_idx, discharge_column])
        peak_element = str(df_clean.loc[peak_idx, element_column])
        warnings.append(f"Peak discharge: {peak_discharge:.2f} m³/s at element '{peak_element}'")
        
        # Step 11: Map elements to spatial features (placeholder)
        element_mapping = map_hms_elements_to_features(unique_elements)
        
        # Return processed data
        hms_data = {
            'df': df_clean,
            'element_column': element_column,
            'datetime_column': datetime_column,
            'discharge_column': discharge_column,
            'unique_elements': unique_elements,
            'element_mapping': element_mapping,
            'statistics': discharge_stats,
            'time_range': time_range,
            'peak_discharge': peak_discharge,
            'peak_element': peak_element,
            'warnings': warnings,
        }
        
        return hms_data
        
    except Exception as e:
        raise ValueError(f"Error processing HMS data: {str(e)}")


# ============================================================================
# HEC-HMS BASIN FILE PARSING
# ============================================================================

def parse_hms_basin_file(basin_file_path: str) -> Dict[str, Any]:
    """
    Parse HEC-HMS .basin file to extract subbasin geometry and metadata.
    
    The .basin file contains:
    - Subbasin centroids (Canvas X/Y in UTM coordinates)
    - Drainage areas (km²)
    - Flow network topology (Downstream connections)
    - Hydrologic parameters (Curve Number, Lag time, etc.)
    
    Args:
        basin_file_path: Path to .basin file
        
    Returns:
        Dict with keys:
            - subbasins: List of dicts with {name, centroid, area_km2, downstream, ...}
            - junctions: List of dicts with {name, centroid, downstream}
            - reaches: List of dicts with {name, centroid, from_centroid, downstream}
            - sinks: List of dicts with {name, centroid}
            - crs: Coordinate system WKT string
            - basin_name: Basin name
    """
    try:
        from shapely.geometry import Point
        
        basin_data = {
            'subbasins': [],
            'junctions': [],
            'reaches': [],
            'sinks': [],
            'crs': None,
            'basin_name': None
        }
        
        with open(basin_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Parse basin name
            if line.startswith('Basin:'):
                basin_data['basin_name'] = line.split(':', 1)[1].strip()
            
            # Parse coordinate system
            elif line.startswith('Coordinate System:'):
                # Extract CRS (can span multiple lines)
                crs_lines = []
                i += 1
                while i < len(lines) and not lines[i].strip().startswith('Terrain:'):
                    crs_lines.append(lines[i].strip())
                    i += 1
                basin_data['crs'] = ' '.join(crs_lines)
                continue
            
            # Parse Subbasin
            elif line.startswith('Subbasin:'):
                subbasin = _parse_basin_element(lines, i, 'Subbasin')
                if subbasin:
                    basin_data['subbasins'].append(subbasin)
                    # Skip to end of this element
                    while i < len(lines) and not lines[i].strip().startswith('End:'):
                        i += 1
            
            # Parse Junction
            elif line.startswith('Junction:'):
                junction = _parse_basin_element(lines, i, 'Junction')
                if junction:
                    basin_data['junctions'].append(junction)
                    while i < len(lines) and not lines[i].strip().startswith('End:'):
                        i += 1
            
            # Parse Reach
            elif line.startswith('Reach:'):
                reach = _parse_basin_element(lines, i, 'Reach')
                if reach:
                    basin_data['reaches'].append(reach)
                    while i < len(lines) and not lines[i].strip().startswith('End:'):
                        i += 1
            
            # Parse Sink
            elif line.startswith('Sink:'):
                sink = _parse_basin_element(lines, i, 'Sink')
                if sink:
                    basin_data['sinks'].append(sink)
                    while i < len(lines) and not lines[i].strip().startswith('End:'):
                        i += 1
            
            i += 1
        
        return basin_data
        
    except Exception as e:
        raise ValueError(f"Error parsing .basin file: {str(e)}")


def _parse_basin_element(lines: List[str], start_idx: int, element_type: str) -> Optional[Dict[str, Any]]:
    """
    Helper function to parse a basin element (Subbasin, Junction, Reach, Sink).
    
    Args:
        lines: All lines from .basin file
        start_idx: Index of the element's header line
        element_type: Type of element ('Subbasin', 'Junction', 'Reach', 'Sink')
        
    Returns:
        Dict with element properties, or None if parsing fails
    """
    from shapely.geometry import Point
    
    try:
        # Parse element name from header line
        header_line = lines[start_idx].strip()
        name = header_line.split(':', 1)[1].strip()
        
        element = {
            'name': name,
            'type': element_type,
            'centroid': None,
            'centroid_geom': None,
            'area_km2': None,
            'downstream': None,
            'lat': None,
            'lon': None,
        }
        
        # Parse element properties
        i = start_idx + 1
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('End:'):
                break
            
            # Parse Canvas X/Y (centroid coordinates)
            if line.startswith('Canvas X:'):
                canvas_x = float(line.split(':', 1)[1].strip())
                element['centroid_x'] = canvas_x
            elif line.startswith('Canvas Y:'):
                canvas_y = float(line.split(':', 1)[1].strip())
                element['centroid_y'] = canvas_y
            
            # Parse Lat/Lon (for validation)
            elif line.startswith('Latitude Degrees:'):
                element['lat'] = float(line.split(':', 1)[1].strip())
            elif line.startswith('Longitude Degrees:'):
                element['lon'] = float(line.split(':', 1)[1].strip())
            
            # Parse drainage area (for Subbasins)
            elif line.startswith('Area:'):
                element['area_km2'] = float(line.split(':', 1)[1].strip())
            
            # Parse downstream connection
            elif line.startswith('Downstream:'):
                element['downstream'] = line.split(':', 1)[1].strip()
            
            # Parse From Canvas X/Y (for Reaches)
            elif line.startswith('From Canvas X:'):
                element['from_centroid_x'] = float(line.split(':', 1)[1].strip())
            elif line.startswith('From Canvas Y:'):
                element['from_centroid_y'] = float(line.split(':', 1)[1].strip())
            
            i += 1
        
        # Create Point geometry from centroid coordinates
        if 'centroid_x' in element and 'centroid_y' in element:
            element['centroid'] = (element['centroid_x'], element['centroid_y'])
            element['centroid_geom'] = Point(element['centroid_x'], element['centroid_y'])
        
        # Create from-point geometry for Reaches
        if 'from_centroid_x' in element and 'from_centroid_y' in element:
            element['from_centroid'] = (element['from_centroid_x'], element['from_centroid_y'])
            element['from_centroid_geom'] = Point(element['from_centroid_x'], element['from_centroid_y'])
        
        return element
        
    except Exception as e:
        print(f"Warning: Could not parse {element_type} at line {start_idx}: {e}")
        return None


# ============================================================================
# HEC-HMS DSS FILE READING
# ============================================================================

def parse_hms_dss_file(dss_file_path: str, pathnames: List[str]) -> pd.DataFrame:
    """
    Parse HEC-HMS .dss file to extract time series data using pydsstools.
    
    DSS (Data Storage System) is HEC's proprietary format for storing time series.
    Pathnames follow the format: /A/B/C/D/E/F/
    - A: Unused (usually empty)
    - B: Location (e.g., "Subbasin-1", "Junction-1")
    - C: Parameter (e.g., "FLOW", "PRECIP-INC")
    - D: Unused (usually empty)
    - E: Time interval (e.g., "15MIN", "1HOUR")
    - F: Version (e.g., "RUN:Run 1", "MET:Met 1")
    
    Args:
        dss_file_path: Path to .dss file
        pathnames: List of DSS pathnames to extract (e.g., ["//Subbasin-1/FLOW//15MIN/RUN:Run 1/"])
        
    Returns:
        pandas.DataFrame with MultiIndex (datetime, pathname) and 'value' column
        
    Raises:
        ImportError: If pydsstools is not installed
        ValueError: If DSS file cannot be read or pathnames not found
    """
    try:
        from pydsstools.heclib.dss import HecDss
        import pandas as pd
        from datetime import datetime
    except ImportError:
        raise ImportError(
            "pydsstools is not installed. Install with: pip install pydsstools\n"
            "Note: pydsstools requires HEC-DSS libraries (may need HEC-DSS Vue installed)"
        )
    
    try:
        # Open DSS file (HecDss is a module, use HecDss.Open())
        dss_file = HecDss.Open(dss_file_path)
        
        # Storage for all time series
        all_data = []
        
        for pathname in pathnames:
            try:
                # Read time series from DSS
                ts = dss_file.read_ts(pathname)
                
                if ts is None:
                    print(f"Warning: Could not read pathname '{pathname}' from DSS file")
                    continue
                
                # Extract time series data
                # pydsstools returns a namedtuple with: times, values, units, type, interval, etc.
                times = ts.pytimes  # List of datetime objects
                values = ts.values  # Numpy array of values
                units = ts.units if hasattr(ts, 'units') else 'Unknown'
                
                # Create DataFrame for this pathname
                df_ts = pd.DataFrame({
                    'datetime': times,
                    'value': values,
                    'pathname': pathname,
                    'units': units
                })
                
                all_data.append(df_ts)
                
                print(f"Loaded {len(times)} timesteps from '{pathname}' ({units})")
                
            except Exception as e:
                print(f"Warning: Error reading pathname '{pathname}': {e}")
                continue
        
        # Close DSS file
        dss_file.close()
        
        if len(all_data) == 0:
            raise ValueError("No data was extracted from DSS file")
        
        # Combine all time series
        df_combined = pd.concat(all_data, ignore_index=True)
        
        # Set MultiIndex for efficient querying
        df_combined = df_combined.set_index(['datetime', 'pathname'])
        df_combined = df_combined.sort_index()
        
        return df_combined
        
    except Exception as e:
        raise ValueError(f"Error reading DSS file: {str(e)}")


def extract_dss_pathnames_from_results_xml(results_xml_path: str) -> Dict[str, List[str]]:
    """
    Extract DSS pathnames from HEC-HMS results XML file.
    
    The results XML contains references to time series data stored in DSS files.
    This function parses the XML to get the pathnames needed for reading DSS.
    
    Args:
        results_xml_path: Path to results XML file (e.g., RUN_Run_1.results.xml)
        
    Returns:
        Dict mapping element names to lists of DSS pathnames
        Example: {
            'Subbasin-1': ['//Subbasin-1/FLOW//15MIN/RUN:Run 1/', ...],
            'Junction-1': ['//Junction-1/FLOW//15MIN/RUN:Run 1/', ...],
        }
    """
    import xml.etree.ElementTree as ET
    
    try:
        tree = ET.parse(results_xml_path)
        root = tree.getroot()
        
        pathnames_by_element = {}
        
        # Parse each basin element
        for basin_element in root.findall('BasinElement'):
            element_name = basin_element.get('name')
            element_type = basin_element.get('type')
            
            pathnames = []
            
            # Find all time series references
            hydrology = basin_element.find('Hydrology')
            if hydrology is not None:
                for time_series in hydrology.findall('TimeSeries'):
                    pathname_elem = time_series.find('DssPathname')
                    if pathname_elem is not None:
                        pathname = pathname_elem.text
                        pathnames.append(pathname)
            
            if pathnames:
                pathnames_by_element[element_name] = pathnames
        
        return pathnames_by_element
        
    except Exception as e:
        raise ValueError(f"Error parsing results XML: {str(e)}")


def load_hms_hydrographs(
    basin_file_path: str,
    results_xml_path: str,
    dss_file_path: str,
    element_types: List[str] = ['Subbasin', 'Junction', 'Sink']
) -> Dict[str, pd.DataFrame]:
    """
    High-level function to load HEC-HMS hydrographs with spatial geometry.
    
    This combines basin geometry parsing with DSS time series extraction.
    
    Args:
        basin_file_path: Path to .basin file
        results_xml_path: Path to results XML file
        dss_file_path: Path to .dss file with time series
        element_types: List of element types to load (default: ['Subbasin', 'Junction', 'Sink'])
        
    Returns:
        Dict mapping element names to DataFrames with columns:
            - datetime: Timestamp
            - discharge: Flow value (m³/s)
            - centroid_x: X coordinate (UTM)
            - centroid_y: Y coordinate (UTM)
            - area_km2: Drainage area (for subbasins)
    """
    try:
        # Step 1: Parse basin file for geometry
        basin_data = parse_hms_basin_file(basin_file_path)
        
        # Step 2: Extract DSS pathnames from results XML
        pathnames_by_element = extract_dss_pathnames_from_results_xml(results_xml_path)
        
        # Step 3: Filter pathnames to get FLOW time series only
        flow_pathnames = []
        for element_name, pathnames in pathnames_by_element.items():
            for pathname in pathnames:
                # Check if this is a flow/discharge pathname
                if '/FLOW/' in pathname and '/FLOW-' not in pathname:
                    flow_pathnames.append(pathname)
        
        # Step 4: Load time series from DSS
        df_dss = parse_hms_dss_file(dss_file_path, flow_pathnames)
        
        # Step 5: Match time series to basin elements
        hydrographs = {}
        
        for element_type in element_types:
            elements_key = element_type.lower() + 's'  # 'subbasins', 'junctions', 'sinks'
            
            for element in basin_data.get(elements_key, []):
                element_name = element['name']
                
                # Find matching time series
                # DSS pathname format: //ELEMENT_NAME/FLOW//15MIN/RUN:Run 1/
                matching_rows = df_dss.index.get_level_values('pathname').str.contains(f"//{element_name}/FLOW//")
                
                if not matching_rows.any():
                    continue
                
                # Extract time series for this element
                df_element = df_dss[matching_rows].copy()
                df_element = df_element.reset_index()
                df_element = df_element.rename(columns={'value': 'discharge'})
                
                # Add spatial metadata
                if element['centroid']:
                    df_element['centroid_x'] = element['centroid'][0]
                    df_element['centroid_y'] = element['centroid'][1]
                
                if element['area_km2']:
                    df_element['area_km2'] = element['area_km2']
                
                df_element['element_type'] = element_type
                
                hydrographs[element_name] = df_element
        
        return hydrographs
        
    except Exception as e:
        raise ValueError(f"Error loading HMS hydrographs: {str(e)}")
