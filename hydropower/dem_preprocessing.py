"""
DEM Preprocessing Utilities using WhiteboxTools

Implements hydrological preprocessing algorithms:
- Depression filling (fill algorithm)
- Stream breach algorithm for flow continuity
- D8 flow direction computation
- Flow accumulation from flow direction
- Optional DEM smoothing/filtering
"""

import os
import tempfile
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from django.conf import settings
from django.core.files.base import ContentFile
import rasterio
import numpy as np

# Import WhiteboxTools
try:
    from whitebox import WhiteboxTools
    wbt = WhiteboxTools()
    # Set verbose mode to False for cleaner output
    wbt.set_verbose_mode(False)
    WHITEBOX_AVAILABLE = True
except ImportError:
    WHITEBOX_AVAILABLE = False
    wbt = None
    print("Warning: WhiteboxTools not available. DEM preprocessing will be disabled.")


class DEMPreprocessor:
    """
    Handles DEM preprocessing for hydrological analysis using WhiteboxTools.
    """
    
    def __init__(self, work_dir: Optional[str] = None):
        """
        Initialize DEM preprocessor.
        
        Args:
            work_dir: Working directory for temporary files. If None, uses system temp.
        """
        if not WHITEBOX_AVAILABLE:
            raise RuntimeError("WhiteboxTools is not available. Please install whitebox package.")
        
        self.work_dir = work_dir or tempfile.mkdtemp(prefix='dem_preprocessing_')
        self.wbt = WhiteboxTools()
        self.wbt.set_verbose_mode(False)
        self.wbt.work_dir = self.work_dir
        
    def fill_depressions(self, input_dem: str, output_filled: str, 
                        flat_increment: Optional[float] = None,
                        max_depth: Optional[float] = None) -> Dict[str, Any]:
        """
        Fill depressions (sinks) in DEM using Wang & Liu (2006) algorithm.
        
        Args:
            input_dem: Path to input DEM file
            output_filled: Path to output filled DEM file
            flat_increment: Optional increment value for flat areas (default: small value)
            max_depth: Optional maximum depression depth to fill in elevation units
            
        Returns:
            Dictionary with processing metadata
        """
        try:
            # Use FillDepressions with Wang & Liu algorithm (default in whitebox)
            self.wbt.fill_depressions(
                dem=input_dem,
                output=output_filled,
                fix_flats=True,  # Apply flat increment to ensure flow routing
                flat_increment=flat_increment,
                max_depth=max_depth
            )
            
            # Get statistics
            stats = self._get_raster_stats(output_filled)
            
            return {
                'success': True,
                'output_file': output_filled,
                'algorithm': 'Wang & Liu (2006) depression filling',
                'stats': stats,
                'parameters': {
                    'flat_increment': flat_increment,
                    'max_depth': max_depth
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def breach_depressions(self, input_dem: str, output_breached: str,
                          max_depth: Optional[float] = None,
                          max_length: Optional[float] = None,
                          fill_pits: bool = True) -> Dict[str, Any]:
        """
        Breach depressions to create flow continuity using least-cost paths.
        
        This is an alternative to filling that creates channels through barriers.
        
        Args:
            input_dem: Path to input DEM file
            output_breached: Path to output breached DEM file
            max_depth: Maximum breach depth in elevation units
            max_length: Maximum breach channel length in map units
            fill_pits: Whether to fill remaining pits after breaching
            
        Returns:
            Dictionary with processing metadata
        """
        try:
            # Use BreachDepressionsLeastCost
            self.wbt.breach_depressions_least_cost(
                dem=input_dem,
                output=output_breached,
                max_cost=max_depth,
                max_dist=max_length,
                fill=fill_pits,
                flat_increment=None
            )
            
            stats = self._get_raster_stats(output_breached)
            
            return {
                'success': True,
                'output_file': output_breached,
                'algorithm': 'Least-cost path breaching',
                'stats': stats,
                'parameters': {
                    'max_depth': max_depth,
                    'max_length': max_length,
                    'fill_pits': fill_pits
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def compute_d8_flow_direction(self, input_dem: str, output_flow_dir: str) -> Dict[str, Any]:
        """
        Compute D8 flow direction from filled/breached DEM.
        
        D8 flow direction assigns flow from each cell to one of 8 neighbors
        in the direction of steepest descent.
        
        Args:
            input_dem: Path to filled/breached DEM
            output_flow_dir: Path to output flow direction raster
            
        Returns:
            Dictionary with processing metadata
        """
        try:
            # Compute D8 flow direction
            self.wbt.d8_pointer(
                dem=input_dem,
                output=output_flow_dir,
                esri_pntr=False  # Use Whitebox encoding (1-128)
            )
            
            stats = self._get_raster_stats(output_flow_dir)
            
            return {
                'success': True,
                'output_file': output_flow_dir,
                'algorithm': 'D8 flow direction (Whitebox encoding)',
                'stats': stats,
                'encoding': 'Whitebox (1=E, 2=NE, 4=N, 8=NW, 16=W, 32=SW, 64=S, 128=SE)'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def compute_flow_accumulation(self, input_dem: str, output_flow_accum: str,
                                   log_transform: bool = False) -> Dict[str, Any]:
        """
        Compute flow accumulation directly from DEM.
        
        Flow accumulation represents the number of upslope cells that drain
        through each cell. This method computes flow accumulation directly from
        the hydrologically conditioned (filled/breached) DEM, which internally
        computes flow directions.
        
        Args:
            input_dem: Path to hydrologically conditioned DEM (filled or breached)
            output_flow_accum: Path to output flow accumulation raster
            log_transform: Whether to apply log transformation for visualization
            
        Returns:
            Dictionary with processing metadata
        """
        try:
            # Compute flow accumulation directly from DEM (pntr=False)
            # This is more reliable than using a separate flow pointer raster
            self.wbt.d8_flow_accumulation(
                i=input_dem,
                output=output_flow_accum,
                out_type='cells',  # Output in number of cells (or 'catchment area', 'specific contributing area')
                pntr=False,  # Input is a DEM, not a D8 pointer raster
                log=log_transform,
                clip=False
            )
            
            stats = self._get_raster_stats(output_flow_accum)
            
            return {
                'success': True,
                'output_file': output_flow_accum,
                'algorithm': 'D8 flow accumulation',
                'stats': stats,
                'parameters': {
                    'output_type': 'cells',
                    'log_transform': log_transform
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def smooth_dem(self, input_dem: str, output_smoothed: str,
                   filter_size: int = 3, method: str = 'mean') -> Dict[str, Any]:
        """
        Apply smoothing filter to DEM (optional preprocessing step).
        
        Args:
            input_dem: Path to input DEM
            output_smoothed: Path to output smoothed DEM
            filter_size: Filter kernel size (3, 5, 7, etc.)
            method: Smoothing method ('mean', 'median', 'gaussian')
            
        Returns:
            Dictionary with processing metadata
        """
        try:
            if method == 'mean':
                self.wbt.mean_filter(
                    i=input_dem,
                    output=output_smoothed,
                    filterx=filter_size,
                    filtery=filter_size
                )
            elif method == 'median':
                self.wbt.median_filter(
                    i=input_dem,
                    output=output_smoothed,
                    filterx=filter_size,
                    filtery=filter_size
                )
            elif method == 'gaussian':
                self.wbt.gaussian_filter(
                    i=input_dem,
                    output=output_smoothed,
                    sigma=1.0
                )
            else:
                raise ValueError(f"Unknown smoothing method: {method}")
            
            stats = self._get_raster_stats(output_smoothed)
            
            return {
                'success': True,
                'output_file': output_smoothed,
                'algorithm': f'{method.capitalize()} smoothing',
                'stats': stats,
                'parameters': {
                    'filter_size': filter_size,
                    'method': method
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def full_preprocessing_pipeline(self, input_dem: str, output_prefix: str,
                                    use_breach: bool = False,
                                    smooth: bool = False,
                                    smooth_params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Run complete DEM preprocessing pipeline.
        
        Pipeline:
        1. Optional: Smooth DEM
        2. Fill or breach depressions
        3. Compute D8 flow direction
        4. Compute flow accumulation
        
        Args:
            input_dem: Path to input DEM
            output_prefix: Prefix for output files (e.g., 'preprocessed_dem')
            use_breach: Use breaching instead of filling (default: False)
            smooth: Apply smoothing before processing (default: False)
            smooth_params: Parameters for smoothing (filter_size, method)
            
        Returns:
            Dictionary with all outputs and metadata
        """
        results = {
            'success': True,
            'outputs': {},
            'errors': []
        }
        
        try:
            # Determine output paths
            base_dir = os.path.dirname(input_dem) if os.path.dirname(input_dem) else self.work_dir
            
            # Step 1: Optional smoothing
            current_dem = input_dem
            if smooth:
                smooth_params = smooth_params or {'filter_size': 3, 'method': 'mean'}
                smoothed_path = os.path.join(base_dir, f"{output_prefix}_smoothed.tif")
                result = self.smooth_dem(input_dem, smoothed_path, **smooth_params)
                if result['success']:
                    results['outputs']['smoothed_dem'] = smoothed_path
                    current_dem = smoothed_path
                else:
                    results['errors'].append(f"Smoothing failed: {result.get('error')}")
            
            # Step 2: Fill or breach depressions
            conditioned_path = os.path.join(base_dir, f"{output_prefix}_conditioned.tif")
            if use_breach:
                result = self.breach_depressions(current_dem, conditioned_path)
                condition_type = 'breached'
            else:
                result = self.fill_depressions(current_dem, conditioned_path)
                condition_type = 'filled'
            
            if result['success']:
                results['outputs'][f'{condition_type}_dem'] = conditioned_path
            else:
                results['success'] = False
                results['errors'].append(f"Depression {condition_type} failed: {result.get('error')}")
                return results
            
            # Step 3: Compute D8 flow direction
            flow_dir_path = os.path.join(base_dir, f"{output_prefix}_flow_direction.tif")
            result = self.compute_d8_flow_direction(conditioned_path, flow_dir_path)
            if result['success']:
                results['outputs']['flow_direction'] = flow_dir_path
            else:
                results['success'] = False
                results['errors'].append(f"Flow direction failed: {result.get('error')}")
                return results
            
            # Step 4: Compute flow accumulation (directly from conditioned DEM)
            flow_accum_path = os.path.join(base_dir, f"{output_prefix}_flow_accumulation.tif")
            result = self.compute_flow_accumulation(conditioned_path, flow_accum_path)
            if result['success']:
                results['outputs']['flow_accumulation'] = flow_accum_path
            else:
                results['success'] = False
                results['errors'].append(f"Flow accumulation failed: {result.get('error')}")
                return results
            
            # Add summary metadata
            results['metadata'] = {
                'input_dem': input_dem,
                'output_prefix': output_prefix,
                'preprocessing_method': condition_type,
                'smoothing_applied': smooth,
                'pipeline_complete': True
            }
            
            return results
            
        except Exception as e:
            results['success'] = False
            results['errors'].append(f"Pipeline error: {str(e)}")
            return results
    
    def _get_raster_stats(self, raster_path: str) -> Dict[str, Any]:
        """
        Get basic statistics for a raster file.
        
        Args:
            raster_path: Path to raster file
            
        Returns:
            Dictionary with raster statistics
        """
        try:
            with rasterio.open(raster_path) as src:
                data = src.read(1, masked=True)
                
                return {
                    'width': src.width,
                    'height': src.height,
                    'min': float(np.nanmin(data)),
                    'max': float(np.nanmax(data)),
                    'mean': float(np.nanmean(data)),
                    'std': float(np.nanstd(data)),
                    'nodata': src.nodata
                }
        except Exception as e:
            return {
                'error': f"Failed to read raster stats: {str(e)}"
            }
    
    def cleanup(self):
        """
        Clean up temporary working directory.
        """
        import shutil
        if self.work_dir and os.path.exists(self.work_dir) and 'dem_preprocessing_' in self.work_dir:
            try:
                shutil.rmtree(self.work_dir)
            except Exception as e:
                print(f"Warning: Failed to clean up work directory: {e}")


def preprocess_dem_for_dataset(dataset_id: int, use_breach: bool = False,
                               smooth: bool = False, cache: bool = True) -> Dict[str, Any]:
    """
    Convenience function to preprocess a DEM from a Dataset model instance.
    
    Args:
        dataset_id: ID of Dataset with type='DEM'
        use_breach: Use breaching instead of filling
        smooth: Apply smoothing filter
        cache: Cache preprocessed results for reuse
        
    Returns:
        Dictionary with processing results and output paths
    """
    from .models import Dataset, RasterLayer
    
    try:
        dataset = Dataset.objects.get(id=dataset_id, dataset_type='DEM')
    except Dataset.DoesNotExist:
        return {
            'success': False,
            'error': f"Dataset {dataset_id} not found or not a DEM"
        }
    
    # Get input DEM path
    input_dem_path = dataset.file.path
    
    # Determine output directory (same as uploads)
    output_dir = os.path.dirname(input_dem_path)
    output_prefix = f"preprocessed_{dataset.id}"
    
    # Initialize preprocessor
    preprocessor = DEMPreprocessor(work_dir=output_dir)
    
    try:
        # Run full pipeline
        results = preprocessor.full_preprocessing_pipeline(
            input_dem=input_dem_path,
            output_prefix=output_prefix,
            use_breach=use_breach,
            smooth=smooth
        )
        
        if results['success'] and cache:
            # Update RasterLayer model with preprocessing metadata
            try:
                raster_layer = dataset.raster_metadata
                # Store preprocessing info in a JSON field (would need to add to model)
                # For now, just return the results
            except RasterLayer.DoesNotExist:
                pass
        
        return results
        
    finally:
        # Don't cleanup if caching (keep files)
        if not cache:
            preprocessor.cleanup()
