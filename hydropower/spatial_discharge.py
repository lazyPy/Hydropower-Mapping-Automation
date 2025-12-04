"""
Spatially-Varying Discharge Module

This module calculates discharge at each location based on flow accumulation (drainage area).
Instead of using a single discharge value for all sites, this implements the hydrological
principle that discharge varies with contributing area:

    Q_site = Q_outlet × (A_site / A_outlet)

Where:
    - Q_site = Discharge at site location (m³/s)
    - Q_outlet = Discharge at basin outlet (from HMS) (m³/s)
    - A_site = Contributing area at site (approximated by flow accumulation cells)
    - A_outlet = Total basin area (max flow accumulation)

This provides realistic discharge values that vary spatially across the watershed,
addressing the issue where all nodes had the same discharge value.
"""

import numpy as np
import rasterio
from rasterio.transform import rowcol
from django.conf import settings
from django.db import transaction
from pathlib import Path
import logging
from typing import Dict, Optional, Tuple, List, Any

logger = logging.getLogger(__name__)


class SpatialDischargeCalculator:
    """
    Calculate spatially-varying discharge based on drainage area (flow accumulation).
    
    This class reads the flow accumulation raster and calculates discharge at any
    location using the drainage-area proportional method.
    """
    
    def __init__(self, raster_layer):
        """
        Initialize with a RasterLayer that has flow accumulation data.
        
        Args:
            raster_layer: RasterLayer model instance with flow_accumulation_path
        """
        self.raster_layer = raster_layer
        self.flow_acc_path = None
        self.flow_acc_data = None
        self.flow_acc_transform = None
        self.flow_acc_nodata = None
        self.max_flow_acc = None
        self.cell_size = None
        self.basin_area_km2 = None
        
        # Load flow accumulation raster
        self._load_flow_accumulation()
    
    def _load_flow_accumulation(self):
        """Load flow accumulation raster into memory for efficient querying."""
        if not self.raster_layer.flow_accumulation_path:
            raise ValueError("Flow accumulation raster not available. Run DEM preprocessing first.")
        
        # Build full path
        flow_acc_path = Path(settings.MEDIA_ROOT) / self.raster_layer.flow_accumulation_path
        
        if not flow_acc_path.exists():
            raise FileNotFoundError(f"Flow accumulation raster not found: {flow_acc_path}")
        
        self.flow_acc_path = str(flow_acc_path)
        
        with rasterio.open(self.flow_acc_path) as src:
            self.flow_acc_data = src.read(1).astype(np.float64)
            self.flow_acc_transform = src.transform
            self.flow_acc_nodata = src.nodata
            self.cell_size = abs(src.res[0])  # Assuming square cells
            
            # Mask nodata values
            if self.flow_acc_nodata is not None:
                self.flow_acc_data = np.ma.masked_equal(self.flow_acc_data, self.flow_acc_nodata)
            
            # Calculate max flow accumulation (basin outlet)
            self.max_flow_acc = float(np.nanmax(self.flow_acc_data))
            
            # Estimate basin area in km² (cell_size in meters, cells × cell_area)
            # Note: flow accumulation = number of upstream cells
            cell_area_m2 = self.cell_size ** 2
            self.basin_area_km2 = (self.max_flow_acc * cell_area_m2) / 1_000_000
        
        logger.info(f"Loaded flow accumulation: max={self.max_flow_acc:.0f} cells, "
                   f"cell_size={self.cell_size:.1f}m, basin_area≈{self.basin_area_km2:.2f}km²")
    
    def get_flow_accumulation_at_point(self, x: float, y: float) -> Optional[float]:
        """
        Get flow accumulation value at a specific coordinate.
        
        Args:
            x, y: Coordinates in raster CRS (EPSG:32651)
        
        Returns:
            Flow accumulation (number of upstream cells), or None if outside bounds
        """
        if self.flow_acc_data is None:
            return None
        
        try:
            row, col = rowcol(self.flow_acc_transform, x, y)
            
            if not (0 <= row < self.flow_acc_data.shape[0] and 
                    0 <= col < self.flow_acc_data.shape[1]):
                return None
            
            value = self.flow_acc_data[row, col]
            
            if isinstance(value, np.ma.core.MaskedConstant) or np.isnan(value):
                return None
            
            return float(value)
            
        except Exception as e:
            logger.debug(f"Error getting flow accumulation at ({x}, {y}): {e}")
            return None
    
    def get_drainage_area_km2(self, x: float, y: float) -> Optional[float]:
        """
        Get contributing drainage area in km² at a specific location.
        
        Args:
            x, y: Coordinates in raster CRS
        
        Returns:
            Drainage area in km², or None if not available
        """
        flow_acc = self.get_flow_accumulation_at_point(x, y)
        
        if flow_acc is None:
            return None
        
        # Convert cells to area
        cell_area_m2 = self.cell_size ** 2
        area_km2 = (flow_acc * cell_area_m2) / 1_000_000
        
        return area_km2
    
    def calculate_discharge(self, x: float, y: float, q_outlet: float) -> Optional[float]:
        """
        Calculate discharge at a location using drainage-area proportional method.
        
        Formula: Q_site = Q_outlet × (A_site / A_outlet)
        
        Args:
            x, y: Coordinates in raster CRS
            q_outlet: Reference discharge at basin outlet (m³/s)
        
        Returns:
            Discharge at location (m³/s), or None if not available
        """
        flow_acc = self.get_flow_accumulation_at_point(x, y)
        
        if flow_acc is None or self.max_flow_acc is None or self.max_flow_acc == 0:
            return None
        
        # Area ratio = flow_acc_site / flow_acc_outlet
        area_ratio = flow_acc / self.max_flow_acc
        
        # Ensure minimum ratio for numerical stability
        area_ratio = max(area_ratio, 0.001)
        
        # Calculate discharge
        q_site = q_outlet * area_ratio
        
        return q_site
    
    def get_discharge_summary_stats(self) -> Dict[str, float]:
        """
        Get summary statistics for the flow accumulation raster.
        
        Returns:
            Dict with statistics: max, min, mean, std, etc.
        """
        if self.flow_acc_data is None:
            return {}
        
        valid_data = self.flow_acc_data.compressed() if isinstance(self.flow_acc_data, np.ma.MaskedArray) else self.flow_acc_data.flatten()
        valid_data = valid_data[valid_data > 0]  # Exclude zero/negative
        
        if len(valid_data) == 0:
            return {}
        
        return {
            'max_cells': float(np.max(valid_data)),
            'min_cells': float(np.min(valid_data)),
            'mean_cells': float(np.mean(valid_data)),
            'median_cells': float(np.median(valid_data)),
            'std_cells': float(np.std(valid_data)),
            'percentile_95': float(np.percentile(valid_data, 95)),
            'percentile_99': float(np.percentile(valid_data, 99)),
            'cell_size_m': self.cell_size,
            'basin_area_km2': self.basin_area_km2
        }


def update_site_pair_discharge_spatial(
    raster_layer,
    q_outlet: float,
    efficiency: float = 0.7
) -> Dict[str, Any]:
    """
    Update all site pairs for a raster layer with spatially-varying discharge.
    
    This function:
    1. Loads flow accumulation raster
    2. For each site pair, calculates discharge based on drainage area at inlet
    3. Recalculates power using the new discharge
    
    Args:
        raster_layer: RasterLayer model instance
        q_outlet: Reference discharge at basin outlet (m³/s), typically from HMS peak
        efficiency: Turbine efficiency factor (default 0.7)
    
    Returns:
        Statistics dict with update counts and discharge range
    """
    from hydropower.models import SitePair
    from hydropower.site_pairing import calculate_head_losses
    
    # Initialize spatial calculator
    try:
        calculator = SpatialDischargeCalculator(raster_layer)
    except Exception as e:
        logger.error(f"Failed to initialize spatial discharge calculator: {e}")
        return {'error': str(e), 'updated': 0, 'failed': 0}
    
    # Get site pairs for this raster
    site_pairs = SitePair.objects.filter(raster_layer=raster_layer)
    total_count = site_pairs.count()
    
    if total_count == 0:
        logger.warning("No site pairs found to update")
        return {'updated': 0, 'failed': 0, 'total': 0}
    
    logger.info(f"Updating {total_count} site pairs with spatially-varying discharge...")
    logger.info(f"Reference Q_outlet: {q_outlet:.2f} m³/s, Basin area: {calculator.basin_area_km2:.2f} km²")
    
    # Process site pairs
    updated = 0
    failed = 0
    discharges = []
    powers = []
    
    # Constants for power calculation
    RHO = 1000.0  # kg/m³
    G = 9.81      # m/s²
    
    with transaction.atomic():
        for site_pair in site_pairs:
            try:
                # Get outlet coordinates (outlet accumulates all upstream flow!)
                outlet_x = site_pair.outlet.geometry.x
                outlet_y = site_pair.outlet.geometry.y
                
                # Calculate discharge at outlet location
                q_site = calculator.calculate_discharge(outlet_x, outlet_y, q_outlet)
                
                if q_site is None:
                    # Fallback: use area ratio based on stream order
                    # Higher order = more upstream area
                    stream_order = site_pair.inlet.stream_order or 1
                    fallback_ratio = min(0.5, stream_order * 0.1)
                    q_site = q_outlet * fallback_ratio
                    logger.debug(f"Site {site_pair.pair_id}: fallback discharge (order={stream_order})")
                
                # Ensure minimum discharge for numerical stability
                q_site = max(q_site, 0.01)  # At least 10 L/s
                
                # Update discharge
                site_pair.discharge = q_site
                discharges.append(q_site)
                
                # Calculate net head (accounting for losses)
                gross_head = site_pair.head or 0
                
                if site_pair.penstock_length and site_pair.penstock_length > 0:
                    losses = calculate_head_losses(
                        q_site,
                        site_pair.penstock_length,
                        gross_head
                    )
                    net_head = max(0, gross_head - losses.get('total_head_loss', 0))
                else:
                    net_head = gross_head
                
                # Calculate power: P = ρ × g × Q × H × η
                eff = site_pair.efficiency or efficiency
                power_kw = RHO * G * q_site * net_head * eff / 1000.0
                
                site_pair.power = max(0, power_kw)
                powers.append(site_pair.power)
                
                # Save
                site_pair.save(update_fields=['discharge', 'power'])
                updated += 1
                
            except Exception as e:
                logger.error(f"Error updating site pair {site_pair.pair_id}: {e}")
                failed += 1
    
    # Calculate statistics
    stats = {
        'updated': updated,
        'failed': failed,
        'total': total_count,
        'q_outlet': q_outlet,
        'basin_area_km2': calculator.basin_area_km2,
        'max_flow_acc': calculator.max_flow_acc,
    }
    
    if discharges:
        stats['discharge_min'] = min(discharges)
        stats['discharge_max'] = max(discharges)
        stats['discharge_mean'] = sum(discharges) / len(discharges)
        stats['discharge_std'] = np.std(discharges)
    
    if powers:
        stats['power_min'] = min(powers)
        stats['power_max'] = max(powers)
        stats['power_total'] = sum(powers)
        stats['power_mean'] = sum(powers) / len(powers)
    
    logger.info(f"Updated {updated}/{total_count} site pairs successfully")
    if discharges:
        logger.info(f"Discharge range: {stats['discharge_min']:.3f} - {stats['discharge_max']:.2f} m³/s")
    if powers:
        logger.info(f"Power range: {stats['power_min']:.1f} - {stats['power_max']:.1f} kW")
        logger.info(f"Total potential: {stats['power_total']:.1f} kW ({stats['power_total']/1000:.2f} MW)")
    
    return stats


def generate_discharge_raster(
    raster_layer,
    q_outlet: float,
    output_path: Optional[str] = None
) -> str:
    """
    Generate a discharge raster layer from flow accumulation.
    
    This creates a raster where each cell has its discharge value (m³/s),
    enabling visualization of discharge variation across the watershed.
    
    Args:
        raster_layer: RasterLayer model instance
        q_outlet: Reference discharge at outlet (m³/s)
        output_path: Optional output path for the discharge raster
    
    Returns:
        Path to the generated discharge raster
    """
    # Load flow accumulation
    flow_acc_path = Path(settings.MEDIA_ROOT) / raster_layer.flow_accumulation_path
    
    if output_path is None:
        output_dir = Path(settings.MEDIA_ROOT) / 'preprocessed' / f'dem_{raster_layer.id}'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / 'discharge_raster.tif')
    
    with rasterio.open(str(flow_acc_path)) as src:
        flow_acc = src.read(1).astype(np.float64)
        profile = src.profile.copy()
        nodata = src.nodata
        
        # Get max flow accumulation
        valid_mask = flow_acc > 0
        if nodata is not None:
            valid_mask &= (flow_acc != nodata)
        
        max_flow_acc = float(np.max(flow_acc[valid_mask]))
        
        # Calculate discharge: Q = Q_outlet × (flow_acc / max_flow_acc)
        discharge = np.zeros_like(flow_acc, dtype=np.float32)
        discharge[valid_mask] = q_outlet * (flow_acc[valid_mask] / max_flow_acc)
        
        # Set nodata for invalid cells
        discharge_nodata = -9999.0
        discharge[~valid_mask] = discharge_nodata
        
        # Update profile for output
        profile.update(
            dtype=rasterio.float32,
            nodata=discharge_nodata,
            compress='lzw'
        )
        
        # Write output
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(discharge.astype(np.float32), 1)
    
    logger.info(f"Generated discharge raster: {output_path}")
    logger.info(f"Discharge range: 0 - {q_outlet:.2f} m³/s")
    
    return output_path
