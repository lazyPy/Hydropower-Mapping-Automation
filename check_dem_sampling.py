"""Check inlet elevation sampling vs DEM NODATA issue."""
import django
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'HYDROPOWER_MAPPING.settings')
django.setup()

from django.conf import settings
from pathlib import Path
import rasterio
from hydropower.models import SitePoint, SitePair, RasterLayer

# Get inlet
sp = SitePoint.objects.filter(raster_layer_id=97, site_type='INLET', site_id='IN_97_8').first()
print(f"Inlet {sp.site_id}:")
print(f"  Database Geom: ({sp.geometry.x:.1f}, {sp.geometry.y:.1f})")
print(f"  Database Elev: {sp.elevation:.2f}m")
print()

# Load original DEM
rl = RasterLayer.objects.get(id=97)
original_dem = Path(settings.MEDIA_ROOT) / rl.dataset.file.name
filled_dem = Path(settings.MEDIA_ROOT) / rl.filled_dem_path

print(f"Original DEM: {original_dem}")
print(f"Filled DEM: {filled_dem}")
print()

# Sample original DEM at inlet location
with rasterio.open(original_dem) as src:
    row, col = src.index(sp.geometry.x, sp.geometry.y)
    val = src.read(1)[row, col]
    print(f"Original DEM at inlet ({sp.geometry.x:.1f}, {sp.geometry.y:.1f}):")
    print(f"  Pixel: row={row}, col={col}")
    print(f"  Value: {val:.2f}m")
    print(f"  Is NODATA: {val < -9999}")
    print()

# Sample filled DEM at inlet location
with rasterio.open(filled_dem) as src:
    row, col = src.index(sp.geometry.x, sp.geometry.y)
    val = src.read(1)[row, col]
    print(f"Filled DEM at inlet ({sp.geometry.x:.1f}, {sp.geometry.y:.1f}):")
    print(f"  Pixel: row={row}, col={col}")
    print(f"  Value: {val:.2f}m")
    print(f"  Is NODATA: {val < -9999}")
    print()

# Check 3x3 window around inlet in both DEMs
print("3x3 window around inlet:")
print()
print("Original DEM:")
with rasterio.open(original_dem) as src:
    row, col = src.index(sp.geometry.x, sp.geometry.y)
    window = src.read(1)[row-1:row+2, col-1:col+2]
    for r in window:
        print("  ", [f"{v:.1f}" if v > -9999 else "NODATA" for v in r])
print()

print("Filled DEM:")
with rasterio.open(filled_dem) as src:
    row, col = src.index(sp.geometry.x, sp.geometry.y)
    window = src.read(1)[row-1:row+2, col-1:col+2]
    for r in window:
        print("  ", [f"{v:.1f}" if v > -9999 else "NODATA" for v in r])
