"""Test DEM preprocessing directly."""
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'HYDROPOWER_MAPPING.settings')
django.setup()

from hydropower.dem_preprocessing import DEMPreprocessor
from hydropower.models import RasterLayer
from pathlib import Path
from django.conf import settings

# Get latest raster
raster = RasterLayer.objects.latest('id')
print(f"Testing with RasterLayer ID: {raster.id}")

# Get input DEM path
dem_path = str(Path(settings.MEDIA_ROOT) / raster.dataset.file.name)
print(f"Input DEM: {dem_path}")
print(f"Exists: {os.path.exists(dem_path)}")

# Set up output directory
output_dir = Path(settings.MEDIA_ROOT) / 'preprocessed' / f'dem_{raster.id}'
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Output dir: {output_dir}")

filled_path = str(output_dir / 'filled_dem.tif')
print(f"Output file: {filled_path}")

# Test preprocessing
preprocessor = DEMPreprocessor()
print("\nFilling depressions...")
result = preprocessor.fill_depressions(dem_path, filled_path)

print(f"\nResult: {result}")
print(f"Output exists: {os.path.exists(filled_path)}")

if os.path.exists(filled_path):
    print(f"File size: {os.path.getsize(filled_path)} bytes")
