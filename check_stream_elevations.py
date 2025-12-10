"""Check if stream network has elevation data."""
import django
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'HYDROPOWER_MAPPING.settings')
django.setup()

from pathlib import Path
from django.conf import settings
import geopandas as gpd
from hydropower.models import SitePoint, RasterLayer

# Get first inlet
sp = SitePoint.objects.filter(raster_layer_id=97, site_type='INLET').first()
print(f"Inlet: {sp.site_id}")
print(f"  Geometry: {sp.geometry}")
print(f"  Elevation: {sp.elevation}m")
print()

# Load stream network
rl = RasterLayer.objects.get(id=97)
gpkg_path = Path(settings.MEDIA_ROOT) / 'preprocessed' / f'dem_{rl.id}' / 'streams.gpkg'
print(f"Stream GPKG: {gpkg_path}")

streams = gpd.read_file(gpkg_path, layer='stream_network')
print(f"Stream network CRS: {streams.crs}")
print(f"Stream columns: {list(streams.columns)}")
print(f"Total stream features: {len(streams)}")

if 'elevation' in streams.columns:
    print(f"\nElevation stats:")
    print(f"  Min: {streams['elevation'].min():.1f}m")
    print(f"  Max: {streams['elevation'].max():.1f}m")
    print(f"  Mean: {streams['elevation'].mean():.1f}m")
    print(f"  Null count: {streams['elevation'].isna().sum()}")
else:
    print("\n❌ No elevation column in stream network!")

# Check stream nodes
nodes = gpd.read_file(gpkg_path, layer='stream_nodes')
print(f"\nStream nodes CRS: {nodes.crs}")
print(f"Stream node columns: {list(nodes.columns)}")
print(f"Total node features: {len(nodes)}")

if 'elevation' in nodes.columns:
    print(f"\nNode elevation stats:")
    print(f"  Min: {nodes['elevation'].min():.1f}m")
    print(f"  Max: {nodes['elevation'].max():.1f}m")
    print(f"  Mean: {nodes['elevation'].mean():.1f}m")
    print(f"  Null count: {nodes['elevation'].isna().sum()}")
    
    # Find node closest to inlet
    inlet_geom = sp.geometry
    nodes['dist'] = nodes.geometry.distance(inlet_geom)
    closest = nodes.loc[nodes['dist'].idxmin()]
    print(f"\nClosest node to inlet:")
    print(f"  Node ID: {closest.get('node_id', 'N/A')}")
    print(f"  Distance: {closest['dist']:.1f}m")
    print(f"  Elevation: {closest.get('elevation', 'N/A'):.1f}m")
    print(f"  Geometry: {closest.geometry}")
else:
    print("\n❌ No elevation column in stream nodes!")
