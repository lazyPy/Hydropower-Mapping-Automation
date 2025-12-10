import os
import sys
import django

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'HYDROPOWER_MAPPING.settings')
django.setup()

from hydropower.models import StreamNetwork, RasterLayer
import statistics

rl = RasterLayer.objects.filter(watershed_delineated=True).order_by('-id').first()
streams = StreamNetwork.objects.filter(raster_layer=rl)

print(f'Total streams: {streams.count()}')
lengths = [s.length_m for s in streams]
print(f'Length (m): min={min(lengths):.1f}, max={max(lengths):.1f}')
print(f'Mean: {statistics.mean(lengths):.1f}m, Median: {statistics.median(lengths):.1f}m')
print(f'Total network length: {sum(lengths):.1f}m ({sum(lengths)/1000:.2f}km)')

# Check how many segments > 100m
long_streams = [l for l in lengths if l > 100]
print(f'\nSegments > 100m: {len(long_streams)} ({len(long_streams)/len(lengths)*100:.1f}%)')

# Check how many segments > 200m
very_long = [l for l in lengths if l > 200]
print(f'Segments > 200m: {len(very_long)} ({len(very_long)/len(lengths)*100:.1f}%)')
