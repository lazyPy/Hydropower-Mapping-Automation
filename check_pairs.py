import os
import sys
import django

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'HYDROPOWER_MAPPING.settings')
django.setup()

from hydropower.models import SitePair, RasterLayer

rl = RasterLayer.objects.filter(watershed_delineated=True).order_by('-id').first()
pairs = SitePair.objects.filter(raster_layer=rl)

print(f'RasterLayer ID: {rl.id}')
print(f'Total site pairs: {pairs.count()}')
print(f'Feasible pairs: {pairs.filter(is_feasible=True).count()}')
print(f'\nTop 5 by power:')
for p in pairs.order_by('-power')[:5]:
    print(f'  ID {p.id}: {p.power:.1f} kW, Head={p.head:.1f}m, Q={p.discharge:.3f} m³/s')

print(f'\nMost recent 5 pairs (by ID):')
for p in pairs.order_by('-id')[:5]:
    print(f'  ID {p.id}: {p.power:.1f} kW, Head={p.head:.1f}m, created={p.created_at}')
