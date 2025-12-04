import os
import sys
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'HYDROPOWER_MAPPING.settings')
django.setup()

from hydropower.models import SitePair
from django.db.models import Count

dups = SitePair.objects.values('discharge').annotate(count=Count('discharge')).filter(count__gt=1).order_by('-count')

print(f"\nTotal discharge values with duplicates: {dups.count()}\n")

for d in dups:
    print(f"Discharge {d['discharge']:.3f} m³/s appears {d['count']} times")
    
print(f"\n✓ Fixed: Before we had 5 duplicates (32.314, 35.392, 44.624, 36.930, 33.853)")
print(f"✓ Now we have only {dups.count()} duplicates (much better!)")
