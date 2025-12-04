#!/usr/bin/env python
"""Check discharge distribution across site pairs."""

import os
import sys
from pathlib import Path

# Set up Django environment
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'HYDROPOWER_MAPPING.settings')

import django
django.setup()

from collections import Counter
from hydropower.models import SitePair

# Get all discharge values
discharges = list(SitePair.objects.values_list('discharge', flat=True))
print(f"Total site pairs: {len(discharges)}")
print(f"Unique discharge values: {len(set(discharges))}")
print()

# Count occurrences
c = Counter(discharges)
print("Discharge value counts:")
for d, count in sorted(c.items()):
    print(f"  {d:.4f} m³/s: {count} sites")

print()
print("Sites with duplicate discharge values:")
for d, count in sorted(c.items()):
    if count > 1:
        sites = SitePair.objects.filter(discharge=d).values_list('pair_id', flat=True)
        print(f"  {d:.4f} m³/s ({count} sites): {list(sites)}")
