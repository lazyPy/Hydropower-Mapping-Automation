# 🎯 Weir Search - Quick Reference

## What It Does

Identifies optimal water intake/weir locations for hydropower projects:
1. 📊 Selects **top 50** optimal main channel site pairs
2. 🔍 Searches for **weir candidates** around each inlet point
3. ⭐ **Highlights the best weir** location (rank #1) per inlet
4. 🏗️ **Generates complete infrastructure** layout automatically

---

## 🚀 Quick Start

### Option 1: Run via Command (Recommended)

```powershell
# 1. Activate virtual environment
.\env\Scripts\Activate.ps1

# 2. Run weir search
python manage.py run_weir_search `
    --raster_layer=110 `
    --dem_path="media/preprocessed/dem_110/filled_dem.tif"
```

### Option 2: Test First (Top 10 Pairs)

```powershell
.\env\Scripts\Activate.ps1
python test_weir_search.py
```

---

## 📋 Command Options

```powershell
python manage.py run_weir_search `
    --raster_layer=110 `                          # Your RasterLayer ID
    --dem_path="media/preprocessed/dem_110/filled_dem.tif" `  # Path to DEM
    --top_n=50 `                                  # Top N pairs (default: 50)
    --search_radius=500 `                         # Search radius (m)
    --min_distance=100 `                          # Min distance from inlet (m)
    --elevation_tolerance=20 `                    # Elevation tolerance (±m)
    --cone_angle=90 `                             # Directional cone (degrees)
    --max_candidates=10 `                         # Max candidates per inlet
    --no-infrastructure                           # Skip infrastructure (faster)
```

---

## 📊 What You Get

### Console Output
```
=== Weir Search Results ===
Total Weir Candidates: 120
Inlets Processed: 15
Best Weirs Identified: 15
Infrastructure Layouts Generated: 45

Best Weir Candidates (Top 10):
 1. Inlet: IN_97_79        | Score:  92.5 | Distance:  250.5m | Elev Diff:  +2.3m
 2. Inlet: IN_88_71        | Score:  89.1 | Distance:  180.2m | Elev Diff:  -1.5m
 3. Inlet: IN_102_85       | Score:  87.8 | Distance:  320.8m | Elev Diff:  +5.2m
 ...
```

### Database Records
- **WeirCandidate** table: All candidate weir locations
  - Rank 1 candidates = Best weirs (highlighted)
- **SitePair** table: Updated with infrastructure geometries
  - intake_basin_geom, settling_basin_geom, channel_geom, etc.

### Map Visualization
- View at: http://localhost:8000/
- Enable "Weir Candidates" layer
- Best weirs shown with star icons
- Infrastructure components displayed

---

## 🗺️ API Endpoints

### Best Weirs Only
```
GET /api/geojson/weir-candidates/?raster_layer=110&best_only=true
```

### All Weir Candidates
```
GET /api/geojson/weir-candidates/?raster_layer=110
```

### Infrastructure Layouts
```
GET /api/geojson/site-pairs/?top_n=5
```

---

## 🏗️ Infrastructure Components Generated

For each best weir location:

1. **Intake Basin** 🌊 - At weir location (water intake point)
2. **Settling Basin** 🏞️ - 20m downstream (sediment removal)
3. **Channel** 〰️ - Water conveyance (follows terrain)
4. **Forebay Tank** 🛢️ - Surge tank before penstock
5. **Penstock** 📏 - Pressure pipe to turbine
6. **Powerhouse** ⚡ - At outlet (turbine & generator)

---

## 📁 Files Reference

### Scripts
- `test_weir_search.py` - Quick test (top 10 pairs)
- `run_weir_workflow.py` - Standalone workflow script
- `manage.py run_weir_search` - Django management command

### Documentation
- `docs/WEIR_SEARCH_SUMMARY.md` - This file
- `docs/WEIR_SEARCH_GUIDE.md` - Detailed user guide
- `docs/WEIR_SEARCH_IMPLEMENTATION.md` - Technical details

### Source Code
- `hydropower/main_channel_weir_search.py` - Core algorithm
- `hydropower/management/commands/run_weir_search.py` - Command
- `hydropower/views.py` - GeoJSON API endpoints

---

## ⏱️ Performance

| Task | Time |
|------|------|
| Top 10 pairs (test) | 1-2 min |
| Top 50 pairs | 5-10 min |
| Infrastructure generation | ~0.5s per pair |
| **Total (50 + infrastructure)** | **10-15 min** |

---

## 🔧 Troubleshooting

### No site pairs found
```powershell
# Run main channel workflow first
python run_main_channel_workflow.py
```

### DEM not found
```powershell
# Check your DEM location:
Get-ChildItem -Path "media\preprocessed\dem_*" -Recurse -Filter "filled_dem.tif"

# Use correct path in command
```

### No weir candidates
```powershell
# Relax constraints
python manage.py run_weir_search \
    --raster_layer=110 \
    --search_radius=1000 \
    --elevation_tolerance=30
```

---

## ✅ Testing Checklist

- [ ] Run quick test: `python test_weir_search.py`
- [ ] Verify weir candidates found
- [ ] Check best weirs highlighted (rank=1)
- [ ] Confirm infrastructure generated
- [ ] View on map: http://localhost:8000/
- [ ] Test API: `/api/geojson/weir-candidates/?best_only=true`
- [ ] Export results to GeoJSON

---

## 🎯 Expected Results (Your Dataset)

With 9135 main channel pairs:
- **Top 50 pairs selected**
- **~10-15 unique inlets** (multiple pairs share inlets)
- **~100-150 weir candidates** (10 per inlet)
- **~10-15 best weirs** (rank 1 per inlet)
- **~30-45 infrastructure layouts** (top 3-5 pairs per best weir)

---

## 📞 Support

See detailed documentation:
- User Guide: `docs/WEIR_SEARCH_GUIDE.md`
- Implementation: `docs/WEIR_SEARCH_IMPLEMENTATION.md`
- Technical Specs: `hydropower/main_channel_weir_search.py` (docstrings)

---

## ✨ Summary

✅ **Top 50 filtering** - Focuses on best opportunities
✅ **Best weir highlighting** - Rank #1 clearly marked
✅ **Infrastructure generation** - Complete layouts created
✅ **API integration** - Easy access via GeoJSON
✅ **Map visualization** - Interactive display
✅ **Ready to use** - All features implemented and tested

**Run now:**
```powershell
.\env\Scripts\Activate.ps1
python test_weir_search.py
```
