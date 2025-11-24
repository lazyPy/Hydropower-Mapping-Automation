# GitHub Copilot Instructions

## Project: Hydropower Potential Mapping System

This document provides guidance for GitHub Copilot when working on this Django + PostGIS project.

---

## Python Virtual Environment

**CRITICAL: Always activate the virtual environment before running Django commands.**

### Virtual Environment Location
- Path: `d:\Desktop\Hydro HEC-HMS\env\`
- Python version: 3.x (as configured in `env/pyvenv.cfg`)

### Command Execution Rules

When running any Django management command, ALWAYS use the virtual environment activation:

**PowerShell (default shell):**
```powershell
.\env\Scripts\Activate.ps1; python manage.py <command>
```

### Package Installation

Always activate virtual environment before installing packages:

```powershell
.\env\Scripts\Activate.ps1; pip install <package_name>
```

---

## Documentation: Use Context7 MCP

**Always use the Context7 MCP server for fetching up-to-date library documentation.**

### When to Use Context7 MCP

Use Context7 (`mcp_upstash_conte_resolve-library-id` and `mcp_upstash_conte_get-library-docs`) when:
- User asks about Django features, APIs, or best practices
- Need documentation for Django REST framework
- Questions about PostgreSQL, PostGIS, or GeoDjango
- Looking up documentation for GIS libraries: rasterio, geopandas, shapely, whitebox
- Questions about Leaflet.js, OpenLayers, or Chart.js
- Need Bootstrap 5 documentation for layout/components
- Questions about pandas, reportlab, or Celery
- User mentions "latest docs" or "check documentation"

---

## Browser Testing: Use Playwright MCP

**Always use Playwright MCP server for browser-based testing and UI validation.**

### When to Use Playwright MCP

Use Playwright MCP (`mcp_microsoft_pla_browser_*` tools) when:
- User asks to test the web application UI
- Need to validate upload forms, map interactions, or export features
- Testing responsive Bootstrap layouts on different screen sizes
- Verifying map rendering (Leaflet/OpenLayers)
- Testing filter panels, popups, and tooltips
- Validating CSV/GeoJSON/PDF export downloads
- End-to-end testing of Upload → Process → Visualize → Export flow
- Taking screenshots for documentation or bug reports

### Playwright MCP Workflow

1. **Navigate to page**:
   ```
   mcp_microsoft_pla_browser_navigate(url="http://localhost:8000/upload/")
   ```

2. **Take snapshot** (preferred over screenshot for actions):
   ```
   mcp_microsoft_pla_browser_snapshot()
   ```

3. **Interact with elements** (click, type, etc.):
   ```
   mcp_microsoft_pla_browser_click(element="Upload DEM button", ref="...")
   mcp_microsoft_pla_browser_type(element="CRS input", ref="...", text="32651")
   ```

4. **Take screenshot** (for visual documentation):
   ```
   mcp_microsoft_pla_browser_take_screenshot(filename="upload-page.png", fullPage=true)
   ```

5. **Validate responsive design**:
   ```
   mcp_microsoft_pla_browser_resize(width=375, height=667)  # Mobile
   mcp_microsoft_pla_browser_resize(width=768, height=1024) # Tablet
   mcp_microsoft_pla_browser_resize(width=1920, height=1080) # Desktop
   ```

### Common Testing Scenarios

**Upload Form Testing:**
```
1. Navigate to upload page
2. Snapshot page
3. Click file upload button
4. Upload test DEM file
5. Verify validation messages
6. Take screenshot
```

**Map Testing:**
```
1. Navigate to map page
2. Wait for map to load
3. Snapshot map
4. Click on site marker
5. Verify popup content (head, discharge, power)
6. Test filter panel (adjust min head slider)
7. Take screenshot of filtered results
```

**Responsive Testing:**
```
1. Resize to mobile (375x667)
2. Verify filter panel collapses
3. Test map touch interactions
4. Take mobile screenshot
5. Resize to desktop (1920x1080)
6. Verify full layout
7. Take desktop screenshot
```

**Export Testing:**
```
1. Navigate to results page
2. Click "Export CSV" button
3. Wait for download
4. Verify file downloaded
5. Click "Export PDF" button
6. Wait for background job completion
```

---

## Django Project Structure

```
d:\Desktop\Hydro HEC-HMS\
├── env/                          # Virtual environment (ALWAYS activate)
├── INPUT DATA/                   # Sample data for testing
│   ├── WATERSHED DATA/
│   │   ├── Terrain Data/        # DEM.tif
│   │   ├── Bridge & River/      # Shapefiles
│   │   └── Projection File/     # 32651.prj
│   └── RAINFALL & DISCHARGE DATA/  # Excel/CSV files
├── docs/
│   ├── PRD.md                   # Product Requirements Document
│   └── TASKS.md                 # Development tasks
├── manage.py                     # Django management script
├── requirements.txt              # Python dependencies
└── <django_project>/            # Django project root (to be created)
    ├── settings.py
    ├── urls.py
    └── wsgi.py
```

---

## Coding Standards

### Django Apps to Create

Based on PRD and TASKS.md:
- `data_upload` - File upload and validation
- `gis_processing` - DEM preprocessing, watershed delineation, site pairing
- `visualization` - Map rendering, filters, charts
- `exports` - CSV/GeoJSON/PDF generation

### Database Models (PostGIS)

All spatial models use EPSG:32651 (UTM Zone 51N):
- `Dataset` - Uploaded files metadata
- `ProcessingRun` - Job tracking
- `RasterLayer` - DEM metadata
- `VectorLayer` - Shapefile metadata
- `TimeSeries` - Rainfall/discharge data
- `HMSRun` - HEC-HMS simulation metadata
- `SitePoint` - Inlet/outlet points (PostGIS Point, SRID=32651)
- `SitePair` - Inlet-outlet pairs (PostGIS LineString, SRID=32651)
- `Export` - Export files metadata

### Code Style

- Follow PEP 8 for Python code
- Use Django best practices (class-based views, model managers, etc.)
- Add docstrings to all functions and classes
- Use type hints where applicable
- Comment complex GIS algorithms
- Use Bootstrap 5 classes for all UI components
- Follow responsive design principles (mobile-first)

---

## Testing Guidelines

### Unit Tests

Always write unit tests for:
- Upload validation functions
- CRS detection and reprojection
- DEM preprocessing algorithms
- Head and power calculations
- Inlet-outlet pairing algorithm

**Run tests:**
```powershell
.\env\Scripts\Activate.ps1; python manage.py test
```

### Integration Tests

Test complete workflows:
- Upload → Validation → Storage
- Upload → Process → Results
- Results → Export (CSV/GeoJSON/PDF)

### Browser Tests (Playwright MCP)

Use Playwright MCP for:
- UI component testing
- Responsive layout validation
- Map interaction testing
- End-to-end workflow testing

---

## Performance Requirements (from PRD)

- Map initial load: < 3 seconds (5k features)
- Client-side filters: < 500 ms
- Server-side filters: < 2 s
- CSV/GeoJSON export: < 10 s (10k features)
- PDF generation: background job for large datasets
- DEM processing: < 30 min end-to-end (standard dataset)

---

## Security Checklist

- ✅ CSRF protection on all forms
- ✅ File upload validation (type, size, content)
- ✅ Filename sanitization (prevent directory traversal)
- ✅ OWASP input validation
- ✅ Antivirus scan on uploads
- ✅ Rate limiting (IP-based)
- ✅ File size quotas
- ❌ No authentication/RBAC in v1 (open access)

---

## Deployment Notes

- OS: Windows
- Shell: PowerShell 5.1
- Python: 3.x (via venv)
- Database: PostgreSQL 14+ with PostGIS extension
- GDAL/PROJ: System-level installation required
- CRS: EPSG:32651 (processing), EPSG:3857 (map display)
- No authentication in v1 (open access)

---

## Quick Reference Commands

```powershell
# Start Django development server
.\env\Scripts\Activate.ps1; python manage.py runserver

# Create database migrations
.\env\Scripts\Activate.ps1; python manage.py makemigrations

# Apply migrations
.\env\Scripts\Activate.ps1; python manage.py migrate

# Create Django app
.\env\Scripts\Activate.ps1; python manage.py startapp <app_name>

# Django shell (test queries, models)
.\env\Scripts\Activate.ps1; python manage.py shell

# Run tests
.\env\Scripts\Activate.ps1; python manage.py test

# Install package
.\env\Scripts\Activate.ps1; pip install <package>

# Freeze dependencies
.\env\Scripts\Activate.ps1; pip freeze > requirements.txt
```

---

## Summary

1. **Always activate `.\env\Scripts\Activate.ps1` before Python commands**
2. **Use Context7 MCP for up-to-date library documentation**
3. **Use Playwright MCP for browser testing and UI validation**
4. **Follow Django + PostGIS best practices**
5. **Test responsive Bootstrap 5 layouts on multiple screen sizes**
6. **No authentication in v1 (open access model)**
7. **Follow PRD and TASKS.md for feature requirements**
8. **Do NOT create summary documentation after completing tasks** - This wastes tokens. Only update TASKS.md checkboxes and provide brief confirmation to user.
