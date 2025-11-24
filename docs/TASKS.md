# Development Tasks

## Hydropower Potential Mapping System - Task Breakdown

Based on PRD v1.0 (2025-11-13)

---

## Phase 0: Environment & Data Setup (Week 1)

### Django Project Setup
- [x] Initialize Django project structure
- [x] Configure Django settings for development and production environments
- [x] Set up Python virtual environment and activate
- [x] Install Django and core dependencies
- [x] Configure static files and media directories
- [x] Set up project-level URLs and base templates

### Database Setup
- [x] Install PostgreSQL 14+ with PostGIS extension
- [x] Create database and enable PostGIS
- [x] Configure Django database connection (PostgreSQL + PostGIS)
- [ ] Test PostGIS queries from Django shell
- [x] Create initial Django migrations
- [x] Set up database backup strategy

### GIS Libraries & Dependencies
- [x] Install GDAL/OGR libraries (system-level) - GDAL 3.8.4 with 210 drivers
- [x] Install PROJ library for coordinate transformations - PROJ 9.5.1 via pyproj 3.7.2
- [x] Install Python GIS packages: `rasterio`, `geopandas`, `shapely` - rasterio 1.4.3, geopandas 1.1.1, shapely 2.1.2
- [x] Install `whitebox` for hydrological processing - whitebox 2.3.6
- [x] Install `pandas` for time series handling - pandas 2.3.3
- [x] Install `reportlab` for PDF generation - reportlab 4.4.4
- [x] Verify all library installations and compatibility - All 9 libraries verified and working

### Sample Data Loading
- [x] Load `DEM.tif` from `INPUT DATA/WATERSHED DATA/Terrain Data/` - 2022x1798 pixels, 10m resolution, 0-1627m elevation
- [x] Load shapefiles: `Subbasins_MatiaoRiver.*`, `Matiao Bridge.*` - 5 subbasins, 1 bridge point, EPSG:32651
- [x] Load projection file `32651.prj` (EPSG:32651) - WGS 84 / UTM Zone 51N
- [x] Load rainfall/discharge Excel files from `INPUT DATA/RAINFALL & DISCHARGE DATA/` - 98 records (15-min intervals), historical data (1951-2008+)
- [x] Validate CRS consistency across all sample data - All data in WGS 84 UTM Zone 51N (DEM lacks EPSG code but correct CRS)
- [x] Document sample data schema and structure - See docs/SAMPLE_DATA_SCHEMA.md

---

## Phase 1: Data Ingestion & Validation (Weeks 2-3)

### File Upload Infrastructure
- [x] Create Django app: `hydropower` (single unified app)
- [x] Design upload form UI with Bootstrap 5
- [x] Implement multi-file upload handler (DEM, shapefiles, CSV/Excel, HEC-HMS)
- [x] Configure file size limits (up to 2 GB for DEM)
- [x] Set up secure file storage directory structure
- [x] Implement file type validation (MIME type checking)
- [ ] Add virus/malware scanning integration
- [x] Sanitize uploaded filenames

### DEM Upload & Validation
- [x] Create DEM upload endpoint
- [x] Validate GeoTIFF format using `rasterio`
- [x] Extract DEM metadata: bounds, resolution, CRS, nodata value
- [x] Detect and validate CRS from `.tif` metadata
- [x] Compute basic statistics: min/max elevation, mean, std dev
- [x] Generate DEM preview/thumbnail
- [x] Store DEM metadata in `RasterLayer` model
- [x] Handle DEM with missing CRS (prompt user for EPSG code)

### Shapefile Upload & Validation
- [x] Create shapefile upload endpoint (accept .shp, .shx, .dbf, .prj as set)
- [x] Validate shapefile completeness (all required files present)
- [x] Parse shapefile using `geopandas`
- [x] Extract CRS from `.prj` file
- [x] Validate geometry types (Point, LineString, Polygon)
- [x] Check for invalid/null geometries
- [x] Store vector metadata in `VectorLayer` model
- [x] Reproject shapefile to EPSG:32651 if CRS differs
- [x] Handle missing `.prj` file (require user CRS input)

### Time Series Upload & Validation
- [x] Create time series upload endpoint (CSV/Excel)
- [x] Parse CSV files using `pandas`
- [x] Parse Excel files using `pandas` (`.xlsx` support)
- [x] Validate datetime column format
- [x] Validate numeric data columns (discharge, rainfall)
- [x] Extract units metadata (m³/s, mm, etc.)
- [x] Validate station/source IDs
- [x] Store time series in `TimeSeries` model
- [x] Handle missing values and data gaps

### HEC-HMS CSV Integration
- [x] Define HEC-HMS CSV schema (element ID, datetime, discharge)
- [x] Create HMS CSV upload endpoint
- [x] Parse HMS CSV files
- [x] Map element IDs/names to spatial features
- [x] Extract event name and return period metadata
- [x] Validate discharge units (m³/s)
- [x] Store HMS data in `HMSRun` model
- [x] Link HMS data to time series table

### CRS Handling
- [x] Implement CRS detection from raster/vector files
- [x] Create CRS selection UI (dropdown with common EPSG codes)
- [x] Implement automatic reprojection to EPSG:32651 using `geopandas`/`rasterio`
- [x] Log CRS transformations for audit trail
- [x] Validate reprojected geometries

### Dataset Management UI
- [x] Create dataset list view (Bootstrap 5 table)
- [x] Display dataset metadata: name, type, size, CRS, upload date
- [x] Add dataset detail view
- [x] Implement dataset delete functionality
- [x] Add dataset tagging/labeling
- [x] Show upload validation status and errors
- [x] Add dataset search/filter functionality

---

## Phase 2: Core Processing (Weeks 4-6)

### DEM Preprocessing
- [x] Implement DEM fill algorithm (depression filling using `whitebox`)
- [x] Implement DEM breach algorithm for flow continuity
- [x] Compute D8 flow direction using `whitebox`
- [x] Compute flow accumulation from flow direction
- [x] Optional: implement DEM smoothing/filtering
- [x] Cache preprocessed rasters for reuse
- [x] Store preprocessing results in PostGIS or file system

### Watershed Delineation
- [x] Define accumulation threshold parameter (configurable)
- [x] Extract stream network based on accumulation threshold
- [x] Delineate watershed boundaries using `whitebox`
- [x] Vectorize stream network to PostGIS LineString layer
- [x] Vectorize watershed boundaries to PostGIS Polygon layer
- [x] Compute watershed statistics (area, perimeter, stream length)
- [x] Validate watershed outputs against input shapefiles

### Stream Network Extraction
- [x] Generate stream nodes (confluences, outlets)
- [x] Assign stream orders (Strahler or Shreve)
- [x] Extract stream segments between nodes
- [x] Store stream network in PostGIS with topology
- [x] Index stream segments for spatial queries

### Inlet–Outlet Pairing Algorithm
- [x] Define configurable parameters: min/max head, min river distance, spacing buffer
- [x] Identify inlet candidates (upstream points on stream network)
- [x] Identify outlet candidates (downstream points on stream network)
- [x] Implement head calculation: H = z_inlet - z_outlet from DEM
- [x] Apply minimum head constraint filter
- [x] Apply minimum river distance constraint
- [x] Apply land proximity to riverbanks constraint
- [x] For each inlet, search downstream for feasible outlets
- [x] Implement head-maximizing or multi-criteria scoring
- [x] De-duplicate overlapping site pairs
- [x] Enforce spacing buffer between sites
- [x] Store inlet/outlet points in `SitePoint` model (PostGIS Point geometry)
- [x] Store site pairs in `SitePair` model (PostGIS LineString geometry)

### Head Calculation
- [x] Extract elevation at inlet coordinates from DEM
- [x] Extract elevation at outlet coordinates from DEM
- [x] Compute head: H = z_inlet - z_outlet
- [x] Validate head values (positive, within expected range)
- [x] Store head in `SitePair` model

### Discharge Association
- [x] Link HEC-HMS discharge data to spatial features by element ID
- [x] Extract representative Q (peak flow or event-average, configurable)
- [x] Handle multiple return periods/scenarios
- [x] Assign discharge to site pairs based on spatial proximity
- [x] Store Q in `SitePair` model
- [ ] **TEST: Run site pairing WITH HMS discharge association (HMSRun ID=1) to populate discharge and power fields**

### Power Computation
- [x] Define constants: ρ = 1000 kg/m³, g = 9.81 m/s²
- [x] Define configurable efficiency factor η (default 0.7, range 0.6–0.85)
- [x] Implement power equation: P = ρ × g × Q × H × η
- [x] Convert power to kW
- [x] Store power in `SitePair` model
- [x] Validate power values (non-negative, within feasible range)

### Asynchronous Job Processing
- [x] Set up Celery for async task processing (optional)
- [x] Configure Redis as Celery broker (optional)
- [x] Create processing job model: `ProcessingRun`
- [x] Implement job status tracking (queued/running/succeeded/failed)
- [x] Create job submission endpoint
- [x] Create job status polling endpoint
- [x] Implement job cancellation
- [x] Log processing errors and warnings
- [x] Store job parameters (JSON) in `ProcessingRun` model
- [x] Link job outputs to datasets

### Result Persistence
- [x] Write site points to PostGIS `sites_point` table
- [x] Write site pairs to PostGIS `sites_pair_line` table
- [x] Create spatial indexes on geometry columns
- [x] Store source references (DEM cell IDs, HMS element/time)
- [x] Compute and store result checksums for reproducibility
- [x] Create job lineage/provenance metadata

---

## Phase 3: Visualization & Mapping (Weeks 7-8)

### Bootstrap 5 Integration
- [x] Include Bootstrap 5 CSS/JS (CDN or bundled)
- [x] Create base template with Bootstrap navbar
- [x] Design responsive grid layout for upload/process/map pages
- [x] Implement collapsible sidebar for filters on small screens
- [x] Style forms, buttons, and tables with Bootstrap components
- [x] Add loading spinners and progress indicators
- [x] Test responsive behavior on mobile (≥320px), tablet, desktop

### Map Setup
- [x] Choose map library: Leaflet.js or OpenLayers
- [x] Include Leaflet/OpenLayers CSS/JS
- [x] Create map container in Django template
- [x] Initialize map with base layer (OSM, satellite, etc.)
- [x] Set initial map center and zoom (Matiao, Pantukan, Davao de Oro area)
- [x] Configure map CRS (display in EPSG:3857 Web Mercator)

### Map Layers
- [x] Create vector tile endpoint for watershed boundaries (PostGIS → GeoJSON)
- [x] Create vector tile endpoint for stream network (PostGIS → GeoJSON)
- [x] Create vector tile endpoint for subbasins (PostGIS → GeoJSON)
- [x] Create vector tile endpoint for bridges/POIs (PostGIS → GeoJSON)
- [x] Create vector tile endpoint for site pairs (PostGIS → GeoJSON)
- [x] Implement on-the-fly CRS transformation (EPSG:32651 → EPSG:3857)
- [x] Add watershed boundary layer to map
- [x] Add stream network layer to map
- [x] Add subbasin layer to map
- [x] Add bridges/POIs layer to map
- [x] Add site pair layer (inlet–outlet lines and point markers)

### Map Styling
- [x] Style watershed boundaries (fill, stroke, opacity)
- [x] Style stream network (color by stream order, width)
- [x] Style subbasins (distinct colors or patterns)
- [x] Style bridges/POIs (icon markers)
- [x] Style site pairs: inlet/outlet points (distinct colors) and connecting lines
- [x] Add color ramp for power output classification
- [x] Create map legend (Bootstrap card component)

### Layer Controls
- [x] Implement layer visibility toggles (checkboxes)
- [x] Add base layer switcher (OSM, satellite, topo)
- [x] Implement layer opacity sliders
- [x] Group layers logically (data layers, result layers)

### Popups & Tooltips
- [x] Implement click event on site pair layer
- [x] Show popup with inlet/outlet coordinates
- [x] Display head (m), discharge (m³/s), power (kW) in popup
- [x] Link to source metadata (DEM, HMS element)
- [ ] **TODO: Test popups with HMS discharge data** (currently discharge=null, power=null)
- [x] Add tooltip on hover (site ID, power)
- [x] Format popup content with Bootstrap styles

### Filter Panel
- [x] Create filter sidebar (Bootstrap offcanvas or card)
- [x] Add minimum head slider/input
- [x] Add minimum discharge slider/input
- [x] Add minimum power slider/input
- [x] Add scenario/return period dropdown (from HMS data)
- [x] Add spatial extent filter (draw bounding box on map)
- [x] **COMPLETE: Connect filter sliders to API endpoint** (form submit handler prevents page refresh)
- [x] Implement server-side filtering for large datasets (API endpoint)
- [x] **COMPLETE: Update map layers when filters change** (auto-apply on slider change + manual "Apply Filters" button)
- [x] Show filter result count (displays "888 sites match filters" when filtered)
- [x] **BROWSER TESTED (COMPLETE)**: 
  - ✅ Power filter: 1,146 sites → 334 sites (min_power >= 5000 kW)
  - ✅ Form submit handler working (no page refresh on "Apply Filters" click)
  - ✅ Badge updates correctly ("334 sites match filters")
  - ✅ Map clusters refresh dynamically
  - ✅ "Reset All" button resets sliders to 0
  - ⚠️ Note: Auto-apply may not trigger for manual slider drag (use "Apply Filters" button)

### Performance Optimization
- [x] Implement clustering for dense site markers (Leaflet.markercluster)
- [x] Use vector tiles for large datasets (PostGIS → MVT)
- [x] Implement layer generalization for zoom levels
- [x] Cache map tiles and GeoJSON responses (Redis optional)
- [x] Lazy-load layers on demand
- [x] Optimize PostGIS queries with spatial indexes
- [x] Target map load < 3 seconds with 5k features

---

## Phase 4: Results & Export (Week 9)

### Tabular Results View
- [ ] Create results table page (Bootstrap 5 DataTable or custom table)
- [ ] Display site pairs in table: ID, inlet coords, outlet coords, H, Q, P
- [ ] Implement table sorting (by H, Q, P, etc.)
- [ ] Implement table filtering (search box, column filters)
- [ ] Implement pagination (server-side for large datasets)
- [ ] Add "View on Map" button for each site pair
- [ ] Style table with Bootstrap striped/hover classes

### CSV Export
- [ ] Create CSV export endpoint
- [ ] Generate CSV with all site attributes (ID, coords, H, Q, P, source refs)
- [ ] Include header row with column names
- [ ] Handle UTF-8 encoding
- [ ] Stream large CSV files for memory efficiency
- [ ] Add "Export CSV" button to results page
- [ ] Target export completion < 10 s for 10k features

### GeoJSON Export
- [ ] Create GeoJSON export endpoint
- [ ] Generate GeoJSON FeatureCollection with site pair geometries and attributes
- [ ] Include CRS metadata in GeoJSON
- [ ] Validate GeoJSON structure
- [ ] Add "Export GeoJSON" button to results page
- [ ] Target export completion < 10 s for 10k features

### PDF Report Generation
- [ ] Set up `reportlab` for PDF generation
- [ ] Design PDF report template: title, summary, map snapshot, charts, table
- [ ] Generate map snapshot (static image from Leaflet/OpenLayers)
- [ ] Include summary metrics: total sites, avg head, avg power, classification counts
- [ ] Add classification charts (distributions of H, Q, P)
- [ ] Include tabular results (top N sites or all sites)
- [ ] Add report metadata: date, parameters, data sources
- [ ] Style PDF with consistent fonts and branding
- [ ] Add "Export PDF" button to results page
- [ ] Handle long-running PDF generation as background job

### Chart Generation
- [ ] Include Chart.js library
- [ ] Create histogram for head distribution
- [ ] Create histogram for discharge distribution
- [ ] Create histogram for power distribution
- [ ] Create bar chart for power classification (low/medium/high)
- [ ] Add charts to results page
- [ ] Include charts in PDF report
- [ ] Make charts interactive (hover tooltips)

### Background Export Jobs
- [ ] Implement async export for large datasets (Celery)
- [ ] Show export job status (queued/running/succeeded/failed)
- [ ] Notify user when export is ready (email or UI notification)
- [ ] Store export files in `Export` model with file path
- [ ] Provide download link for completed exports
- [ ] Auto-delete old export files (configurable retention period)

---

## Phase 5: Hardening & Validation (Weeks 10-11)

### Quality Assurance
- [ ] Write unit tests for upload validation functions
- [ ] Write unit tests for DEM preprocessing functions
- [ ] Write unit tests for inlet–outlet pairing algorithm
- [ ] Write unit tests for head and power calculations
- [ ] Write integration tests for upload → process → visualize → export flow
- [ ] Test with multiple DEM sizes and resolutions
- [ ] Test with various shapefile formats and CRS
- [ ] Test with different HEC-HMS CSV schemas
- [ ] Validate results against manual calculations (benchmark sites)
- [ ] Perform regression testing on sample data

### Performance Testing
- [ ] Benchmark DEM preprocessing time for 2 GB DEM
- [ ] Benchmark site pairing algorithm for 10k candidates
- [ ] Benchmark map load time with 5k site features
- [ ] Benchmark filter update time (client-side and server-side)
- [ ] Benchmark export time for 10k features (CSV, GeoJSON, PDF)
- [ ] Profile database queries and optimize slow queries
- [ ] Load test with concurrent users (optional)
- [ ] Optimize critical paths to meet performance targets

### Security Review
- [ ] Validate all file upload inputs (type, size, content)
- [ ] Implement CSRF protection for all forms
- [ ] Sanitize file names and paths to prevent directory traversal
- [ ] Validate and sanitize all user inputs (OWASP Top 10)
- [ ] Implement rate limiting on upload/processing endpoints
- [ ] Add file size quotas for anonymous users
- [ ] Review GDAL/rasterio calls for injection vulnerabilities
- [ ] Test virus scanning integration
- [ ] Review Django security settings (SECRET_KEY, DEBUG, ALLOWED_HOSTS)
- [ ] Enable HTTPS in production

### Data Integrity & Lineage
- [ ] Implement file checksums (SHA256) for uploaded files
- [ ] Store checksums in `Dataset` model
- [ ] Validate checksums on file retrieval
- [ ] Track CRS transformations in audit log
- [ ] Store processing parameters in `ProcessingRun` model
- [ ] Link outputs to source datasets (lineage tracking)
- [ ] Ensure reproducibility: same inputs + params → same results

### Logging & Monitoring
- [ ] Configure structured logging (JSON logs)
- [ ] Log all upload events (file, user, timestamp)
- [ ] Log all processing jobs (start, end, status, errors)
- [ ] Log CRS transformations and reprojects
- [ ] Log export events
- [ ] Set up error monitoring (Sentry or similar)
- [ ] Monitor database query performance (Django Debug Toolbar in dev)
- [ ] Create job metrics dashboard (optional)
- [ ] Set up alerts for failed jobs or errors

### Accessibility
- [ ] Ensure WCAG AA compliance for all UI components
- [ ] Add keyboard navigation for map controls
- [ ] Add ARIA labels for screen readers
- [ ] Test with screen reader (NVDA or JAWS)
- [ ] Ensure sufficient color contrast (4.5:1 ratio)
- [ ] Add alt text for images and map snapshots
- [ ] Test navigation with keyboard only

### Documentation
- [ ] Write user guide: Upload → Process → Visualize → Export workflow
- [ ] Document supported file formats and size limits
- [ ] Document CRS handling and EPSG:32651 requirement
- [ ] Document configurable parameters (thresholds, efficiency factor)
- [ ] Write developer documentation: setup, architecture, data model
- [ ] Document API endpoints (upload, processing, export)
- [ ] Add code comments and docstrings
- [ ] Create README.md with quick start instructions
- [ ] Document deployment steps (Django, PostgreSQL, GDAL, env setup)
- [ ] Align documentation with academic write-up

### Academic Alignment
- [ ] Validate head calculation matches Eq. 8 in research paper
- [ ] Validate power calculation matches Eq. 10 in research paper
- [ ] Confirm efficiency factor range and default (0.6–0.85, default 0.7)
- [ ] Validate site pairing algorithm matches research methodology
- [ ] Compare results with manual GIS analysis (if available)
- [ ] Document deviations or assumptions
- [ ] Prepare figures/tables for academic publication

### Deployment Preparation
- [ ] Set up production environment (server, OS, Python, PostgreSQL, GDAL)
- [ ] Configure environment variables (SECRET_KEY, DATABASE_URL, etc.)
- [ ] Set up static file serving (collectstatic, Whitenoise or nginx)
- [ ] Set up media file storage (local or cloud)
- [ ] Configure PostgreSQL with PostGIS in production
- [ ] Set up GDAL/PROJ in production environment
- [ ] Configure Celery and Redis (if used)
- [ ] Set up HTTPS with SSL certificate
- [ ] Configure backup and restore procedures
- [ ] Test deployment on staging environment
- [ ] Create deployment checklist

---

## Public Deployment Considerations (Open Access)

### Rate Limiting & Quotas
- [ ] Implement IP-based rate limiting for upload endpoints
- [ ] Set max file uploads per IP per hour
- [ ] Set max processing jobs per IP per day
- [ ] Display quota status to users
- [ ] Add CAPTCHA for upload form (optional, to prevent bots)

### Resource Management
- [ ] Set max concurrent processing jobs
- [ ] Implement job queue prioritization
- [ ] Auto-cleanup old datasets (configurable retention, e.g., 7 days)
- [ ] Auto-cleanup old exports (configurable retention, e.g., 24 hours)
- [ ] Monitor disk usage and alert on low space

### User Experience
- [ ] Add help text and tooltips throughout UI
- [ ] Create FAQ page
- [ ] Add example dataset download link
- [ ] Show processing time estimates
- [ ] Add "Contact Us" or feedback form (optional)

---

## Optional Future Enhancements (Post-v1)

- [ ] Add user authentication and role-based access control
- [ ] Add admin dashboard for dataset management
- [ ] Support HEC-HMS DSS file format
- [ ] Implement network optimization for site selection
- [ ] Add cost estimation module
- [ ] Integrate environmental/social impact assessment
- [ ] Add detailed energy economics calculations
- [ ] Implement automated HEC-HMS model calibration
- [ ] Support multiple river basins
- [ ] Add real-time monitoring integration
- [ ] Create mobile app

---

## Risk Mitigation Tasks

### Large DEM Handling
- [ ] Implement DEM tiling for processing
- [ ] Use raster overviews for visualization
- [ ] Optimize stream threshold tuning UI
- [ ] Implement chunked processing for large rasters

### CRS Mismatch Handling
- [ ] Mandatory CRS selection/confirmation UI
- [ ] Automated reproject with user consent
- [ ] Audit log for all CRS transformations
- [ ] CRS validation before processing

### HMS Data Variability
- [ ] Define strict CSV schema and document
- [ ] Create HMS CSV template for download
- [ ] Implement element name mapping adapter
- [ ] Support multiple HMS export formats

---

## Definition of Done

Each task is considered complete when:
- [ ] Code is written and tested
- [ ] Unit tests pass (where applicable)
- [ ] Integration tests pass (where applicable)
- [ ] Code review completed (if team workflow)
- [ ] Documentation updated
- [ ] Acceptance criteria met (per PRD Section 4)
- [ ] No critical bugs or blockers
- [ ] Deployed to staging/dev environment and verified

---

**Total Estimated Duration: 10–11 weeks**

**Next Steps:**
1. Review and prioritize tasks with stakeholders
2. Assign tasks to team members (or self if solo)
3. Set up project tracking (GitHub Issues, Jira, Trello, etc.)
4. Begin Phase 0: Environment & Data Setup
