# Product Requirements Document (PRD)

## Hydropower Potential Mapping using HEC-HMS and Head-Matched Inlet–Outlet Site Pairing along Matiao, Pantukan, Davao de Oro, Philippines Riverbanks

Date: 2025-11-13  
Owner: Product (Technical PM)  
Engineering: Full-stack (Django/PostGIS), GIS/Geo, Data/HEC-HMS  
Stakeholders: Research team, LGU/Policy, Academe partners

---

## 1. Project Overview

This project delivers a web application that automates identification and mapping of hydropower potential sites along the Matiao, Pantukan, Davao de Oro, Philippines riverbanks. It integrates hydrologic simulations from HEC-HMS, deterministic Python algorithms, and GIS-based spatial analysis on a Django + PostgreSQL/PostGIS stack. The system ingests topographic and hydrometeorological data, computes flow direction/accumulation and watershed delineation, automatically pairs feasible inlet–outlet sites, computes hydraulic head and theoretical power, and visualizes results on an interactive map with export and reporting features.

Scope includes data ingestion (DEM, shapefiles, rainfall/discharge), HEC-HMS output integration, spatial processing (whitebox/rasterio/geopandas), algorithmic pairing of inlet–outlet sites, map visualization (Leaflet/OpenLayers), charting (Chart.js), and export to CSV/GeoJSON/PDF.

Out of scope for v1: network optimization and cost estimation, hydraulic structure design, environmental/social impact assessment, detailed energy economics, automated HEC-HMS model calibration.

Access model for v1: open access (no user authentication or admin module).

---

## 2. Objectives & Key Outcomes

- Automate hydropower site identification using DEM and HEC-HMS-derived flows.
- Standardize processing workflow and reduce manual GIS effort for researchers and LGUs.
- Provide transparent, reproducible computations for head and theoretical power across candidate sites.
- Deliver an interactive, filterable map and tabular reports supporting screening-level decisions.
- Ensure data governance, CRS consistency (EPSG:32651), and exportability for downstream analysis.

Key outcomes by v1:
- End-to-end pipeline from upload → processing → visualization → export for the Matiao River.
- Validated set of inlet–outlet site pairs with computed head and power, aligned with hydrologic events/scenarios from HEC-HMS outputs.

---

## 3. Target Users / Stakeholders

- Researchers (Hydrology/GIS): Prepare datasets, run analyses, interpret outputs.
- Engineers (Water/Power): Review candidate sites, apply thresholds, export data.
- Policymakers/LGU Planners: Explore results on map, review summaries/charts.
 
Access model:
- Open access (no authentication/admin). All users can upload, process, visualize, and export.

---

## 4. Core Features and Functional Requirements

### 4.1 Data Input & Upload

Supported inputs (initially via UI upload; programmatic ingestion later):
- DEM: GeoTIFF `.tif` (e.g., `INPUT DATA/WATERSHED DATA/Terrain Data/DEM.tif`)
- Vector layers: ESRI Shapefile set `.shp/.shx/.dbf/.prj` (e.g., `Subbasins_MatiaoRiver.*`, `Matiao Bridge.*`)
- Projection: `.prj` (EPSG:32651 from `WATERSHED DATA/Projection File/32651.prj`)
- Rainfall/Discharge time series: CSV or Excel (`.csv`, `.xlsx`) from `INPUT DATA/RAINFALL & DISCHARGE DATA/`
- HEC-HMS outputs: exported CSV time series (preferred v1) from basin elements (Subbasin/Reach/Outlet). DSS support may be added later.

Validation (server-side):
- File types and completeness for shapefile components; valid GeoTIFF with numeric elevation values.
- CRS detection; if missing, require EPSG code selection; if present but different, reproject to target CRS (EPSG:32651). 
- Time series schema: datetime column, units metadata (e.g., m³/s for Q), station/source IDs.
- Size limits (configurable), virus scan of uploads, and safe file names.

Acceptance criteria:
- System rejects incomplete shapefiles; shows clear error.
- System reprojects to EPSG:32651 or blocks with fix instructions.
- Time series parsed with source/unit metadata persisted in DB.
- All uploads are indexed and visible in a dataset management table.

### 4.2 Automated Processing

Core computations:
- Flow direction (D8), flow accumulation, depression breaching/filling as needed.
- Watershed delineation and stream extraction (threshold on accumulation).
- Deterministic inlet–outlet pairing algorithm over stream network.
- Head calculation and theoretical power computation per pair.
- Integration with HEC-HMS outputs to associate discharge $Q$ with spatial features.

Equations:
- Head (Eq. 8): $H = z_{\text{inlet}} - z_{\text{outlet}}$ from DEM elevations.
- Power (Eq. 10): $P = \rho \cdot g \cdot Q \cdot H \cdot \eta$
  - Constants: $\rho = 1000\,\text{kg/m}^3$, $g = 9.81\,\text{m/s}^2$
  - Efficiency default (configurable): $\eta \in [0.6, 0.85]$, default 0.7 for screening.

HEC-HMS integration (v1):
- Accept CSV exports of discharge at subbasin outlets/reaches for selected storm events/return periods.
- Map time series to spatial features by element ID/name, date/time; extract representative Q (e.g., peak flow or event-average configurable).

Algorithmic pairing (high level):
- Identify stream network nodes and segments with sufficient accumulation.
- Generate inlet candidates upstream and outlet candidates downstream, respecting minimum river distance, minimum head, and land proximity to riverbanks.
- For each inlet, search downstream reach for feasible outlet candidates meeting constraints; select head-maximizing or multi-criteria scoring.
- De-duplicate overlapping sites; enforce spacing buffer.

Acceptance criteria:
- Processing job can run asynchronously; status visible (queued/running/succeeded/failed).
- Each site pairing has inlet/outlet coordinates, H, Q, P, and references to data sources (DEM cell IDs, HEC-HMS element/time).
- Reproducible results given same inputs and parameters.

### 4.3 Visualization & Mapping

- Interactive map using Leaflet.js or OpenLayers with base layers.
- Overlays: watershed boundaries, stream network, subbasins, bridges/POIs, candidate sites (inlet–outlet lines and point markers).
- Filter panels: by minimum head, discharge, or power; by scenario or return period; by spatial extent.
- Popups/tooltips: show inlet/outlet coordinates, head (m), discharge (m³/s), power (kW), and links to source metadata.
- Legend and layer controls; CRS handling (display map in EPSG:4326 Web Mercator with on-the-fly transforms from EPSG:32651).

Acceptance criteria:
- Map loads under 3 seconds with 5k site features using clustering/generalization.
- Filters update results within 500 ms for client-side filters; server-side aggregation within 2 s.
- Clicking a site shows accurate computed attributes.
 - Responsive UI with Bootstrap 5: grid-based layout adapts for mobile (≥320px), tablet, and desktop; filter panel collapses on small screens; map and popups remain touch-accessible.

### 4.4 Results & Reporting

- Tabular results view with sorting, filtering, and pagination.
- Export formats: CSV (attributes), GeoJSON (geometry + attributes), PDF (summary report with charts).
- Charts (Chart.js): distributions of head, discharge, and power; classification by thresholds.

Acceptance criteria:
- Exports complete within 10 s for up to 10k features; longer runs as background jobs with notification.
- PDF report includes map snapshot, summary metrics, and classification counts.

### 4.5 System Management (Removed in v1)

No authentication or admin module in v1. Minimal dataset handling occurs implicitly within the Upload → Process → Visualize → Export flow; advanced dataset lifecycle and RBAC are out of scope.

---

## 5. Data Inputs and Processing Workflow

Data sources (as present in workspace):
- DEM: `INPUT DATA/WATERSHED DATA/Terrain Data/DEM.tif` (+ sidecars `.tfw`, `.aux.xml`, `.ovr`).
- Projection file: `INPUT DATA/WATERSHED DATA/Projection File/32651.prj` (EPSG:32651; UTM Zone 51N).
- Shapefiles: `Subbasins_MatiaoRiver.*`, `Matiao Bridge.*` under `Bridge & River/`.
- Hydro-meteorological: Excel files in `INPUT DATA/RAINFALL & DISCHARGE DATA/`. System will ingest `.xlsx` and `.csv` (pandas) in v1.

Workflow steps:
1) Upload & Validation: parse metadata, detect CRS; reproject to EPSG:32651; index in DB.
2) Preprocessing: DEM fill; compute D8 flow direction and accumulation; optional smoothing.
3) Watershed/Streams: delineate watershed; extract stream network at accumulation threshold.
4) HMS Association: import HEC-HMS CSV flows; link to spatial features (by element ID/name) and event.
5) Site Pairing: generate candidates; compute head from DEM; associate Q from HMS; compute P.
6) Persist & Index: write results to PostGIS layers (`sites_point`, `sites_pair_line`, related tables).
7) Visualize: render map layers; expose filtering endpoints.
8) Export/Report: generate files; store and link to run.

Configurable parameters:
- Accumulation threshold for stream extraction.
- Minimum/maximum head; minimum river distance between inlet/outlet.
- Efficiency factor and Q selection rule (peak vs. percentile).

---

## 6. System Architecture

High-level components:
- Web App: Django (REST endpoints + Django Templates UI); no authentication (open access v1).
- Database: PostgreSQL + PostGIS for spatial storage and queries.
- Processing: Python workers (Celery optional for async) using `rasterio`, `geopandas`, `shapely`, `whitebox`, `GDAL`, `pandas`.
- Storage: File system for raw uploads; DB for vectorized outputs; caching (Redis, optional) for tiles/queries.
- Frontend: Bootstrap 5 for layout/components; Leaflet.js/OpenLayers for maps; Chart.js for charts; server-side rendered pages.

Architecture flow:
- Upload → Storage (FS) → Validation → Preprocess (DEM/flows) → Pairing algorithm → PostGIS layers → Map & API → Exports.
- CRS: All processing in EPSG:32651; map served in EPSG:3857 with on-the-fly transform.

Security & Ops:
- Input sanitization; size limits; virus scanning. Logging, monitoring, and job audit trail.
- Environment isolation via Python venv; GDAL/PROJ configured.

---

## 7. User Flows

1) Upload → Process → Visualize → Export
- Upload: select DEM, shapefiles, rainfall/discharge, HMS CSV. Confirm CRS and parameters.
- Process: start job; watch status; view logs/errors if any.
- Visualize: map loads watershed/streams/sites; adjust filters; inspect popups.
- Export: choose CSV/GeoJSON/PDF; download; optionally save report.

2) Admin Data Management (Removed in v1)
 - Not applicable. No admin module or authentication in v1.

---

## 8. Non-functional Requirements

- Performance: ingest up to 2 GB DEM; produce 5–10k site pairs per run. Map initial load < 3 s; filter < 2 s.
- Scalability: optional Celery + Redis for queueing; horizontal worker scale.
- Reliability: retries for transient processing errors; resumable jobs where feasible.
- Security: CSRF protection; upload sanitization; OWASP input validation; antivirus scan on upload. No RBAC in v1.
- Data integrity: CRS metadata enforced; lineage tracked; checksums for files.
- Observability: structured logs; job metrics; request/DB traces in staging/prod.
- Accessibility: WCAG AA for UI components; keyboard navigation for map controls.
- Internationalization: i18n-ready strings; units displayed consistently.
 - Responsive design: Bootstrap 5 grid and utilities provide consistent mobile/tablet/desktop layouts.

---

## 9. Success Metrics / KPIs

- Coverage: % of river length processed without gaps at selected thresholds.
- Accuracy: |H| and |P| variance vs. benchmark/manual checks (target < 5% for H on test sites).
- Throughput: average processing time for standard dataset (< 30 min end-to-end on reference machine).
- Usability: time to first result for new user (< 10 min with provided data).
- Reliability: job success rate (> 95%); mean time to recovery for failed jobs (< 5 min with retry).
- Engagement: downloads/exports per session; filters used per session.

---

## 10. Timeline / Phased Development Plan

Phase 0 – Environment & Data (1 week)
- Set up Django + PostGIS; GDAL/PROJ; validate venv; load sample data.

Phase 1 – Ingestion & Validation (2 weeks)
- Upload flows for DEM/shapefile/time series; CRS handling; dataset management UI.

Phase 2 – Core Processing (3 weeks)
- DEM preprocessing; flow direction/accumulation; watershed and streams; deterministic pairing; power computation.

Phase 3 – Visualization (2 weeks)
- Bootstrap 5 integration; Leaflet/OpenLayers map; layers and legends; filters and popups; performance tuning.

Phase 4 – Reporting & Export (1 week)
- Tabular results; CSV/GeoJSON export; PDF report with charts and map snapshot.

Phase 5 – Hardening & Validation (1–2 weeks)
- QA, documentation, academic write-up alignment; performance/stability improvements; security review.

Risks & mitigations:
- Large DEMs: use overviews/tiling; stream threshold tuning; chunked processing.
- CRS mismatches: mandatory selection/confirmation; automated reproject with audit.
- HMS data variability: define strict CSV schema; add adapter layer for element naming.

---

## 11. Data Model (Initial Sketch)

- `Dataset`: id, name, type (DEM/Vector/TimeSeries/HMS), path, size, checksum, crs, owner, created_at.
- `ProcessingRun`: id, params (JSON), status, logs, started_at, finished_at, dataset_ids.
- `RasterLayer`: id, dataset_id, stats, bounds, resolution.
- `VectorLayer`: id, dataset_id, layer_name, geom_type, srid.
- `TimeSeries`: id, dataset_id, source_id, variable (Q/Rainfall), unit, timestamps, indexing.
- `HMSRun`: id, event_name, return_period, element_mapping, time_window.
- `SitePoint` (inlet/outlet): id, type, geom(Point, 32651), z, q, attrs.
- `SitePair`: id, inlet_id, outlet_id, geom(LineString, 32651), H, Q, P, eta, constraints, source_refs.
- `Export`: id, run_id, type (CSV/GeoJSON/PDF), path, created_at.

---

## 12. Acceptance Test Cases (Samples)

- Upload DEM (.tif): accept valid GeoTIFF; compute stats; store CRS=32651.
- Upload shapefile set: reject if missing .prj; accept after CRS confirmation; reproject if needed.
- Parse HMS CSV: map element names to spatial features; extract peak Q; store units.
- Run processing: job completes; at least N sites produced given thresholds; reproducible results.
- Map filters: min head=10 m returns only sites with H ≥ 10 m; performance within target.
- Export: CSV row count equals site count; GeoJSON features match map; PDF contains charts and legend.

---

## 13. Tech Stack & Libraries

- Backend: Python Django
- DB: PostgreSQL + PostGIS
- GIS: rasterio, geopandas, shapely, whitebox, GDAL
- Frontend: Bootstrap 5, Leaflet.js or OpenLayers, Chart.js
- File handling/reporting: pandas, reportlab

---

## 14. Open Questions (to confirm early)

- HMS output format: confirm CSV schema and naming conventions; DSS timeline if needed.
- Default thresholds: min head, min spacing, stream extraction threshold values for Matiao, Pantukan, Davao de Oro context.
- Efficiency factor: choose default and per-site variability policy.
- Publication: which runs/results are publishable to viewer role?

---

## 15. References

- EPSG:32651 UTM Zone 51N (from `32651.prj`).
- HEC-HMS documentation for export options and element naming.
- WhiteboxTools documentation for flow direction/accumulation and watershed delineation.
 - Bootstrap 5 documentation for layout and components.
