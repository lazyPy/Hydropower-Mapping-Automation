#!/usr/bin/env python
r"""
Hydropower Data Processing Console Script

This script processes watershed data (DEM, shapefiles, discharge) and populates
the database for map visualization. Run this script whenever INPUT DATA changes.

Usage:
    .\env\Scripts\Activate.ps1; python "INPUT DATA/process_data.py"
    
Options:
    --stream-threshold  Flow accumulation threshold for stream extraction (default: 20)
    --help             Show this help message

Note: Existing data is automatically cleared and reprocessed on each run.

Requirements:
    - DEM file in: INPUT DATA/WATERSHED DATA/Terrain Data/DEM.tif
    - Shapefiles in: INPUT DATA/WATERSHED DATA/Bridge & River/*.shp
    - Discharge data in: INPUT DATA/RAINFALL & DISCHARGE DATA/*.xlsx or *.csv
    - CRS file in: INPUT DATA/WATERSHED DATA/Projection File/32651.prj
"""

import os
import sys
import argparse
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from django.utils import timezone
from typing import Dict, Optional, List, Tuple

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'HYDROPOWER_MAPPING.settings')
import django
django.setup()

# Django imports (after setup)
from django.conf import settings
from django.contrib.gis.geos import GEOSGeometry, Point as GEOSPoint
from django.utils import timezone
from django.db import transaction

# Project imports
from hydropower.models import Dataset, RasterLayer, VectorLayer, StreamNetwork, WatershedPolygon, SitePair, HMSRun, TimeSeries
from hydropower.dem_preprocessing import DEMPreprocessor
from hydropower.watershed_delineation import WatershedDelineator
from hydropower.discharge_association import DischargeAssociator, DischargeConfig
from hydropower.site_pairing import InletOutletPairing, PairingConfig
from hydropower.utils import (
    parse_hms_csv, parse_hms_excel, 
    detect_hms_element_column, detect_hms_discharge_column
)

# Console colors (optional, works on Windows 10+)
try:
    import colorama
    colorama.init()
    GREEN = colorama.Fore.GREEN
    YELLOW = colorama.Fore.YELLOW
    RED = colorama.Fore.RED
    BLUE = colorama.Fore.BLUE
    CYAN = colorama.Fore.CYAN
    RESET = colorama.Style.RESET_ALL
    BOLD = colorama.Style.BRIGHT
except ImportError:
    GREEN = YELLOW = RED = BLUE = CYAN = RESET = BOLD = ""


class ConsoleLogger:
    """Enhanced console output with progress tracking"""
    
    def __init__(self, log_file: Optional[str] = None):
        self.step_num = 0
        self.total_steps = 7  # Actual implemented steps
        self.start_time = datetime.now()
        
        # Set up file logging
        if log_file:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_file),
                    logging.StreamHandler()
                ]
            )
        else:
            logging.basicConfig(level=logging.INFO)
        
        self.logger = logging.getLogger('process_data')
    
    def header(self, text: str):
        """Print header"""
        print(f"\n{BOLD}{BLUE}{'='*80}{RESET}")
        print(f"{BOLD}{BLUE}{text.center(80)}{RESET}")
        print(f"{BOLD}{BLUE}{'='*80}{RESET}\n")
    
    def step(self, description: str):
        """Print step header"""
        self.step_num += 1
        print(f"\n{BOLD}{CYAN}[Step {self.step_num}/{self.total_steps}] {description}{RESET}")
        print(f"{'-'*80}")
        self.logger.info(f"Step {self.step_num}/{self.total_steps}: {description}")
    
    def success(self, message: str):
        """Print success message"""
        print(f"{GREEN}[OK] {message}{RESET}")
        self.logger.info(f"SUCCESS: {message}")
    
    def warning(self, message: str):
        """Print warning message"""
        print(f"{YELLOW}[WARNING] {message}{RESET}")
        self.logger.warning(message)
    
    def error(self, message: str):
        """Print error message"""
        print(f"{RED}[ERROR] {message}{RESET}")
        self.logger.error(message)
    
    def info(self, message: str):
        """Print info message"""
        print(f"  {message}")
        self.logger.info(message)
    
    def summary(self, stats: Dict):
        """Print processing summary"""
        elapsed = datetime.now() - self.start_time
        
        print(f"\n{BOLD}{GREEN}{'='*80}{RESET}")
        print(f"{BOLD}{GREEN}Processing Complete!{RESET}")
        print(f"{BOLD}{GREEN}{'='*80}{RESET}\n")
        
        print(f"\n{BOLD}Summary:{RESET}")
        for key, value in stats.items():
            print(f"  - {key}: {BOLD}{value}{RESET}")
        
        print(f"\n{BOLD}Total Time:{RESET} {elapsed.total_seconds():.1f} seconds")
        print(f"\n{BOLD}{CYAN}Next Steps:{RESET}")
        print(f"  1. Start Django server: {BOLD}.\\env\\Scripts\\Activate.ps1; python manage.py runserver{RESET}")
        print(f"  2. Open browser: {BOLD}http://localhost:8000{RESET}")
        print(f"  3. View hydropower sites on the map!")
        print(f"\n{GREEN}{'='*80}{RESET}\n")


class DataProcessor:
    """Main data processing orchestrator"""
    
    def __init__(self, console: ConsoleLogger, config: Dict):
        self.console = console
        self.config = config
        self.stats = {
            'DEM Processed': 'No',
            'Watersheds Delineated': 0,
            'Stream Segments': 0,
            'Site Pairs Generated': 0,
            'Discharge Records': 0,
            'HMS Runs': 0,
        }
        
        # File paths (updated for Claveria data structure)
        self.input_data_dir = Path(__file__).resolve().parent
        self.dem_path = self.input_data_dir / 'WATERSHED DATA' / 'Terrain Data' / 'DEM.tif'
        self.shapefile_dir = self.input_data_dir / 'WATERSHED DATA' / 'Bridge & River'
        self.discharge_dir = self.input_data_dir / 'RAINFALL & DISCHARGE DATA'
        self.crs_file = self.input_data_dir / 'WATERSHED DATA' / 'Projection File' / '32651.prj'
        self.hms_project_dir = self.input_data_dir / 'HEC-HMS' / 'Claveria Hec-Hms'
    
    def validate_input_files(self) -> bool:
        """Step 1: Validate all required input files exist"""
        self.console.step("Validating Input Files")
        
        valid = True
        
        # Check DEM
        if self.dem_path.exists():
            size_mb = self.dem_path.stat().st_size / (1024 * 1024)
            self.console.success(f"DEM found: {self.dem_path.name} ({size_mb:.1f} MB)")
        else:
            self.console.error(f"DEM not found: {self.dem_path}")
            valid = False
        
        # Check shapefiles
        shapefiles = list(self.shapefile_dir.glob('*.shp')) if self.shapefile_dir.exists() else []
        if shapefiles:
            self.console.success(f"Found {len(shapefiles)} shapefile(s)")
            for shp in shapefiles:
                self.console.info(f"  - {shp.name}")
        else:
            self.console.warning("No shapefiles found (optional)")
        
        # Check discharge data
        discharge_files = []
        if self.discharge_dir.exists():
            discharge_files = list(self.discharge_dir.glob('*.xlsx')) + list(self.discharge_dir.glob('*.csv'))
        
        if discharge_files:
            self.console.success(f"Found {len(discharge_files)} discharge file(s)")
            for df in discharge_files:
                self.console.info(f"  - {df.name}")
        else:
            self.console.warning("No discharge data found (will use default values)")
        
        # Check CRS
        if self.crs_file.exists():
            self.console.success(f"CRS file found: {self.crs_file.name}")
        else:
            self.console.warning("CRS file not found, will auto-detect from DEM")
        
        if not valid:
            self.console.error("Required files missing! Please check INPUT DATA folder structure.")
        
        return valid
    
    def clear_existing_data(self):
        """Clear existing processed data before reprocessing"""
        self.console.step("Clearing Existing Data")
        
        try:
            with transaction.atomic():
                # Delete in reverse dependency order
                deleted_counts = {}
                deleted_counts['Site Pairs'] = SitePair.objects.all().delete()[0]
                deleted_counts['Watersheds'] = WatershedPolygon.objects.all().delete()[0]
                deleted_counts['Streams'] = StreamNetwork.objects.all().delete()[0]
                deleted_counts['Raster Layers'] = RasterLayer.objects.all().delete()[0]
                deleted_counts['Vector Layers'] = VectorLayer.objects.all().delete()[0]
                deleted_counts['Time Series'] = TimeSeries.objects.all().delete()[0]
                deleted_counts['HMS Runs'] = HMSRun.objects.all().delete()[0]
                deleted_counts['Datasets'] = Dataset.objects.all().delete()[0]
                
                for model, count in deleted_counts.items():
                    if count > 0:
                        self.console.success(f"Deleted {count} {model}")
                
                self.console.success("Database cleared successfully")
        except Exception as e:
            self.console.error(f"Failed to clear database: {e}")
            raise
    
    def load_and_validate_dem(self) -> Tuple[Dataset, RasterLayer]:
        """Step 2: Load DEM and create database records"""
        self.console.step("Loading DEM File")
        
        try:
            import rasterio
            import shutil
            
            # Copy DEM to MEDIA_ROOT/uploads if not already there
            media_root = Path(settings.MEDIA_ROOT)
            upload_dir = media_root / 'uploads' / datetime.now().strftime('%Y/%m/%d')
            upload_dir.mkdir(parents=True, exist_ok=True)
            
            dem_in_media = upload_dir / self.dem_path.name
            self.console.info(f"Copying DEM to media storage...")
            shutil.copy2(self.dem_path, dem_in_media)
            dem_to_use = dem_in_media
            
            # Read DEM metadata
            with rasterio.open(dem_to_use) as src:
                self.console.info(f"Dimensions: {src.width} x {src.height} pixels")
                self.console.info(f"Bounds: {src.bounds}")
                self.console.info(f"CRS: {src.crs}")
                
                # Create Dataset record with relative path
                file_path = str(dem_to_use.relative_to(media_root))
                
                dataset = Dataset.objects.create(
                    name=f"DEM_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    dataset_type='DEM',
                    file=file_path,
                    file_size=dem_to_use.stat().st_size,
                    original_filename=self.dem_path.name,
                    crs=str(src.crs),
                    validation_status='VALID',
                    description='Console-uploaded DEM for hydropower analysis'
                )
                
                # Create RasterLayer metadata (stats() returns list for all bands, [0] = band 1)
                all_stats = src.stats()
                band_stats = all_stats[0]  # Get stats for band 1
                raster = RasterLayer.objects.create(
                    dataset=dataset,
                    width=src.width,
                    height=src.height,
                    bands=src.count,
                    bounds_minx=src.bounds.left,
                    bounds_miny=src.bounds.bottom,
                    bounds_maxx=src.bounds.right,
                    bounds_maxy=src.bounds.top,
                    pixel_size_x=src.res[0],
                    pixel_size_y=src.res[1],
                    nodata_value=src.nodata,
                    min_value=band_stats.min,
                    max_value=band_stats.max,
                    mean_value=band_stats.mean,
                    std_value=band_stats.std,
                )
                
                self.console.success(f"DEM loaded: Dataset ID {dataset.id}, Raster ID {raster.id}")
                self.console.info(f"Elevation range: {band_stats.min:.1f}m - {band_stats.max:.1f}m")
                
                return dataset, raster
                
        except Exception as e:
            self.console.error(f"Failed to load DEM: {e}")
            raise
    
    def preprocess_dem(self, raster: RasterLayer) -> bool:
        """Step 3: Preprocess DEM (fill depressions, compute flow)"""
        self.console.step("Preprocessing DEM (Fill, Flow Direction, Flow Accumulation)")
        
        try:
            preprocessor = DEMPreprocessor()
            
            # Get input DEM path from dataset (now in MEDIA_ROOT)
            dem_path = str(Path(settings.MEDIA_ROOT) / raster.dataset.file.name)
            
            # Set up output paths
            output_dir = Path(settings.MEDIA_ROOT) / 'preprocessed' / f'dem_{raster.id}'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            filled_path = str(output_dir / 'filled_dem.tif')
            flow_dir_path = str(output_dir / 'flow_direction.tif')
            flow_accum_path = str(output_dir / 'flow_accumulation.tif')
            
            # 1. Fill depressions
            self.console.info("Filling depressions...")
            fill_result = preprocessor.fill_depressions(dem_path, filled_path)
            if not fill_result.get('success'):
                raise Exception(f"Depression filling failed: {fill_result.get('error')}")
            self.console.success("Depression filling complete")
            
            # 2. Compute D8 flow direction
            self.console.info("Computing D8 flow direction...")
            flow_dir_result = preprocessor.compute_d8_flow_direction(filled_path, flow_dir_path)
            if not flow_dir_result.get('success'):
                raise Exception(f"Flow direction failed: {flow_dir_result.get('error')}")
            self.console.success("Flow direction computed")
            
            # 3. Compute flow accumulation
            self.console.info("Computing flow accumulation...")
            flow_accum_result = preprocessor.compute_flow_accumulation(flow_dir_path, flow_accum_path)
            if not flow_accum_result.get('success'):
                raise Exception(f"Flow accumulation failed: {flow_accum_result.get('error')}")
            self.console.success("Flow accumulation computed")
            
            # Display flow accumulation statistics to help users choose threshold
            import rasterio
            with rasterio.open(flow_accum_path) as src:
                flow_data = src.read(1)
                flow_data = flow_data[flow_data > 0]  # Exclude no-flow areas
                
                if len(flow_data) > 0:
                    flow_max = float(np.max(flow_data))
                    flow_95 = float(np.percentile(flow_data, 95))
                    flow_median = float(np.median(flow_data))
                    
                    self.console.info(f"\nFlow Accumulation Statistics:")
                    self.console.info(f"  Maximum: {flow_max:.0f} cells")
                    self.console.info(f"  95th percentile: {flow_95:.0f} cells")
                    self.console.info(f"  Median: {flow_median:.0f} cells")
                    self.console.info(f"  Recommended threshold range: 2-{int(flow_max * 0.6)}")
                    
                    # Store for later use in adaptive threshold
                    raster.flow_accum_max = flow_max
                    raster.flow_accum_95percentile = flow_95
                    raster.save()
            
            # Update RasterLayer record
            raster.is_preprocessed = True
            raster.preprocessing_method = 'FILLED'
            raster.preprocessing_date = timezone.now()
            raster.filled_dem_path = str(Path(filled_path).relative_to(settings.MEDIA_ROOT))
            raster.flow_direction_path = str(Path(flow_dir_path).relative_to(settings.MEDIA_ROOT))
            raster.flow_accumulation_path = str(Path(flow_accum_path).relative_to(settings.MEDIA_ROOT))
            raster.save()
            
            self.stats['DEM Processed'] = 'Yes'
            self.console.success("DEM preprocessing complete!")
            
            return True
            
        except Exception as e:
            self.console.error(f"DEM preprocessing failed: {e}")
            return False
    
    def delineate_watersheds(self, raster: RasterLayer) -> bool:
        """Step 4: Delineate watersheds and extract stream network"""
        self.console.step("Delineating Watersheds and Extracting Streams")
        
        try:
            stream_threshold = self.config.get('stream_threshold', 1000)
            self.console.info(f"Stream threshold: {stream_threshold} cells")
            
            # Adaptive threshold validation
            if hasattr(raster, 'flow_accum_max') and raster.flow_accum_max:
                flow_max = raster.flow_accum_max
                flow_95 = getattr(raster, 'flow_accum_95percentile', flow_max * 0.1)
                
                if stream_threshold > flow_max * 0.8:
                    recommended = max(2, int(flow_95 * 1.5))
                    self.console.warning(f"Threshold {stream_threshold} is too high (max flow: {flow_max:.0f})!")
                    self.console.info(f"  Recommended threshold: {recommended}")
                    self.console.info(f"  Using recommended value instead...")
                    stream_threshold = recommended
                elif stream_threshold < 2:
                    self.console.warning(f"Threshold {stream_threshold} is very low, may generate excessive streams")
                    self.console.info(f"  Consider using threshold >= 4 for better performance")
            
            delineator = WatershedDelineator(
                raster_layer=raster,
                stream_threshold=stream_threshold
            )
            
            # Extract streams
            self.console.info("Extracting stream network...")
            stream_raster = delineator.extract_stream_network()
            self.console.success("Stream network extracted")
            
            # Vectorize streams
            self.console.info("Vectorizing stream network...")
            stream_gdf = delineator.vectorize_stream_network(stream_raster)
            stream_count = len(stream_gdf)
            self.console.success(f"Vectorized {stream_count} stream segments")
            
            # Delineate watersheds
            self.console.info("Delineating watershed boundaries...")
            watershed_raster = delineator.delineate_watersheds(stream_raster)
            self.console.success("Watershed boundaries delineated")
            
            # Vectorize watersheds
            self.console.info("Vectorizing watershed polygons...")
            watershed_gdf = delineator.vectorize_watersheds(watershed_raster)
            watershed_count = len(watershed_gdf)
            self.console.success(f"Vectorized {watershed_count} watersheds")
            
            # Save to database
            self.console.info("Saving to database...")
            
            # Save streams
            for idx, row in stream_gdf.iterrows():
                StreamNetwork.objects.create(
                    raster_layer=raster,
                    stream_order=int(row.get('strahler_order', row.get('stream_order', 1))),
                    length_m=float(row.get('length_m', 0)),
                    stream_threshold=stream_threshold,
                    geometry=GEOSGeometry(row.geometry.wkt, srid=32651)
                )
            
            # Save watersheds
            for idx, row in watershed_gdf.iterrows():
                WatershedPolygon.objects.create(
                    raster_layer=raster,
                    watershed_id=int(row.get('watershed_id', idx)),
                    area_m2=float(row.get('area_m2', 0)),
                    area_km2=float(row.get('area_m2', 0)) / 1_000_000,
                    perimeter_m=float(row.geometry.length),
                    stream_threshold=stream_threshold,
                    geometry=GEOSGeometry(row.geometry.wkt, srid=32651)
                )
            
            # Update RasterLayer
            raster.watershed_delineated = True
            raster.watershed_count = watershed_count
            raster.stream_count = stream_count
            raster.stream_threshold = stream_threshold
            raster.watershed_delineation_date = timezone.now()
            raster.save()
            
            self.stats['Watersheds Delineated'] = watershed_count
            self.stats['Stream Segments'] = stream_count
            
            self.console.success("Watershed delineation complete!")
            
            return True
            
        except Exception as e:
            self.console.error(f"Watershed delineation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_hms_project(self) -> bool:
        """Load HEC-HMS project data (.basin file, DSS time series, results XML)"""
        try:
            from hydropower.utils import (
                parse_hms_basin_file,
                extract_dss_pathnames_from_results_xml,
                parse_hms_dss_file
            )
            import xml.etree.ElementTree as ET
            
            # Find HMS project files
            basin_file = self.hms_project_dir / 'Claveria.basin'
            results_xml = self.hms_project_dir / 'results' / 'RUN_Run_1.results.xml'
            dss_file = self.hms_project_dir / 'Run_1.dss'
            
            if not basin_file.exists():
                self.console.warning("HMS .basin file not found")
                return False
            
            if not results_xml.exists():
                self.console.warning("HMS results XML not found")
                return False
            
            # Step 1: Parse basin file for geometry
            self.console.info("Parsing HMS .basin file for element geometry...")
            basin_data = parse_hms_basin_file(str(basin_file))
            
            num_subbasins = len(basin_data.get('subbasins', []))
            num_junctions = len(basin_data.get('junctions', []))
            num_reaches = len(basin_data.get('reaches', []))
            num_sinks = len(basin_data.get('sinks', []))
            
            self.console.success(f"Parsed HMS basin: {num_subbasins} subbasins, {num_junctions} junctions, {num_reaches} reaches, {num_sinks} sinks")
            
            # Step 2: Parse results XML for statistics
            self.console.info("Parsing HMS results XML...")
            tree = ET.parse(str(results_xml))
            root = tree.getroot()
            
            run_name = root.find('RunName').text if root.find('RunName') is not None else 'Run 1'
            start_time_str = root.find('StartTime').text if root.find('StartTime') is not None else None
            end_time_str = root.find('EndTime').text if root.find('EndTime') is not None else None
            
            # Parse start/end times
            start_time = None
            end_time = None
            if start_time_str and end_time_str:
                try:
                    import dateutil.parser
                    start_time = dateutil.parser.parse(start_time_str, dayfirst=True)
                    end_time = dateutil.parser.parse(end_time_str, dayfirst=True)
                    if start_time.tzinfo is None:
                        start_time = timezone.make_aware(start_time)
                    if end_time.tzinfo is None:
                        end_time = timezone.make_aware(end_time)
                except:
                    pass
            
            # Create HMS dataset record
            media_root = Path(settings.MEDIA_ROOT)
            upload_dir = media_root / 'uploads' / 'hms' / datetime.now().strftime('%Y/%m/%d')
            upload_dir.mkdir(parents=True, exist_ok=True)
            
            dataset = Dataset.objects.create(
                name=f"HMS_Claveria_{run_name.replace(' ', '_')}",
                dataset_type='HMS',
                file='',  # No single file, it's a project folder
                file_size=0,
                original_filename='Claveria HEC-HMS Project',
                validation_status='VALID',
                description=f"HEC-HMS project: {run_name} for Claveria watershed"
            )
            
            # Extract element statistics from XML
            elements_stats = {}
            for basin_element in root.findall('BasinElement'):
                element_name = basin_element.get('name')
                element_type = basin_element.get('type')
                
                stats_dict = {}
                stats_section = basin_element.find('Statistics')
                if stats_section is not None:
                    for stat in stats_section.findall('StatisticMeasure'):
                        stat_type = stat.get('type')
                        stat_value = stat.get('value')
                        stat_units = stat.get('units', '')
                        try:
                            stats_dict[stat_type] = float(stat_value)
                        except:
                            pass
                
                elements_stats[element_name] = {
                    'type': element_type,
                    'stats': stats_dict
                }
            
            # Find peak discharge across all elements
            peak_discharge = 0
            peak_element = ''
            for elem_name, elem_data in elements_stats.items():
                outflow_max = elem_data['stats'].get('Outflow Maximum', 0)
                if outflow_max > peak_discharge:
                    peak_discharge = outflow_max
                    peak_element = elem_name
            
            # Create HMSRun record
            hms_run = HMSRun.objects.create(
                dataset=dataset,
                event_name=run_name,
                return_period='',
                num_elements=len(elements_stats),
                num_timesteps=0,
                start_time=start_time,
                end_time=end_time,
                peak_discharge=peak_discharge if peak_discharge > 0 else None,
                peak_element=peak_element
            )
            
            self.console.success(f"Created HMS run: {run_name}, peak discharge: {peak_discharge:.2f} m³/s at {peak_element}")
            
            # Step 3: Load DSS time series (if file exists)
            total_records = 0
            if dss_file.exists():
                try:
                    self.console.info("Loading DSS time series data...")
                    
                    # Extract pathnames from results XML
                    pathnames_by_element = extract_dss_pathnames_from_results_xml(str(results_xml))
                    
                    # Filter to get FLOW pathnames only
                    flow_pathnames = []
                    for element_name, pathnames in pathnames_by_element.items():
                        for pathname in pathnames:
                            if '/FLOW/' in pathname and '/FLOW-' not in pathname:
                                flow_pathnames.append(pathname)
                    
                    self.console.info(f"Found {len(flow_pathnames)} flow time series pathnames")
                    
                    # Read DSS file
                    df_dss = parse_hms_dss_file(str(dss_file), flow_pathnames)
                    
                    # Save to TimeSeries table
                    for (dt, pathname), row in df_dss.iterrows():
                        # Extract element name from pathname (//ELEMENT_NAME/...)
                        element_name = pathname.split('/')[2] if len(pathname.split('/')) > 2 else 'Unknown'
                        
                        try:
                            if dt.tzinfo is None:
                                dt = timezone.make_aware(dt)
                            
                            TimeSeries.objects.create(
                                dataset=dataset,
                                station_id=element_name,
                                datetime=dt,
                                value=float(row['value']),
                                units=row.get('units', 'm3/s'),
                                data_type='DISCHARGE'
                            )
                            total_records += 1
                        except Exception as e:
                            pass  # Skip invalid rows
                    
                    self.console.success(f"Loaded {total_records} discharge records from DSS file")
                    
                except Exception as e:
                    self.console.warning(f"Could not load DSS time series: {e}")
                    self.console.info("Using statistics from results XML instead")
            
            # Update stats
            self.stats['HMS Runs'] = 1
            self.stats['Discharge Records'] = total_records
            
            return True
            
        except Exception as e:
            self.console.error(f"Failed to load HMS project: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_discharge_data(self) -> bool:
        """Step 5: Load discharge/rainfall data and HEC-HMS project (if available)"""
        self.console.step("Loading Discharge Data and HEC-HMS Project")
        
        # Check for HEC-HMS project first
        hms_project_loaded = False
        if self.hms_project_dir.exists():
            self.console.info("HEC-HMS project folder detected")
            hms_project_loaded = self.load_hms_project()
        
        # Load regular discharge files
        discharge_files = []
        if self.discharge_dir.exists():
            discharge_files = list(self.discharge_dir.glob('*.xlsx')) + list(self.discharge_dir.glob('*.csv'))
        
        if not discharge_files and not hms_project_loaded:
            self.console.warning("No discharge data or HMS project found, will use estimated values")
            self.stats['Discharge Records'] = 0
            self.stats['HMS Runs'] = 0
            return True
        
        try:
            import pandas as pd
            import shutil
            
            total_records = 0
            hms_run_count = 0
            
            for file_path in discharge_files:
                self.console.info(f"Loading {file_path.name}...")
                
                # Copy file to MEDIA_ROOT/uploads if not already there
                media_root = Path(settings.MEDIA_ROOT)
                upload_dir = media_root / 'uploads' / 'discharge' / datetime.now().strftime('%Y/%m/%d')
                upload_dir.mkdir(parents=True, exist_ok=True)
                
                file_in_media = upload_dir / file_path.name
                if not file_in_media.exists():
                    shutil.copy2(file_path, file_in_media)
                
                # Detect if this is an HMS file
                is_hms_file = 'hms' in file_path.name.lower() or 'output' in file_path.name.lower()
                
                # Parse file using appropriate method
                try:
                    if is_hms_file:
                        if file_path.suffix == '.xlsx':
                            df = parse_hms_excel(str(file_path))
                        else:
                            df = parse_hms_csv(str(file_path))
                        
                        # Detect HMS-specific columns
                        element_col = detect_hms_element_column(df)
                        discharge_col = detect_hms_discharge_column(df)
                        
                        if element_col and discharge_col:
                            self.console.info(f"  Detected HMS format: element='{element_col}', discharge='{discharge_col}'")
                        else:
                            self.console.warning("  HMS columns not detected, falling back to generic parsing")
                            is_hms_file = False
                    
                    if not is_hms_file:
                        # Generic discharge data
                        if file_path.suffix == '.xlsx':
                            df = pd.read_excel(file_path)
                        else:
                            df = pd.read_csv(file_path)
                except Exception as e:
                    self.console.warning(f"  Failed to parse {file_path.name}: {e}, trying generic parsing")
                    if file_path.suffix == '.xlsx':
                        df = pd.read_excel(file_path)
                    else:
                        df = pd.read_csv(file_path)
                    is_hms_file = False
                
                # Create Dataset record
                file_relative_path = str(file_in_media.relative_to(media_root))
                dataset_type = 'HMS' if is_hms_file else 'TIMESERIES'
                
                dataset = Dataset.objects.create(
                    name=f"{'HMS_' if is_hms_file else 'Discharge_'}{file_path.stem}",
                    dataset_type=dataset_type,
                    file=file_relative_path,
                    file_size=file_path.stat().st_size,
                    original_filename=file_path.name,
                    validation_status='VALID',
                    description=f"Console-uploaded {'HMS simulation' if is_hms_file else 'discharge'} data"
                )
                
                # Process HMS file
                if is_hms_file and element_col and discharge_col:
                    # Extract HMS metadata
                    elements = df[element_col].unique()
                    event_name = file_path.stem.replace('_', ' ').title()
                    
                    # Detect return period from filename
                    return_period = ''
                    for part in file_path.stem.split('_'):
                        if 'year' in part.lower() or 'yr' in part.lower():
                            return_period = part
                            break
                    
                    # Detect time columns
                    time_col = None
                    for col in ['Date', 'Time', 'DateTime', 'Timestamp', 'date', 'time']:
                        if col in df.columns:
                            time_col = col
                            break
                    
                    # Get peak discharge
                    peak_discharge = df[discharge_col].max()
                    peak_idx = df[discharge_col].idxmax()
                    peak_element = df.loc[peak_idx, element_col] if not pd.isna(peak_idx) else ''
                    
                    # Create HMSRun record
                    hms_run = HMSRun.objects.create(
                        dataset=dataset,
                        event_name=event_name,
                        return_period=return_period,
                        num_elements=len(elements),
                        num_timesteps=len(df),
                        peak_discharge=float(peak_discharge) if not pd.isna(peak_discharge) else None,
                        peak_element=str(peak_element)
                    )
                    hms_run_count += 1
                    
                    self.console.info(f"  Created HMS run: {event_name} ({return_period})")
                    self.console.info(f"  Elements: {len(elements)}, Peak: {peak_discharge:.2f} m³/s at {peak_element}")
                    
                    # Create TimeSeries records for HMS data
                    for idx, row in df.iterrows():
                        try:
                            # Parse datetime
                            if time_col:
                                dt = pd.to_datetime(row[time_col])
                            else:
                                dt = datetime.now() + pd.Timedelta(hours=idx)
                            
                            if dt.tzinfo is None:
                                dt = timezone.make_aware(dt, timezone.get_current_timezone())
                            
                            TimeSeries.objects.create(
                                dataset=dataset,
                                station_id=str(row[element_col]),
                                datetime=dt,
                                value=float(row[discharge_col]),
                                units='m3/s',
                                data_type='DISCHARGE'
                            )
                            total_records += 1
                        except Exception as e:
                            pass  # Skip invalid rows silently
                
                else:
                    # Generic discharge data (non-HMS)
                    for idx, row in df.iterrows():
                        try:
                            # Parse datetime and make timezone-aware
                            dt = pd.to_datetime(row.get('timestamp', row.get('datetime', datetime.now())))
                            if dt.tzinfo is None:
                                dt = timezone.make_aware(dt, timezone.get_current_timezone())
                            
                            TimeSeries.objects.create(
                                dataset=dataset,
                                station_id=str(row.get('station_id', f'station_{idx}')),
                                datetime=dt,
                                value=float(row.get('discharge', row.get('value', 0))),
                                units='m3/s',
                                data_type='DISCHARGE'
                            )
                            total_records += 1
                        except Exception as e:
                            pass  # Skip invalid rows silently
                
                self.console.success(f"Loaded {len(df)} records from {file_path.name}")
            
            self.stats['Discharge Records'] = total_records
            self.stats['HMS Runs'] = hms_run_count
            self.console.success(f"Total discharge records loaded: {total_records}")
            if hms_run_count > 0:
                self.console.success(f"HMS simulation runs processed: {hms_run_count}")
            
            return True
            
        except Exception as e:
            self.console.error(f"Failed to load discharge data: {e}")
            self.console.warning("Continuing without discharge data...")
            return True  # Non-critical error
    
    def run_site_pairing(self, raster: RasterLayer) -> bool:
        """Step 6: Run inlet-outlet site pairing algorithm"""
        self.console.step("Running Site Pairing Algorithm")
        
        try:
            # Get stream network
            streams = StreamNetwork.objects.filter(raster_layer=raster)
            if not streams.exists():
                self.console.error("No stream network found. Run watershed delineation first.")
                return False
            
            self.console.info(f"Processing {streams.count()} stream segments...")
            
            # Export streams to temporary GeoDataFrame for processing
            import geopandas as gpd
            from shapely import wkt
            
            stream_data = []
            for stream in streams:
                stream_data.append({
                    'geometry': wkt.loads(stream.geometry.wkt),
                    'stream_order': stream.stream_order,
                    'length_m': stream.length_m,
                })
            
            streams_gdf = gpd.GeoDataFrame(stream_data, crs='EPSG:32651')
            
            # Export to temporary GeoPackage (no column name length restrictions)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.gpkg', delete=False) as tmp:
                temp_gpkg = tmp.name
            
            streams_gdf.to_file(temp_gpkg, driver='GPKG')
            self.console.info(f"Exported stream network to temporary file")
            
            # Configure pairing algorithm (optimized for small watersheds)
            pairing_config = PairingConfig(
                min_head=5.0,           # Lower for micro-hydro sites
                max_head=200.0,         # More realistic for small terrain
                min_river_distance=50.0,    # Shorter minimum distance
                max_river_distance=2000.0,  # Appropriate scale for small watershed
                min_stream_order=1,     # Accept first-order streams (headwaters)
                max_candidates_per_type=50,   # Further reduced to 50 inlets/outlets for faster processing
                max_outlets_per_inlet=3,      # Only check top 3 outlets per inlet
                efficiency=0.7          # Standard 70% efficiency
            )
            
            # Initialize pairing algorithm
            pairing = InletOutletPairing(config=pairing_config)
            
            # Load DEM and streams
            dem_path = str(settings.MEDIA_ROOT / raster.filled_dem_path)
            pairing.load_dem(dem_path)
            pairing.load_stream_network(temp_gpkg)
            
            # Load watershed boundary for validation
            # PRIORITY: Use subbasin shapefile if available (more accurate than auto-delineated)
            self.console.info("Loading watershed boundaries for validation...")
            subbasin_path = Path(settings.MEDIA_ROOT) / 'vector_cache' / 'subbasin_Claveria Subbasin.geojson'
            
            if subbasin_path.exists():
                # Use the HEC-HMS subbasin shapefile - more accurate
                pairing.load_watershed_from_shapefile(str(subbasin_path))
                if pairing.watershed_gdf is not None:
                    self.console.info(f"  ✓ Using subbasin shapefile ({pairing.watershed_gdf['area_km2'].sum():.1f} km²)")
            else:
                # Fallback to delineated watersheds
                pairing.load_watershed_polygons(raster.id)
                if pairing.watershed_gdf is not None:
                    self.console.info(f"  Loaded {len(pairing.watershed_gdf)} delineated watershed(s)")
                else:
                    self.console.warning("  No watershed boundaries found - sites may fall outside basin")
            
            # Run pairing
            self.console.info("Identifying inlet-outlet pairs...")
            pairs_list = pairing.run_pairing()
            
            # Clean up temporary file
            import os
            try:
                if os.path.exists(temp_gpkg):
                    os.remove(temp_gpkg)
            except:
                pass
            
            if len(pairs_list) == 0:
                self.console.warning("No feasible site pairs found with current constraints")
                return True
            
            self.console.success(f"Found {len(pairs_list)} candidate site pairs")
            
            # Save to database using the built-in method
            self.console.info("Saving site pairs to database...")
            site_point_ids, site_pair_ids = pairing.save_to_postgis(pairs_list, raster.id)
            
            # Update RasterLayer
            raster.site_pairing_completed = True
            raster.site_pair_count = len(pairs_list)
            raster.site_pairing_date = timezone.now()
            raster.save()
            
            self.stats['Site Pairs Generated'] = len(pairs_list)
            
            # Show summary statistics
            if len(pairs_list) > 0:
                heads = [p.get('head_m', p.get('head', 0)) for p in pairs_list]
                powers = [p.get('power_kw', p.get('power', 0)) for p in pairs_list]
                
                self.console.info(f"\nSite Pair Statistics:")
                self.console.info(f"  Total pairs: {len(pairs_list)}")
                if heads and any(heads):
                    self.console.info(f"  Avg head: {sum(heads)/len(heads):.1f}m")
                    self.console.info(f"  Max head: {max(heads):.1f}m")
                if powers and any(powers):
                    self.console.info(f"  Avg power: {sum(powers)/len(powers):.1f} kW")
                    self.console.info(f"  Total potential: {sum(powers):.1f} kW")
            
            self.console.success("Site pairing complete!")
            
            return True
            
        except Exception as e:
            self.console.error(f"Site pairing failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def associate_discharge_and_power(self, raster: RasterLayer) -> bool:
        """Step 9: Associate discharge data with site pairs and calculate power"""
        self.console.step("Associating Discharge & Calculating Power")
        
        try:
            # Check if we have HMS data
            hms_run = HMSRun.objects.first()
            if not hms_run:
                self.console.warning("No HMS data available. Power will not be calculated.")
                return True
            
            # Get site pairs for this raster layer
            site_pairs = SitePair.objects.filter(raster_layer=raster)
            if not site_pairs.exists():
                self.console.warning("No site pairs to process.")
                return True
            
            self.console.info(f"Processing {site_pairs.count()} site pairs...")
            
            # Clean invalid discharge values (IEEE float nodata)
            invalid_ts = TimeSeries.objects.filter(value__lt=-1e30)
            if invalid_ts.exists():
                count = invalid_ts.count()
                invalid_ts.delete()
                self.console.info(f"Cleaned {count} invalid discharge records")
            
            # Get representative discharge (peak from HMS)
            from django.db.models import Max
            peak_data = TimeSeries.objects.filter(
                dataset=hms_run.dataset,
                data_type='DISCHARGE',
                value__gt=0
            ).aggregate(peak=Max('value'))
            
            representative_q = peak_data.get('peak', 0)
            if not representative_q or representative_q <= 0:
                self.console.warning("No valid discharge data found. Using default Q=10 m³/s")
                representative_q = 10.0  # Default fallback
            
            self.console.info(f"Representative discharge: {representative_q:.2f} m³/s")
            
            # Constants for power calculation
            RHO = 1000.0  # Water density kg/m³
            G = 9.81      # Gravity m/s²
            
            updated = 0
            for site_pair in site_pairs:
                # Assign discharge
                site_pair.discharge = representative_q
                site_pair.hms_run = hms_run
                
                # Calculate net head (with losses if penstock exists)
                gross_head = site_pair.head
                if site_pair.penstock_length and site_pair.penstock_length > 0:
                    from hydropower.site_pairing import calculate_head_losses
                    losses = calculate_head_losses(
                        site_pair.discharge,
                        site_pair.penstock_length,
                        gross_head
                    )
                    net_head = gross_head - losses.get('total_head_loss', 0)
                else:
                    net_head = gross_head
                
                # Calculate power: P = ρ × g × Q × H × η
                efficiency = site_pair.efficiency or 0.7
                power = RHO * G * site_pair.discharge * net_head * efficiency / 1000  # kW
                
                site_pair.power = max(0, power)  # Ensure non-negative
                site_pair.save(update_fields=['discharge', 'power', 'hms_run'])
                updated += 1
            
            self.console.success(f"Updated {updated} site pairs with discharge and power")
            
            # Show power summary
            from django.db.models import Min, Max as DjMax, Avg, Sum
            power_stats = site_pairs.aggregate(
                min_p=Min('power'),
                max_p=DjMax('power'),
                avg_p=Avg('power'),
                total_p=Sum('power')
            )
            
            self.console.info(f"\nPower Output Summary:")
            self.console.info(f"  Range: {power_stats['min_p']:.1f} - {power_stats['max_p']:.1f} kW")
            self.console.info(f"  Average: {power_stats['avg_p']:.1f} kW")
            self.console.info(f"  Total potential: {power_stats['total_p']:.1f} kW ({power_stats['total_p']/1000:.2f} MW)")
            
            return True
            
        except Exception as e:
            self.console.error(f"Discharge association failed: {e}")
            import traceback
            traceback.print_exc()
            return True  # Non-critical, continue anyway
    
    def load_shapefiles(self, dataset: Dataset) -> bool:
        """Load shapefiles (bridges, subbasins, outlets) and cache as GeoJSON for map display"""
        self.console.info("Loading Shapefiles...")
        
        try:
            import geopandas as gpd
            import json
            
            shapefiles = list(self.shapefile_dir.glob('*.shp')) if self.shapefile_dir.exists() else []
            
            if not shapefiles:
                self.console.warning("No shapefiles found - skipping")
                return True
            
            # Create cache directory
            cache_dir = Path(settings.MEDIA_ROOT) / 'vector_cache'
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            loaded_count = 0
            for shp_path in shapefiles:
                try:
                    self.console.info(f"Loading: {shp_path.name}")
                    
                    # Read shapefile with geopandas
                    gdf = gpd.read_file(str(shp_path))
                    
                    if gdf.empty:
                        self.console.warning(f"  Empty shapefile: {shp_path.name}")
                        continue
                    
                    # Reproject to EPSG:32651 if needed
                    if gdf.crs is None:
                        gdf.set_crs(epsg=32651, inplace=True)
                        self.console.info(f"  Set CRS to EPSG:32651")
                    elif gdf.crs.to_epsg() != 32651:
                        gdf = gdf.to_crs(epsg=32651)
                        self.console.info(f"  Reprojected to EPSG:32651")
                    
                    # Determine layer type from filename
                    name_lower = shp_path.stem.lower()
                    if 'bridge' in name_lower:
                        layer_type = 'bridge'
                    elif 'outlet' in name_lower:
                        layer_type = 'outlet'
                    elif 'subbasin' in name_lower or 'basin' in name_lower:
                        layer_type = 'subbasin'
                    elif 'river' in name_lower or 'stream' in name_lower:
                        layer_type = 'river'
                    else:
                        layer_type = 'other'
                    
                    # Get geometry type
                    geom_type = gdf.geometry.geom_type.iloc[0].upper() if len(gdf) > 0 else 'POLYGON'
                    
                    # Store the layer type in the cache filename for easy lookup
                    cache_file = cache_dir / f'{layer_type}_{shp_path.stem}.geojson'
                    
                    # Convert to GeoJSON and save
                    geojson_str = gdf.to_json()
                    with open(cache_file, 'w') as f:
                        f.write(geojson_str)
                    
                    loaded_count += 1
                    self.console.success(f"  Cached {len(gdf)} features as '{layer_type}' ({geom_type})")
                    self.console.info(f"    → {cache_file}")
                    
                except Exception as e:
                    self.console.warning(f"  Failed to load {shp_path.name}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            self.console.success(f"Loaded {loaded_count} shapefile(s) to cache")
            return True
            
        except ImportError:
            self.console.warning("geopandas not available - skipping shapefiles")
            return True
        except Exception as e:
            self.console.error(f"Shapefile loading failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def process_all(self) -> bool:
        """Main processing workflow"""
        try:
            # Step 1: Validate inputs
            if not self.validate_input_files():
                return False
            
            # Step 2: Clear existing data (always)
            self.clear_existing_data()
            
            # Step 3: Load DEM
            dataset, raster = self.load_and_validate_dem()
            
            # Step 4: Load shapefiles (bridges, subbasins, outlets)
            self.load_shapefiles(dataset)
            
            # Step 5: Preprocess DEM
            if not self.preprocess_dem(raster):
                return False
            
            # Step 6: Delineate watersheds
            if not self.delineate_watersheds(raster):
                return False
            
            # Step 7: Load discharge data (optional)
            self.load_discharge_data()
            
            # Step 8: Run site pairing
            if not self.run_site_pairing(raster):
                return False
            
            # Step 9: Associate discharge and calculate power
            self.associate_discharge_and_power(raster)
            
            # Success!
            return True
            
        except Exception as e:
            self.console.error(f"Processing failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main entry point"""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Process hydropower data from INPUT DATA folder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--stream-threshold', type=int, default=20,
                       help='Flow accumulation threshold for stream extraction (default: 20)')
    
    args = parser.parse_args()
    
    # Validate parameters
    if args.stream_threshold <= 0:
        print(f"{RED}[ERROR] Stream threshold must be a positive integer (got: {args.stream_threshold}){RESET}")
        print(f"{YELLOW}[TIP] Typical values: 4-10 (detailed), 100-1000 (moderate), 5000+ (coarse){RESET}")
        return 1
    
    # Set up console logger
    log_dir = project_root / 'logs'
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f'preprocessing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    console = ConsoleLogger(str(log_file))
    
    # Print header
    console.header("HYDROPOWER DATA PROCESSING")
    print(f"{BOLD}Started:{RESET} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{BOLD}Log file:{RESET} {log_file}")
    print(f"{BOLD}Config:{RESET}")
    print(f"  • Stream threshold: {args.stream_threshold}")
    print(f"  • Auto-refresh: Enabled (clears existing data)")
    
    # Create processor
    config = {
        'stream_threshold': args.stream_threshold,
    }
    
    processor = DataProcessor(console, config)
    
    # Run processing
    success = processor.process_all()
    
    # Print summary
    if success:
        console.summary(processor.stats)
        return 0
    else:
        console.error("\nProcessing failed! Check log file for details.")
        console.error(f"Log: {log_file}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
