"""
Simplified URL configuration for map-only hydropower app.
All data processing is done via console script (INPUT DATA/process_data.py).
"""
from django.urls import path
from . import views

app_name = 'hydropower'

urlpatterns = [
    # Map view (main page)
    path('', views.map_view, name='map_view'),
    
    # GeoJSON API endpoints for map layers
    path('api/geojson/site-pairs/', views.geojson_site_pairs, name='geojson_site_pairs'),
    path('api/geojson/watersheds/', views.geojson_watersheds, name='geojson_watersheds'),
    path('api/geojson/streams/', views.geojson_streams, name='geojson_streams'),
    path('api/geojson/subbasins/', views.geojson_subbasins, name='geojson_subbasins'),
    path('api/geojson/bridges/', views.geojson_bridges, name='geojson_bridges'),
    
    # MVT (Mapbox Vector Tiles) API endpoints for large datasets
    path('api/mvt/site-pairs/<int:z>/<int:x>/<int:y>.pbf', views.mvt_site_pairs, name='mvt_site_pairs'),
]
