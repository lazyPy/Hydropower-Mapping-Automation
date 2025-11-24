"""
URL configuration for HYDROPOWER_MAPPING project.
Simplified for map-only application.
"""

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from hydropower import views as hydropower_views

urlpatterns = [
    path("admin/", admin.site.urls),
    
    # Root URL - redirect to map view
    path("", hydropower_views.map_view, name="home"),
    
    # Hydropower app URLs
    path("", include("hydropower.urls")),
]

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

