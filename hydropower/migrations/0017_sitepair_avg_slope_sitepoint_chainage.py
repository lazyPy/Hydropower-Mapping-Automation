# Generated migration for adding chainage and slope fields
# Implements Objective 2: Inlet-Outlet Pairing improvements

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('hydropower', '0016_diversionpath_pairingdecision_processinglog'),
    ]

    operations = [
        # Add chainage field to SitePoint - distance along stream from outlet
        migrations.AddField(
            model_name='sitepoint',
            name='chainage',
            field=models.FloatField(
                null=True, 
                blank=True, 
                help_text='Distance along stream from basin outlet (meters)'
            ),
        ),
        
        # Add flow accumulation for FA-based discharge
        migrations.AddField(
            model_name='sitepoint',
            name='flow_accumulation',
            field=models.FloatField(
                null=True, 
                blank=True, 
                help_text='Flow accumulation at this point (upstream cell count)'
            ),
        ),
        
        # Add nodal discharge (FA-based Q)
        migrations.AddField(
            model_name='sitepoint',
            name='nodal_discharge',
            field=models.FloatField(
                null=True, 
                blank=True, 
                help_text='Nodal discharge based on flow accumulation (m³/s)'
            ),
        ),
        
        # Add stream segment ID for same-channel pairing
        migrations.AddField(
            model_name='sitepoint',
            name='stream_segment_id',
            field=models.IntegerField(
                null=True, 
                blank=True, 
                help_text='ID of the stream segment this point belongs to'
            ),
        ),
        
        # Add average slope to SitePair (head/length)
        migrations.AddField(
            model_name='sitepair',
            name='avg_slope',
            field=models.FloatField(
                null=True, 
                blank=True, 
                help_text='Average slope: head / river_distance (m/m)'
            ),
        ),
        
        # Add slope in percent for display
        migrations.AddField(
            model_name='sitepair',
            name='avg_slope_percent',
            field=models.FloatField(
                null=True, 
                blank=True, 
                help_text='Average slope as percentage (head / river_distance × 100)'
            ),
        ),
        
        # Add inlet elevation (z_inlet) for quick access
        migrations.AddField(
            model_name='sitepair',
            name='z_inlet',
            field=models.FloatField(
                null=True, 
                blank=True, 
                help_text='Inlet elevation (meters above sea level)'
            ),
        ),
        
        # Add outlet elevation (z_outlet) for quick access
        migrations.AddField(
            model_name='sitepair',
            name='z_outlet',
            field=models.FloatField(
                null=True, 
                blank=True, 
                help_text='Outlet elevation (meters above sea level)'
            ),
        ),
        
        # Add inlet nodal Q for pair-specific discharge
        migrations.AddField(
            model_name='sitepair',
            name='inlet_discharge',
            field=models.FloatField(
                null=True, 
                blank=True, 
                help_text='Nodal discharge at inlet point (m³/s)'
            ),
        ),
        
        # Add outlet nodal Q for pair-specific discharge
        migrations.AddField(
            model_name='sitepair',
            name='outlet_discharge',
            field=models.FloatField(
                null=True, 
                blank=True, 
                help_text='Nodal discharge at outlet point (m³/s)'
            ),
        ),
        
        # Add channel path geometry (actual along-river path)
        migrations.AddField(
            model_name='sitepair',
            name='channel_path_geom',
            field=models.TextField(
                blank=True,
                help_text='Channel path geometry as WKT (follows stream network)'
            ),
        ),
        
        # Add index for stream_segment_id for efficient same-channel queries
        migrations.AddIndex(
            model_name='sitepoint',
            index=models.Index(
                fields=['stream_segment_id', 'chainage'],
                name='hydropower_sp_stream_chainage_idx'
            ),
        ),
    ]
