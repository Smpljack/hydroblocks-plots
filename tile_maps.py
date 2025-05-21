#%%
import xarray as xr
import numpy as np
import cartopy.feature as cfeature
from bokeh.models import HoverTool, ColorBar,BasicTicker
from bokeh.palettes import all_palettes
from bokeh.plotting import figure
from bokeh.io import output_notebook, show, save
from bokeh.layouts import row, grid, gridplot
output_notebook()
import datashader as ds
import pandas as pd
import bokeh
import os
from process_utils import load_hr_mdata


def create_datashader_map(
        combined_df, variable=None, unit=None, var_scale=1, 
        title=None, cmap=all_palettes['Viridis'][256], 
        width=800, height=600, resolution=1, show_borders=True, 
        vmin=None, vmax=None, agg_func='mean', shared_figure=None):
    """
    Create a high-resolution map visualization using Datashader, 
    combining multiple DataArrays into a single plot.
    
    Parameters
    ----------
    combined_df : pandas.DataFrame
        DataFrame containing the combined data from multiple DataArrays.
    variable : str, optional
        Variable to plot. If None, will use the first DataArray's name.
    unit : str, optional
        Unit for the color scale. If None, will not display the unit.
    var_scale : float, optional
        Scale factor for the variable. If None, will not scale the variable.
    title : str, optional
        Title for the plot. If None, will display variable name as title.
    cmap : list or str, optional
        Colormap to use for visualization. Default is Viridis256.
    width : int, optional
        Width of the Bokeh figure in pixels. Default is 800.
    height : int, optional
        Height of the Bokeh figure in pixels. Default is 600.
    resolution : float, optional
        Resolution multiplier for the Datashader canvas. 
        A value of 2 means the canvas will be
        twice as large as the figure in each dimension. 
        Higher values give better resolution
        but require more memory. Default is 2.
    show_borders : bool, optional
        Whether to show coastlines and borders. Default is True.
    vmin : float, optional
        Minimum value for the color scale. 
        If None, will use the minimum value of the DataArray.
    vmax : float, optional
        Maximum value for the color scale. 
        If None, will use the maximum value of the DataArray.
    agg_func : str, optional
        Aggregation function to use. Can be either 'mean' or 'nearest'.
        Default is 'mean'.
    shared_figure : bokeh.plotting.figure.Figure, optional
        If provided, the new plot will share some elements with the shared_figure.
        Default is None.
    Returns
    -------
    bokeh.plotting.figure.Figure
        A Bokeh figure containing the combined visualization.
    """
    # Get the overall domain bounds
    x_range = (combined_df['lon'].min(), combined_df['lon'].max())
    y_range = (combined_df['lat'].min(), combined_df['lat'].max())
    
    # Calculate the correct aspect ratio for PlateCarree projection
    # At the equator, 1 degree of latitude = 1 degree of longitude
    # We use the cosine of the mean latitude to account 
    # for the convergence of meridians
    mean_lat = np.mean(y_range)
    lat_span = y_range[1] - y_range[0]
    lon_span = x_range[1] - x_range[0]
    
    # Calculate the correct aspect ratio based on the actual lat/lon spans
    # and the cosine of the mean latitude
    aspect_ratio = (lon_span / lat_span) * np.cos(np.radians(mean_lat))
    
    # Adjust the figure dimensions to maintain the correct aspect ratio
    if width/height > aspect_ratio:
        # Figure is too wide
        width = int(height * aspect_ratio)
    else:
        # Figure is too tall
        height = int(width / aspect_ratio)
    
    # Create a Bokeh figure with PlateCarree projection
    p = figure(
        title=title if title is not None else variable,
        x_axis_label='Longitude',
        y_axis_label='Latitude',
        width=width,
        height=height,
        x_range=x_range,
        y_range=y_range,
        tools='pan,wheel_zoom,box_zoom,reset,save',
        match_aspect=True,  # This ensures the aspect ratio matches the projection
        aspect_ratio=aspect_ratio,  # Explicitly set the aspect ratio
        min_border_bottom=60,  # Add space for the colorbar
    )
    if shared_figure is not None:
        p.x_range = shared_figure.x_range
        p.y_range = shared_figure.y_range

    # Create Datashader canvas with higher resolution
    canvas_width = int(width * resolution)
    canvas_height = int(height * resolution)
    cvs = ds.Canvas(plot_width=canvas_width, plot_height=canvas_height, 
                    x_range=x_range, y_range=y_range)
    
    # Aggregate the data based on the specified aggregation function
    if agg_func == 'mean':
        agg = cvs.points(combined_df, 'lon', 'lat', ds.mean(variable))
    elif agg_func == 'nearest':
        agg = cvs.points(combined_df, 'lon', 'lat', ds.first(variable))
    else:
        raise ValueError("agg_func must be either 'mean' or 'nearest'")
    
    # Create a source for the image
    source = bokeh.models.ColumnDataSource({
        'image': [agg.values*var_scale],
        'x': [x_range[0]],
        'y': [y_range[0]],
        'dw': [x_range[1] - x_range[0]],
        'dh': [y_range[1] - y_range[0]],
        'values': [agg.values*var_scale]  # Add the aggregated values
    })
    
    if vmin is None:
        vmin = float(agg.min())
    if vmax is None:
        vmax = float(agg.max())
    color_mapper = bokeh.models.LinearColorMapper(
        palette=cmap,
        low=vmin,
        high=vmax,
        nan_color='white'
    )
    
    # Add the image to the Bokeh figure
    p.image(
        image='image',
        x='x',
        y='y',
        dw='dw',
        dh='dh',
        source=source,
        color_mapper=color_mapper
    )
    
    # Calculate tick positions and labels
    num_ticks = 8
    
    color_bar = ColorBar(
        color_mapper=color_mapper,
        border_line_color=None,
        location=(0,0),
        title=f'{variable} [{unit}]' if unit is not None else '-',
        width=width-100,
        height=20,
        orientation='horizontal',
        ticker=BasicTicker(desired_num_ticks=num_ticks),
    )
    
    # Add the colorbar below the plot with proper spacing
    p.add_layout(color_bar, 'below')
    
    # Add hover tool
    hover = HoverTool(
        tooltips=[
            ('Value', '@values'),
            ('Latitude', '$y'),
            ('Longitude', '$x')
        ],
        mode='mouse'
    )
    p.add_tools(hover)
    
    # Add coastlines and borders if requested
    if show_borders:
        
        # Get coastlines and borders with appropriate scale
        states_provinces = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces_lines',
            scale='50m',
            facecolor='none')
        coastlines = cfeature.COASTLINE.with_scale('50m')
        borders = cfeature.BORDERS.with_scale('50m')
        
        # Prepare lists for MultiLine glyph
        xs_list = []
        ys_list = []
        
        # Process each feature
        for feature in [coastlines, borders, states_provinces]:
            for geom in feature.geometries():
                # Skip if geometry is outside our bounds
                bounds = geom.bounds
                if (bounds[2] < x_range[0] or bounds[0] > x_range[1] or 
                    bounds[3] < y_range[0] or bounds[1] > y_range[1]):
                    continue
                
                if geom.geom_type == 'LineString':
                    coords = list(geom.coords)
                    xs, ys = zip(*coords)
                    xs_list.append(xs)
                    ys_list.append(ys)
                elif geom.geom_type == 'MultiLineString':
                    for line in geom.geoms:
                        coords = list(line.coords)
                        xs, ys = zip(*coords)
                        xs_list.append(xs)
                        ys_list.append(ys)
        
        # Add all lines at once using MultiLine
        if xs_list:  # Only add if we have lines to draw
            p.multi_line(xs_list, ys_list, line_color='black', line_width=1)
    
    return p 

#%%
# Base directories
base_paths_disag = {
    'mdata': '/archive/m2p/awg/2023.04_orog_disag/'
             'c96L33_am4p0_cmip6Diag_orog_disag/'
             'gfdl.ncrc5-intel23-classic-prod-openmp/pp/land_ptid/ts/'
             'monthly/1yr/',
    'ptile': '/archive/Marc.Prange/ptiles/',
    'tid': '/archive/Rui.Wang/lightning_test_20250422/',
    'dem': '/archive/Rui.Wang/lightning_test_20250422/'
}
base_paths_ctrl = {
    'mdata': '/archive/m2p/awg/2023.04/'
             'c96L33_am4p0_cmip6Diag_a2p_irrHB_repro/'
             'gfdl.ncrc5-intel23-classic-prod-openmp/pp/'
             'land_ptid/ts/monthly/1yr/',
    'ptile': '/archive/Marc.Prange/ptiles/',
    'tid': '/archive/Rui.Wang/lightning_test_20250422/',
    'dem': '/archive/Rui.Wang/lightning_test_20250422/'
}
# Setup parameters
year_range = range(1980, 1990)
vars = ['precip', 'Tgrnd']
units = ['mm/day', 'K']
project_width = 100 # number of sub-grid pixels to project high-res data onto
project_height = 100
use_multiprocessing = True
num_threads = os.cpu_count()
# Select grid cells and create locstrs
conus_lat_range = (25, 50)
conus_lon_range = (-125+360, -67+360)
europe_lat_range = (35, 70)
europe_lon_range = (350, 35)
#%%
mdata_df_disag = load_hr_mdata(
    europe_lon_range, europe_lat_range, year_range, vars, base_paths_disag,
    project_width=project_width, project_height=project_height, 
    use_multiprocessing=use_multiprocessing, num_threads=num_threads,
    add_tile_elevation=False)
mdata_df_ctrl = load_hr_mdata(
    europe_lon_range, europe_lat_range, year_range, vars, base_paths_ctrl,
    project_width=project_width, project_height=project_height, 
    use_multiprocessing=use_multiprocessing, num_threads=num_threads,
    add_tile_elevation=True)
#%%
# Create datashader maps
plot1 = create_datashader_map(
    mdata_df_disag, variable='precip', unit='mm/day', var_scale=86400,
    resolution=1, show_borders=True,
    cmap=all_palettes['Viridis'][256], agg_func='mean',
    vmin=0, vmax=6)
plot2 = create_datashader_map(
    mdata_df_ctrl, variable='precip', unit='mm/day', var_scale=86400,
    resolution=1, show_borders=True,
    cmap=all_palettes['Viridis'][256], agg_func='mean',
    vmin=0, vmax=6, shared_figure=plot1)
precip_diff = mdata_df_disag.copy()
precip_diff['precip'] = mdata_df_disag['precip'] - mdata_df_ctrl['precip']
plot3 = create_datashader_map(
    precip_diff, variable='precip', unit='mm/day', var_scale=86400,
    title='precip (disag - ctrl)', resolution=1, show_borders=True,
    cmap=all_palettes['RdBu'][11], agg_func='mean',
    vmin=-2, vmax=2, shared_figure=plot1)
plot4 = create_datashader_map(
    mdata_df_ctrl, variable='elevation', unit='m', resolution=1, show_borders=True,
    cmap=all_palettes['Viridis'][256], agg_func='mean', shared_figure=plot1)
plot5 = create_datashader_map(
    mdata_df_ctrl, variable='tid', unit='-', resolution=1, show_borders=True,
    cmap=all_palettes['Iridescent'][mdata_df_ctrl['tid'].max()],
    vmin=0, vmax=mdata_df_ctrl['tid'].max(), agg_func='nearest', shared_figure=plot1)
all_plots = gridplot([[plot1, plot2, plot3], [plot4, plot5]])
show(all_plots)
#%%
# Save the plots as HTML
# save(all_plots, filename='tile_maps_orog_disag_lr.html')