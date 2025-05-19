#%%
import xarray as xr
import numpy as np
import cartopy.feature as cfeature
from bokeh.models import HoverTool, ColorBar,BasicTicker
from bokeh.palettes import all_palettes
from bokeh.plotting import figure
from bokeh.io import output_notebook, show, save
from bokeh.layouts import row
output_notebook()
import datashader as ds
import pandas as pd
import bokeh
import os
from process_utils import (get_dem_terrain_data_for_locstrs, 
                           load_ptile_data, get_tid_hr_data_for_locstrs,
                           map_mdata_to_tid_hr_list)


def create_locstrs(cube_face, grid_xt, grid_yt):
    """
    Create a list of location strings for a given cube face and grid range.
    
    Parameters
    ----------
    cube_face : int
        The cube face number.
    grid_xt : numpy array
        The range of x coordinates.
    grid_yt : numpy array
        The range of y coordinates.
    
    Returns
    -------
    list
        A list of location strings.
    """
    locstrs = [f'tile:{cube_face},is:{i},js:{j}' 
    for i, j in zip(grid_yt, grid_xt) 
    if os.path.exists(
        '/archive/Rui.Wang/lightning_test_20250422/'
        f'tile:{cube_face},is:{i},js:{j}')]
    return locstrs

def create_datashader_map(
        combined_df, title=None, unit=None, cmap=all_palettes['Viridis'][256], 
        width=800, height=600, resolution=1, show_borders=True, 
        vmin=None, vmax=None, agg_func='mean'):
    """
    Create a high-resolution map visualization using Datashader, 
    combining multiple DataArrays into a single plot.
    
    Parameters
    ----------
    combined_df : pandas.DataFrame
        DataFrame containing the combined data from multiple DataArrays.
    title : str, optional
        Title for the plot. If None, will use the first DataArray's name.
    unit : str, optional
        Unit for the color scale. If None, will not display the unit.
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
        title=title if title is not None else combined_df.columns[-1],
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
    
    # Create Datashader canvas with higher resolution
    canvas_width = int(width * resolution)
    canvas_height = int(height * resolution)
    cvs = ds.Canvas(plot_width=canvas_width, plot_height=canvas_height, 
                    x_range=x_range, y_range=y_range)
    
    # Get the value column name
    value_col = combined_df.columns[-1]
    
    # Aggregate the data based on the specified aggregation function
    if agg_func == 'mean':
        agg = cvs.points(combined_df, 'lon', 'lat', ds.mean(value_col))
    elif agg_func == 'nearest':
        agg = cvs.points(combined_df, 'lon', 'lat', ds.first(value_col))
    else:
        raise ValueError("agg_func must be either 'mean' or 'nearest'")
    
    # Create a source for the image
    source = bokeh.models.ColumnDataSource({
        'image': [agg.values],
        'x': [x_range[0]],
        'y': [y_range[0]],
        'dw': [x_range[1] - x_range[0]],
        'dh': [y_range[1] - y_range[0]],
        'values': [agg.values]  # Add the aggregated values
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
        title=f'{value_col} [{unit}]' if unit is not None else value_col,
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

def combine_da_list_to_df(da_list):
    """
    Combine a list of DataArrays into a single pandas DataFrame.
    The pandas DataFrame will have the same number of rows as the number of 
    high-resolution pixels in the combined DataArrays.
    
    Parameters
    ----------
    da_list : list of xarray.DataArray
        List of DataArrays to combine.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the combined data from all DataArrays.
    """
    if not da_list:
        raise ValueError("da_list cannot be empty")
    
    # Combine all DataArrays into a single DataFrame
    dfs = []
    for da in da_list:
        df = da.to_dataframe().reset_index()
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

def get_grid_indices_in_range(mdata, lon_range, lat_range):
    """
    Find grid indices (grid_xt, grid_yt) for points within a specified lat/lon range.
    
    Parameters
    ----------
    mdata : xarray.Dataset or xarray.DataArray
        Model data containing geolat_t and geolon_t coordinates
    lon_range : tuple
        Tuple of (min_lon, max_lon) in degrees
    lat_range : tuple
        Tuple of (min_lat, max_lat) in degrees
    
    Returns
    -------
    tuple
        Two numpy arrays containing the grid_xt and grid_yt indices that fall within
        the specified lat/lon range
    """
    # Get the lat/lon coordinates
    lats = mdata.geolat_t
    lons = mdata.geolon_t
    
    # Create masks for points within the ranges
    lat_mask = (lats >= lat_range[0]) & (lats <= lat_range[1])
    lon_mask = (lons >= lon_range[0]) & (lons <= lon_range[1])
    
    # Combine masks to find points within both ranges
    combined_mask = lat_mask & lon_mask
    
    # Get the indices where the mask is True
    grid_yt_indices, grid_xt_indices = np.where(combined_mask)
    
    # Add 1 to the indices to match the grid indices
    return grid_xt_indices+1, grid_yt_indices+1

def cube_sphere_face_min_max_lon_lat(cube_face):
    """
    Get the minimum and maximum longitude and latitude for a given cube face.
    """
    if cube_face == 1:
        return (0.8439675569534302, 359.8221130371094, -35.174068450927734, 44.26972579956055)
    elif cube_face == 2:
        return (35.39055252075195, 124.60938262939453, -35.53116989135742, 44.60742950439453)
    elif cube_face == 3:
        return (0.16366428136825562, 359.95452880859375, 36.0015983581543, 84.08895874023438)
    elif cube_face == 4:
        return (125.39061737060547, 204.6916046142578, -44.60742950439453, 40.94393539428711)
    elif cube_face == 5:
        return (235.32156372070312, 304.6094665527344, -41.65079116821289, 44.60742950439453)
    elif cube_face == 6:
        return (0.39228981733322144, 359.9735107421875, -89.26538848876953, -36.722476959228516)
    else:
        raise ValueError(f'Invalid cube face: {cube_face}')
    
def get_cube_faces_in_range(lon_range, lat_range):
    """
    Determine which cube sphere faces contain data within the given lat/lon ranges.
    
    Parameters
    ----------
    lon_range : tuple
        Tuple of (min_lon, max_lon) in degrees
    lat_range : tuple
        Tuple of (min_lat, max_lat) in degrees
    
    Returns
    -------
    list
        List of cube face numbers (1-6) that contain data within the specified ranges
    """
    # Initialize list to store relevant cube faces
    relevant_faces = []
    
    # Check each cube face
    for face in range(1, 7):
        face_min_lon, face_max_lon, face_min_lat, face_max_lat = cube_sphere_face_min_max_lon_lat(face)
        
        # Check if there's any overlap between the ranges
        lon_overlap = (lon_range[0] <= face_max_lon and lon_range[1] >= face_min_lon)
        lat_overlap = (lat_range[0] <= face_max_lat and lat_range[1] >= face_min_lat)
        
        # If both longitude and latitude ranges overlap, add this face
        if lon_overlap and lat_overlap:
            relevant_faces.append(face)
    
    return relevant_faces

#%%
# Setup parameters
year_range = range(1980, 2015)
var = 'precip'
unit = 'mm/day'
var_scale = 86400
project_width = 100 # number of sub-grid pixels to project high-res data onto
project_height = 100
use_multiprocessing = True
num_threads = os.cpu_count()
# Select grid cells and create locstrs
conus_lat_range = (25, 50)
conus_lon_range = (-125+360, -67+360)
cube_faces = get_cube_faces_in_range(conus_lon_range, conus_lat_range)
# Load the model data
mdata_paths = {cube_face: ['/archive/m2p/awg/2023.04_orog_disag/'
            'c96L33_am4p0_cmip6Diag_orog_disag/'
            'gfdl.ncrc5-intel23-classic-prod-openmp/pp/land_ptid/ts/'
            f'monthly/1yr/land_ptid.{year}01-{year}12.{var}.tile{cube_face}.nc' 
            for year in year_range] for cube_face in cube_faces}
mdata = {cube_face: 
         xr.open_mfdataset(mdata_paths[cube_face])[f'{var}'].mean('time').load()*var_scale 
         for cube_face in cube_faces}
grid_indices = {cube_face: get_grid_indices_in_range(
    mdata[cube_face], conus_lon_range, conus_lat_range) for cube_face in cube_faces}
# Remove cube faces for which no grid indices are found
for cube_face in cube_faces:
    if grid_indices[cube_face][0].size == 0:
        print(f'No grid indices found for cube face {cube_face}')
        cube_faces.remove(cube_face)
        del mdata[cube_face]
        del grid_indices[cube_face]
locstrs = {cube_face: create_locstrs(
    cube_face, grid_indices[cube_face][0], grid_indices[cube_face][1]) 
    for cube_face in cube_faces}
# Load the ptile data
ptile_data = {cube_face: load_ptile_data(cube_face) for cube_face in cube_faces}
#%%

#%%
# Load and reprojectthe tid data
tid_hr_data_list = {cube_face: get_tid_hr_data_for_locstrs(
    locstrs[cube_face], num_threads=num_threads, use_multiprocessing=use_multiprocessing,
    project_width=project_width, project_height=project_height) for cube_face in cube_faces}
#%%
# Prepare lists of high-res data for each cell
mdata_loc_hr_list = {cube_face: map_mdata_to_tid_hr_list(
    mdata[cube_face], ptile_data[cube_face], tid_hr_data_list[cube_face], locstrs[cube_face]) 
    for cube_face in cube_faces}
#%%
# Prepare lists of high-res DEM elevation data for each cell
dem_data_hr_list = {cube_face: get_dem_terrain_data_for_locstrs(
    locstrs[cube_face], num_threads=num_threads, use_multiprocessing=use_multiprocessing,
    project_width=project_width, project_height=project_height) for cube_face in cube_faces}
#%%
# Combine the data into dataframes (1D, HR pixel-by-pixel)
mdata_df = pd.concat([combine_da_list_to_df(mdata_loc_hr_list[cube_face]) for cube_face in cube_faces], ignore_index=True)
dem_data_df = pd.concat([combine_da_list_to_df(dem_data_hr_list[cube_face]) for cube_face in cube_faces], ignore_index=True)
tid_data_df = pd.concat([combine_da_list_to_df(tid_hr_data_list[cube_face]) for cube_face in cube_faces], ignore_index=True)
#%%
# Create datashader maps
plot1 = create_datashader_map(
    mdata_df, title=var, unit=unit, resolution=1, show_borders=True, 
    cmap=all_palettes['Viridis'][256], agg_func='mean',
    vmin=0, vmax=6)
plot2 = create_datashader_map(
    dem_data_df, title='tile elevation', unit='m', resolution=1, show_borders=True,
    cmap=all_palettes['Viridis'][256], agg_func='mean')
plot3 = create_datashader_map(
    tid_data_df, title='tile ID', resolution=1, show_borders=True,
    cmap=all_palettes['Iridescent'][tid_data_df['tile ID'].max()],
    vmin=0, vmax=tid_data_df['tile ID'].max(), agg_func='nearest')
all_plots = row(plot1, plot2, plot3)
#%%
# Save the plots as HTML
save(all_plots, filename='tile_maps_orog_disag_lr.html')