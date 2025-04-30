#%%
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import h5py
from bokeh.models import HoverTool, ColorBar,BasicTicker
from bokeh.palettes import all_palettes
from bokeh.plotting import figure
from bokeh.io import output_notebook, show, save
from bokeh.layouts import row
output_notebook()
import datashader as ds
import pandas as pd
import tqdm
from tqdm.contrib.concurrent import thread_map
import bokeh
import os

from process_utils import process_grid_cell
    

def load_ptile_data(cube_face):
    """
    Load the ptiles data for a given cube face.
    
    Parameters
    ----------
    cube_face : int
        Cube face number
    
    Returns
    -------
    h5py.File
        The ptiles data.
    """
    ptile_filename = f'/archive/Marc.Prange/ptiles/ptiles.face{cube_face}.h5'
    ptile_file = h5py.File(ptile_filename, 'r')
    return ptile_file

def get_mdata_hr_for_loc_and_time(
        mdata, grid_xt_range, grid_yt_range, cube_face, num_threads=None, 
        use_multiprocessing=True):
    """
    Get high-resolution data for multiple locations 
    and times using multiprocessing.
    
    Parameters
    ----------
    mdata : xarray.DataArray
        Input data array
    grid_xt_range : range
        Range of x coordinates
    grid_yt_range : range
        Range of y coordinates
    cube_face : int
        Cube face number
    num_threads : int, optional
        Number of threads to use. If None, uses cpu_count() - 1
    use_multiprocessing : bool, optional
        Whether to use multiprocessing. If False, processes sequentially.
    
    Returns
    -------
    tuple
        (mdata_loc_hr_list, dem_data_hr_list, tid_hr_data_list)
    """
    if num_threads is None:
        num_threads = os.cpu_count()
    
    locstrs = [f'tile:{cube_face},is:{i},js:{j}' 
    for i in grid_yt_range for j in grid_xt_range 
    if os.path.exists(
        '/archive/Rui.Wang/lightning_test_20250422/'
        f'tile:{cube_face},is:{i},js:{j}')
    ]
    ptile_data = load_ptile_data(cube_face)
    # Create argument tuples for each grid cell
    args_list = [(
        mdata, locstr, ptile_data['grid_data'][locstr]['soil/tile'][()]) 
        for locstr in locstrs 
        if 'soil/tile' in ptile_data['grid_data'][locstr].keys()]
    ptile_data.close()
    if use_multiprocessing:
        # Process grid cells in parallel
        results = thread_map(
            process_grid_cell, args_list, max_workers=num_threads)
    else:
        # Process grid cells sequentially
        results = []
        for args in tqdm.tqdm(args_list, desc="Processing tiles"):
            results.append(process_grid_cell(args))
    
    # Separate the results
    mdata_loc_hr_list = []
    dem_data_hr_list = []
    tid_hr_data_list = []
    
    for result in results:
        if result is not None:
            mdata_loc_hr, dem_data_hr, tid_hr_data = result
            mdata_loc_hr_list.append(mdata_loc_hr)
            dem_data_hr_list.append(dem_data_hr)
            tid_hr_data_list.append(tid_hr_data)
    
    return mdata_loc_hr_list, dem_data_hr_list, tid_hr_data_list

def create_datashader_map(
        combined_df, title=None, unit=None, cmap=all_palettes['Viridis'][256], 
        width=800, height=600, resolution=2, show_borders=True, 
        vmin=None, vmax=None):
    """
    Create a high-resolution map visualization using Datashader, 
    combining multiple DataArrays into a single plot.
    
    Parameters
    ----------
    combined_df : pandas.DataFrame
        DataFrame containing the combined data from multiple DataArrays.
    title : str, optional
        Title for the plot. If None, will use the first DataArray's name.
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
    
    # Aggregate the data
    agg = cvs.points(combined_df, 'lon', 'lat', ds.mean(value_col))
    
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
#%%
# Prepare the data
cube_face = 5
year_range = range(1980, 2015)
var = 'precip'
unit = 'mm/day'
var_scale = 86400
mdata_paths = ['/archive/m2p/awg/2023.04_orog_disag/'
            'c96L33_am4p0_cmip6Diag_orog_disag/'
            'gfdl.ncrc5-intel23-classic-prod-openmp/pp/land_ptid/ts/'
            f'monthly/1yr/land_ptid.{year}01-{year}12.{var}.tile{cube_face}.nc' 
            for year in year_range]
print("Loading mdata")
mdata = xr.open_mfdataset(mdata_paths)[f'{var}'].mean('time')*var_scale
print("Finished loading mdata")
grid_xt_range = range(1, 25)
grid_yt_range = range(15, 80)
# Prepare lists of high-res data for each pixel to plot
print("Loading high-res data and mapping out model data")
mdata_loc_hr_list, dem_data_hr_list, tid_hr_data_list = \
    get_mdata_hr_for_loc_and_time(mdata, grid_xt_range, grid_yt_range, cube_face, 
    use_multiprocessing=True, num_threads=os.cpu_count())
print("Combining high-res data into dataframes")
mdata_df = combine_da_list_to_df(mdata_loc_hr_list)
dem_data_df = combine_da_list_to_df(dem_data_hr_list)
tid_data_df = combine_da_list_to_df(tid_hr_data_list)
print("Finished combining dataframes")
#%%
# Create a datashader map
plot1 = create_datashader_map(
    mdata_df, title=var, unit=unit, resolution=1, show_borders=True, 
    cmap=all_palettes['Viridis'][256])
plot2 = create_datashader_map(
    dem_data_df, title='tile elevation', unit='m', resolution=1, show_borders=True,
    cmap=all_palettes['Viridis'][256])
plot3 = create_datashader_map(
    tid_data_df, title='tile ID', resolution=1, show_borders=True,
    cmap=all_palettes['Iridescent'][tid_data_df['tile ID'].max()],
    vmin=0, vmax=tid_data_df['tile ID'].max())
all_plots = row(plot1, plot2, plot3)
#%%
# Save the plots as HTML
save(all_plots, filename='tile_maps_orog_disag_lr.html')

# if __name__ == '__main__':
#     main()

# %%
