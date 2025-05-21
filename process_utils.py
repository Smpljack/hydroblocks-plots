#%%
import xarray as xr
import numpy as np
import rasterio
from rasterio.warp import reproject
import h5py
import tqdm
from tqdm.contrib.concurrent import thread_map
import os
from scipy.spatial import cKDTree
import pandas as pd


def combine_ds_list_to_df(ds_list):
    """
    Combine a list of DataSets into a single pandas DataFrame.
    The pandas DataFrame will have the same number of rows as the number of 
    high-resolution pixels in the combined DataFrames of the Dataset.
    Each variable in the Dataset will be a column in the DataFrame.
    """
    if not ds_list:
        raise ValueError("ds_list cannot be empty")
    
    # Combine all DataFrames into a single DataFrame
    dfs = []
    for ds in ds_list:
        dfs.append(ds.to_dataframe().reset_index())
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

def get_grid_indices_in_range(mdata, lon_range, lat_range):
    """
    Find grid indices (grid_xt, grid_yt) for points within a 
    specified lat/lon range.
    
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
        Two numpy arrays containing the grid_xt and grid_yt indices that fall 
        within the specified lat/lon range.
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
    Determine which cube sphere faces contain data within the given 
    lat/lon ranges.
    
    Parameters
    ----------
    lon_range : tuple
        Tuple of (min_lon, max_lon) in degrees
    lat_range : tuple
        Tuple of (min_lat, max_lat) in degrees
    
    Returns
    -------
    list
        List of cube face numbers (1-6) that contain data within the 
        specified ranges.
    """
    # Initialize list to store relevant cube faces
    relevant_faces = []
    
    # Check each cube face
    for face in range(1, 7):
        face_min_lon, face_max_lon, face_min_lat, face_max_lat = \
            cube_sphere_face_min_max_lon_lat(face)
        
        # Check if there's any overlap between the ranges
        lon_overlap = (lon_range[0] <= face_max_lon and 
                       lon_range[1] >= face_min_lon)
        lat_overlap = (lat_range[0] <= face_max_lat and 
                       lat_range[1] >= face_min_lat)
        
        # If both longitude and latitude ranges overlap, add this face
        if lon_overlap and lat_overlap:
            relevant_faces.append(face)
    
    return relevant_faces

def load_hr_mdata(lon_range, lat_range, year_range, vars, base_paths,
                  project_width=100, project_height=100, 
                  use_multiprocessing=True, num_threads=os.cpu_count(),
                  add_tile_elevation=False):
    """
    Load high-resolution model data for a given lat/lon range and year range.
    
    Parameters
    ----------
    lon_range : tuple
        Tuple of (min_lon, max_lon) in degrees
    lat_range : tuple
        Tuple of (min_lat, max_lat) in degrees
    year_range : tuple
        Tuple of (min_year, max_year)
    vars : list
        List of variables to load
    base_paths : dict
        Dictionary of base paths with keys 'mdata', 'ptile', 'tid', and 'dem'
    project_width : int, optional
        The width of the projected grid.
    project_height : int, optional
        The height of the projected grid.
    use_multiprocessing : bool, optional
        Whether to use multiprocessing.
    num_threads : int, optional
        The number of threads to use.
    add_tile_elevation : bool, optional
        Whether to interpolate DEM elevation data to the model data grid.
    
    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the model data.
    """
    print(
        'Loading model data for:\n'
        f'lon_range: {lon_range}\n'
        f'lat_range: {lat_range}\n'
        f'year_range: {year_range}\n'
        f'vars: {vars}')
    cube_faces = get_cube_faces_in_range(lon_range, lat_range)
    print(f'Considering cube faces {cube_faces}')
    # Load the model data
    mdata_paths = {cube_face: [os.path.join(base_paths['mdata'],
                f'land_ptid.{year}01-{year}12.{var}.tile{cube_face}.nc') 
                for year in year_range for var in vars] 
                for cube_face in cube_faces}
    print(f'Loading model data...')
    mdata = {cube_face: 
            xr.open_mfdataset(mdata_paths[cube_face]).mean('time').load()
            for cube_face in cube_faces}
    print(f'Getting cubic-sphere grid indices for region of interest...')
    grid_indices = {cube_face: get_grid_indices_in_range(
        mdata[cube_face], lon_range, lat_range) for cube_face in cube_faces}
    # Remove cube faces for which no grid indices are found
    for cube_face in cube_faces:
        if grid_indices[cube_face][0].size == 0:
            print(f'No grid indices found for cube face {cube_face}')
            cube_faces.remove(cube_face)
            del mdata[cube_face]
            del grid_indices[cube_face]
    print(f'Finding grid indices with existing high-resolution tile data...')
    locstrs = {cube_face: create_locstrs(
        cube_face, grid_indices[cube_face][0], grid_indices[cube_face][1]) 
        for cube_face in cube_faces}
    print(f'Loading ptiles data...')
    ptile_data = {cube_face: load_ptile_data(cube_face, base_paths['ptile']) 
                  for cube_face in cube_faces}
    print(f'Loading high-resolution tile ID data for each cube face...')
    tid_hr_data_list = {cube_face: get_tid_hr_data_for_locstrs(
        locstrs[cube_face], base_paths['tid'], num_threads=num_threads, 
        use_multiprocessing=use_multiprocessing,
        project_width=project_width, project_height=project_height)
        for cube_face in cube_faces}
    print(f'Mapping model data to high-resolution tile ID data for each cube face...')
    mdata_loc_hr_list = {cube_face: map_mdata_to_tid_hr_list(
        mdata[cube_face], ptile_data[cube_face], tid_hr_data_list[cube_face], 
        locstrs[cube_face])
        for cube_face in cube_faces}
    print(f'Combining model data into a single DataFrame for each cube face...')
    mdata_df = pd.concat(
        [combine_ds_list_to_df(mdata_loc_hr_list[cube_face]) 
         for cube_face in cube_faces], ignore_index=True)
    tid_data_df = pd.concat(
        [combine_ds_list_to_df(tid_hr_data_list[cube_face]) 
         for cube_face in cube_faces], ignore_index=True)
    mdata_df['tid'] = tid_data_df['tile ID']
    if add_tile_elevation:
        print(f'Loading DEM/terrain data for each cube face...')
        dem_data_hr_list = {cube_face: get_dem_terrain_data_for_locstrs(
            locstrs[cube_face], base_paths['dem'], num_threads=num_threads, 
            use_multiprocessing=use_multiprocessing,
            project_width=project_width, project_height=project_height) 
            for cube_face in cube_faces}
        dem_data_df = pd.concat(
            [combine_ds_list_to_df(dem_data_hr_list[cube_face]) 
             for cube_face in cube_faces], ignore_index=True)
        print(f'Interpolating DEM/terrain data to model data grid...')
        dem_points = np.column_stack((dem_data_df['lon'].values, dem_data_df['lat'].values))
        dem_values = dem_data_df['elevation'].values
        mdata_points = np.column_stack((mdata_df['lon'].values, mdata_df['lat'].values))
        # Build KDTree and find nearest neighbors
        tree = cKDTree(dem_points)
        distances, indices = tree.query(mdata_points, k=1)
        # Interpolate using nearest neighbor
        mdata_df['elevation'] = dem_values[indices]
    return mdata_df

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

def load_ptile_data(cube_face, base_path):
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
    ptile_filename = os.path.join(base_path, f'ptiles.face{cube_face}.h5')
    ptile_file = h5py.File(ptile_filename, 'r')
    return ptile_file

def reproject_data(src, project_width=None, project_height=None):
    """
    Reproject data using rasterio.warp.reproject.
    
    Parameters
    ----------
    src : rasterio.DatasetReader
        The source raster.
    project_width : int, optional
        The width of the projected raster.
    project_height : int, optional
        The height of the projected raster.
    
    Returns
    -------
    src : rasterio.DatasetReader
        The source raster.
    width : int
        The width of the reprojected raster.
    height : int
        The height of the reprojected raster.
    transform : rasterio.transform.Affine
        The transform of the reprojected raster.
    """
    data = src.read(1)
    # Calculate the new transform
    scale_x = src.width / project_width
    scale_y = src.height / project_height
    new_transform = rasterio.transform.Affine(
        src.transform.a * scale_x,
        src.transform.b,
        src.transform.c,
        src.transform.d,
        src.transform.e * scale_y,
        src.transform.f
    )
    
    # Create destination array
    destination = np.zeros((project_height, project_width), 
                           dtype=data.dtype)
    
    # Reproject using nearest neighbor
    reproject(
        source=data,
        destination=destination,
        src_transform=src.transform,
        dst_transform=new_transform,
        src_crs=src.crs,
        dst_crs=src.crs,
        resampling=rasterio.enums.Resampling.nearest
    )
    
    # Update the data array and dimensions
    data = destination
    width = project_width
    height = project_height
    transform = new_transform
    return data, width, height, transform

def load_tid_hr_data_mp(args):
    """
    Wrapper for load_tid_hr_data to be used with multiprocessing.

    Parameters
    ----------
    args : tuple
        The location string, project width, and project height.

    Returns
    -------
    tid_hr_data_da : xarray.DataArray
        The high-resolution tile ID data with lon/lat coordinates.
    """
    locstr, project_width, project_height, base_path = args
    return load_tid_hr_data(locstr, project_width, project_height, base_path)

def load_tid_hr_data(
        locstr, base_path, project_width=None, project_height=None):
    """
    Load the high-resolution tile ID data for a given location.
    
    Parameters
    ----------
    locstr : str
        The location string.
    base_path : str
        The base path to the tile ID data.
    project_width : int, optional
        The width of the projected raster.
    project_height : int, optional
        The height of the projected raster.
    
    Returns
    -------
    tid_hr_data_da : xarray.DataArray
        The high-resolution tile ID data with lon/lat coordinates.
    """
    tid_filename = os.path.join(base_path, f'{locstr}/tiles.tif')
    with rasterio.open(tid_filename) as src:
        # Reproject the data if requested
        if project_width is not None and project_height is not None:
            tid_hr_data_array, width, height, transform = reproject_data(
                src, project_width, project_height)
        else:
            tid_hr_data_array = src.read(1)
            width = src.width
            height = src.height
            transform = src.transform
        
        # Generate coordinates
        x = np.arange(0, width)
        y = np.arange(0, height)
        xx, yy = np.meshgrid(x, y)
        lon_hr, lat_hr = rasterio.transform.xy(transform, yy, xx)
        # Convert to 2D arrays
        lon_hr = np.array(lon_hr).reshape(height, width)
        lat_hr = np.array(lat_hr).reshape(height, width)
        
    tid_hr_data_da = xr.DataArray(
        data=tid_hr_data_array,
        coords={'lon': (('x', 'y'), lon_hr),
                'lat': (('x', 'y'), lat_hr)},
        dims=['x', 'y'],
        name='tile ID'
    )
    return tid_hr_data_da

def get_tid_hr_data_for_locstrs(
        locstrs, base_path, num_threads=None, use_multiprocessing=True,
        project_width=None, project_height=None):
    """
    Wrapper for loading the high-resolution tile ID data for 
    multiple locations.

    Parameters
    ----------
    locstrs : list
        The location strings.
    base_path : str
        The base path to the tile ID data.
    num_threads : int, optional
        The number of threads to use.
    use_multiprocessing : bool, optional
        Whether to use multiprocessing.
    project_width : int, optional
        The width of the projected raster.
    project_height : int, optional
        The height of the projected raster.
    
    Returns
    -------
    tid_hr_data_list : list
        The high-resolution tile ID data for each location.
    """
    # Create argument tuples for each grid cell
    args_list = [(locstr, base_path, project_width, project_height) 
                 for locstr in locstrs]
    if use_multiprocessing:
        if num_threads is None:
            num_threads = os.cpu_count()
        # Process grid cells in parallel
        results = thread_map(
            load_tid_hr_data_mp, args_list, max_workers=num_threads)
    else:
        # Process grid cells sequentially
        results = []
        for args in tqdm.tqdm(args_list, desc="Processing tiles"):
            results.append(load_tid_hr_data(*args))
    return results

def map_mdata_to_tid_hr_for_loc(mdata_loc, soil_tiles, tid_hr_data_loc):
    """
    Map out the model data to the high-resolution tile ID data.
    
    Parameters
    ----------
    mdata_loc : xarray.DataArray
        The model data.
    soil_tiles : list
        The soil tiles.
    tid_hr_data_loc : xarray.DataArray
        The high-resolution tile ID data.
    
    Returns
    -------
    mdata_loc_hr_da : xarray.DataArray
        The high-resolution model data.
    """
    variables = [var for var in mdata_loc.data_vars 
                 if var not in ('average_DT')]
    mdata_loc_hr = {var: np.zeros_like(tid_hr_data_loc)*np.nan 
                    for var in variables}
    for var in variables:
        for ti in soil_tiles:
            mdata_loc_hr[var] = np.where(
                tid_hr_data_loc==ti, 
                mdata_loc[var].isel(ptid=ti).values, 
                mdata_loc_hr[var])
    mdata_loc_hr_da = xr.Dataset(
        data_vars={var: (('x', 'y'), mdata_loc_hr[var]) 
                   for var in variables},
        coords={'lon': (('x', 'y'), tid_hr_data_loc.lon.data),
                'lat': (('x', 'y'), tid_hr_data_loc.lat.data)}
    )
    return mdata_loc_hr_da

def map_mdata_to_tid_hr_list(mdata, ptile_data, tid_hr_data_list, locstrs):
    """
    Map out the model data to the high-resolution tile ID data given a list of
    high-resolution tile ID data and an mdata array.
    
    Parameters
    ----------
    mdata : xarray.DataArray
        Input data array
    ptile_data : h5py.File
        Ptiles data
    locstrs : list
        List of location strings
    project_width : int, optional
        Width of the projected grid.
    project_height : int, optional
        Height of the projected grid.
    
    Returns
    -------
    list
        List of high-resolution data arrays
    """
    # Create argument tuples for each grid cell
    args_list = [(
        mdata.sel(grid_xt=int(locstr.split(',')[2].split(':')[1]), 
                  grid_yt=int(locstr.split(',')[1].split(':')[1])), 
        ptile_data['grid_data'][locstr]['soil/tile'][()],
        tid_hr_data_list[locstrs.index(locstr)]) 
        for locstr in locstrs 
        if 'soil/tile' in ptile_data['grid_data'][locstr].keys()]
    results = []
    for args in tqdm.tqdm(args_list, desc="Processing tiles"):
        results.append(map_mdata_to_tid_hr_for_loc(*args))
    
    return results

def read_dem_terrain_data_mp(args):
    """
    Wrapper for read_dem_terrain_data to be used with multiprocessing.

    Parameters
    ----------
    args : tuple
        The location string, project width, and project height.
    
    Returns
    -------
    dem_data_da : xarray.DataArray
        The high-resolution DEM/terrain data.
    """
    locstr, base_path, project_width, project_height = args
    return read_dem_terrain_data(
        locstr, base_path, project_width, project_height)

def read_dem_terrain_data(
        locstr, base_path, project_width=None, project_height=None):
    """
    Read the high-resolution DEM/terrain data for a given location.
    
    Parameters
    ----------
    locstr : str
        The location string.
    base_path : str
        The base path to the DEM/terrain data.
    project_width : int, optional
        The width of the projected raster.
    project_height : int, optional
        The height of the projected raster.
    
    Returns
    -------
    dem_data_da : xarray.DataArray
        The high-resolution DEM/terrain data.
    """
    dem_filename = os.path.join(base_path, f'{locstr}/dem_latlon.tif')
    with rasterio.open(dem_filename) as src:
        if project_width is not None and project_height is not None:
            dem_data_array, width, height, transform = reproject_data(
                src, project_width, project_height)
        else:
            dem_data_array = src.read(1)
            width = src.width
            height = src.height
            transform = src.transform
        
        x = np.arange(0, width)
        y = np.arange(0, height)
        xx, yy = np.meshgrid(x, y)
        demlon, demlat = rasterio.transform.xy(transform, yy, xx)
        demlon = np.array(demlon).reshape(height, width)
        demlat = np.array(demlat).reshape(height, width)
    
    dem_data_da = xr.Dataset(
        data_vars={'elevation': 
                   (('x', 'y'), 
                    np.where(dem_data_array <= 0, np.nan, dem_data_array))},
        coords={'lon': (('x', 'y'), demlon),
                'lat': (('x', 'y'), demlat)},
    )
    return dem_data_da

def get_dem_terrain_data_for_locstrs(
        locstrs, base_path, num_threads=None, use_multiprocessing=True,
        project_width=None, project_height=None):
    """
    Wrapper for loading the DEM/terrain data for multiple locations.
    
    Parameters
    ----------
    locstrs : list
        List of location strings
    base_path : str
        The base path to the DEM/terrain data.
    num_threads : int, optional
        Number of threads to use. If None, uses cpu_count() - 1
    use_multiprocessing : bool, optional
        Whether to use multiprocessing. If False, processes sequentially.
    project_width : int, optional
        Width of the projected grid.
    project_height : int, optional
        Height of the projected grid.
    """
    args_list = [(locstr, base_path, project_width, project_height) 
                 for locstr in locstrs]
    if use_multiprocessing:
        if num_threads is None:
            num_threads = os.cpu_count()
        # Process grid cells in parallel
        results = thread_map(
            read_dem_terrain_data_mp, args_list, max_workers=num_threads)
    else:
        # Process grid cells sequentially
        results = []
        for args in tqdm.tqdm(args_list, desc="Processing tiles"):
            results.append(read_dem_terrain_data(*args))
    return results