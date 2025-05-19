#%%
import xarray as xr
import numpy as np
import rasterio
from rasterio.warp import reproject
import h5py
import tqdm
from tqdm.contrib.concurrent import thread_map
import os


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

def load_tid_hr_data(locstr, base_path, project_width=None, project_height=None):
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
    Wrapper for loading the high-resolution tile ID data for multiple locations.

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
    mdata_loc_hr = np.zeros_like(tid_hr_data_loc)*np.nan
    for ti in soil_tiles:
        mdata_loc_hr[tid_hr_data_loc==ti]=mdata_loc.data[ti]
    mdata_loc_hr_da = xr.DataArray(
        data=mdata_loc_hr,
        coords={'lon': (('x', 'y'), tid_hr_data_loc.lon.data),
                'lat': (('x', 'y'), tid_hr_data_loc.lat.data)},
        dims=['x', 'y'],
        name=mdata_loc.name
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
    return read_dem_terrain_data(locstr, base_path, project_width, project_height)

def read_dem_terrain_data(locstr, base_path, project_width=None, project_height=None):
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
    
    dem_data_da = xr.DataArray(
        data=np.where(dem_data_array <= 0, np.nan, dem_data_array),
        coords={'lon': (('x', 'y'), demlon),
                'lat': (('x', 'y'), demlat)},
        dims=['x', 'y'],
        name='elevation'
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