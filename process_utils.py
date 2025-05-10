#%%
import xarray as xr
import numpy as np
import rasterio
from rasterio.warp import reproject
import h5py

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

def load_hr_latlon_data(locstr):
    """
    Load the high-resolution lat/lon data for a given location.
    
    Parameters
    ----------
    locstr : str
        The location string.
    
    Returns
    -------
    lon_hr : np.ndarray
        The high-resolution longitude data.
    lat_hr : np.ndarray
        The high-resolution latitude data.
    """
    latlon_path = ('/archive/Rui.Wang/lightning_test_20250422/'
                  f'{locstr}/mask_latlon.tif')
    with rasterio.open(latlon_path) as src:
        x = np.arange(0, src.width)
        y = np.arange(0, src.height)
        xx, yy = np.meshgrid(x, y)
        lon_hr, lat_hr = rasterio.transform.xy(src.transform, yy, xx)
        lon_hr = np.array(lon_hr)
        lat_hr = np.array(lat_hr)
    return lon_hr, lat_hr


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

def load_tid_hr_data(locstr, project_width=None, project_height=None):
    """
    Load the high-resolution tile ID data for a given location.
    
    Parameters
    ----------
    locstr : str
        The location string.
    project_width : int, optional
        If provided, reproject the data to this width using nearest neighbor resampling.
    project_height : int, optional
        If provided, reproject the data to this height using nearest neighbor resampling.
    
    Returns
    -------
    tid_hr_data_da : xarray.DataArray
        The high-resolution tile ID data with lon/lat coordinates.
    """
    tid_filename = ('/archive/Rui.Wang/lightning_test_20250422/'
                    f'{locstr}/tiles.tif')
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

def get_loc_ptile_data(ptile_data, locstr):
    """
    Get the ptile data for a given location.
    
    Parameters
    ----------
    ptile_data : xarray.Dataset
        The ptile data.
    locstr : str
        The location string.
    
    Returns
    -------
    ptile_data_loc : xarray.Dataset
        The ptile data for the given location.
    """
    return ptile_data[locstr]

def get_mdata_loc_hr(mdata_loc, soil_tiles, tid_hr_data_loc):
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

def read_dem_terrain_data(args):
    """
    Read the high-resolution DEM/terrain data for a given location.
    
    Parameters
    ----------
    args : tuple
        Tuple containing (locstr, project_width, project_height)
    Returns
    -------
    dem_data_da : xarray.DataArray
        The high-resolution DEM/terrain data.
    """
    locstr, project_width, project_height = args
    dem_filename = ('/archive/Rui.Wang/lightning_test_20250422/'
                    f'{locstr}/dem_latlon.tif')
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

def get_mdata_hr_for_loc(
        mdata_loc, locstr, return_tid_hr_data=True, soil_tiles=None,
        project_width=None, project_height=None):
    """
    Get the high-resolution model data for a given location.
    
    Parameters
    ----------
    mdata_loc : xarray.DataArray
        The model data.
    locstr : str
        The location string.
    return_tid_hr_data : bool, optional
        Whether to return the high-resolution tile ID data.
    soil_tiles : list, optional
        The soil tiles.
    project_width : int, optional
        Width of the projected grid.
    project_height : int, optional
        Height of the projected grid.

    Returns
    -------
    tuple
        Returns (mdata_loc_hr_da, tid_hr_data_da) 
        if return_tid_hr_data is True, otherwise returns mdata_loc_hr_da.
    """
    # Get high-res lat/lon
    # lon_hr, lat_hr = load_hr_latlon_data(locstr)
    # Get high-res map of tile ID
    tid_hr_da = load_tid_hr_data(locstr, project_width, project_height)
    # Map out model data to high-res
    mdata_loc_hr_da = get_mdata_loc_hr(mdata_loc, soil_tiles, tid_hr_da)
    if return_tid_hr_data:
        return mdata_loc_hr_da, tid_hr_da
    else:
        return mdata_loc_hr_da

def process_grid_cell(args):
    """
    Process a single grid cell. This function is designed to be called 
    by multiprocessing.
    
    Parameters
    ----------
    args : tuple
        Tuple containing (mdata, locstr, soil_tiles)
    
    Returns
    -------
    tuple or None
        Returns (mdata_loc_hr, dem_data_hr, tid_hr_data) 
        if successful, None if failed
    """
    try:
        mdata, locstr, soil_tiles, project_width, project_height = args
        i = int(locstr.split(',')[1].split(':')[1])
        j = int(locstr.split(',')[2].split(':')[1])
        mdata_loc = mdata.sel(grid_xt=j, grid_yt=i)
        mdata_loc_hr_da, tid_hr_da = get_mdata_hr_for_loc(
            mdata_loc, locstr, return_tid_hr_data=True, 
            soil_tiles=soil_tiles, project_width=project_width, 
            project_height=project_height)
        # dem_data_hr = read_dem_terrain_data(locstr)
        return mdata_loc_hr_da, tid_hr_da
    except Exception as e:
        print(f"Error processing {locstr}: {str(e)}", flush=True)
        return None
