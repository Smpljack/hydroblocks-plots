import xarray as xr
import numpy as np
import rasterio

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

def load_tid_hr_data(locstr):
    """
    Load the high-resolution tile ID data for a given location.
    
    Parameters
    ----------
    locstr : str
        The location string.
    
    Returns
    -------
    tid_hr_data_array : np.ndarray
        The high-resolution tile ID data.
    """
    tid_filename = ('/archive/Rui.Wang/lightning_test_20250422/'
                    f'{locstr}/tiles.tif')
    with rasterio.open(tid_filename) as src:
        tid_hr_data_array = src.read(1)
    return tid_hr_data_array

def get_loc_ptile_data(ptile_data, locstr):
    return ptile_data[locstr]

def get_mdata_loc_hr(mdata_loc, soil_tiles, tid_hr_data_loc):
    mdata_loc_hr = np.zeros_like(tid_hr_data_loc)*np.nan
    for ti in soil_tiles:
        mdata_loc_hr[tid_hr_data_loc==ti]=mdata_loc[ti]
    return mdata_loc_hr

def read_dem_terrain_data(locstr):
    """
    Read the high-resolution DEM/terrain data for a given location.
    
    Parameters
    ----------
    locstr : str
        The location string.
    
    Returns
    -------
    dem_data_da : xarray.DataArray
        The high-resolution DEM/terrain data.
    """
    dem_filename = ('/archive/Rui.Wang/lightning_test_20250422/'
                    f'{locstr}/dem_latlon.tif')
    with rasterio.open(dem_filename) as src:
        dem_data_array = src.read(1)
        x = np.arange(0, src.width)
        y = np.arange(0, src.height)
        xx, yy = np.meshgrid(x, y)
        demlon, demlat = rasterio.transform.xy(src.transform, yy, xx)
        demlon = np.array(demlon)
        demlat = np.array(demlat)
    
    dem_data_da = xr.DataArray(
        data=np.where(dem_data_array <= 0, np.nan, dem_data_array),
        coords={'lon': (('x', 'y'), demlon),
                'lat': (('x', 'y'), demlat)},
        dims=['x', 'y'],
        name='elevation'
    )
    return dem_data_da

def get_mdata_hr_for_loc(
        mdata_loc, locstr, return_tid_hr_data=True, soil_tiles=None):
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
    
    Returns
    -------
    tuple
        Returns (mdata_loc_hr_da, tid_hr_data_da) 
        if return_tid_hr_data is True, otherwise returns mdata_loc_hr_da.
    """
    # Get high-res lat/lon
    lon_hr, lat_hr = load_hr_latlon_data(locstr)
    # Get high-res map of tile ID
    tid_hr_data = load_tid_hr_data(locstr)
    # Map out model data to high-res
    mdata_loc_hr = get_mdata_loc_hr(mdata_loc, soil_tiles, tid_hr_data)
    mdata_loc_hr_da = xr.DataArray(
        coords={'lon': (('x', 'y'), lon_hr),
                'lat': (('x', 'y'), lat_hr)},
        data=mdata_loc_hr,
        dims=['x', 'y'],
        name=mdata_loc.name
    )
    if return_tid_hr_data:
        tid_hr_data_da = xr.DataArray(
            data=np.where(tid_hr_data == -9.999e3, np.nan, tid_hr_data),
            coords={'lon': (('x', 'y'), lon_hr),
                    'lat': (('x', 'y'), lat_hr)},
            dims=['x', 'y'],
            name='tile ID'
        )
        return mdata_loc_hr_da, tid_hr_data_da
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
        mdata, locstr, soil_tiles = args
        i = int(locstr.split(',')[1].split(':')[1])
        j = int(locstr.split(',')[2].split(':')[1])
        mdata_loc = mdata.sel(grid_xt=j, grid_yt=i)
        mdata_loc_hr, tid_hr_data = get_mdata_hr_for_loc(
            mdata_loc, locstr, return_tid_hr_data=True, 
            soil_tiles=soil_tiles)
        dem_data_hr = read_dem_terrain_data(locstr)
        return mdata_loc_hr, dem_data_hr, tid_hr_data
    except Exception as e:
        print(f"Error processing {locstr}: {str(e)}", flush=True)
        return None