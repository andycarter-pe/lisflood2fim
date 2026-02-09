# ************************************************************
# LISFLOOD-FP to FIM
# Script 02 - build_netcdf_03
#
# Created by: Andy Carter, PE
# Created - 2026.02.07
# ************************************************************

# ************************************************************
import rioxarray as rxr
import xarray as xr
import numpy as np
import os
import re
import subprocess

import argparse
import configparser
import time
import datetime
import warnings
# ************************************************************


# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist" % arg)
    else:
        # File exists so return the directory
        return arg
        return open(arg, 'r')  # return an open file handle
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


# ----------------
def fn_str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'true', 't', '1'}:
        return True
    elif value.lower() in {'false', 'f', '0'}:
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected. Got '{value}'.")
# ----------------


# --------------------------------------------------
def fn_convert_to_tif(str_input_filepath, str_outfolder, str_crs="EPSG:5070"):

    os.makedirs(str_outfolder, exist_ok=True)

    da = rxr.open_rasterio(str_input_filepath, masked=True).squeeze()

    da = da.rio.write_crs(str_crs, inplace=False)

    if da.rio.nodata is None:
        da = da.rio.write_nodata(-9999, inplace=False)

    da = da.astype(np.float32)

    base = os.path.splitext(os.path.basename(str_input_filepath))[0]
    str_outfilepath = os.path.join(str_outfolder, f"{base}.tif")

    # ---- Windows-safe guard ----
    if os.path.exists(str_outfilepath):
        #print(f"Skipping existing: {str_outfilepath}")
        return

    da.rio.to_raster(
        str_outfilepath,
        driver="GTiff",
        dtype="float32",
        compress="LZW",
        tiled=True,
        nodata=da.rio.nodata,
    )

    #print("Wrote:", str_outfilepath)
# --------------------------------------------------


# -----------------
def fn_compress_netcdf_nccopy(str_netcdf_filepath):
    
    str_output_netcdf_filepath = str_netcdf_filepath[:-7] + ".nc"
    
    # Construct the nccopy command
    # Note:  uses the nccopy command which is installed in this environment
    command = ['nccopy', '-d', '5', str_netcdf_filepath, str_output_netcdf_filepath]
    
    # Call the nccopy command using subprocess
    try:
        subprocess.run(command, check=True)
        #print(f"Compressed file created: {str_output_netcdf_filepath}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        
    # Delete the uncompressed wsel
    if os.path.exists(str_netcdf_filepath):
        os.remove(str_netcdf_filepath)
# -----------------


# --------------------
def fn_get_sorted_geotiffs_and_intensities(folder):
    # regex to capture intensity like 88p5mm or 167p4mm
    pattern = re.compile(r'_(\d+p\d+)mm', re.IGNORECASE)

    records = []

    for fname in os.listdir(folder):
        if fname.lower().endswith(".tif"):
            match = pattern.search(fname)
            if match:
                intensity_str = match.group(1).replace("p", ".")
                intensity = float(intensity_str)

                full_path = os.path.join(folder, fname)
                records.append((intensity, full_path))

    # sort by intensity (ascending)
    records.sort(key=lambda x: x[0])

    filepaths = [r[1] for r in records]
    intensities = [r[0] for r in records]

    return filepaths, intensities
# --------------------


# .........................................................
def fn_build_netcdf_03(
    str_global_config_file_path,
    str_local_config_file_path,
    b_print_output
):

    #warnings.filterwarnings("ignore", category=UserWarning)

    # ---- Header output ----
    if b_print_output:
        print(f"""
+=================================================================+
|                 BUILD FLOOD INUNDATION NETCDF                   |
|                Created by Andy Carter, PE of                    |
|             Center for Water and the Environment                |
|                 University of Texas at Austin                   |
+-----------------------------------------------------------------+
  ---(g) INPUT GLOBAL CONFIGURATION FILE: {str_global_config_file_path}
  ---(c) LOCAL CONFIGURATION FILE:  {str_local_config_file_path}
  ---[r] PRINT OUTPUT: {b_print_output}
===================================================================
""")
    else:
        print("Script 03: Build Flood Inundation NetCDF")

    # ==================================================================
    # READ GLOBAL CONFIG
    # ==================================================================
    global_config = configparser.ConfigParser()
    global_config.read(str_global_config_file_path)

    global_section_schema = {
        'lisflood_settings': ['downscale']
    }

    dict_global_params = {}

    for section_name, keys in global_section_schema.items():
        if section_name not in global_config:
            raise KeyError(f"Missing [{section_name}] section in GLOBAL config")

        section = global_config[section_name]
        dict_global_params.update({
            key: section.get(key, '')
            for key in keys
        })

    # ==================================================================
    # READ LOCAL CONFIG
    # ==================================================================
    local_config = configparser.ConfigParser()
    local_config.read(str_local_config_file_path)

    local_section_schema = {
        'run_parameters': [
            'catchment',
            'out_root_folder'
        ]
    }

    dict_local_params = {}

    for section_name, keys in local_section_schema.items():
        if section_name not in local_config:
            raise KeyError(f"Missing [{section_name}] section in LOCAL config")

        section = local_config[section_name]
        dict_local_params.update({
            key: section.get(key, '')
            for key in keys
        })

    # COMBINE (local overrides global if collision)
    dict_all_params = {
        **dict_global_params,
        **dict_local_params
    }
    
    print(dict_all_params)
    
    if b_print_output:
        print('  -- STEP 1: Finding Depth Rasters')
    
    # From local config
    str_out_root_folder = dict_all_params['out_root_folder']
    str_catchment = dict_all_params['catchment']
    
    # -------------------
    # Root folder to search
    root_dir = os.path.join(str_out_root_folder, str_catchment, "02_lisflood_input")
    
    # Output folder
    str_outfolder = os.path.join(root_dir, str_catchment + "-geotiffs")
    
    os.makedirs(str_outfolder, exist_ok=True)
    
    # --------------------------------------------------
    # Walk directory and process files ending with "0001.wd"
    for root, dirs, files in os.walk(root_dir):
        for fname in files:
            # **** HARD CODED to 0001.wd ****
            if fname.endswith("0001.wd"):
                in_path = os.path.join(root, fname)
                #print(in_path)
                fn_convert_to_tif(in_path, str_outfolder)
                
    # From folder r'E:\condition_terrain\cat-2421551\02_lisflood_input\cat-2421551-geotiffs' get list of
    # geotiff files.  Then parse out a coresponding intensity like cat-2421551_88p5mm-0001.tif = 88.5
    # and cat-2421551_167p4mm-0001.tif = 167.4.  Return a list of filepaths in acssecending order.
    # also return a list of intensity values.
    
    folder = str_outfolder
    list_filepaths, list_intensities = fn_get_sorted_geotiffs_and_intensities(folder)

    list_intensities = np.round(list_intensities, 1)  # rounds to 1 decimal place (try to fix machine precision issues)
    
    
    if b_print_output:
        print('  -- STEP 2: Writing NetCDF')
        
    # -----------------------------
    # INPUTS
    # -----------------------------
    str_terrain = os.path.join(str_out_root_folder, str_catchment, "01_stream_delineation", "dem_clipped_5070.asc")
    str_crs = "EPSG:5070"
    
    str_id = str_catchment
    str_type = 'depth'
    str_intensity_units = 'mm h-1'
    str_raster_vertical_units = 'meters'
    
    # -----------------------------
    # LOAD TERRAIN
    # -----------------------------
    da_terrain = rxr.open_rasterio(str_terrain)
    da_terrain = da_terrain.rio.write_crs(str_crs, inplace=False)
    da_terrain = da_terrain.astype(np.float32).squeeze()
    da_terrain = da_terrain.rename("terrain")
    
    # -----------------------------
    # LOAD DEPTH STACK
    # -----------------------------
    da_list = []
    for tiff_file in list_filepaths:
        da = rxr.open_rasterio(tiff_file)
        
        # Capture CRS (assumes all rasters have same CRS)
        crs = da.rio.crs
        
        # Replace 0 with NaN
        da = da.where(da != 0)
        
        # Assign variable name
        da.name = str_type
        
        # Remove unnecessary dimensions (e.g., band=1)
        da = da.squeeze()
        
        # Expand along intensity dimension
        da_list.append(da.expand_dims(dim='intensity'))
    
    # Concatenate along intensity dimension
    da_stack = xr.concat(da_list, dim='intensity')
    
    # Assign intensity values as coordinates
    da_stack = da_stack.assign_coords(intensity=("intensity", list_intensities))
    
    # -----------------------------
    # CREATE DATASET AND ALIGN TERRAIN
    # -----------------------------
    ds = da_stack.to_dataset(name=str_type)
    
    # Reproject / resample terrain to match depth stack
    da_terrain_aligned = da_terrain.rio.reproject_match(ds[str_type])
    ds['terrain'] = da_terrain_aligned
    
    # -----------------------------
    # ASSIGN CRS AND UNITS
    # -----------------------------
    ds.rio.write_crs(crs, inplace=True)
    ds[str_type].attrs['units'] = str_raster_vertical_units
    ds['terrain'].attrs['units'] = str_raster_vertical_units
    ds.coords['intensity'].attrs['units'] = str_intensity_units
    
    # -----------------------------
    # OUTPUT FOLDER
    # -----------------------------
    input_file = list_filepaths[0]
    
    input_dir = os.path.dirname(input_file)          # folder containing files
    run_dir = os.path.dirname(input_dir)              # one level up
    project_dir = os.path.dirname(run_dir)            # ← one more level up
    
    str_netcdf_folder = os.path.join(project_dir,f"03_{str_type}_nc")
    os.makedirs(str_netcdf_folder, exist_ok=True)
    
    str_netcdf_filename = f"{str_type}_{str_id}_big.nc"
    str_netcdf_filepath = os.path.join(str_netcdf_folder,str_netcdf_filename)
    
    # -----------------------------
    # SAVE NETCDF
    # -----------------------------
    ds.to_netcdf(str_netcdf_filepath)
    
    # -----------------------------
    # ADD GLOBAL ATTRIBUTES
    # -----------------------------
    ds.attrs['00_stream_id'] = str_id
    ds.attrs['01_type'] = str_type
    ds.attrs['02_description'] = 'LISFLOOD-FP Precipitation Raster Stack'
    ds.attrs['03_author'] = 'Created by Andy Carter, PE'
    ds.attrs['04_organization'] = 'Center for Water and the Environment'
    ds.attrs['05_institution'] = 'University of Texas at Austin'
    ds.attrs['06_history'] = (f"Created on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    ds.attrs['07_raster_folder'] = folder
    
    ds.to_netcdf(str_netcdf_filepath)
    
    # -----------------------------
    # COMPRESS NETCDF
    # -----------------------------
    if b_print_output:
        print('  -- STEP 3: Compressing NetCDF')
    
    fn_compress_netcdf_nccopy(str_netcdf_filepath)
    
# .........................................................


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == '__main__':

    flt_start_run = time.time()
    
    parser = argparse.ArgumentParser(description='========= BUILD FLOOD INUNDATION NETCDF =========')
    
    parser.add_argument('-g',
                        dest = "str_global_config_file_path",
                        help=r'REQUIRED: Global configuration filepath Example:C:\Users\civil\dev\lisflood2fim\config\global_config.ini',
                        required=True,
                        metavar='FILE',
                        type=lambda x: is_valid_file(parser, x))
    
    parser.add_argument('-c',
                        dest = "str_local_config_file_path",
                        help=r'REQUIRED: Global configuration filepath Example:C:\Users\civil\dev\lisflood2fim\config\local_config.ini',
                        required=True,
                        metavar='FILE',
                        type=lambda x: is_valid_file(parser, x))
    
    parser.add_argument('-r',
                        dest = "b_print_output",
                        help=r'OPTIONAL: Print output messages Default: True',
                        required=False,
                        default=True,
                        metavar='T/F',type=fn_str_to_bool)
    
    args = vars(parser.parse_args())
    
    str_global_config_file_path = args['str_global_config_file_path']
    str_local_config_file_path = args['str_local_config_file_path']
    b_print_output = args['b_print_output']

    fn_build_netcdf_03(str_global_config_file_path, str_local_config_file_path, b_print_output)

    flt_end_run = time.time()
    flt_time_pass = (flt_end_run - flt_start_run) // 1
    time_pass = datetime.timedelta(seconds=flt_time_pass)
    
    print('Compute Time: ' + str(time_pass))
 #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~