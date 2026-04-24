# ************************************************************
# LISFLOOD-FP to FIM
# Script 04 - heal netCDF FIM
#
# Step 1 — Computing a DEM difference raster between the raw clipped DEM 
# and the burned/hydro-enforced DEM to identify modified terrain cells.
#
# Step 2 — Applying that correction back into the netCDF: raising terrain values
# and reducing depth values accordingly, clipping any resulting negatives to NoData.
#
# Step 3 — Removing shallow pixels below a X-inch threshold.
#
# Step 4 — Denoising by removing small disconnected wet-pixel 
# clusters (< XX pixels, D8 connectivity).
#
# Created by: Andy Carter, PE
# Created - 2026.04.24
# ************************************************************

# ************************************************************
import numpy as np
import rioxarray as rxr
import os
import netCDF4 as nc
from rasterio.enums import Resampling
import shutil
from typing import Tuple
from scipy.ndimage import label

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


# ...................................
def fn_dem_difference(clipped_path, burned_path, out_diff, tolerance, b_print_verbose):

    if b_print_verbose:
        print(f"Clipped DEM : {clipped_path}")
        print(f"Burned DEM  : {burned_path}")
        print(f"Tolerance   : {tolerance}")

    da_c = rxr.open_rasterio(clipped_path, masked=True).squeeze()
    da_b = rxr.open_rasterio(burned_path,  masked=True).squeeze()

    # ── Sanity checks ─────────────────────────────────────────────────────────
    if da_c.rio.crs != da_b.rio.crs:
        print(f"WARNING: CRS mismatch!\n  clipped={da_c.rio.crs}\n  burned={da_b.rio.crs}")
    if da_c.rio.transform() != da_b.rio.transform():
        print("WARNING: Transform mismatch — grids may not align perfectly.")
    if da_c.shape != da_b.shape:
        raise ValueError(
            f"Shape mismatch: clipped={da_c.shape}, burned={da_b.shape}. "
            "Rasters must have identical dimensions."
        )

    # ── Compute difference (NaN propagates through nodata cells) ─────────────
    diff = da_c - da_b

    # ── Statistics (valid cells only) ─────────────────────────────────────────
    valid = diff.values[~np.isnan(diff.values)]
    n_total = valid.size
    n_diff  = int(np.sum(np.abs(valid) > tolerance))
    n_same  = n_total - n_diff

    if b_print_verbose:
        print(f"\nValid cells : {n_total:,}")
        print(f"  Different : {n_diff:,}  ({100 * n_diff / n_total:.2f} %)")
        print(f"  Identical : {n_same:,}  ({100 * n_same / n_total:.2f} %)")
        if n_total > 0:
            print(f"  Min diff  : {valid.min():.4f} m")
            print(f"  Max diff  : {valid.max():.4f} m")
            print(f"  Mean diff : {valid.mean():.4f} m")
            print(f"  Std diff  : {valid.std():.4f} m")

    # ── Write difference raster ───────────────────────────────────────────────
    diff = diff.astype(np.float32)
    diff.rio.write_nodata(-9999.0, inplace=True)
    diff.rio.to_raster(out_diff)

    if b_print_verbose:
        print(f"\nDifference raster → {out_diff}")
        print("Done.")
# ...................................


# ---------------------------
def fn_load_diff_raster(str_diff_path: str, int_target_rows: int, int_target_cols: int, b_print_output) -> np.ndarray:
    """
    Load the difference raster and resample to match the netCDF grid if needed.
    Returns a float64 2D array (rows=y, cols=x) with NaN where nodata.
    """
    da_diff = rxr.open_rasterio(str_diff_path, masked=True).squeeze()

    if da_diff.shape != (int_target_rows, int_target_cols):
        if b_print_output:
            print(f"  Resampling diff raster: {da_diff.shape} → ({int_target_rows},{int_target_cols})")
        da_diff = da_diff.rio.reproject(
            da_diff.rio.crs,
            shape=(int_target_rows, int_target_cols),
            resampling=Resampling.bilinear
        )

    # masked=True already converts nodata → NaN; just ensure float64
    return da_diff.values.astype(np.float64)
# ---------------------------


# ................................
def fn_apply_dem_correction(str_nc_in, str_diff, str_nc_out, b_print_output) -> None:

    if b_print_output:
        print(f"Input netCDF  : {str_nc_in}")
        print(f"Diff raster   : {str_diff}")
        print(f"Output netCDF : {str_nc_out}")

    shutil.copy2(str_nc_in, str_nc_out)
    
    if b_print_output:
        print("\nCopied source netCDF → output.")

    with nc.Dataset(str_nc_out, "r+") as ds:

        int_ny, int_nx = ds.variables["terrain"].shape
        int_n_bands    = ds.variables["depth"].shape[0]  # (intensity, y, x)

        # ── Load and align difference raster ─────────────────────────────────
        if b_print_output:
            print(f"Loading diff raster ({int_ny} rows × {int_nx} cols)...")
        arr_diff       = fn_load_diff_raster(str_diff, int_ny, int_nx, b_print_output)
        bool_diff_valid = ~np.isnan(arr_diff)
        
        if b_print_output:
            print(f"  Diff valid cells : {bool_diff_valid.sum():,} / {arr_diff.size:,}")

        # ── 1. Terrain: terrain += difference ────────────────────────────────
        if b_print_output:
            print("\nCorrecting terrain layer...")
        var_terrain     = ds.variables["terrain"]
        arr_terrain     = var_terrain[:].astype(np.float64)

        flt_terrain_nd  = var_terrain._FillValue if hasattr(var_terrain, "_FillValue") else -9999.0
        bool_terrain_nd = np.isclose(arr_terrain, flt_terrain_nd, atol=1e-3) | np.isnan(arr_terrain)

        bool_apply = bool_diff_valid & ~bool_terrain_nd
        arr_terrain[bool_apply] += arr_diff[bool_apply]
        arr_terrain[bool_terrain_nd] = flt_terrain_nd
        var_terrain[:] = arr_terrain.astype(np.float32)

        if b_print_output:
            print(f"  Terrain cells updated : {bool_apply.sum():,}")
    
            # ── 2. Depth bands: depth -= difference; clip negatives to NoData ────
            print(f"\nCorrecting {int_n_bands} depth bands...")
        var_depth = ds.variables["depth"]

        int_total_corrected = 0
        int_total_clipped   = 0

        for int_band in range(int_n_bands):
            arr_band     = var_depth[int_band, :, :].astype(np.float64)
            bool_band_nd = np.isnan(arr_band)

            bool_wet        = arr_band > 0.0
            bool_apply_band = bool_diff_valid & ~bool_band_nd & bool_wet
            arr_band[bool_apply_band] -= arr_diff[bool_apply_band]

            bool_neg = bool_apply_band & (arr_band < 0.0)
            arr_band[bool_neg] = np.nan

            int_total_corrected += int(bool_apply_band.sum())
            int_total_clipped   += int(bool_neg.sum())

            var_depth[int_band, :, :] = arr_band.astype(np.float32)

            if b_print_output:
                print(f"  Band {int_band+1:>2}/{int_n_bands}  "
                      f"corrected={bool_apply_band.sum():,}  "
                      f"clipped_to_nodata={bool_neg.sum():,}")
                
        if b_print_output:
            print("\nSummary:")
            print(f"  Total depth cells corrected    : {int_total_corrected:,}")
            print(f"  Total depth cells → NoData (<0): {int_total_clipped:,}")
            print(f"\nOutput written → {str_nc_out}")
            print("Done.")
# ................................


# --------------------------------------
def fn_mask_shallow_depth(input_path: str, output_path: str, threshold_inches: float, b_print_output) -> None:
    
    INCHES_TO_METERS = 0.0254          # 1 inch = 0.0254 m
    
    threshold_m = threshold_inches * INCHES_TO_METERS
    
    if b_print_output:
        print(f"Threshold : {threshold_inches} inches = {threshold_m:.4f} m")
        print(f"Input     : {input_path}")
        print(f"Output    : {output_path}")

    # Copy the whole file first so all metadata, CRS, and other variables
    # are preserved exactly — then we only touch the 'depth' variable.
    shutil.copy2(input_path, output_path)
    
    if b_print_output:
        print("Copied source file → output (preserving all metadata).")

    with nc.Dataset(output_path, "r+") as ds:
        depth_var = ds.variables["depth"]          # shape: (intensity, y, x)
        data = depth_var[:]                        # masked array

        # Build mask: True where depth is valid AND below threshold
        below = (
            (~np.ma.getmaskarray(data))            # currently valid pixels …
            & (data.data < threshold_m)            # … that are too shallow
        )

        n_masked = int(below.sum())
        n_total  = int(data.size)
        
        if b_print_output:
            print(f"Pixels below {threshold_inches}″: {n_masked:,} / {n_total:,} "
                  f"({100 * n_masked / n_total:.2f} %)")

        # Apply: set shallow pixels to NaN (the variable's _FillValue)
        data[below] = np.nan

        # Write back — netCDF4-python respects the existing _FillValue
        depth_var[:] = data
        
    if b_print_output:
        print("Done. Shallow depth pixels set to NaN (no data).")
# --------------------------------------


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def fn_denoise_band(arr_band, int_min_pixels):
    # type: (np.ndarray, int) -> Tuple[np.ndarray, int, int]
    """
    Remove connected wet-pixel components smaller than int_min_pixels.

    Parameters
    ----------
    arr_band       : 2D float array for one depth band (NaN = nodata)
    int_min_pixels : minimum component size to retain

    Returns
    -------
    arr_clean      : corrected band (small components set to NaN)
    int_n_removed  : number of components removed
    int_px_removed : number of pixels set to NaN
    """
    
    # D8 connectivity kernel (all 8 neighbours)
    ARR_D8 = np.ones((3, 3), dtype=int)
    
    # Wet mask: valid, non-NaN, depth > 0
    bool_wet = (~np.isnan(arr_band)) & (arr_band > 0.0)

    if not bool_wet.any():
        return arr_band, 0, 0

    # Label connected components with D8 connectivity
    arr_labels, int_n_components = label(bool_wet, structure=ARR_D8)

    arr_clean      = arr_band.copy()
    int_n_removed  = 0
    int_px_removed = 0

    # Count pixels per component using bincount (fast)
    arr_counts = np.bincount(arr_labels.ravel())   # index 0 = background

    for int_comp in range(1, int_n_components + 1):
        if arr_counts[int_comp] < int_min_pixels:
            arr_clean[arr_labels == int_comp] = np.nan
            int_n_removed  += 1
            int_px_removed += int(arr_counts[int_comp])

    return arr_clean, int_n_removed, int_px_removed
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
def fn_denoise_depth_layers(str_nc_in, str_nc_out, int_min_pixels, b_print_output):
    # type: (str, str, int) -> None

    if b_print_output:
        print("Input netCDF    : {}".format(str_nc_in))
        print("Output netCDF   : {}".format(str_nc_out))
        print("Min pixels keep : {}".format(int_min_pixels))

    shutil.copy2(str_nc_in, str_nc_out)
    
    if b_print_output:
        print("\nCopied source netCDF -> output.")

    with nc.Dataset(str_nc_out, "r+") as ds:
        var_depth   = ds.variables["depth"]          # (intensity, y, x)
        int_n_bands = var_depth.shape[0]

        int_total_components_removed = 0
        int_total_pixels_removed     = 0

        for int_band in range(int_n_bands):
            arr_band = var_depth[int_band, :, :].astype(np.float64)

            arr_clean, int_n_removed, int_px_removed = fn_denoise_band(arr_band, int_min_pixels)

            var_depth[int_band, :, :] = arr_clean.astype(np.float32)

            int_total_components_removed += int_n_removed
            int_total_pixels_removed     += int_px_removed

            if b_print_output:
                print("  Band {:>2}/{}  components_removed={:,}  pixels_removed={:,}".format(
                    int_band + 1, int_n_bands, int_n_removed, int_px_removed))

    if b_print_output:
        print("\nSummary:")
        print("  Total components removed : {:,}".format(int_total_components_removed))
        print("  Total pixels -> NoData   : {:,}".format(int_total_pixels_removed))
        print("\nOutput written -> {}".format(str_nc_out))
        print("Done.")
# ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,


# ==============================
def fn_replace_terrain(str_nc_path, str_clipped_dem_path, b_print_output) -> None:
    """
    Replace the 'terrain' variable in the netCDF with values from the
    original (unburned) clipped DEM, resampled to match the netCDF grid.
    """
    if b_print_output:
        print(f"  Replacing terrain with unburned DEM: {str_clipped_dem_path}")

    with nc.Dataset(str_nc_path, "r+") as ds:
        var_terrain = ds.variables["terrain"]
        int_ny, int_nx = var_terrain.shape

        # Load and align the clipped (unburned) DEM to the netCDF grid
        da_clipped = rxr.open_rasterio(str_clipped_dem_path, masked=True).squeeze()

        if da_clipped.shape != (int_ny, int_nx):
            if b_print_output:
                print(f"    Resampling clipped DEM: {da_clipped.shape} → ({int_ny}, {int_nx})")
            da_clipped = da_clipped.rio.reproject(
                da_clipped.rio.crs,
                shape=(int_ny, int_nx),
                resampling=Resampling.bilinear
            )

        arr_clipped = da_clipped.values.astype(np.float64)

        # Preserve existing nodata mask from the netCDF terrain
        flt_terrain_nd  = var_terrain._FillValue if hasattr(var_terrain, "_FillValue") else -9999.0
        arr_terrain_old = var_terrain[:].astype(np.float64)
        bool_nd         = np.isclose(arr_terrain_old, flt_terrain_nd, atol=1e-3) | np.isnan(arr_terrain_old)

        # Write unburned values, keeping nodata cells intact
        arr_new = arr_clipped.copy()
        arr_new[bool_nd] = flt_terrain_nd

        var_terrain[:] = arr_new.astype(np.float32)

        if b_print_output:
            n_updated = int((~bool_nd).sum())
            print(f"    Terrain cells replaced : {n_updated:,}")
            print(f"    NoData cells preserved : {int(bool_nd.sum()):,}")
# ==============================


# .........................................................
def fn_heal_netcdf_04(
    str_global_config_file_path,
    str_local_config_file_path,
    b_print_output
):
    
    # difference in meters between basre terrain and burned terrain that 
    # requires adjustment  -- HARDCODED
    flt_depth_diff_tolerance = 0.01

    #warnings.filterwarnings("ignore", category=UserWarning)

    # ---- Header output ----
    if b_print_output:
        print(f"""
+=================================================================+
|                  HEAL FLOOD INUNDATION NETCDF                   |
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
        print(" -- Script 04: Heal Flood Inundation NetCDF")

    # ==================================================================
    # READ GLOBAL CONFIG
    # ==================================================================
    global_config = configparser.ConfigParser()
    global_config.read(str_global_config_file_path)

    global_section_schema = {
        'lisflood_settings': ['downscale']
    }
    
    # TODO: 2026.04.24 -- paramaters from global config
    #global_section_schema = {
    #    'clean_fim_stack': ['min_depth_threshold_inches'],
    #    'clean_fim_stack': ['min_connected_pixels'],
    #}

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
    
    #print(dict_all_params)
    
    # From local config
    str_out_root_folder = dict_all_params['out_root_folder']
    str_catchment = dict_all_params['catchment']
    
    # -------------------
    # Root folder to search
    root_dir_01 = os.path.join(str_out_root_folder, str_catchment, "01_stream_delineation")
    root_dir_03 = os.path.join(str_out_root_folder, str_catchment, "03_depth_nc")
    
    # Excpected dem filepaths
    str_clipped_dem_filepath = os.path.join(root_dir_01, "00_dem_clipped_5070.tif")
    str_burned_dem_filepath = os.path.join(root_dir_01, "00_dem_burn_roads_5070_d4fixed.tif")
    str_ouput_diff_filepath = os.path.join(root_dir_01, "04_dem_difference.tif")
    
    # Excpected FIM netCDF
    str_burned_netcdf_filename = 'depth_' + str_catchment + ".nc"
    str_burned_netcdf_filepath = os.path.join(root_dir_03, str_burned_netcdf_filename)
    
    # Output folder
    str_outfolder = os.path.join(str_out_root_folder, str_catchment, "04_healed_depth_nc")
    os.makedirs(str_outfolder, exist_ok=True)
    
    str_healed_netcdf_filepath = os.path.join(str_outfolder, str_burned_netcdf_filename)
    
    
    # --------------------------------------------------
    if b_print_output:
        print('  -- STEP 1: Determing burned terrain cells')

    fn_dem_difference(str_clipped_dem_filepath, str_burned_dem_filepath, 
                      str_ouput_diff_filepath, flt_depth_diff_tolerance, b_print_output)
    
    # --------------------------------------------------
    if b_print_output:
        print('  -- STEP 2: Healing depth layers')
    
    fn_apply_dem_correction(str_burned_netcdf_filepath, str_ouput_diff_filepath,
                            str_healed_netcdf_filepath, b_print_output)
    
    # --------------------------------------------------
    # **** Note hardcoded minimum depth ****
    FLT_THRESHOLD_INCHES = 3.0
    
    if b_print_output:
        print('  -- STEP 3: Removing shallow depth pixels')
    
    str_shallow_remove_netcdf_filepath = str_healed_netcdf_filepath[:-3] + "_shallow_removed.nc"
    
    fn_mask_shallow_depth(str_healed_netcdf_filepath, str_shallow_remove_netcdf_filepath, 
                          FLT_THRESHOLD_INCHES, b_print_output)
    
    
    # --------------------------------------------------
    if b_print_output:
        print('  -- STEP 4: Removing noisey disconected pixels')
    
    # **** Note hardcoded max pixel cluster removal ****
    INT_MIN_PIXELS = 15 #size of D8 disconnected cluster to removal
    

    str_denoised_netcdf_filepath = str_healed_netcdf_filepath[:-3] + "_denoised.nc"
    
    fn_denoise_depth_layers(str_shallow_remove_netcdf_filepath, str_denoised_netcdf_filepath,
                            INT_MIN_PIXELS, b_print_output)
    
    # --------------------------------------------------
    ''' TODO -- 20260424 -- There is an error here!!
    if b_print_output:
        print('  -- STEP 5: Replacing terrain with unburned DEM')

    fn_replace_terrain(str_denoised_netcdf_filepath, 
                       str_clipped_dem_filepath, b_print_output)
    '''
    
# .........................................................


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == '__main__':

    flt_start_run = time.time()
    
    parser = argparse.ArgumentParser(description='========== HEAl FLOOD INUNDATION NETCDF =========')
    
    parser.add_argument('-g',
                        dest = "str_global_config_file_path",
                        help=r'REQUIRED: Global configuration filepath Example:C:\Users\civil\dev\lisflood2fim\config\global_config.ini',
                        required=True,
                        metavar='FILE',
                        type=lambda x: is_valid_file(parser, x))
    
    parser.add_argument('-c',
                        dest = "str_local_config_file_path",
                        help=r'REQUIRED: LOCAL configuration filepath Example:C:\Users\civil\dev\lisflood2fim\config\local_config.ini',
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

    fn_heal_netcdf_04(str_global_config_file_path, str_local_config_file_path, b_print_output)

    flt_end_run = time.time()
    flt_time_pass = (flt_end_run - flt_start_run) // 1
    time_pass = datetime.timedelta(seconds=flt_time_pass)
    
    print('Compute Time: ' + str(time_pass))
 #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~