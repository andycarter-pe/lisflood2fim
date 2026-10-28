# ************************************************************
# LISFLOOD-FP to FIM
# Script 05 - upscale netCDF FIM to high resolution terrain
#
# Takes a coarse (healed) LISFLOOD-FP depth stack netCDF and a high
# resolution terrain (local GeoTIFF or the statewide S3 VRT) and creates
# a high resolution depth stack netCDF in EPSG:5070.
#
# Step 1 — Coarse water surface (WSE = terrain + depth) at each coarse
#          cell center, per intensity.
#
# Step 2 — Halo: extend the WSE into dry coarse cells that touch wet
#          cells (lowest wet D8 neighbor) so that low fine-resolution
#          terrain along the wet/dry edge can flood.
#
# Step 3 — Bilinear interpolation of the WSE from coarse cell centers to
#          fine cell centers. Only wet (or halo) centers contribute; the
#          weights are re-normalized so an edge cell is never pulled toward
#          a NaN / zero neighbor.
#
# Step 4 — Fine depth = WSE - fine terrain. Remove depths below a
#          X-inch threshold and anything outside the HUC12 (where the
#          coarse terrain is NaN).
#
# Step 5 — Connectivity: remove fine wet clusters (D8) that do not touch
#          a fine cell inside an originally wet coarse cell (i.e. ponds
#          created only by the halo).
#
# Step 6 — Post process: depth never decreases with intensity (running
#          max), so a cell wet at a lower intensity stays wet at higher ones.
#
# output_downsample = 1/2/3/4 writes the output at 1, 1/4, 1/9 or 1/16 of
# the full resolution pixels; the fine terrain is block-averaged to that
# grid first.
#
# Run on its own after lisflood2fim_huc12_00 (it is not part of that
# stack).  Settings come from the [upscale] section of the global config
# and the terrain from [datasource] vrt_terrain; the input is
# <out_root_folder>/<catchment>/04_healed_depth_nc/depth_<catchment>.nc and
# the output goes to 05_upscaled_depth_nc/depth_<catchment>.nc
#
# Created by: Andy Carter, PE
# Created - 2026.10.28
# Revised - 2026.10.28 -- reads the global / local config like scripts 01-04;
#                          per-intensity stability variables carried over
# ************************************************************

# ************************************************************
import rioxarray as rxr
import xarray as xr
import numpy as np
import os

import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds
from rasterio.transform import from_origin
from scipy.ndimage import label

import argparse
import configparser
import time
import datetime
# ************************************************************


# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist" % arg)
    else:
        return arg
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
def fn_coarse_grid_info(ds):
    # Returns coarse cell size and the outer bounds (left, top) of the grid.
    # x/y coordinates in the netCDF are cell centers.
    arr_x = ds['x'].values
    arr_y = ds['y'].values

    flt_dx = float(arr_x[1] - arr_x[0])
    flt_dy = float(arr_y[0] - arr_y[1])   # y is descending (north up)

    if not np.isclose(flt_dx, flt_dy, rtol=1e-6):
        raise ValueError(f"Non-square coarse cells: dx={flt_dx}, dy={flt_dy}")

    flt_left = float(arr_x[0]) - flt_dx / 2.0
    flt_top = float(arr_y[0]) + flt_dy / 2.0

    return flt_dx, flt_left, flt_top
# --------------------------------------------------


# --------------------------------------------------
def fn_load_fine_terrain(str_terrain_path, ds, flt_left, flt_top,
                         flt_fine_res, int_ny_fine, int_nx_fine, b_print_output):
    # Reads the high resolution terrain (any CRS, local file or /vsis3/ VRT),
    # clipped to the netCDF extent, and resamples it (bilinear) onto the
    # fine EPSG:5070 grid that nests exactly inside the coarse grid.
    # For /vsis3/ paths, AWS credentials are taken from the usual GDAL
    # sources: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY env vars,
    # AWS_PROFILE, or ~/.aws/credentials.

    str_crs = "EPSG:5070"

    flt_right = flt_left + int_nx_fine * flt_fine_res
    flt_bottom = flt_top - int_ny_fine * flt_fine_res

    # Read band 1 only with rasterio. The statewide VRT has bands of mixed
    # dtype, which rioxarray.open_rasterio refuses to open.
    # buffer the window a little so the bilinear resample has data at the edge
    flt_buffer = 4 * flt_fine_res

    with rasterio.open(str_terrain_path) as src:
        tup_bounds = transform_bounds(str_crs, src.crs,
                                      flt_left - flt_buffer, flt_bottom - flt_buffer,
                                      flt_right + flt_buffer, flt_top + flt_buffer,
                                      densify_pts=21)

        window = from_bounds(*tup_bounds, transform=src.transform)
        int_col0 = max(int(np.floor(window.col_off)), 0)
        int_row0 = max(int(np.floor(window.row_off)), 0)
        int_col1 = min(int(np.ceil(window.col_off + window.width)), src.width)
        int_row1 = min(int(np.ceil(window.row_off + window.height)), src.height)
        if int_col1 <= int_col0 or int_row1 <= int_row0:
            raise ValueError("Terrain does not overlap the netCDF extent")
        window = Window(int_col0, int_row0, int_col1 - int_col0, int_row1 - int_row0)

        arr_src = src.read(1, window=window, masked=True).astype(np.float64).filled(np.nan)
        transform_src = src.window_transform(window)
        crs_src = src.crs

        if b_print_output:
            print(f"  -- Terrain source: {src.count} band(s) {src.dtypes}, using band 1")
            print(f"  -- Terrain source window: {arr_src.shape[1]} x {arr_src.shape[0]} "
                  f"cells ({crs_src})")

    transform_fine = from_origin(flt_left, flt_top, flt_fine_res, flt_fine_res)

    arr_fine = np.full((int_ny_fine, int_nx_fine), np.nan, dtype=np.float64)
    reproject(source=arr_src, destination=arr_fine,
              src_transform=transform_src, src_crs=crs_src, src_nodata=np.nan,
              dst_transform=transform_fine, dst_crs=str_crs, dst_nodata=np.nan,
              resampling=Resampling.bilinear)

    return arr_fine, transform_fine
# --------------------------------------------------


# --------------------------------------------------
def fn_add_halo(arr_wse, arr_huc_mask, int_halo_cells):
    # Extends WSE into dry coarse cells adjacent (D8) to wet cells, one ring
    # at a time. Each new cell gets the LOWEST WSE of its wet / already
    # extended D8 neighbors. The minimum (not the mean) keeps a dry cell
    # between a high pool and a low channel (e.g. a road embankment) from
    # receiving the pool's water surface. Only cells inside the HUC12 are
    # extended.

    int_ny, int_nx = arr_wse.shape
    arr_out = arr_wse.copy()

    for _ in range(int_halo_cells):
        arr_pad = np.pad(arr_out, 1, mode='constant', constant_values=np.nan)
        arr_nbr_min = np.full(arr_out.shape, np.inf)
        for int_di in (-1, 0, 1):
            for int_dj in (-1, 0, 1):
                if int_di == 0 and int_dj == 0:
                    continue
                arr_nbr = arr_pad[1 + int_di:1 + int_di + int_ny, 1 + int_dj:1 + int_dj + int_nx]
                arr_nbr_min = np.fmin(arr_nbr_min, np.where(np.isfinite(arr_nbr), arr_nbr, np.inf))

        arr_new = (~np.isfinite(arr_out)) & np.isfinite(arr_nbr_min) & arr_huc_mask
        if not arr_new.any():
            break

        arr_out[arr_new] = arr_nbr_min[arr_new]

    return arr_out
# --------------------------------------------------


# --------------------------------------------------
def fn_bilinear_wet_only(arr_wse, flt_ratio, int_nx_out, int_row0, int_row1):
    # Bilinear interpolation from coarse cell centers to the output cell
    # centers of output rows [int_row0, int_row1). flt_ratio is the output
    # cell size divided by the coarse cell size (e.g. 1/18, or 2/18 at -d 2).
    # The output grid shares its upper-left corner with the coarse grid.
    #
    # NaN coarse centers get zero weight and the remaining weights are
    # re-normalized, so a fine cell between a wet and a dry center takes the
    # wet WSE instead of being dragged toward NaN (edge handling).
    #
    # Outside the outermost centers (half a coarse cell at the grid border)
    # the nearest center row/column is used (index clamped).

    int_ny_c, int_nx_c = arr_wse.shape

    # fractional coarse index of each output cell center
    arr_fy = (np.arange(int_row0, int_row1) + 0.5) * flt_ratio - 0.5
    arr_fx = (np.arange(int_nx_out) + 0.5) * flt_ratio - 0.5

    arr_fy = np.clip(arr_fy, 0, int_ny_c - 1)
    arr_fx = np.clip(arr_fx, 0, int_nx_c - 1)

    arr_i0 = np.minimum(np.floor(arr_fy).astype(np.int64), int_ny_c - 2) if int_ny_c > 1 else np.zeros(arr_fy.shape, np.int64)
    arr_j0 = np.minimum(np.floor(arr_fx).astype(np.int64), int_nx_c - 2) if int_nx_c > 1 else np.zeros(arr_fx.shape, np.int64)
    arr_i1 = np.minimum(arr_i0 + 1, int_ny_c - 1)
    arr_j1 = np.minimum(arr_j0 + 1, int_nx_c - 1)

    arr_wy = (arr_fy - arr_i0)[:, None]
    arr_wx = (arr_fx - arr_j0)[None, :]

    arr_valid = np.isfinite(arr_wse)
    arr_val0 = np.where(arr_valid, arr_wse, 0.0)

    arr_num = np.zeros((arr_fy.size, arr_fx.size), dtype=np.float64)
    arr_den = np.zeros_like(arr_num)

    for arr_i, arr_w_i in ((arr_i0, 1.0 - arr_wy), (arr_i1, arr_wy)):
        for arr_j, arr_w_j in ((arr_j0, 1.0 - arr_wx), (arr_j1, arr_wx)):
            arr_w = arr_w_i * arr_w_j * arr_valid[np.ix_(arr_i, arr_j)]
            arr_num += arr_w * arr_val0[np.ix_(arr_i, arr_j)]
            arr_den += arr_w

    with np.errstate(invalid='ignore', divide='ignore'):
        arr_out = np.where(arr_den > 1e-9, arr_num / arr_den, np.nan)

    # A fine cell sitting exactly on a dry center line can get den==0 even
    # though a wet neighbor is within the stencil; that's handled by the
    # halo, which makes those centers valid.
    return arr_out
# --------------------------------------------------


# --------------------------------------------------
def fn_expand(arr_coarse, flt_ratio, int_ny_out, int_nx_out):
    # Boolean coarse mask -> output grid: each output cell takes the value of
    # the coarse cell holding its center. Output cells whose center falls
    # outside the coarse grid (padding at -d 4) are False.
    int_ny_c, int_nx_c = arr_coarse.shape
    arr_ri = np.floor((np.arange(int_ny_out) + 0.5) * flt_ratio + 1e-9).astype(np.int64)
    arr_ci = np.floor((np.arange(int_nx_out) + 0.5) * flt_ratio + 1e-9).astype(np.int64)
    arr_out = arr_coarse[np.ix_(np.minimum(arr_ri, int_ny_c - 1), np.minimum(arr_ci, int_nx_c - 1))]
    return arr_out & (arr_ri < int_ny_c)[:, None] & (arr_ci < int_nx_c)[None, :]
# --------------------------------------------------


# --------------------------------------------------
def fn_block_mean(arr, int_block):
    # Mean of int_block x int_block blocks (NaN ignored), padding the
    # bottom / right edge with NaN when the shape is not a multiple.
    if int_block == 1:
        return arr
    int_ny, int_nx = arr.shape
    int_ny_o = -(-int_ny // int_block)
    int_nx_o = -(-int_nx // int_block)
    arr_pad = np.full((int_ny_o * int_block, int_nx_o * int_block), np.nan, dtype=arr.dtype)
    arr_pad[:int_ny, :int_nx] = arr
    arr_blk = arr_pad.reshape(int_ny_o, int_block, int_nx_o, int_block)
    arr_cnt = np.isfinite(arr_blk).sum(axis=(1, 3))
    arr_sum = np.nansum(arr_blk, axis=(1, 3))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(arr_cnt > 0, arr_sum / arr_cnt, np.nan)
# --------------------------------------------------


# --------------------------------------------------
def fn_enforce_monotonic(arr_depth, arr_intensity):
    # Depth must not decrease as intensity increases. Running maximum along
    # increasing intensity: a cell wet at a lower intensity stays wet (at
    # least as deep) at every higher intensity. Returns counts of cells that
    # were dry and made wet, and wet cells that were deepened.
    arr_order = np.argsort(arr_intensity)
    arr_sorted = arr_depth[arr_order]
    arr_fixed = np.fmax.accumulate(arr_sorted, axis=0)

    int_wet_added = int((np.isnan(arr_sorted) & np.isfinite(arr_fixed)).sum())
    int_deepened = int((np.isfinite(arr_sorted) & (arr_fixed > arr_sorted)).sum())

    arr_out = np.empty_like(arr_depth)
    arr_out[arr_order] = arr_fixed
    return arr_out, int_wet_added, int_deepened
# --------------------------------------------------


# --------------------------------------------------
def fn_upscale_one_intensity(arr_depth_c, arr_terrain_c, arr_dem_f, arr_huc_f,
                             flt_ratio, int_halo_cells, flt_min_depth_m,
                             b_connectivity, int_rows_per_block=2048):

    arr_huc_c = np.isfinite(arr_terrain_c)
    arr_wet_c = np.isfinite(arr_depth_c) & (arr_depth_c > 0) & arr_huc_c

    # Step 1 -- coarse WSE at cell centers
    arr_wse_c = np.where(arr_wet_c, arr_terrain_c + arr_depth_c, np.nan)

    # Step 2 -- halo into dry neighbors
    if int_halo_cells > 0:
        arr_wse_c = fn_add_halo(arr_wse_c, arr_huc_c, int_halo_cells)

    # Step 3 + 4 -- bilinear WSE to fine grid, minus fine terrain
    int_ny_f, int_nx_f = arr_dem_f.shape
    arr_depth_f = np.full((int_ny_f, int_nx_f), np.nan, dtype=np.float32)

    for int_r0 in range(0, int_ny_f, int_rows_per_block):
        int_r1 = min(int_r0 + int_rows_per_block, int_ny_f)
        arr_wse_f = fn_bilinear_wet_only(arr_wse_c, flt_ratio, int_nx_f, int_r0, int_r1)
        arr_d = arr_wse_f - arr_dem_f[int_r0:int_r1]
        arr_keep = (arr_d >= flt_min_depth_m) & arr_huc_f[int_r0:int_r1]
        arr_depth_f[int_r0:int_r1] = np.where(arr_keep, arr_d, np.nan).astype(np.float32)

    # Step 5 -- drop fine clusters that never touch an originally wet coarse cell
    if b_connectivity and int_halo_cells > 0:
        arr_wet_f = np.isfinite(arr_depth_f)
        arr_labels, int_n = label(arr_wet_f, structure=np.ones((3, 3), dtype=bool))
        if int_n > 0:
            arr_seed = arr_wet_f & fn_expand(arr_wet_c, flt_ratio, int_ny_f, int_nx_f)
            arr_keep_ids = np.zeros(int_n + 1, dtype=bool)
            arr_keep_ids[np.unique(arr_labels[arr_seed])] = True
            arr_keep_ids[0] = False
            arr_depth_f[~arr_keep_ids[arr_labels]] = np.nan

    return arr_depth_f
# --------------------------------------------------


# .........................................................
def fn_upscale_netcdf(str_netcdf_path, str_terrain_path, str_output_path,
                      flt_target_res, int_halo_cells, flt_min_depth_in,
                      b_connectivity, int_downsample, b_monotonic, b_print_output):

    str_crs = "EPSG:5070"
    INCHES_TO_METERS = 0.0254
    flt_min_depth_m = flt_min_depth_in * INCHES_TO_METERS

    if b_print_output:
        print(f"  ---    INPUT NETCDF: {str_netcdf_path}")
        print(f"  ---    TERRAIN: {str_terrain_path}")
        print(f"  ---    OUTPUT NETCDF: {str_output_path}")
        print(f"  ---    HALO CELLS: {int_halo_cells}  MIN DEPTH: {flt_min_depth_in} in  "
              f"CONNECTIVITY: {b_connectivity}")
        print(f"  ---    OUTPUT DOWNSAMPLE: {int_downsample}  MONOTONIC FIX: {b_monotonic}")
        print("===================================================================")

    ds = xr.open_dataset(str_netcdf_path)

    flt_dx, flt_left, flt_top = fn_coarse_grid_info(ds)

    # fine grid nests exactly inside the coarse grid
    int_factor = max(1, int(round(flt_dx / flt_target_res)))
    flt_fine_res = flt_dx / int_factor

    int_ny_c = ds.sizes['y']
    int_nx_c = ds.sizes['x']
    int_ny_f = int_ny_c * int_factor
    int_nx_f = int_nx_c * int_factor

    # output grid: full resolution (-d 1) or int_downsample x int_downsample
    # blocks of it (-d 2 = 1/4 of the pixels, -d 3 = 1/9, -d 4 = 1/16)
    flt_out_res = flt_fine_res * int_downsample
    flt_ratio = flt_out_res / flt_dx
    int_ny_o = -(-int_ny_f // int_downsample)
    int_nx_o = -(-int_nx_f // int_downsample)

    if b_print_output:
        print(f"  -- Coarse: {int_nx_c} x {int_ny_c} @ {flt_dx:.3f} m")
        print(f"  -- Fine:   {int_nx_f} x {int_ny_f} @ {flt_fine_res:.3f} m "
              f"({int_factor} x {int_factor} per coarse cell)")
        if int_downsample > 1:
            print(f"  -- Output: {int_nx_o} x {int_ny_o} @ {flt_out_res:.3f} m "
                  f"(fine terrain block-averaged {int_downsample} x {int_downsample})")
        print("  -- STEP 1: Loading high resolution terrain")

    arr_dem_full, _ = fn_load_fine_terrain(str_terrain_path, ds, flt_left, flt_top,
                                           flt_fine_res, int_ny_f, int_nx_f,
                                           b_print_output)

    # terrain on the output grid (mean of the full resolution cells)
    arr_dem_f = fn_block_mean(arr_dem_full, int_downsample)
    del arr_dem_full
    transform_fine = from_origin(flt_left, flt_top, flt_out_res, flt_out_res)
    int_ny_f, int_nx_f = int_ny_o, int_nx_o

    arr_terrain_c = ds['terrain'].values.astype(np.float64)
    arr_huc_f = (fn_expand(np.isfinite(arr_terrain_c), flt_ratio, int_ny_f, int_nx_f)
                 & np.isfinite(arr_dem_f))

    arr_intensity = ds['intensity'].values
    arr_out = np.full((arr_intensity.size, int_ny_f, int_nx_f), np.nan, dtype=np.float32)

    if b_print_output:
        print("  -- STEP 2: Upscaling depth layers")

    for int_k, flt_intensity in enumerate(arr_intensity):
        arr_depth_c = ds['depth'].isel(intensity=int_k).values.astype(np.float64)
        arr_out[int_k] = fn_upscale_one_intensity(arr_depth_c, arr_terrain_c, arr_dem_f,
                                                  arr_huc_f, flt_ratio, int_halo_cells,
                                                  flt_min_depth_m, b_connectivity)
        if b_print_output:
            print(f"     -- intensity {flt_intensity:>7.1f}: "
                  f"{int(np.isfinite(arr_depth_c).sum()):,} coarse wet -> "
                  f"{int(np.isfinite(arr_out[int_k]).sum()):,} fine wet")

    # -----------------------------
    # POST PROCESS: depth non-decreasing with intensity
    # -----------------------------
    if b_monotonic:
        arr_out, int_wet_added, int_deepened = fn_enforce_monotonic(arr_out, arr_intensity)
        if b_print_output:
            print("  -- STEP 3: Enforcing depth non-decreasing with intensity")
            print(f"     -- {int_wet_added:,} dry cells made wet, {int_deepened:,} wet cells deepened")

    # -----------------------------
    # BUILD OUTPUT DATASET
    # -----------------------------
    arr_x_f = flt_left + (np.arange(int_nx_f) + 0.5) * flt_out_res
    arr_y_f = flt_top - (np.arange(int_ny_f) + 0.5) * flt_out_res

    da_depth = xr.DataArray(arr_out, dims=('intensity', 'y', 'x'),
                            coords={'intensity': arr_intensity, 'y': arr_y_f, 'x': arr_x_f},
                            name='depth')
    da_terrain = xr.DataArray(np.where(arr_huc_f, arr_dem_f, np.nan).astype(np.float32),
                              dims=('y', 'x'), coords={'y': arr_y_f, 'x': arr_x_f},
                              name='terrain')

    ds_out = xr.Dataset({'depth': da_depth, 'terrain': da_terrain})
    ds_out = ds_out.rio.write_crs(str_crs)
    ds_out = ds_out.rio.write_transform(transform_fine)
    ds_out = ds_out.rio.write_coordinate_system()   # x/y standard_name, units, axis

    # grid_mapping + AREA_OR_POINT as in the coarse netCDF so GDAL / QGIS
    # pick up the CRS and cell registration
    for str_var in ('depth', 'terrain'):
        ds_out[str_var].attrs['AREA_OR_POINT'] = 'Area'
        ds_out[str_var].attrs['units'] = 'meters'
        ds_out[str_var].attrs['grid_mapping'] = 'spatial_ref'
    ds_out.coords['intensity'].attrs = dict(ds['intensity'].attrs)
    if 'units' not in ds_out.coords['intensity'].attrs:
        ds_out.coords['intensity'].attrs['units'] = 'mm h-1'

    # per-intensity variables (stable_flag, time_to_stable, sim_time,
    # qout_ratio_final from build_netcdf_03) carried over as they are
    for str_var, da_var in ds.data_vars.items():
        if da_var.dims == ('intensity',):
            ds_out[str_var] = ('intensity', da_var.values)
            ds_out[str_var].attrs = dict(da_var.attrs)

    # -----------------------------
    # GLOBAL ATTRIBUTES (carry over from input, then update)
    # -----------------------------
    ds_out.attrs = dict(ds.attrs)
    ds_out.attrs['01_type'] = 'depth'
    ds_out.attrs['02_description'] = 'LISFLOOD-FP Precipitation Raster Stack - upscaled to high resolution terrain'
    ds_out.attrs['06_history'] = (f"Upscaled on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    ds_out.attrs['09_source_netcdf'] = os.path.basename(str_netcdf_path)
    ds_out.attrs['10_source_terrain'] = str_terrain_path
    ds_out.attrs['11_upscale_method'] = ('bilinear WSE from coarse cell centers (wet-only, '
                                         're-normalized weights), depth = WSE - fine terrain')
    ds_out.attrs['12_coarse_res_m'] = flt_dx
    ds_out.attrs['13_fine_res_m'] = flt_out_res
    ds_out.attrs['14_halo_cells'] = int_halo_cells
    ds_out.attrs['15_min_depth_inches'] = flt_min_depth_in
    ds_out.attrs['16_connectivity_filter'] = str(b_connectivity)
    ds_out.attrs['17_output_downsample'] = int_downsample
    ds_out.attrs['18_monotonic_fix'] = str(b_monotonic)

    ds.close()

    # -----------------------------
    # SAVE NETCDF (compressed, chunked)
    # -----------------------------
    if b_print_output:
        print("  -- STEP 4: Writing compressed NetCDF")

    str_out_dir = os.path.dirname(os.path.abspath(str_output_path))
    os.makedirs(str_out_dir, exist_ok=True)

    # grid_mapping lives in .encoding after rio.write_crs; passing an explicit
    # encoding dict replaces it, so it is set as an attribute above and the
    # .encoding copy is cleared here to avoid a conflict on write.
    for str_var in ('depth', 'terrain'):
        ds_out[str_var].encoding.pop('grid_mapping', None)

    int_chunk = min(512, int_ny_f, int_nx_f)
    dict_encoding = {
        'x': {'_FillValue': None},
        'y': {'_FillValue': None},
        'depth': {'zlib': True, 'complevel': 5, 'dtype': 'float32',
                  '_FillValue': np.float32(np.nan), 'chunksizes': (1, int_chunk, int_chunk)},
        'terrain': {'zlib': True, 'complevel': 5, 'dtype': 'float32',
                    '_FillValue': np.float32(np.nan), 'chunksizes': (int_chunk, int_chunk)},
    }
    ds_out.to_netcdf(str_output_path, encoding=dict_encoding)

    return str_output_path
# .........................................................


# .........................................................
def fn_upscale_netcdf_05(
    str_global_config_file_path,
    str_local_config_file_path,
    b_print_output
):

    # ---- Header output ----
    if b_print_output:
        print(f"""
+=================================================================+
|        UPSCALE LISFLOOD-FP DEPTH NETCDF TO HIGH-RES TERRAIN      |
|                Created by Andy Carter, PE of                    |
|             Center for Water and the Environment                |
|                 University of Texas at Austin                   |
+-----------------------------------------------------------------+
  ---(g) INPUT GLOBAL CONFIGURATION FILE: {str_global_config_file_path}
  ---(c) LOCAL CONFIGURATION FILE:  {str_local_config_file_path}
  ---[r] PRINT OUTPUT: {b_print_output}""")
    else:
        print(" -- Script 05: Upscale Flood Inundation NetCDF")

    # ==================================================================
    # READ GLOBAL CONFIG
    # ==================================================================
    global_config = configparser.ConfigParser()
    global_config.read(str_global_config_file_path)

    if 'datasource' not in global_config:
        raise KeyError("Missing [datasource] section in GLOBAL config")
    str_terrain_path = global_config.get('datasource', 'vrt_terrain', fallback='').strip()
    if str_terrain_path == '':
        raise KeyError("Missing vrt_terrain in [datasource] of GLOBAL config")

    # [upscale] is optional; the fallbacks match the old command line defaults
    flt_target_res   = global_config.getfloat('upscale', 'target_res_m', fallback=3.0)
    int_halo_cells   = global_config.getint('upscale', 'halo_cells', fallback=1)
    flt_min_depth_in = global_config.getfloat('upscale', 'min_depth_inches', fallback=3.0)
    b_connectivity   = global_config.getboolean('upscale', 'connectivity_filter', fallback=True)
    int_downsample   = global_config.getint('upscale', 'output_downsample', fallback=1)
    b_monotonic      = global_config.getboolean('upscale', 'monotonic_fix', fallback=True)

    if int_downsample not in (1, 2, 3, 4):
        raise ValueError(f"[upscale] output_downsample must be 1, 2, 3 or 4; got {int_downsample}")

    # ==================================================================
    # READ LOCAL CONFIG
    # ==================================================================
    local_config = configparser.ConfigParser()
    local_config.read(str_local_config_file_path)

    if 'run_parameters' not in local_config:
        raise KeyError("Missing [run_parameters] section in LOCAL config")

    str_catchment = local_config.get('run_parameters', 'catchment', fallback='').strip()
    str_out_root_folder = local_config.get('run_parameters', 'out_root_folder', fallback='').strip()

    # healed netCDF from heal_netcdf_04 in, 05_upscaled_depth_nc out
    str_netcdf_path = os.path.join(str_out_root_folder, str_catchment,
                                   "04_healed_depth_nc", f"depth_{str_catchment}.nc")
    if not os.path.isfile(str_netcdf_path):
        raise FileNotFoundError(f"Healed depth netCDF not found: {str_netcdf_path}")

    str_output_path = os.path.join(str_out_root_folder, str_catchment,
                                   "05_upscaled_depth_nc", f"depth_{str_catchment}.nc")

    return fn_upscale_netcdf(str_netcdf_path, str_terrain_path, str_output_path,
                             flt_target_res, int_halo_cells, flt_min_depth_in,
                             b_connectivity, int_downsample, b_monotonic, b_print_output)
# .........................................................


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == '__main__':

    flt_start_run = time.time()

    parser = argparse.ArgumentParser(description='========= UPSCALE FLOOD INUNDATION NETCDF =========')

    parser.add_argument('-g',
                        dest = "str_global_config_file_path",
                        help=r'REQUIRED: Global configuration filepath Example:/app/lisflood2fim/config/demo_global_config.ini',
                        required=False,
                        default='/app/lisflood2fim/config/demo_global_config.ini',
                        metavar='FILE',
                        type=lambda x: is_valid_file(parser, x))

    parser.add_argument('-c',
                        dest = "str_local_config_file_path",
                        help=r'REQUIRED: LOCAL configuration filepath Example:/app/lisflood2fim/config/demo_local_config.ini',
                        required=False,
                        default='/app/lisflood2fim/config/demo_local_config.ini',
                        metavar='FILE',
                        type=lambda x: is_valid_file(parser, x))

    parser.add_argument('-r',
                        dest = "b_print_output",
                        help=r'OPTIONAL: Print output messages Default: True',
                        required=False,
                        default=True,
                        metavar='T/F',type=fn_str_to_bool)

    args = vars(parser.parse_args())

    fn_upscale_netcdf_05(args['str_global_config_file_path'],
                         args['str_local_config_file_path'],
                         args['b_print_output'])

    flt_end_run = time.time()
    flt_time_pass = (flt_end_run - flt_start_run) // 1
    time_pass = datetime.timedelta(seconds=flt_time_pass)

    print('Compute Time: ' + str(time_pass))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
