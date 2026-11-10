# ************************************************************
# LISFLOOD-FP to FIM
# Script 06 - build PMTiles from the upscaled depth netCDF
#
# Takes the upscaled depth stack netCDF from upscale_netcdf_05 and writes
# one Terrain-RGB (Terrarium) PMTiles archive per intensity layer, so each
# depth grid can be served as raster tiles (e.g. MapLibre) with no server.
#
# Step 1 — Read the intensity list (and stable_flag) from the netCDF.
#
# Step 2 — For each intensity layer: reproject the depth to the Web
#          Mercator (EPSG:3857) tile grid at max_zoom (bilinear when the
#          tile pixel is finer than the source cell, average when coarser).
#          Each lower zoom down to min_zoom is the 2 x 2 mean of the zoom
#          above (dry = 0).  Cut into 256 x 256 tiles and encode each tile
#          with any wet cell as a Terrarium PNG.  Fully dry tiles are not
#          stored.
#
# Step 3 — Write a catalog.json listing each layer (file, intensity,
#          stable_flag, size) and the bounds of the catchment.
#
# Encoding (Terrarium): depth_m = (R*256 + G + B/256) - 32768
#                       alpha = 0 where dry (depth 0 / no data)
#
# Runs as the last step of lisflood2fim_huc12_00 (after upscale_netcdf_05),
# or on its own with the same -g / -c / -r arguments.  Settings come from
# the [pmtiles] section of the global config; the input is
# <out_root_folder>/<catchment>/05_upscaled_depth_nc/depth_<catchment>.nc and
# the output goes to
# <out_root_folder>/<catchment>/06_pmtiles/depth_<catchment>_i<NN>_<in/hr>inhr.pmtiles
#
# Requires (beyond the geo environment): pillow, pmtiles
#
# Created by: Andy Carter, PE
# Created - 2026.11.10 -- from netcdf_to_pmtiles_batch.py; single catchment,
#                          reads the global / local config like scripts 01-05
# ************************************************************

# ************************************************************
import xarray as xr
import rioxarray as rxr
import numpy as np
import os
import io
import json
import math

from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_origin
from rasterio.warp import reproject, transform_bounds
from PIL import Image
from pmtiles.tile import Compression, TileType, zxy_to_tileid
from pmtiles.writer import Writer
from concurrent.futures import ProcessPoolExecutor, as_completed

import argparse
import configparser
import time
import datetime
# ************************************************************

# web mercator constants
INT_TILE_SIZE = 256
FLT_ORIGIN = 20037508.342789244     # web mercator half-world (m)
FLT_MAX_LAT = 85.0511287798066      # web mercator latitude limit
FLT_MM_PER_IN = 25.4


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
def fn_encode_terrarium(arr_depth_m):
    # Depth (m) -> RGBA uint8 Terrarium encoding.  Dry cells (depth <= 0)
    # are written as 0 m with alpha 0 (transparent).
    arr_v = arr_depth_m.astype(np.float64) + 32768.0
    arr_r = np.floor(arr_v / 256.0)
    arr_g = np.floor(arr_v) % 256
    arr_b = np.floor((arr_v - np.floor(arr_v)) * 256.0)
    arr_a = np.where(arr_depth_m > 0, 255, 0).astype(np.uint8)

    arr_rgba = np.concatenate([np.stack([arr_r, arr_g, arr_b], -1).astype(np.uint8),
                               arr_a[..., None]], -1)
    arr_rgba[arr_a == 0] = (128, 0, 0, 0)    # dry = 0 m, transparent
    return arr_rgba
# --------------------------------------------------


# --------------------------------------------------
def fn_lonlat_to_tile(flt_lon, flt_lat, int_zoom):
    # Web mercator (XYZ) tile column / row holding a lon / lat point
    int_n = 2 ** int_zoom
    flt_lat = max(min(flt_lat, FLT_MAX_LAT), -FLT_MAX_LAT)
    flt_lat_rad = math.radians(flt_lat)

    int_x = int(math.floor((flt_lon + 180.0) / 360.0 * int_n))
    int_y = int(math.floor((1.0 - math.asinh(math.tan(flt_lat_rad)) / math.pi) / 2.0 * int_n))

    int_x = min(max(int_x, 0), int_n - 1)
    int_y = min(max(int_y, 0), int_n - 1)
    return int_x, int_y
# --------------------------------------------------


# --------------------------------------------------
def fn_read_depth_layer(str_netcdf_path, int_idx):
    # Returns the depth of one intensity layer (dry / no data = 0), its
    # transform (outer corner) and CRS.  x / y in the netCDF are cell centers.
    with xr.open_dataset(str_netcdf_path) as ds:
        crs_src = ds.rio.crs
        if crs_src is None:
            # upscale_netcdf_05 writes NAD83 / Conus Albers
            crs_src = CRS.from_epsg(5070)

        arr_depth = ds['depth'].isel(intensity=int_idx).values.astype(np.float32)
        arr_x = ds['x'].values
        arr_y = ds['y'].values

    # NaN would poison bilinear / average at the wet edge
    arr_depth = np.nan_to_num(arr_depth, nan=0.0)
    arr_depth[arr_depth < 0] = 0

    flt_res = float(arr_x[1] - arr_x[0])
    transform_src = from_origin(float(arr_x[0]) - flt_res / 2.0,
                                float(arr_y[0]) + flt_res / 2.0,
                                flt_res, flt_res)

    return arr_depth, transform_src, crs_src, flt_res
# --------------------------------------------------


# --------------------------------------------------
def fn_mosaic_half(arr_mosaic, int_x0, int_y0, int_x1, int_y1):
    # Mosaic of tiles [x0..x1, y0..y1] at zoom z -> mosaic at zoom z-1
    # (mean of 2 x 2 pixels).  The mosaic is padded with 0 (dry) so it starts
    # and ends on an even tile.
    int_nx0, int_ny0 = int_x0 // 2, int_y0 // 2
    int_nx1, int_ny1 = int_x1 // 2, int_y1 // 2

    int_pad_left = (int_x0 - 2 * int_nx0) * INT_TILE_SIZE
    int_pad_top = (int_y0 - 2 * int_ny0) * INT_TILE_SIZE
    int_w = (int_nx1 - int_nx0 + 1) * 2 * INT_TILE_SIZE
    int_h = (int_ny1 - int_ny0 + 1) * 2 * INT_TILE_SIZE

    arr_pad = np.zeros((int_h, int_w), np.float32)
    arr_pad[int_pad_top:int_pad_top + arr_mosaic.shape[0],
            int_pad_left:int_pad_left + arr_mosaic.shape[1]] = arr_mosaic

    arr_half = arr_pad.reshape(int_h // 2, 2, int_w // 2, 2).mean(axis=(1, 3))
    return arr_half, int_nx0, int_ny0, int_nx1, int_ny1
# --------------------------------------------------


# --------------------------------------------------
def fn_mosaic_to_png_tiles(arr_mosaic, int_z, int_x0, int_y0, int_x1, int_y1):
    # Cuts the mosaic into 256 x 256 tiles; returns [(tile_id, png bytes)]
    # for every tile with a wet cell.  Fully dry tiles are not stored.
    list_tiles = []
    for int_tx in range(int_x0, int_x1 + 1):
        for int_ty in range(int_y0, int_y1 + 1):
            int_r0 = (int_ty - int_y0) * INT_TILE_SIZE
            int_c0 = (int_tx - int_x0) * INT_TILE_SIZE
            arr_sub = arr_mosaic[int_r0:int_r0 + INT_TILE_SIZE, int_c0:int_c0 + INT_TILE_SIZE]
            if not (arr_sub > 0).any():
                continue

            buf = io.BytesIO()
            Image.fromarray(fn_encode_terrarium(arr_sub), "RGBA").save(buf, "PNG", compress_level=6)
            list_tiles.append((zxy_to_tileid(int_z, int_tx, int_ty), buf.getvalue()))
    return list_tiles
# --------------------------------------------------


# .........................................................
def fn_build_one_pmtiles(str_netcdf_path, int_idx, str_output_path,
                         int_min_zoom, int_max_zoom, dict_meta_extra):
    # Tiles one intensity layer of the netCDF into a single PMTiles file.
    # Returns the number of tiles written.

    crs_web = CRS.from_epsg(3857)

    arr_depth, transform_src, crs_src, flt_res = fn_read_depth_layer(str_netcdf_path, int_idx)
    int_ny, int_nx = arr_depth.shape

    flt_left, flt_top = transform_src.c, transform_src.f
    flt_west, flt_south, flt_east, flt_north = transform_bounds(
        crs_src, "EPSG:4326",
        flt_left, flt_top - int_ny * flt_res,
        flt_left + int_nx * flt_res, flt_top,
        densify_pts=21)

    # -----------------------------
    # max zoom mosaic: reproject the depth to the web mercator tile grid
    # -----------------------------
    int_x0, int_y0 = fn_lonlat_to_tile(flt_west, flt_north, int_max_zoom)
    int_x1, int_y1 = fn_lonlat_to_tile(flt_east, flt_south, int_max_zoom)

    flt_tile_m = 2.0 * FLT_ORIGIN / 2 ** int_max_zoom
    flt_px = flt_tile_m / INT_TILE_SIZE
    int_w = (int_x1 - int_x0 + 1) * INT_TILE_SIZE
    int_h = (int_y1 - int_y0 + 1) * INT_TILE_SIZE

    transform_dst = Affine(flt_px, 0, -FLT_ORIGIN + int_x0 * flt_tile_m,
                           0, -flt_px, FLT_ORIGIN - int_y0 * flt_tile_m)
    arr_mosaic = np.zeros((int_h, int_w), np.float32)

    # bilinear when the tile pixel is finer than the source, else average
    resampling = Resampling.bilinear if flt_px <= flt_res else Resampling.average
    reproject(arr_depth, arr_mosaic,
              src_transform=transform_src, src_crs=crs_src,
              dst_transform=transform_dst, dst_crs=crs_web,
              resampling=resampling, src_nodata=None, dst_nodata=0)
    del arr_depth

    # -----------------------------
    # lower zooms: 2 x 2 mean of the zoom above (dry = 0), so a small
    # catchment still shows at a zoom where it is less than one pixel
    # (a direct reproject returns nothing there)
    # -----------------------------
    dict_png_by_zoom = {}
    for int_z in range(int_max_zoom, int_min_zoom - 1, -1):
        if int_z < int_max_zoom:
            arr_mosaic, int_x0, int_y0, int_x1, int_y1 = fn_mosaic_half(
                arr_mosaic, int_x0, int_y0, int_x1, int_y1)
        dict_png_by_zoom[int_z] = fn_mosaic_to_png_tiles(arr_mosaic, int_z, int_x0, int_y0,
                                                         int_x1, int_y1)
    del arr_mosaic

    str_tmp_path = str_output_path + ".tmp"
    int_tiles = 0

    with open(str_tmp_path, "wb") as file_out:
        writer = Writer(file_out)

        # pmtiles wants tiles written in tile-id order (low zoom first)
        for int_z in range(int_min_zoom, int_max_zoom + 1):
            for int_tile_id, bytes_png in sorted(dict_png_by_zoom[int_z]):
                writer.write_tile(int_tile_id, bytes_png)
                int_tiles += 1
            del dict_png_by_zoom[int_z]

        dict_header = {
            "tile_type": TileType.PNG,
            "tile_compression": Compression.NONE,
            "min_zoom": int_min_zoom,
            "max_zoom": int_max_zoom,
            "min_lon_e7": int(flt_west * 1e7),
            "min_lat_e7": int(flt_south * 1e7),
            "max_lon_e7": int(flt_east * 1e7),
            "max_lat_e7": int(flt_north * 1e7),
            "center_zoom": max(int_min_zoom, min(13, int_max_zoom)),
            "center_lon_e7": int((flt_west + flt_east) / 2.0 * 1e7),
            "center_lat_e7": int((flt_south + flt_north) / 2.0 * 1e7),
        }
        dict_meta = {
            "name": os.path.basename(str_output_path),
            "format": "png",
            "encoding": "terrarium",
            "units": "meters",
            "description": "Flood depth, Terrarium RGB, alpha=0 = dry",
            "minzoom": int_min_zoom,
            "maxzoom": int_max_zoom,
            "layer_index": int_idx,
        }
        dict_meta.update(dict_meta_extra)
        writer.finalize(dict_header, dict_meta)

    os.replace(str_tmp_path, str_output_path)

    tup_bounds = (flt_west, flt_south, flt_east, flt_north)
    return int_tiles, tup_bounds
# .........................................................


# --------------------------------------------------
def fn_worker(dict_task):
    # Runs one layer; errors are returned (not raised) so the other layers
    # still get built.
    flt_start = time.time()
    try:
        int_tiles, tup_bounds = fn_build_one_pmtiles(dict_task['nc'], dict_task['idx'],
                                                     dict_task['out'], dict_task['min_zoom'],
                                                     dict_task['max_zoom'], dict_task['meta'])
        return dict_task, int_tiles, tup_bounds, time.time() - flt_start, ''
    except Exception as exc:
        str_tmp_path = dict_task['out'] + ".tmp"
        if os.path.exists(str_tmp_path):
            try:
                os.remove(str_tmp_path)
            except OSError:
                pass
        return dict_task, 0, None, time.time() - flt_start, f"{type(exc).__name__}: {exc}"
# --------------------------------------------------


# .........................................................
def fn_build_pmtiles(str_netcdf_path, str_out_folder, str_catchment,
                     int_min_zoom, int_max_zoom, int_workers, b_print_output):

    if b_print_output:
        print(f"  ---    INPUT NETCDF: {str_netcdf_path}")
        print(f"  ---    OUTPUT FOLDER: {str_out_folder}")
        print(f"  ---    ZOOM: {int_min_zoom} to {int_max_zoom}  WORKERS: {int_workers}")
        print("===================================================================")

    os.makedirs(str_out_folder, exist_ok=True)

    # -----------------------------
    # STEP 1: intensities in the netCDF
    # -----------------------------
    with xr.open_dataset(str_netcdf_path) as ds:
        arr_intensity = ds['intensity'].values.astype(np.float64)
        if 'stable_flag' in ds.data_vars:
            arr_stable = ds['stable_flag'].values
        else:
            arr_stable = np.full(arr_intensity.size, -1)

    list_tasks = []
    for int_idx, flt_mm in enumerate(arr_intensity):
        flt_inhr = flt_mm / FLT_MM_PER_IN
        str_name = f"depth_{str_catchment}_i{int_idx:02d}_{flt_inhr:.2f}inhr.pmtiles"
        list_tasks.append({
            'nc': str_netcdf_path,
            'idx': int_idx,
            'out': os.path.join(str_out_folder, str_name),
            'min_zoom': int_min_zoom,
            'max_zoom': int_max_zoom,
            'meta': {'huc12': str_catchment,
                     'intensity_mm_h': float(flt_mm),
                     'intensity_in_hr': round(flt_inhr, 4),
                     'stable_flag': int(arr_stable[int_idx]) if np.isfinite(arr_stable[int_idx]) else -1}})

    if b_print_output:
        print(f"  -- STEP 1: {len(list_tasks)} intensity layer(s) to tile")
        print("  -- STEP 2: Building PMTiles")

    # -----------------------------
    # STEP 2: one PMTiles per layer
    # -----------------------------
    list_results = []
    int_done = 0

    def fn_report(tup_result):
        dict_task, int_tiles, tup_bounds, flt_secs, str_err = tup_result
        str_name = os.path.basename(dict_task['out'])
        if b_print_output:
            if str_err == '':
                flt_mb = os.path.getsize(dict_task['out']) / 1e6
                print(f"     -- [{int_done}/{len(list_tasks)}] {str_name}  {int_tiles:,} tiles  "
                      f"{flt_mb:.1f} MB  {flt_secs:.0f}s")
            else:
                print(f"     -- [{int_done}/{len(list_tasks)}] ERROR {str_name}: {str_err}")

    if int_workers <= 1:
        for dict_task in list_tasks:
            tup_result = fn_worker(dict_task)
            int_done += 1
            list_results.append(tup_result)
            fn_report(tup_result)
    else:
        with ProcessPoolExecutor(max_workers=int_workers) as pool:
            list_futures = [pool.submit(fn_worker, dict_task) for dict_task in list_tasks]
            for future in as_completed(list_futures):
                tup_result = future.result()
                int_done += 1
                list_results.append(tup_result)
                fn_report(tup_result)

    # -----------------------------
    # STEP 3: catalog of this catchment's layers
    # -----------------------------
    list_layers = []
    list_errors = []
    tup_all_bounds = None

    for dict_task, int_tiles, tup_bounds, flt_secs, str_err in list_results:
        if str_err != '':
            list_errors.append((os.path.basename(dict_task['out']), str_err))
            continue
        if tup_all_bounds is None:
            tup_all_bounds = tup_bounds
        else:
            tup_all_bounds = (min(tup_all_bounds[0], tup_bounds[0]), min(tup_all_bounds[1], tup_bounds[1]),
                              max(tup_all_bounds[2], tup_bounds[2]), max(tup_all_bounds[3], tup_bounds[3]))
        list_layers.append({'index': dict_task['idx'],
                            'file': os.path.basename(dict_task['out']),
                            'intensity_mm_h': dict_task['meta']['intensity_mm_h'],
                            'intensity_in_hr': dict_task['meta']['intensity_in_hr'],
                            'stable_flag': dict_task['meta']['stable_flag'],
                            'n_tiles': int_tiles,
                            'size_mb': round(os.path.getsize(dict_task['out']) / 1e6, 2)})

    list_layers.sort(key=lambda d: d['index'])

    dict_catalog = {'generated': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'huc12': str_catchment,
                    'source_netcdf': os.path.basename(str_netcdf_path),
                    'encoding': 'terrarium',
                    'min_zoom': int_min_zoom,
                    'max_zoom': int_max_zoom,
                    'bounds': list(tup_all_bounds) if tup_all_bounds is not None else None,
                    'layers': list_layers}

    str_catalog_path = os.path.join(str_out_folder, 'catalog.json')
    with open(str_catalog_path, 'w') as file_catalog:
        json.dump(dict_catalog, file_catalog, indent=1)

    if b_print_output:
        print(f"  -- STEP 3: Catalog written: {str_catalog_path}")

    if list_errors:
        for str_name, str_err in list_errors:
            print(f"  -- FAILED: {str_name} - {str_err}")
        raise RuntimeError(f"{len(list_errors)} of {len(list_tasks)} PMTiles layer(s) failed")

    return str_out_folder
# .........................................................


# .........................................................
def fn_build_pmtiles_06(
    str_global_config_file_path,
    str_local_config_file_path,
    b_print_output
):

    # ---- Header output ----
    if b_print_output:
        print(f"""
+=================================================================+
|          BUILD PMTILES FROM UPSCALED DEPTH NETCDF               |
|                Created by Andy Carter, PE of                    |
|             Center for Water and the Environment                |
|                 University of Texas at Austin                   |
+-----------------------------------------------------------------+
  ---(g) INPUT GLOBAL CONFIGURATION FILE: {str_global_config_file_path}
  ---(c) LOCAL CONFIGURATION FILE:  {str_local_config_file_path}
  ---[r] PRINT OUTPUT: {b_print_output}""")
    else:
        print(" -- Script 06: Build PMTiles")

    # ==================================================================
    # READ GLOBAL CONFIG
    # ==================================================================
    global_config = configparser.ConfigParser()
    global_config.read(str_global_config_file_path)

    # [pmtiles] is optional; the fallbacks match the config files
    int_min_zoom = global_config.getint('pmtiles', 'min_zoom', fallback=3)
    int_max_zoom = global_config.getint('pmtiles', 'max_zoom', fallback=15)
    int_workers  = global_config.getint('pmtiles', 'workers', fallback=1)

    if not (0 <= int_min_zoom <= int_max_zoom <= 22):
        raise ValueError(f"[pmtiles] needs 0 <= min_zoom <= max_zoom <= 22; "
                         f"got {int_min_zoom} and {int_max_zoom}")

    # ==================================================================
    # READ LOCAL CONFIG
    # ==================================================================
    local_config = configparser.ConfigParser()
    local_config.read(str_local_config_file_path)

    if 'run_parameters' not in local_config:
        raise KeyError("Missing [run_parameters] section in LOCAL config")

    str_catchment = local_config.get('run_parameters', 'catchment', fallback='').strip()
    str_out_root_folder = local_config.get('run_parameters', 'out_root_folder', fallback='').strip()

    # upscaled netCDF from upscale_netcdf_05 in, 06_pmtiles out
    str_netcdf_path = os.path.join(str_out_root_folder, str_catchment,
                                   "05_upscaled_depth_nc", f"depth_{str_catchment}.nc")
    if not os.path.isfile(str_netcdf_path):
        raise FileNotFoundError(f"Upscaled depth netCDF not found: {str_netcdf_path}")

    str_out_folder = os.path.join(str_out_root_folder, str_catchment, "06_pmtiles")

    return fn_build_pmtiles(str_netcdf_path, str_out_folder, str_catchment,
                            int_min_zoom, int_max_zoom, int_workers, b_print_output)
# .........................................................


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == '__main__':

    flt_start_run = time.time()

    parser = argparse.ArgumentParser(description='========= BUILD PMTILES FROM UPSCALED DEPTH NETCDF =========')

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

    fn_build_pmtiles_06(args['str_global_config_file_path'],
                        args['str_local_config_file_path'],
                        args['b_print_output'])

    flt_end_run = time.time()
    flt_time_pass = (flt_end_run - flt_start_run) // 1
    time_pass = datetime.timedelta(seconds=flt_time_pass)

    print('Compute Time: ' + str(time_pass))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
