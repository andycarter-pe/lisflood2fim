# ************************************************************
# LISFLOOD-FP to FIM
# Script 01 - prepare_input_layers_01
#
# Created by: Andy Carter, PE
# Created - 2026.02.04
# Revised - 2026.02.18 -- Revised for lateral watersheds
# Revised - 2026.03.04 -- Line 843 -- Whitebox $HOME bypass
# Revised - 2026.04.18 -- Refactored for single HUC-12 watershed
#                          (removes NextGen divide/flowpath lookups
#                           and all lateral watershed logic)
# ************************************************************

# ************************************************************
import os

import numpy as np
import rasterio
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, mapping, LineString, box
from shapely.ops import linemerge

import rioxarray
from whitebox import WhiteboxTools
import subprocess

import shapely
from rasterio.mask import mask
from shapely.ops import unary_union
import fiona

from rasterio.features import rasterize
from netCDF4 import Dataset

import pyogrio
from rasterio.enums import Resampling

import argparse
import configparser
import time
import datetime
import warnings

import shutil
from pathlib import Path
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


# ---------------------
def fn_get_huc12_gdf(str_huc12_path, str_huc12_id):
    """
    Returns a GeoDataFrame containing the feature(s) matching str_huc12_id
    from a local vector file (shapefile, GeoJSON, GPKG, etc.).
    Matches against the 'huc12' field (case-insensitive column search).

    Parameters
    ----------
    str_huc12_path : str
        Path to the local vector file containing HUC-12 polygons.
    str_huc12_id : str
        The 12-digit HUC-12 code to select (e.g. '120702050402').

    Returns
    -------
    GeoDataFrame
        Filtered to the matching HUC-12 feature(s), reprojected to EPSG:5070.
    """
    gdf = gpd.read_file(str_huc12_path)

    # Normalise column names to lowercase for a robust match
    gdf.columns = [c.lower() for c in gdf.columns]

    if 'huc12' not in gdf.columns:
        raise KeyError(
            f"No 'huc12' column found in {str_huc12_path}. "
            f"Available columns: {list(gdf.columns)}"
        )

    # Match as string (strips accidental leading-zero issues)
    mask = gdf['huc12'].astype(str).str.strip() == str(str_huc12_id).strip()
    result = gdf[mask].copy()

    if result.empty:
        raise ValueError(
            f"No feature found with huc12 == '{str_huc12_id}' in {str_huc12_path}"
        )

    # Ensure EPSG:5070 (same CRS used throughout the rest of the script)
    if result.crs is None:
        result = result.set_crs("EPSG:4326")
    result = result.to_crs("EPSG:5070")

    return result
# ---------------------


# -----------------
def fn_create_terrain_tif(str_catchment_id,
                          str_outfolder,
                          gdf,
                          str_vrt_terrain,
                          flt_buffer,
                          b_print_output):

    if not gdf.empty:
        if b_print_output:
            print(f"  -- STEP 2: Found catchment: {str_catchment_id}")

        # Merge all polygons into one
        merged_polygon = unary_union(gdf.geometry)

        # Buffer the merged polygon by flt_buffer (EPSG:5070 metres)
        buffered_polygon = merged_polygon.buffer(flt_buffer)

        # Convert to EPSG:4326 (lat/lon)
        gdf_buffered = gpd.GeoDataFrame(geometry=[buffered_polygon], crs=gdf.crs)
        gdf_buffered = gdf_buffered.to_crs("EPSG:4326")

        # Get bounding box coordinates (minx, miny, maxx, maxy)
        bbox = gdf_buffered.geometry.iloc[0].bounds
        minx, miny, maxx, maxy = bbox

        # Create bounding box GeoDataFrame
        divide_bbox = (minx, miny, maxx, maxy)
        divide_geom = gpd.GeoDataFrame({'geometry': [box(*divide_bbox)]}, crs="EPSG:4326")

        # Open VRT
        with rasterio.open(str_vrt_terrain) as src:
            out_image, out_transform = mask(src, divide_geom.geometry, crop=True, indexes=1)

            if out_image.ndim == 2:
                out_image = out_image[np.newaxis, :, :]

            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "count": 1,
                "dtype": out_image.dtype
            })

            output_file = os.path.join(str_outfolder, "dem_bbox_buffered_4326.tif")
            with rasterio.open(output_file, "w", **out_meta) as dest:
                dest.write(out_image[0], 1)
    else:
        print(f" No feature found with huc12: {str_catchment_id}")
        output_file = ''

    return output_file
# -----------------


# ................
def fn_compute_last_point(geom):
    if geom.geom_type == "LineString":
        return Point(geom.coords[-1])
    elif geom.geom_type == "MultiLineString":
        return Point(list(geom.geoms[-1].coords)[-1])
    return None
# ................


# ----------------
def fn_get_clipped_roads(polygon, str_url_roads):
    """
    Extract roads from a FlatGeobuf that intersect a polygon.

    Parameters
    ----------
    polygon : GeoDataFrame
        Polygon(s) to clip against. Can be any CRS.
    str_url_roads : str
        Path to the FlatGeobuf file (EPSG:4326 assumed).

    Returns
    -------
    GeoDataFrame
        Roads clipped to the polygon, in EPSG:5070.
    """
    polygon_4326 = polygon.to_crs(epsg=4326)
    minx, miny, maxx, maxy = polygon_4326.total_bounds

    roads = gpd.read_file(
        str_url_roads,
        bbox=(minx, miny, maxx, maxy)
    )

    roads_clipped = gpd.clip(roads, polygon_4326)
    gdf_roads_clipped_5070 = roads_clipped.to_crs(epsg=5070)

    return gdf_roads_clipped_5070
# ----------------


# -----------
def fn_two_digit_string(n: int) -> str:
    return f"{n:02d}_"
# -----------


# -----------
def fn_build_path(folder: str, prefix: str, name: str) -> str:
    return os.path.abspath(os.path.join(folder, f"{prefix}{name}"))
# -----------


# ---------------
def fn_get_first_last(line: LineString):
    coords = list(line.coords)
    return Point(coords[0]), Point(coords[-1])
# ---------------


# -------------
def fn_append_headwater_streams(str_stream_vector_filepath):
    streams = gpd.read_file(str_stream_vector_filepath)

    if streams.crs is None:
        streams = streams.set_crs("EPSG:5070")

    all_vertices = []
    for geom in streams.geometry:
        f, l = fn_get_first_last(geom)
        all_vertices.extend([f.wkt, l.wkt])

    unique_flags = []
    for geom in streams.geometry:
        first, _ = fn_get_first_last(geom)
        if all_vertices.count(first.wkt) == 1:
            unique_flags.append(True)
        else:
            unique_flags.append(False)

    streams["is_head"] = unique_flags
    streams.to_file(str_stream_vector_filepath)
# -------------


# -----------
def fn_compute_contrib_area(group):
    group = group.copy()
    group["flow_acc"] = pd.to_numeric(group["flow_acc"])
    group = group.sort_values("flow_acc").copy()

    diffs = group["flow_acc"].diff()
    is_head = bool(group["is_head"].iloc[0])

    if is_head:
        group["contrib_area"] = diffs.fillna(group["flow_acc"].iloc[0])
    else:
        group["contrib_area"] = diffs.fillna(0)

    return group
# -----------


# --------------------
def fn_resolve_coincident_points(group):
    sum_contrib = group["contrib_area"].sum()
    flow_acc = group["flow_acc"].iloc[0]
    inc_value = sum_contrib - flow_acc
    max_idx = group["contrib_area"].idxmax()
    group["contrib_area"] = 0
    group.loc[max_idx, "contrib_area"] = inc_value
    return group
# --------------------


# ---------------------
def fn_raster_edge_cells_to_bci(str_raster_path, str_slope, write_gpkg=True):
    """
    Identify raster border and nodata-adjacent cells and write LISFLOOD BCI file.
    """
    base_dir = os.path.dirname(str_raster_path)
    base_name = os.path.splitext(os.path.basename(str_raster_path))[0]

    gpkg_path = os.path.join(base_dir, f"{base_name}_edge_cells.gpkg")
    bci_path = os.path.join(base_dir, f"{base_name}_edge_cells.bci")

    with rasterio.open(str_raster_path) as src:
        data = src.read(1)
        nodata = src.nodata
        transform = src.transform
        crs = src.crs
        height, width = data.shape

    if nodata is None:
        valid = ~np.isnan(data)
    else:
        valid = data != nodata

    border_mask = np.zeros_like(valid, dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True

    border_cells = valid & border_mask

    adjacent_to_nodata = np.zeros_like(valid, dtype=bool)
    adjacent_to_nodata[1:, :]  |= ~valid[:-1, :]
    adjacent_to_nodata[:-1, :] |= ~valid[1:, :]
    adjacent_to_nodata[:, 1:]  |= ~valid[:, :-1]
    adjacent_to_nodata[:, :-1] |= ~valid[:, 1:]

    edge_cells = valid & adjacent_to_nodata
    final_mask = border_cells | edge_cells
    rows, cols = np.where(final_mask)

    xs, ys, points = [], [], []
    for r, c in zip(rows, cols):
        x, y = rasterio.transform.xy(transform, r, c, offset="center")
        xs.append(x)
        ys.append(y)
        points.append(Point(x, y))

    if write_gpkg:
        gdf = gpd.GeoDataFrame({"x": xs, "y": ys}, geometry=points, crs=crs)
        gdf.to_file(gpkg_path, layer="edge_cells", driver="GPKG")

    with open(bci_path, "w") as f:
        for x, y in zip(xs, ys):
            f.write(f"F {x} {y} FREE {str_slope}\n")

    return bci_path
# ---------------------


# ---------------------
def fn_max_intensity_from_aoi_atlas_14(str_polygon_path, str_atlas_14_1000yr_5min):
    gdf = gpd.read_file(str_polygon_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    polygon = gdf.geometry.iloc[0]

    rds = rioxarray.open_rasterio(str_atlas_14_1000yr_5min, masked=True)
    if rds.rio.crs is None:
        rds.rio.write_crs("EPSG:4326", inplace=True)
    if gdf.crs != rds.rio.crs:
        gdf = gdf.to_crs(rds.rio.crs)
        polygon = gdf.geometry.iloc[0]

    rds_clipped = rds.rio.clip([mapping(polygon)], gdf.crs, drop=True)
    flt_max_value = float(rds_clipped.max().values)
    flt_peak_rainfall_inhr = flt_max_value / 100
    return flt_peak_rainfall_inhr
# ---------------------


# ---------------------
def fn_min_intensity_from_aoi_atlas_14(str_polygon_path, str_atlas_14_1yr_24hr):
    gdf = gpd.read_file(str_polygon_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    polygon = gdf.geometry.iloc[0]

    rds = rioxarray.open_rasterio(str_atlas_14_1yr_24hr, masked=True)
    if rds.rio.crs is None:
        rds.rio.write_crs("EPSG:4326", inplace=True)
    if gdf.crs != rds.rio.crs:
        gdf = gdf.to_crs(rds.rio.crs)
        polygon = gdf.geometry.iloc[0]

    rds_clipped = rds.rio.clip([mapping(polygon)], gdf.crs, drop=True)
    flt_min_value = float(rds_clipped.min().values)
    flt_min_rainfall_inhr = flt_min_value / (1000 * 24)
    return flt_min_rainfall_inhr
# ---------------------


# --------
def fn_q_from_intensity(flt_rain_rate_mmhr, flt_max_cell_value, flt_pixel_area_base_terrain):
    flt_outflow_cfs = (flt_rain_rate_mmhr * flt_max_cell_value * flt_pixel_area_base_terrain * 35.3146667) / (1000 * 3600)
    return int(round(flt_outflow_cfs))
# --------


# --------
def fn_intensity_from_q(flt_outflow_cfs, flt_max_cell_value, flt_pixel_area_base_terrain):
    flt_rain_rate_mmhr = (1000 * 3600 * flt_outflow_cfs) / (flt_max_cell_value * flt_pixel_area_base_terrain * 35.3146667)
    return flt_rain_rate_mmhr
# --------


# .................
def fn_logarithmic_progression(flt_min_q, flt_max_q, int_log_q_steps):
    values = np.ceil(np.logspace(np.log10(flt_min_q), np.log10(flt_max_q), num=int_log_q_steps)).astype(int)
    values[0], values[-1] = flt_min_q, flt_max_q
    return values
# .................


# -------------------
def fn_get_fa_stats(fa):
    pixel_width, pixel_height = fa.rio.resolution()
    flt_pixel_area_base_terrain = abs(pixel_width * pixel_height)
    flt_max_cell_value = float(fa.max(skipna=True))
    return flt_pixel_area_base_terrain, flt_max_cell_value
# -------------------


# ---------------------
def fn_create_precip_nc(
        str_catchment,
        str_dem_asc_clipped_filepath,
        str_gpkg_filepath,
        str_out_folder_streams,
        flt_rain_mm,
        int_timesteps,
        fill_value=-999.0,
        flt_min_rain=0.10):

    str_rain_mm = str(flt_rain_mm).replace('.', 'p')

    str_nc_pnt_rainfall_out = os.path.join(
        str_out_folder_streams,
        f"{str_catchment}_rainfall_pnt_{str_rain_mm}mm.nc"
    )

    with rasterio.open(str_dem_asc_clipped_filepath) as src:
        ncols = src.width
        nrows = src.height
        transform = src.transform
        dem_crs = src.crs
        cellsize = transform.a
        x0 = transform.c
        y0 = transform.f

    x = x0 + cellsize * (np.arange(ncols) + 0.5)
    y = y0 - cellsize * (np.arange(nrows) + 0.5)
    time = np.arange(0, int_timesteps, 1)

    gdf = gpd.read_file(str_gpkg_filepath, layer="flow_points")

    shapes = (
        (geom, value)
        for geom, value in zip(gdf.geometry, gdf["contrib_ar"])
    )

    contrib_grid = rasterize(
        shapes=shapes,
        out_shape=(nrows, ncols),
        transform=transform,
        fill=fill_value,
        dtype="float32"
    )

    rainfall = np.full((int_timesteps, nrows, ncols), flt_min_rain, dtype=np.float32)
    valid = contrib_grid != fill_value
    rainfall[:, valid] = contrib_grid[valid] * flt_rain_mm

    total_rainfall = np.full((nrows, ncols), fill_value, dtype=np.float32)
    total_rainfall[valid] = contrib_grid[valid] * flt_rain_mm * int_timesteps

    nc = Dataset(str_nc_pnt_rainfall_out, "w", format="NETCDF4_CLASSIC")
    nc.createDimension("time", None)
    nc.createDimension("y", nrows)
    nc.createDimension("x", ncols)

    tv = nc.createVariable("time", "f8", ("time",))
    yv = nc.createVariable("y", "f8", ("y",))
    xv = nc.createVariable("x", "f8", ("x",))

    rf = nc.createVariable(
        "rainfall_depth", "f4", ("time", "y", "x"),
        fill_value=fill_value, zlib=True, complevel=4,
        chunksizes=(1, nrows, ncols)
    )
    tr = nc.createVariable(
        "total_rainfall_depth", "f4", ("y", "x"),
        fill_value=fill_value, zlib=True, complevel=4
    )

    nc.title = "Gridded Rainfall"
    nc.source = "Python-generated"
    nc.references = "TUFLOW NetCDF Rainfall Format"
    nc.comment = (
        f"{str_catchment} -- Point-based rainfall: "
        f"contrib_ar--{str(flt_rain_mm)} mm per timestep"
    )

    tv.units = "hours"
    yv.units = "m"
    xv.units = "m"

    tv[:] = time
    yv[:] = y
    xv[:] = x
    rf[:, :, :] = rainfall
    tr[:, :] = total_rainfall

    nc.close()

    return str_nc_pnt_rainfall_out
# ---------------------


# ---------------
def fn_format_params(params):
    VALUE_COL = 24
    lines = []
    for key, value in params:
        tag = f"{key}"
        if value is None or value == "":
            lines.append(tag)
        else:
            pad = max(1, VALUE_COL - len(tag))
            lines.append(tag + (" " * pad) + str(value))
    return "\n".join(lines)
# ---------------


# ----------------
def fn_extract_run_name(filename):
    name = Path(filename).stem
    left, right = name.split("_rainfall_pnt_")
    return f"{left}_{right}"
# ----------------


# ----------------------
def fn_fix_diagonal_burns(str_dem_before_path, str_dem_after_path, str_dem_fixed_path):
    """
    Post-process a DEM burned by BurnStreamsAtRoads to remove diagonal-only
    pixel connections that break LISFLOOD-FP's D4 (cardinal-only) solver.
    """
    import numpy as np
    import rasterio

    with rasterio.open(str_dem_before_path) as src:
        elev_before = src.read(1).astype(np.float64)
        nodata      = src.nodata
        profile     = src.profile.copy()
        height, width = elev_before.shape

    with rasterio.open(str_dem_after_path) as src:
        elev_after = src.read(1).astype(np.float64)

    if nodata is not None:
        valid = (elev_before != nodata) & (elev_after != nodata)
    else:
        valid = ~(np.isnan(elev_before) | np.isnan(elev_after))

    burned = valid & (elev_after < elev_before - 1e-6)

    elev_fixed  = elev_after.copy()
    int_bridges = 0
    diagonals   = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    changed = True
    while changed:
        changed = False
        for r in range(height):
            for c in range(width):
                if not burned[r, c]:
                    continue
                for dr, dc in diagonals:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < height and 0 <= nc < width):
                        continue
                    if not burned[nr, nc]:
                        continue

                    r1, c1 = r,  nc
                    r2, c2 = nr, c

                    in1 = 0 <= r1 < height and 0 <= c1 < width
                    in2 = 0 <= r2 < height and 0 <= c2 < width

                    card1_burned = in1 and burned[r1, c1]
                    card2_burned = in2 and burned[r2, c2]

                    if card1_burned or card2_burned:
                        continue

                    bridge_r, bridge_c = None, None
                    for br, bc in [(r1, c1), (r2, c2)]:
                        if 0 <= br < height and 0 <= bc < width and valid[br, bc]:
                            bridge_r, bridge_c = br, bc
                            break

                    if bridge_r is None:
                        continue

                    bridge_elev = min(elev_fixed[r, c], elev_fixed[nr, nc])

                    if bridge_elev < elev_fixed[bridge_r, bridge_c]:
                        elev_fixed[bridge_r, bridge_c] = bridge_elev
                        burned[bridge_r, bridge_c] = True
                        int_bridges += 1
                        changed = True

    profile.update(dtype=rasterio.float64, count=1)
    with rasterio.open(str_dem_fixed_path, "w", **profile) as dst:
        dst.write(elev_fixed, 1)

    return str_dem_fixed_path, int_bridges
# ----------------------


# ...............................
def fn_condition_terrain(dict_filepaths,
                         str_out_folder_streams,
                         str_whitebox_path,
                         flt_threshold,
                         polygon,
                         str_url_roads):

    wbt = WhiteboxTools()
    wbt.set_whitebox_dir("/opt/whitebox_tools")
    wbt.verbose = False
    wbt.work_dir = str_out_folder_streams

    # Step 1 -- Breach Depressions Least Cost
    cmd = [
        str_whitebox_path,
        "--run=BreachDepressionsLeastCost",
        f"--dem={dict_filepaths['dem_clipped']}",
        f"-o={dict_filepaths['dem_breach']}",
        "--dist=2000",
        "--max_cost=500",
        "--flat_increment=0.001"
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    # Step 2 -- Flow Accumulation (Pass 01)
    wbt.fill_depressions(dict_filepaths['dem_breach'], dict_filepaths['dem_filled'])
    wbt.d8_pointer(dict_filepaths['dem_filled'], dict_filepaths['dem_fdir'])
    wbt.d8_flow_accumulation(dict_filepaths['dem_filled'], dict_filepaths['dem_fa'], out_type="cells")

    # Step 3 -- Stream Network (Pass 01)
    cmd = [
        str_whitebox_path,
        "--run=ExtractStreams",
        f"--flow_accum={dict_filepaths['dem_fa']}",
        f"--d8_pntr={dict_filepaths['dem_fdir']}",
        f"--output={dict_filepaths['stream_raster']}",
        f"--threshold={flt_threshold}",
        "--zero_background"
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    wbt.verbose = False
    wbt.raster_streams_to_vector(dict_filepaths['stream_raster'],
                                 dict_filepaths['dem_fdir'],
                                 dict_filepaths['stream_vector'])

    # Step 4 -- Burn Streams at Roads
    gdf_roads_clipped_5070 = fn_get_clipped_roads(polygon, str_url_roads)
    gdf_roads_clipped_5070.to_file(dict_filepaths['clipped_roads'], driver="ESRI Shapefile")

    cmd = [
        str_whitebox_path,
        "--run=BurnStreamsAtRoads",
        f"--dem={dict_filepaths['dem_breach']}",
        f"--streams={dict_filepaths['stream_vector']}",
        f"--roads={dict_filepaths['clipped_roads']}",
        f"--output={dict_filepaths['dem_burn_roads']}",
        "--width=20"
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    # Step 4b -- Fix diagonal burns for D4 compatibility
    str_dem_burn_fixed = dict_filepaths['dem_burn_roads'].replace('.tif', '_d4fixed.tif')
    str_dem_burn_fixed, n_bridges = fn_fix_diagonal_burns(
        dict_filepaths['dem_clipped'],
        dict_filepaths['dem_burn_roads'],
        str_dem_burn_fixed
    )
    dict_filepaths['dem_burn_roads'] = str_dem_burn_fixed

    # Step 5 -- Flow Accumulation (Pass 02)
    wbt.fill_depressions(dict_filepaths['dem_burn_roads'], dict_filepaths['dem_filled'])
    wbt.d8_pointer(dict_filepaths['dem_filled'], dict_filepaths['dem_fdir'])
    wbt.d8_flow_accumulation(dict_filepaths['dem_filled'], dict_filepaths['dem_fa'], out_type="cells")

    # Step 6 -- Stream Network (Pass 02)
    cmd = [
        str_whitebox_path,
        "--run=ExtractStreams",
        f"--flow_accum={dict_filepaths['dem_fa']}",
        f"--d8_pntr={dict_filepaths['dem_filled']}",
        f"--output={dict_filepaths['stream_raster']}",
        f"--threshold={flt_threshold}",
        "--zero_background"
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    wbt.verbose = False
    wbt.raster_streams_to_vector(dict_filepaths['stream_raster'],
                                 dict_filepaths['dem_fdir'],
                                 dict_filepaths['stream_vector'])
# ...............................


# .........................................................
def fn_prepare_input_layers_01(
        str_global_config_file_path,
        str_local_config_file_path,
        b_print_output):

    import warnings
    import configparser

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="pyogrio")

    if b_print_output:
        print(f"""
+=================================================================+
|              PREPARE INPUT LAYERS FOR LISFLOOD-FP               |
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
        print(" -- Script 01: Prepare input layers for LISFLOOD-FP")

    # ==================================================================
    # READ GLOBAL CONFIG
    # ==================================================================
    global_config = configparser.ConfigParser()
    global_config.read(str_global_config_file_path)

    global_section_schema = {
        'datasource': [
            'url_huc12',          # path to local HUC-12 vector file
            'vrt_terrain',
            'url_roads',
            'atlas_14_1000yr_5min',
            'atlas_14_1yr_24hr'
        ],
        'lisflood_settings': [
            'downscale',
            'terrain_buffer_m',
            'fpfric',
            'initial_tstep',
            'depththresh',
            'max_Froude',
            'outflow_boundary_slope',
            'stream_threshold_sq_mi'
        ],
        'flow_parameters': [
            'num_steps',
            'timesteps',
            'output_step',
            'mass_balance_step'
        ],
        'stable_run_paramters': [
            'window',
            'min_Qout_ratio',
            'max_Qout_ratio',
            'max_rolling_avg_stability'
        ]
    }

    dict_global_params = {}
    for section_name, keys in global_section_schema.items():
        if section_name not in global_config:
            raise KeyError(f"Missing [{section_name}] section in GLOBAL config")
        section = global_config[section_name]
        dict_global_params.update({key: section.get(key, '') for key in keys})

    # ==================================================================
    # READ LOCAL CONFIG
    # ==================================================================
    local_config = configparser.ConfigParser()
    local_config.read(str_local_config_file_path)

    local_section_schema = {
        'run_parameters': [
            'catchment',        # 12-digit HUC-12 code, e.g. 120702050402
            'out_root_folder'
        ]
    }

    dict_local_params = {}
    for section_name, keys in local_section_schema.items():
        if section_name not in local_config:
            raise KeyError(f"Missing [{section_name}] section in LOCAL config")
        section = local_config[section_name]
        dict_local_params.update({key: section.get(key, '') for key in keys})

    # COMBINE (local overrides global if collision)
    dict_all_params = {**dict_global_params, **dict_local_params}

    # --------------- Make folders / paths
    # 'catchment' is now the 12-digit HUC-12 code
    str_catchment = dict_all_params['catchment']

    str_out_root_folder = os.path.abspath(dict_all_params['out_root_folder'])
    str_out_folder = os.path.join(str_out_root_folder, str_catchment)
    os.makedirs(str_out_folder, exist_ok=True)

    str_out_folder_streams    = os.path.join(str_out_folder, '01_stream_delineation')
    str_out_folder_streams_02 = os.path.join(str_out_folder, '02_lisflood_input')
    os.makedirs(str_out_folder_streams, exist_ok=True)
    os.makedirs(str_out_folder_streams_02, exist_ok=True)

    str_header = fn_two_digit_string(0)

    dict_files = {
        "dem_clipped":    "dem_clipped_5070.tif",
        "dem_breach":     "dem_breach_5070.tif",
        "dem_filled":     "dem_filled.tif",
        "dem_fdir":       "fdir.tif",
        "dem_fa":         "flow_accum.tif",
        "stream_raster":  "streams.tif",
        "stream_vector":  "streams.shp",
        "stream_points":  "stream_vertices_area.shp",
        "stream_points_acc": "stream_vert_acc_pnt.shp",
        "clipped_roads":  "clipped_roads_ln_5070.shp",
        "dem_burn_roads": "dem_burn_roads_5070.tif",
    }

    dict_paths_base = {
        key: fn_build_path(str_out_folder_streams, str_header, filename)
        for key, filename in dict_files.items()
    }

    str_polygon_filepath     = os.path.abspath(os.path.join(str_out_folder_streams, 'watershed_ar_4326.geojson'))
    str_gpkg_filepath        = os.path.abspath(os.path.join(str_out_folder_streams, str_catchment + '.gpkg'))
    str_dem_asc_clipped_filepath = os.path.abspath(os.path.join(str_out_folder_streams, 'dem_clipped_5070.asc'))

    os.environ["PROJ_LIB"] = "/opt/miniconda/envs/geo/share/proj"
    str_whitebox_path = "/opt/whitebox_tools/whitebox_tools"

    # --- Config values ---
    str_url_huc12      = dict_all_params['url_huc12']
    str_vrt_terrain    = dict_all_params['vrt_terrain']
    str_url_roads      = dict_all_params['url_roads']
    int_terrain_buffer_m = int(dict_all_params['terrain_buffer_m'])
    int_downscale      = int(dict_all_params['downscale'])

    # Pixel size assumptions (3 m native terrain)
    int_pixel_size = 3

    flt_stream_threshold_sq_mi = float(dict_all_params['stream_threshold_sq_mi'])
    int_cell_to_start_stream = int(
        flt_stream_threshold_sq_mi * 2589988 / (int_pixel_size * int_pixel_size)
    )
    flt_threshold = int_cell_to_start_stream / (int_downscale * int_downscale)

    # ==================================================================
    # STEP 1: Load HUC-12 polygon
    # ==================================================================
    if b_print_output:
        print('  -- STEP 1: Finding terrain')

    polygon = fn_get_huc12_gdf(str_url_huc12, str_catchment)

    # ==================================================================
    # STEP 2: Clip terrain to HUC-12 bounding box (buffered)
    # ==================================================================
    str_dem_4326_filepath = fn_create_terrain_tif(
        str_catchment,
        str_out_folder_streams,
        polygon,
        str_vrt_terrain,
        int_terrain_buffer_m,
        b_print_output
    )

    dem = rioxarray.open_rasterio(str_dem_4326_filepath, masked=True)

    # Reproject to EPSG:5070
    dem_5070 = dem.rio.reproject("EPSG:5070")

    # Downscale
    res_x, res_y = dem_5070.rio.resolution()
    new_resolution = (res_x * int_downscale, res_y * int_downscale)
    dem_5070_downscaled = dem_5070.rio.reproject(
        dem_5070.rio.crs,
        resolution=new_resolution,
        resampling=Resampling.average
    )

    # Ensure CRS match for clipping
    if polygon.crs != dem_5070_downscaled.rio.crs:
        polygon = polygon.to_crs(dem_5070_downscaled.rio.crs)

    # Clip downscaled DEM to HUC-12 polygon
    dem_clipped_5070 = dem_5070_downscaled.rio.clip(
        polygon.geometry.apply(mapping),
        polygon.crs,
        from_disk=True
    )
    dem_clipped_5070.rio.to_raster(dict_paths_base['dem_clipped'])

    # Save rectangular (bbox) downscaled DEM for reference
    str_dem_bbox_filepath = os.path.join(str_out_folder_streams, "dem_bbox_buffered_5070.tif")
    dem_5070_downscaled.rio.to_raster(str_dem_bbox_filepath)

    # ==================================================================
    # STEP 3: Stream Conditioning
    # ==================================================================
    if b_print_output:
        print('  -- STEP 3: Stream Conditioning')

    fn_condition_terrain(
        dict_paths_base,
        str_out_folder_streams,
        str_whitebox_path,
        flt_threshold,
        polygon,
        str_url_roads
    )

    # ==================================================================
    # STEP 4: Determine rainfall locations
    # ==================================================================
    if b_print_output:
        print('  -- STEP 4: Determine rainfall locations')

    fn_append_headwater_streams(dict_paths_base['stream_vector'])

    streams = gpd.read_file(dict_paths_base['stream_vector'])
    streams = streams.set_crs("EPSG:5070")

    fa = rioxarray.open_rasterio(dict_paths_base['dem_fa'], masked=True)
    if fa.rio.crs.to_string() != "EPSG:5070":
        fa = fa.rio.reproject("EPSG:5070", resampling=Resampling.nearest)

    point_records = []
    for idx, row in streams.iterrows():
        geom = row.geometry
        fid  = row["FID"] if "FID" in row else idx

        if geom.geom_type == "LineString":
            coords = list(geom.coords)
        elif geom.geom_type == "MultiLineString":
            coords = [c for line in geom for c in line.coords]
        else:
            continue

        for x, y in coords:
            flow_val = fa.sel(x=x, y=y, method="nearest").values.item()
            point_records.append({
                "geometry": Point(x, y),
                "FID": fid,
                "flow_acc": float(flow_val)
            })

    points_gdf = gpd.GeoDataFrame(point_records, crs="EPSG:5070")
    points_gdf.to_file(dict_paths_base['stream_points'])

    points_gdf = points_gdf.merge(
        streams[["FID", "is_head"]],
        on="FID",
        how="left"
    )

    points_gdf_contrib = (
        points_gdf
        .groupby("FID", group_keys=False)
        .apply(fn_compute_contrib_area, include_groups=False)
    )
    points_gdf_contrib.to_file(dict_paths_base['stream_points_acc'])

    dup_mask = points_gdf_contrib.duplicated(subset="geometry", keep=False)
    dupes    = points_gdf_contrib[dup_mask].copy()
    uniques  = points_gdf_contrib[~dup_mask].copy()

    dupes = (
        dupes
        .groupby("geometry", group_keys=False)
        .apply(fn_resolve_coincident_points, include_groups=False)
    )
    points_gdf_contrib = pd.concat([uniques, dupes], ignore_index=True)

    if 'FID' in points_gdf_contrib.columns:
        points_gdf_contrib = points_gdf_contrib.drop(columns=['FID'])

    points_gdf_contrib.to_file(dict_paths_base['stream_points_acc'])

    # Save watershed polygon files
    # Reproject to 4326 for GeoJSON output
    polygon_4326 = polygon.to_crs("EPSG:4326")
    polygon_4326.to_file(str_polygon_filepath, driver="GeoJSON")

    str_polygon_shp_filepath = os.path.join(str_out_folder_streams, 'watershed_ar_4326.shp')
    polygon_4326.to_file(str_polygon_shp_filepath, driver="ESRI Shapefile")

    # Build summary GeoPackage
    if os.path.exists(str_gpkg_filepath):
        os.remove(str_gpkg_filepath)

    vector_layers = [
        (str_polygon_filepath,               "watershed"),
        (dict_paths_base['stream_vector'],   "streams"),
        (dict_paths_base['stream_points_acc'], "flow_points")
    ]

    for vector_path, layer_name in vector_layers:
        gdf = gpd.read_file(vector_path)
        if layer_name == "streams":
            gdf['catchment'] = str_catchment
            gdf['threshold'] = flt_threshold
        if layer_name == "watershed":
            gdf['terrain_clip']       = dict_paths_base['dem_clipped']
            gdf['terrain_source']     = str_vrt_terrain
            gdf['huc12_source']       = str_url_huc12
        gdf.to_file(str_gpkg_filepath, layer=layer_name, driver="GPKG")

    # ==================================================================
    # STEP 5: Outflow Boundary Slope (BCI)
    # ==================================================================
    if b_print_output:
        print('  -- STEP 5: Outflow Boundary Slope')

    str_slope   = dict_all_params['outflow_boundary_slope']
    str_bci_path = fn_raster_edge_cells_to_bci(dict_paths_base['dem_clipped'], str_slope)

    src = Path(str_bci_path)
    dst = src.parents[1] / "02_lisflood_input" / src.name
    shutil.copy2(src, dst)

    # ==================================================================
    # STEP 6: Convert terrain to ASC for LISFLOOD-FP
    # ==================================================================
    if b_print_output:
        print('  -- STEP 6: Prep terrain for LISFLOOD')

    dem = rioxarray.open_rasterio(dict_paths_base['dem_burn_roads'], masked=True)

    if dem.rio.count > 1:
        dem = dem.sel(band=1)
    if dem.rio.nodata is None:
        dem.rio.write_nodata(-9999, inplace=True)

    dem.rio.to_raster(str_dem_asc_clipped_filepath, driver='AAIGrid')

    src = Path(str_dem_asc_clipped_filepath)
    dst = src.parents[1] / "02_lisflood_input" / src.name
    shutil.copy2(src, dst)

    # ==================================================================
    # STEP 7: Bounding rain intensities
    # ==================================================================
    if b_print_output:
        print('  -- STEP 7: Determine bounding rain intensities')

    str_atlas_14_1000yr_5min = dict_all_params['atlas_14_1000yr_5min']
    str_atlas_14_1yr_24hr    = dict_all_params['atlas_14_1yr_24hr']
    int_num_steps            = int(dict_all_params['num_steps'])

    flt_peak_rainfall_inhr = fn_max_intensity_from_aoi_atlas_14(str_polygon_filepath, str_atlas_14_1000yr_5min)
    flt_min_rainfall_inhr  = fn_min_intensity_from_aoi_atlas_14(str_polygon_filepath, str_atlas_14_1yr_24hr)

    flt_peak_rain_rate_mmhr = flt_peak_rainfall_inhr * 25.4
    flt_min_rain_rate_mmhr  = flt_min_rainfall_inhr  * 25.4

    fa = rioxarray.open_rasterio(dict_paths_base['dem_fa'], masked=True)
    flt_pixel_area_base_terrain, flt_max_cell_value = fn_get_fa_stats(fa)

    flt_max_q_cfs = fn_q_from_intensity(flt_peak_rain_rate_mmhr, flt_max_cell_value, flt_pixel_area_base_terrain)
    flt_min_q_cfs = fn_q_from_intensity(flt_min_rain_rate_mmhr,  flt_max_cell_value, flt_pixel_area_base_terrain)

    arr_log_flows_cfs = fn_logarithmic_progression(flt_min_q_cfs, flt_max_q_cfs, int_num_steps)

    arr_intensity_mm_hr = np.array([
        fn_intensity_from_q(q, flt_max_cell_value, flt_pixel_area_base_terrain)
        for q in arr_log_flows_cfs
    ]).round(1)

    # ==================================================================
    # STEP 8: Write parameter and rainfall NetCDF files
    # ==================================================================
    if b_print_output:
        print('  -- STEP 8: Writing parameter and rainfall data')

    int_timesteps        = int(dict_all_params['timesteps'])
    int_output_step      = int(dict_all_params['output_step'])
    flt_mass_balance_step = float(dict_all_params['mass_balance_step'])

    str_simtime   = str(int_timesteps * 3600)
    str_saveint   = str(float(int_output_step * 3600))
    str_mass_step = str(flt_mass_balance_step * 3600)

    str_dem_filename = Path(str_dem_asc_clipped_filepath).name
    str_bci_filename = Path(str_bci_path).name

    src = Path(str_bci_path)
    dst = src.parents[1] / "02_lisflood_input" / src.name
    shutil.copy2(src, dst)

    for flt_rain_mm in arr_intensity_mm_hr:
        str_nc_pnt_rainfall_out = fn_create_precip_nc(
            str_catchment,
            str_dem_asc_clipped_filepath,
            str_gpkg_filepath,
            str_out_folder_streams,
            flt_rain_mm,
            int_timesteps
        )

        src_rainfall      = Path(str_nc_pnt_rainfall_out)
        str_rainfall_filename = src_rainfall.name
        dst_rainfall      = src_rainfall.parents[1] / "02_lisflood_input" / src_rainfall.name
        shutil.copy2(src_rainfall, dst_rainfall)

        str_run_name = fn_extract_run_name(str_rainfall_filename)

        params = [
            ("DEMfile",        str_dem_filename),
            ("bcifile",        str_bci_filename),
            ("resroot",        str_run_name),
            ("dirroot",        str_run_name),
            ("sim_time",       str_simtime),
            ("initial_tstep",  dict_all_params['initial_tstep']),
            ("massint",        str_mass_step),
            ("saveint",        str_saveint),
            ("fpfric",         dict_all_params['fpfric']),
            ("elevoff",        None),
            ("acceleration",   None),
            ("SGC_Enable",     None),
            ("dynamicrainfile", str_rainfall_filename),
            ("depththresh",    dict_all_params['depththresh']),
            ("max_Froude",     dict_all_params['max_Froude']),
        ]

        str_param_filename = str_run_name + ".par"
        dst_param = src_rainfall.parents[1] / "02_lisflood_input" / str_param_filename
        with open(dst_param, "w", encoding="utf-8") as f:
            f.write(fn_format_params(params))

# .........................................................


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == '__main__':

    flt_start_run = time.time()

    parser = argparse.ArgumentParser(
        description='========= PREPARE INPUT LAYERS FOR LISFLOOD-FP (HUC-12) ========='
    )

    parser.add_argument('-g',
                        dest="str_global_config_file_path",
                        help=r'REQUIRED: Global configuration filepath '
                             r'Example: /app/lisflood2fim/config/demo_global_config.ini',
                        required=False,
                        default='/app/lisflood2fim/config/demo_global_config.ini',
                        metavar='FILE',
                        type=lambda x: is_valid_file(parser, x))

    parser.add_argument('-c',
                        dest="str_local_config_file_path",
                        help=r'REQUIRED: LOCAL configuration filepath '
                             r'Example: /app/lisflood2fim/config/demo_local_config.ini',
                        required=False,
                        default='/app/lisflood2fim/config/demo_local_config.ini',
                        metavar='FILE',
                        type=lambda x: is_valid_file(parser, x))

    parser.add_argument('-r',
                        dest="b_print_output",
                        help=r'OPTIONAL: Print output messages  Default: True',
                        required=False,
                        default=True,
                        metavar='T/F',
                        type=fn_str_to_bool)

    args = vars(parser.parse_args())

    str_global_config_file_path = args['str_global_config_file_path']
    str_local_config_file_path  = args['str_local_config_file_path']
    b_print_output              = args['b_print_output']

    fn_prepare_input_layers_01(
        str_global_config_file_path,
        str_local_config_file_path,
        b_print_output
    )

    flt_end_run  = time.time()
    flt_time_pass = (flt_end_run - flt_start_run) // 1
    time_pass = datetime.timedelta(seconds=flt_time_pass)

    print('Compute Time: ' + str(time_pass))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~