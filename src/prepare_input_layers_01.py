# ************************************************************
# LISFLOOD-FP to FIM
# Script 01 - prepare_input_layers_01
#
# Created by: Andy Carter, PE
# Created - 2026.02.04
# Revised - 2026.02.18 -- Revised for lateral waterhseds
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


# ---------------------
def fn_get_divide_gdf(fgb_path, divide_id):
    """
    Returns a GeoDataFrame containing the feature(s) for a given divide_id
    from a FlatGeobuf (.fgb) file.
    """
    records = []

    with fiona.open(fgb_path, layer=None) as src:
        crs_dict = src.crs
        # Convert dict to EPSG string if possible
        if crs_dict:
            try:
                crs_str = crs_dict.get('init', None)
                if crs_str:
                    crs_str = crs_str.replace('+init=', '').upper()  # EPSG:XXXX
                else:
                    crs_str = None
            except Exception:
                crs_str = None
        else:
            crs_str = None

        for feature in src:
            if feature["properties"].get("divide_id") == divide_id:
                records.append(feature)

    if not records:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=crs_str)

    gdf = gpd.GeoDataFrame.from_features(records, crs=crs_str)
    return gdf
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

        # Buffer the merged polygon by flt_buffer (EPSG:5070 meters)
        buffered_polygon = merged_polygon.buffer(flt_buffer)

        # Convert to EPSG:4326 (lat/lon)
        gdf_buffered = gpd.GeoDataFrame(geometry=[buffered_polygon], crs=gdf.crs)
        gdf_buffered = gdf_buffered.to_crs("EPSG:4326")

        # Get bounding box coordinates (minx, miny, maxx, maxy)
        bbox = gdf_buffered.geometry.iloc[0].bounds
        minx, miny, maxx, maxy = bbox

        # Create bounding box GeoDataFrame
        divide_bbox = (minx, miny, maxx, maxy)  # <-- use tuple, not set
        divide_geom = gpd.GeoDataFrame({'geometry':[box(*divide_bbox)]}, crs="EPSG:4326")

        # Open VRT
        with rasterio.open(str_vrt_terrain) as src:
            # Clip band 1 to polygon
            out_image, out_transform = mask(src, divide_geom.geometry, crop=True, indexes=1)

            # Ensure 3D array for writing
            if out_image.ndim == 2:
                out_image = out_image[np.newaxis, :, :]

            # Update metadata
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "count": 1,
                "dtype": out_image.dtype
            })

            # Save clipped DEM

            output_file = os.path.join(str_outfolder, "dem_bbox_buffered_4326.tif")
            with rasterio.open(output_file, "w", **out_meta) as dest:
                dest.write(out_image[0], 1)
    else:
        print(f" No feature found with divide_id: {str_catchment_id}")
        output_file = ''

    return(output_file)
# -----------------


# ................
# Function to get last point of a LineString or MultiLineString
def fn_compute_last_point(geom):
    if geom.geom_type == "LineString":
        return Point(geom.coords[-1])
    elif geom.geom_type == "MultiLineString":
        return Point(list(geom.geoms[-1].coords)[-1])
    return None
# ................


# ----------------------------
def fn_get_divides_lateral_gdf(str_wb,
                               polygon,
                               str_url_divides,
                               str_url_flowpaths,
                               dict_all_params):
    
    
    flt_perct_bottom_line = float(dict_all_params['perct_bottom_line'])
    flt_perct_top_line = float(dict_all_params['perct_top_line'])

    # --- Read only flowpaths within polygon bbox ---
    bbox = tuple(polygon.total_bounds)
    flowpaths = gpd.read_file(str_url_flowpaths, bbox=bbox)

    # --- Select target line ---
    target_line = flowpaths.loc[flowpaths["id"] == str_wb]

    if target_line.empty:
        raise ValueError(f"No line with id '{str_wb}' found.")

    target_geom = target_line.geometry.iloc[0]

    # Merge MultiLineString if needed
    if target_geom.geom_type == "MultiLineString":
        target_geom = linemerge(target_geom)

    target_length = target_geom.length

    candidates = flowpaths.copy()
    candidates["last_point"] = candidates.geometry.apply(fn_compute_last_point)

    # Filter lines whose last point intersects target
    mask = candidates["last_point"].apply(lambda pt: pt.intersects(target_geom))
    lines_touching_target = candidates.loc[mask].copy()

    # Compute normalized measure
    lines_touching_target["measure"] = (
        lines_touching_target["last_point"]
        .apply(lambda pt: target_geom.project(pt) / target_length)
    )

    # Drop temp column and sort
    lines_touching_target = (
        lines_touching_target
        .drop(columns="last_point")
        .sort_values("measure")
        .reset_index(drop=True)
    )

    # --- Remove lateral inflow streams that are too close to beginning
    # or ending of the main stream ---
    lines_touching_target = (
        lines_touching_target[
            (lines_touching_target["measure"] >= flt_perct_bottom_line) &
            (lines_touching_target["measure"] <= flt_perct_top_line)
        ]
        .reset_index(drop=True)
    )

    # --- List of lateral inflow streams
    list_id_lateral = lines_touching_target["id"].tolist()

    # --- Add the mainstream to the list
    list_id_lateral.append(str_wb)
    
    int_catchment_count = len(list_id_lateral)

    # --- convert 'wb-' to 'cat-'
    list_id_lateral_cat = ["cat-" + item[3:] for item in list_id_lateral]

    gdf_list = []

    for divide_id in list_id_lateral_cat:
        gdf = fn_get_divide_gdf(str_url_divides, divide_id)

        if not gdf.empty:
            gdf_list.append(gdf)

    if not gdf_list:
        raise ValueError("No polygons were returned.")

    # --- Combine into single GeoDataFrame ---
    combined = gpd.GeoDataFrame(
        pd.concat(gdf_list, ignore_index=True),
        crs=gdf_list[0].crs
    )

    # --- Merge into single polygon geometry ---
    merged_polygon = unary_union(combined.geometry)

    # Wrap back into GeoDataFrame
    merged_gdf = gpd.GeoDataFrame(
        geometry=[merged_polygon],
        crs=combined.crs
    )
    
    return(int_catchment_count, merged_gdf)
# ----------------------------


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
    
    # Assume your polygon GeoDataFrame is called 'polygon' in EPSG:5070
    # Reproject to EPSG:4326
    polygon_4326 = polygon.to_crs(epsg=4326)
    
    # Get bounding box of the polygon
    minx, miny, maxx, maxy = polygon_4326.total_bounds
    bbox_geom = box(minx, miny, maxx, maxy)
    
    # Read only features intersecting the bounding box using pyogrio
    roads = gpd.read_file(
        str_url_roads, 
        bbox=(minx, miny, maxx, maxy)  # limits read to bbox
    )
    
    # Clip to exact polygon (still fast because only a subset is loaded)
    roads_clipped = gpd.clip(roads, polygon_4326)
    
    # Ensure CRS is EPSG:4326
    gdf_roads_clipped_5070 = roads_clipped.to_crs(epsg=5070)

    return(gdf_roads_clipped_5070)
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
# Function to get first and last vertex
def fn_get_first_last(line: LineString):
    coords = list(line.coords)
    return Point(coords[0]), Point(coords[-1])
# ---------------


# -------------
def fn_append_headwater_streams(str_stream_vector_filepath):
    # Load your streams shapefile
    streams = gpd.read_file(str_stream_vector_filepath)
    
    # Ensure CRS is set (important for geometry operations)
    if streams.crs is None:
        streams = streams.set_crs("EPSG:5070")
    
    # Extract all vertices (first + last) from every line
    all_vertices = []
    for geom in streams.geometry:
        f, l = fn_get_first_last(geom)
        all_vertices.extend([f.wkt, l.wkt])  # use WKT string for fast comparison
    
    # Now flag whether the first vertex is unique
    unique_flags = []
    for geom in streams.geometry:
        first, _ = fn_get_first_last(geom)
        if all_vertices.count(first.wkt) == 1:
            unique_flags.append(True)   # unique upstream start
        else:
            unique_flags.append(False)  # shared vertex
    
    # Add column to GeoDataFrame
    streams["is_head"] = unique_flags
    
    # --- Save to shapefile ---
    streams.to_file(str_stream_vector_filepath)
# -------------


# -----------
def fn_compute_contrib_area(group):
    # ensure numeric and sort ascending by flow_acc
    group = group.copy()
    group["flow_acc"] = pd.to_numeric(group["flow_acc"])
    group = group.sort_values("flow_acc").copy()

    # diffs: NaN for first row, then (current - previous)
    diffs = group["flow_acc"].diff()

    # assume is_head is consistent within group; pick the first value
    is_head = bool(group["is_head"].iloc[0])

    if is_head:
        # first keeps original flow_acc, others are differences
        group["contrib_area"] = diffs.fillna(group["flow_acc"].iloc[0])
    else:
        # first becomes zero, others are differences
        group["contrib_area"] = diffs.fillna(0)

    return group
# -----------


# --------------------
def fn_resolve_coincident_points(group):
    # Step 1: sum of contrib_area
    sum_contrib = group["contrib_area"].sum()

    # Step 2: flow_acc from the first row
    flow_acc = group["flow_acc"].iloc[0]

    # Step 3: compute inc_value
    inc_value = sum_contrib - flow_acc

    # Step 4: find index of largest contrib_area
    max_idx = group["contrib_area"].idxmax()

    # Reset contrib_area
    group["contrib_area"] = 0
    group.loc[max_idx, "contrib_area"] = inc_value

    return group
# --------------------


# ---------------------
def fn_raster_edge_cells_to_bci(
    str_raster_path,
    str_slope,
    write_gpkg=True):
    
    """
    Identify raster border and nodata-adjacent cells and write LISFLOOD BCI file.

    Parameters
    ----------
    str_raster_path : str
        Path to input raster
    str_slope : str, optional
        Slope value written to BCI file (default "0.01")
    write_gpkg : bool, optional
        If True, also writes a GeoPackage of edge cells

    Returns
    -------
    bci_path : str
        Path to written BCI file
    """

    base_dir = os.path.dirname(str_raster_path)
    base_name = os.path.splitext(os.path.basename(str_raster_path))[0]

    gpkg_path = os.path.join(base_dir, f"{base_name}_edge_cells.gpkg")
    bci_path = os.path.join(base_dir, f"{base_name}_edge_cells.bci")

    # ------------------------------------------------------------------
    # READ RASTER
    # ------------------------------------------------------------------
    with rasterio.open(str_raster_path) as src:
        data = src.read(1)
        nodata = src.nodata
        transform = src.transform
        crs = src.crs
        height, width = data.shape

    # ------------------------------------------------------------------
    # VALID DATA MASK
    # ------------------------------------------------------------------
    if nodata is None:
        valid = ~np.isnan(data)
    else:
        valid = data != nodata

    # ------------------------------------------------------------------
    # CELLS ON RASTER BORDER
    # ------------------------------------------------------------------
    border_mask = np.zeros_like(valid, dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True

    border_cells = valid & border_mask

    # ------------------------------------------------------------------
    # 4-DIRECTION ADJACENCY TO NODATA (N, S, E, W)
    # ------------------------------------------------------------------
    adjacent_to_nodata = np.zeros_like(valid, dtype=bool)

    adjacent_to_nodata[1:, :] |= ~valid[:-1, :]   # North
    adjacent_to_nodata[:-1, :] |= ~valid[1:, :]   # South
    adjacent_to_nodata[:, 1:] |= ~valid[:, :-1]   # West
    adjacent_to_nodata[:, :-1] |= ~valid[:, 1:]   # East

    edge_cells = valid & adjacent_to_nodata

    # ------------------------------------------------------------------
    # COMBINE CRITERIA
    # ------------------------------------------------------------------
    final_mask = border_cells | edge_cells
    rows, cols = np.where(final_mask)

    # ------------------------------------------------------------------
    # CONVERT TO POINTS (CELL CENTERS)
    # ------------------------------------------------------------------
    xs, ys, points = [], [], []

    for r, c in zip(rows, cols):
        x, y = rasterio.transform.xy(transform, r, c, offset="center")
        xs.append(x)
        ys.append(y)
        points.append(Point(x, y))

    # ------------------------------------------------------------------
    # WRITE GEOPACKAGE (OPTIONAL)
    # ------------------------------------------------------------------
    if write_gpkg:
        gdf = gpd.GeoDataFrame(
            {"x": xs, "y": ys},
            geometry=points,
            crs=crs
        )
        gdf.to_file(gpkg_path, layer="edge_cells", driver="GPKG")

    # ------------------------------------------------------------------
    # WRITE BCI FILE
    # ------------------------------------------------------------------
    with open(bci_path, "w") as f:
        for x, y in zip(xs, ys):
            f.write(f"F {x} {y} FREE {str_slope}\n")

    #print(f"Total unique edge cells: {len(xs)}")
    return bci_path
# ---------------------


# ---------------------
def fn_max_intensity_from_aoi_atlas_14(str_polygon_path,str_atlas_14_1000yr_5min):

    # Load polygon (assumes one feature)
    gdf = gpd.read_file(str_polygon_path)

    # If the CRS is missing, assign a default (usually EPSG:4326)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    polygon = gdf.geometry.iloc[0]
    
    # Load raster using rioxarray (which extends xarray for spatial work)
    rds = rioxarray.open_rasterio(str_atlas_14_1000yr_5min, masked=True)
    
    # If the raster CRS is missing, set it
    if rds.rio.crs is None:
        rds.rio.write_crs("EPSG:4326", inplace=True)

    # Ensure CRS matches
    if gdf.crs != rds.rio.crs:
        gdf = gdf.to_crs(rds.rio.crs)
        polygon = gdf.geometry.iloc[0]
    
    # Clip raster to polygon
    rds_clipped = rds.rio.clip([mapping(polygon)], gdf.crs, drop=True)
    
    # Compute the maximum pixel value
    flt_max_value = float(rds_clipped.max().values)
    
    flt_peak_rainfall_inhr = flt_max_value/100

    return(flt_peak_rainfall_inhr)
# ---------------------


# ---------------------
def fn_min_intensity_from_aoi_atlas_14(str_polygon_path,str_atlas_14_1yr_24hr):

    # Load polygon (assumes one feature)
    gdf = gpd.read_file(str_polygon_path)

    # If the CRS is missing, assign a default (usually EPSG:4326)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    polygon = gdf.geometry.iloc[0]
    
    # Load raster using rioxarray (which extends xarray for spatial work)
    rds = rioxarray.open_rasterio(str_atlas_14_1yr_24hr, masked=True)
    
    # If the raster CRS is missing, set it
    if rds.rio.crs is None:
        rds.rio.write_crs("EPSG:4326", inplace=True)

    # Ensure CRS matches
    if gdf.crs != rds.rio.crs:
        gdf = gdf.to_crs(rds.rio.crs)
        polygon = gdf.geometry.iloc[0]
    
    # Clip raster to polygon
    rds_clipped = rds.rio.clip([mapping(polygon)], gdf.crs, drop=True)
    
    # Compute the min pixel value
    flt_min_value = float(rds_clipped.min().values)
    
    flt_min_rainfall_inhr = flt_min_value / (1000 * 24) # average of 24 hour duration

    return(flt_min_rainfall_inhr)
# ---------------------


# --------
def fn_q_from_intensity(flt_rain_rate_mmhr, flt_max_cell_value, flt_pixel_area_base_terrain):
    # from a given intensity (mm per hour), determine the flow (cfs)
    flt_outflow_cfs = (flt_rain_rate_mmhr * flt_max_cell_value * flt_pixel_area_base_terrain * 35.3146667) / (1000 * 3600)
    return int(round(flt_outflow_cfs))
# --------


# --------
def fn_intensity_from_q(flt_outflow_cfs, flt_max_cell_value, flt_pixel_area_base_terrain):
    # from a given target flow (cfs), determine the desired intensity (mm-hr)
    flt_rain_rate_mmhr = (1000 * 3600 * flt_outflow_cfs) / (flt_max_cell_value * flt_pixel_area_base_terrain * 35.3146667)
    return(flt_rain_rate_mmhr)
# --------


# .................
def fn_logarithmic_progression(flt_min_q, flt_max_q, int_log_q_steps):
    values = np.ceil(np.logspace(np.log10(flt_min_q), np.log10(flt_max_q), num=int_log_q_steps)).astype(int)
    values[0], values[-1] = flt_min_q, flt_max_q  # Ensure first and last values are exactly start and end
    
    # returns a numpy array of log progression flows
    return values
# .................


# -------------------
def fn_get_fa_stats(fa):
    """
    Given a flow accumulation raster (rioxarray DataArray),
    return:
      - flt_pixel_area_base_terrain: pixel area in square meters
      - flt_max_cell_value: maximum flow accumulation value (ignoring NaNs)
    """
    # Get pixel size (width, height) in meters
    pixel_width, pixel_height = fa.rio.resolution()
    
    # Compute absolute pixel area (handles negative heights for north-up rasters)
    flt_pixel_area_base_terrain = abs(pixel_width * pixel_height)
    
    # Compute max cell value (ignoring NaNs and masked values)
    flt_max_cell_value = float(fa.max(skipna=True))
    
    return(flt_pixel_area_base_terrain, flt_max_cell_value)
# -------------------


# ---------------------
def fn_create_precip_nc(
    str_catchment,
    str_dem_asc_clipped_filepath,
    str_gpkg_filepath,
    str_out_folder_streams,
    flt_rain_mm,
    int_timesteps,
    fill_value = -999.0,
    flt_min_rain = 0.10):

    str_rain_mm = str(flt_rain_mm).replace('.', 'p')
    
    str_nc_pnt_rainfall_out = os.path.join(
        str_out_folder_streams,
        f"{str_catchment}_rainfall_pnt_{str_rain_mm}mm.nc"
    )
    
    # --------------------------------------------------
    # Read grid geometry from DEM
    # --------------------------------------------------
    with rasterio.open(str_dem_asc_clipped_filepath) as src:
        ncols = src.width
        nrows = src.height
        transform = src.transform
        dem_crs = src.crs
        cellsize = transform.a
        x0 = transform.c
        y0 = transform.f
    
    # Cell centres
    x = x0 + cellsize * (np.arange(ncols) + 0.5)
    y = y0 - cellsize * (np.arange(nrows) + 0.5)
    time = np.arange(0, int_timesteps, 1)
    
    # --------------------------------------------------
    # Read point rainfall data
    # --------------------------------------------------
    gdf = gpd.read_file(str_gpkg_filepath, layer="flow_points")
    
    # --------------------------------------------------
    # Rasterize contrib_ar to DEM grid
    # --------------------------------------------------
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
    
    # --------------------------------------------------
    # Create rainfall arrays
    # --------------------------------------------------
    
    # tiny rain in all cells to get them all wet
    rainfall = np.full(
        (int_timesteps, nrows, ncols),
        flt_min_rain,
        dtype=np.float32)
    
    valid = contrib_grid != fill_value
     
    # Apply rainfall to point cells only
    rainfall[:, valid] = contrib_grid[valid] * flt_rain_mm
    
    # Total rainfall over all timesteps
    total_rainfall = np.full(
        (nrows, ncols),
        fill_value,
        dtype=np.float32
    )
    
    total_rainfall[valid] = contrib_grid[valid] * flt_rain_mm * int_timesteps
    
    # --------------------------------------------------
    # Create NetCDF
    # --------------------------------------------------
    nc = Dataset(str_nc_pnt_rainfall_out, "w", format="NETCDF4_CLASSIC")
    
    nc.createDimension("time", None)
    nc.createDimension("y", nrows)
    nc.createDimension("x", ncols)
    
    tv = nc.createVariable("time", "f8", ("time",))
    yv = nc.createVariable("y", "f8", ("y",))
    xv = nc.createVariable("x", "f8", ("x",))
    
    # --- Compressed rainfall cube ---
    rf = nc.createVariable(
        "rainfall_depth",
        "f4",
        ("time", "y", "x"),
        fill_value=fill_value,
        zlib=True,
        complevel=4,
        chunksizes=(1, nrows, ncols)
    )
    
    # --- Compressed total rainfall ---
    tr = nc.createVariable(
        "total_rainfall_depth",
        "f4",
        ("y", "x"),
        fill_value=fill_value,
        zlib=True,
        complevel=4
    )
    
    # --------------------------------------------------
    # Global attributes
    # --------------------------------------------------
    nc.title = "Gridded Rainfall"
    nc.source = "Python-generated"
    nc.references = "TUFLOW NetCDF Rainfall Format"
    nc.comment = f"{str_catchment} -- Point-based rainfall: contrib_ar--{str(flt_rain_mm)} mm per  timestep"
    
    tv.units = "hours"
    yv.units = "m"
    xv.units = "m"
    
    # --------------------------------------------------
    # Write data
    # --------------------------------------------------
    tv[:] = time
    yv[:] = y
    xv[:] = x
    rf[:, :, :] = rainfall
    tr[:, :] = total_rainfall
    
    nc.close()
    
    #print("Created:", str_nc_pnt_rainfall_out)

    return(str_nc_pnt_rainfall_out)
# ---------------------


# ---------------
def fn_format_params(params):
    """
    params: list of (key, value) tuples.
            Use None (or '') for keys that have no value.
    returns: multi-line formatted string
    """

    VALUE_COL = 24  # 1-based column index where value should start
    lines = []

    for key, value in params:
        tag = f"{key}"

        if value is None or value == "":
            # key only line
            lines.append(tag)
        else:
            # number of spaces so value starts in column 24
            pad = max(1, VALUE_COL - len(tag))
            lines.append(tag + (" " * pad) + str(value))

    return "\n".join(lines)
# --------------


# ----------------
def fn_extract_run_name(filename):
    """
    From 'cat-2402262_rainfall_pnt_333p8mm.nc'
    return 'cat-2402262_333p8mm'
    """
    name = Path(filename).stem  # remove .nc
    left, right = name.split("_rainfall_pnt_")
    return f"{left}_{right}"
# ----------------


# ...............................
def fn_condition_terrain(dict_filepaths, 
                         str_out_folder_streams,
                         str_whitebox_path,
                         flt_threshold,
                         polygon,
                         str_url_roads):

    wbt = WhiteboxTools()
    wbt.verbose = False  # print tool messages

    # Now set the working directory
    wbt.work_dir = str_out_folder_streams

    # Step 1 -- Breach Depressions Least Cost
    cmd = [
        str_whitebox_path,
        "--run=BreachDepressionsLeastCost",
        f"--dem={dict_filepaths['dem_clipped']}",
        f"-o={dict_filepaths['dem_breach']}",
        f"--dist=2000",
        f"--max_cost=500",
        f"--flat_increment=0.001"]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    # Step 2 -- Flow Accumulation (Pass 01)
    # --- Fill DEM depressions ---
    wbt.fill_depressions(dict_filepaths['dem_breach'], dict_filepaths['dem_filled'])

    # --- Flow direction ---
    wbt.d8_pointer(dict_filepaths['dem_filled'], dict_filepaths['dem_fdir'])

    # --- Flow accumulation (D8) ---
    wbt.d8_flow_accumulation(dict_filepaths['dem_filled'], dict_filepaths['dem_fa'], out_type="cells")

    # Step 3 -- Stream Network (Pass 01)
    # --- Extract stream locations

    cmd = [
        str_whitebox_path,
        "--run=ExtractStreams",
        f"--flow_accum={dict_filepaths['dem_fa']}",
        f"--d8_pntr={dict_filepaths['dem_fdir']}",
        f"--output={dict_filepaths['stream_raster']}",
        f"--threshold={flt_threshold}",
        "--zero_background"]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    # Convert raster streams to vector
    wbt.verbose = False  # print tool messages
    wbt.raster_streams_to_vector(dict_filepaths['stream_raster'],
                                 dict_filepaths['dem_fdir'],
                                 dict_filepaths['stream_vector'])

    # Step 4 -- Burn Streams at Road 

    # note -- needs road vector extraction from fgb
    gdf_roads_clipped_5070 = fn_get_clipped_roads(polygon, str_url_roads)

    # Save as Shapefile
    gdf_roads_clipped_5070.to_file(dict_filepaths['clipped_roads'], driver="ESRI Shapefile")


    cmd = [
        str_whitebox_path,
        "--run=BurnStreamsAtRoads",
        f"--dem={dict_filepaths['dem_breach']}",          # Input DEM (breached or preprocessed)
        f"--streams={dict_filepaths['stream_vector']}",  # Vector streams (preliminary pass 1)
        f"--roads={dict_filepaths['clipped_roads']}",      # Vector roads (centerlines)
        f"--output={dict_filepaths['dem_burn_roads']}",    # Output DEM with burned culverts
        f"--width=20"           # Maximum road embankment width in map units
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    # Step 5 -- Flow Accumulation (Pass 02)

    # --- Fill DEM depressions ---
    wbt.fill_depressions(dict_filepaths['dem_burn_roads'], dict_filepaths['dem_filled'])

    # --- Flow direction ---
    wbt.d8_pointer(dict_filepaths['dem_filled'], dict_filepaths['dem_fdir'])

    # --- Flow accumulation (D8) ---
    wbt.d8_flow_accumulation(dict_filepaths['dem_filled'], dict_filepaths['dem_fa'], out_type="cells")


    # Step 6 -- Stream Network (Pass 02)
    # --- Extract stream locations

    cmd = [
        str_whitebox_path,
        "--run=ExtractStreams",
        f"--flow_accum={dict_filepaths['dem_fa']}",
        f"--d8_pntr={dict_filepaths['dem_filled']}",
        f"--output={dict_filepaths['stream_raster']}",
        f"--threshold={flt_threshold}",
        "--zero_background"]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    # Convert raster streams to vector
    wbt.verbose = False  # print tool messages
    wbt.raster_streams_to_vector(dict_filepaths['stream_raster'],
                                 dict_filepaths['dem_fdir'],
                                 dict_filepaths['stream_vector'])
# ...............................


# .........................................................
def fn_prepare_input_layers_01(
    str_global_config_file_path,
    str_local_config_file_path,
    b_print_output
):
    import warnings
    import configparser

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore",category=RuntimeWarning,module="pyogrio")

    # ---- Header output ----
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
            'url_divides',
            'url_flowpaths',
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
            'perct_bottom_line',
            'perct_top_line',
            'stream_threshold_sq_mi'
        ],
        'flow_parameters': [
            'num_steps',
            'timesteps',
            'output_step',
            'mass_balance_step'
        ]
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

    
    #print(dict_all_params)
    # --------------- Make folders absolute
    str_catchment = dict_all_params['catchment']
    str_wb = 'wb-' + str_catchment[4:]
    
    str_out_root_folder = os.path.abspath(dict_all_params['out_root_folder'])
    str_out_folder = os.path.join(str_out_root_folder, str_catchment)
    os.makedirs(str_out_folder, exist_ok=True)

    str_out_folder_streams = os.path.join(str_out_folder, '01_stream_delineation')
    str_out_folder_streams_02 = os.path.join(str_out_folder, '02_lisflood_input')
    os.makedirs(str_out_folder_streams, exist_ok=True)
    os.makedirs(str_out_folder_streams_02, exist_ok=True)

    # --- output file names (absolute)
    str_header = fn_two_digit_string(0)
    
    dict_files = {
        "dem_clipped": "dem_clipped_5070.tif",
        "dem_breach": "dem_breach_5070.tif",
        "dem_filled": "dem_filled.tif",
        "dem_fdir": "fdir.tif",
        "dem_fa": "flow_accum.tif",
        "stream_raster": "streams.tif",
        "stream_vector": "streams.shp",
        "stream_points": "stream_vertices_area.shp",
        "stream_points_acc": "stream_vert_acc_pnt.shp",
        "clipped_roads": "clipped_roads_ln_5070.shp",
        "dem_burn_roads": "dem_burn_roads_5070.tif",
    }
    
    dict_paths_base = {
        key: fn_build_path(str_out_folder_streams, str_header, filename)
        for key, filename in dict_files.items()
    }
    
    str_header_lateral  = fn_two_digit_string(1)
    
    dict_paths_lateral = {
        key: fn_build_path(str_out_folder_streams, str_header_lateral, filename)
        for key, filename in dict_files.items()
    }
    
    ##str_dem_clipped_lateral_filepath = os.path.abspath(os.path.join(str_out_folder_streams, 'dem_clipped_lateral_5070.tif'))

    str_polygon_filepath = os.path.abspath(os.path.join(str_out_folder_streams, 'watershed_ar_4326.geojson'))
    str_gpkg_filepath = os.path.abspath(os.path.join(str_out_folder_streams, str_catchment + '.gpkg'))

    str_dem_asc_clipped_filepath = os.path.abspath(os.path.join(str_out_folder_streams, 'dem_clipped_5070.asc'))
    str_nc_pnt_rainfall_out = os.path.abspath(os.path.join(str_out_folder_streams, 'dem_clipped_5070.asc'))

    str_dem_asc_clipped_filepath_02 = os.path.abspath(os.path.join(str_out_folder_streams_02, 'dem_clipped_5070.asc'))
    str_nc_pnt_rainfall_out_02 = os.path.abspath(os.path.join(str_out_folder_streams_02, 'dem_clipped_5070.asc'))

    
    # other environment settings
    # TODO -- can probably eliminate once I am using the Docker Container
    #os.environ["PROJ_LIB"] = r"C:\Users\civil\anaconda3\envs\grib-env\Lib\site-packages\pyproj\proj_dir\share\proj"
    #str_whitebox_path = r"C:\Users\civil\anaconda3\envs\grib-env\Lib\site-packages\whitebox\whitebox_tools.exe"
    
    os.environ["PROJ_LIB"] = "/opt/miniconda/envs/geo/share/proj"
    str_whitebox_path = "/opt/whitebox_tools/whitebox_tools"
    
    # --- Load polygon ---
    str_url_divides = dict_all_params['url_divides']
    str_url_flowpaths = dict_all_params['url_flowpaths']
    str_vrt_terrain = dict_all_params['vrt_terrain']
    str_url_roads = dict_all_params['url_roads']
    int_terrain_buffer_m = int(dict_all_params['terrain_buffer_m'])
    int_downscale = int(dict_all_params['downscale'])
    
    # -- ASSUMES BASE TERRAIN OF 3 meters --
    # *** Note:  Assuming each pixel is 3m x 3m -- HARDCODED
    int_pixel_size = 3
    
    flt_stream_threshold_sq_mi = float(dict_all_params['stream_threshold_sq_mi'])
    int_cell_to_start_stream = int(flt_stream_threshold_sq_mi * 2589988 / (int_pixel_size * int_pixel_size))
    
    # *** Number of cells (at downscaled resolution) where stream begins
    flt_threshold = (int_cell_to_start_stream / (int_downscale * int_downscale)) # cell count to start streams
    
    if b_print_output:
        print('  -- STEP 1: Finding terrain')
    polygon = fn_get_divide_gdf(str_url_divides, str_catchment)
    if polygon.crs is None:
        polygon.set_crs("EPSG:5070", inplace=True)
    
    # determine lateral watersheds
    int_catchment_count, gdf_lateral = fn_get_divides_lateral_gdf(str_wb,
                                                                  polygon,
                                                                  str_url_divides,
                                                                  str_url_flowpaths,
                                                                  dict_all_params)
    
    if gdf_lateral.crs is None:
        gdf_lateral.set_crs("EPSG:5070", inplace=True)
    
    # --- Load raster DEM (created in EPSG:4326) ---
    str_dem_4326_filepath = fn_create_terrain_tif(
        str_catchment,
        str_out_folder_streams,
        gdf_lateral,
        str_vrt_terrain,
        int_terrain_buffer_m,
        b_print_output)
        
    dem = rioxarray.open_rasterio(str_dem_4326_filepath, masked=True)

    # --- Reproject DEM to EPSG:5070 ---
    dem_5070 = dem.rio.reproject("EPSG:5070")
    
    # --- Downscale DEM: int_downscale (like 2x) pixel size using average ---
    res_x, res_y = dem_5070.rio.resolution()
    new_resolution = (res_x * int_downscale, res_y * int_downscale)
    
    resampling_method = Resampling.average
    
    dem_5070_downscaled = dem_5070.rio.reproject(
        dem_5070.rio.crs,
        resolution=new_resolution,
        resampling=resampling_method)
    
    # --- Ensure CRS match for clipping ---
    if gdf_lateral.crs != dem_5070_downscaled.rio.crs:
        gdf_lateral = gdf_lateral.to_crs(dem_5070_downscaled.rio.crs)
        
    # --- Clip downscaled DEM ---
    dem_clipped_5070 = dem_5070_downscaled.rio.clip(
        polygon.geometry.apply(mapping),
        polygon.crs,
        from_disk=True)
    
    # --- Save clipped, downscaled DEM ---
    dem_clipped_5070.rio.to_raster(dict_paths_base['dem_clipped'])
    
    # --- Clip downscaled lateral DEM ---
    if int_catchment_count > 1:
        dem_clipped_lateral_5070 = dem_5070_downscaled.rio.clip(
            gdf_lateral.geometry.apply(mapping),
            gdf_lateral.crs,
            from_disk=True)
    
        dem_clipped_lateral_5070.rio.to_raster(dict_paths_lateral['dem_clipped'])
    
    # --- Save rectangular (bbox) downscaled DEM ---
    str_dem_bbox_filepath = os.path.join(
        str_out_folder_streams,
        "dem_bbox_buffered_5070.tif"
    )
    
    dem_5070_downscaled.rio.to_raster(str_dem_bbox_filepath)
    
    
    # ..............
    # TODO - 2026.02.04 -- Possible function (whitebox conditioning)
    if b_print_output:
        print('  -- STEP 3: Stream Conditioning')
    
    # condition the base polygon (single catchment)
    fn_condition_terrain(dict_paths_base, 
                         str_out_folder_streams,
                         str_whitebox_path,
                         flt_threshold,
                         polygon,
                         str_url_roads)
        
    
    # condition the terrain with lateral watersheds
    if int_catchment_count > 1:
        fn_condition_terrain(dict_paths_lateral, 
                             str_out_folder_streams,
                             str_whitebox_path,
                             flt_threshold,
                             gdf_lateral,
                             str_url_roads)
    
    # ..............
    if b_print_output:
        print('  -- STEP 4: Determine rainfall loactions')
     
    # Each vertex in the "streams" shapefile is whithin the center of a cell in the 
    # flow accumulation grid.  Create a point shapefile that is all the vertex points
    # in all these lines. Create an attribute as to which line (FID from streams.shp)
    # and flow_ac value from the flow_accum.tif raster
    
    # The 'streams' has an FID 1,2,3,4 ... and a value of is_head (T/F).
    # For each unique 'FID' in points_gdf, get list of matching points in 'points_gdf'
    # If 'is_head' is True, then determine the lowest 'flow_acc' in the matching point list
    # and then subtract that lowest value from the other points creating a new coloumn value
    # of 'contib_area'.  The point with the lowest value should remain unchanged.
    
    # If 'is_head' is False, then determine the lowest 'flow_acc' in the matching point list
    # and then subtract that lowest value from the other points creating a new coloumn value
    # of 'contib_area', including the lowest point.  The lowest point should be zero.
    
    fn_append_headwater_streams(dict_paths_base['stream_vector'])
    
    # --- Load streams ---
    streams = gpd.read_file(dict_paths_base['stream_vector'])
    streams = streams.set_crs("EPSG:5070")
    
    # --- Load flow accumulation raster ---
    fa = rioxarray.open_rasterio(dict_paths_base['dem_fa'], masked=True)
    
    # --- If raster is not EPSG:5070, reproject it ---
    if fa.rio.crs.to_string() != "EPSG:5070":
        # from rioxarray.enums import Resampling
        fa = fa.rio.reproject("EPSG:5070", resampling=Resampling.nearest)
    
    # --- Prepare list for vertices ---
    point_records = []
    
    for idx, row in streams.iterrows():
        geom = row.geometry
        fid = row["FID"] if "FID" in row else idx
    
        # Extract vertices
        if geom.geom_type == "LineString":
            coords = list(geom.coords)
        elif geom.geom_type == "MultiLineString":
            coords = [c for line in geom for c in line.coords]
        else:
            continue
    
        for x, y in coords:
            # Sample flow accumulation raster at vertex
            flow_val = fa.sel(x=x, y=y, method="nearest").values.item()
            point_records.append({
                "geometry": Point(x, y),
                "FID": fid,
                "flow_acc": float(flow_val)
            })
    
    # --- Create GeoDataFrame in EPSG:5070 ---
    points_gdf = gpd.GeoDataFrame(point_records, crs="EPSG:5070")
    
    # --- Save to shapefile ---
    points_gdf.to_file(dict_paths_base['stream_points'])
    
    # left join stream points_gdf with stream on 'FID' to add "is_head" value to
    # every point on points_gdf
    
    points_gdf = points_gdf.merge(
        streams[["FID", "is_head"]],   # only bring in needed cols
        on="FID", 
        how="left"
    )
    
    # Apply to the full GeoDataFrame
    #points_gdf_contrib = points_gdf.groupby("FID", group_keys=False).apply(fn_compute_contrib_area)
    # TODO -- this may me an issue with the precip file -- 2026.02.05
    points_gdf_contrib = (points_gdf.groupby("FID", group_keys=False).apply(fn_compute_contrib_area, include_groups=False))

    # --- Save to shapefile ---
    points_gdf_contrib.to_file(dict_paths_base['stream_points_acc'])
    
    # --- Identify duplicate geometries ---
    dup_mask = points_gdf_contrib.duplicated(subset="geometry", keep=False)
    
    # --- Split into duplicates and uniques ---
    dupes = points_gdf_contrib[dup_mask].copy()
    uniques = points_gdf_contrib[~dup_mask].copy()
    
    # --- Process only duplicates ---
    #dupes = dupes.groupby("geometry", group_keys=False).apply(fn_resolve_coincident_points)
    dupes = (dupes.groupby("geometry", group_keys=False).apply(fn_resolve_coincident_points, include_groups=False))
    
    # --- Combine back ---
    points_gdf_contrib = pd.concat([uniques, dupes], ignore_index=True)
    
    #total_contrib = points_gdf_contrib["contrib_area"].sum()
    #print("Total contrib_area (duplicates removed):", total_contrib)
    
    # Drop the 'FID' column if it exists
    if 'FID' in points_gdf_contrib.columns:
        points_gdf_contrib = points_gdf_contrib.drop(columns=['FID'])
    
    points_gdf_contrib.to_file(dict_paths_base['stream_points_acc'])
    
    polygon.to_file(str_polygon_filepath, driver="GeoJSON")
    
    # create a shapefile of the basin in EPSG:4326
    str_polygon_shp_filepath = os.path.join(str_out_folder_streams, 'watershed_ar_4326.shp')
    polygon.to_file(str_polygon_shp_filepath, driver="ESRI Shapefile")
    
    # ----------
    # create a summary geopackage
    
    # Delete existing GPKG if it exists
    if os.path.exists(str_gpkg_filepath):
        os.remove(str_gpkg_filepath)
    
    # List of your vector layers and desired layer names
    vector_layers = [
        (str_polygon_filepath, "watershed"),
        (dict_paths_base['stream_vector'], "streams"),
        (dict_paths_base['stream_points_acc'], "flow_points")
    ]
    
    # Loop through layers and write them to the same GPKG
    for vector_path, layer_name in vector_layers:
        gdf = gpd.read_file(vector_path)
        if layer_name == "streams":
            gdf['catchment']=str_catchment
            gdf['threshold']=flt_threshold
        if layer_name == "watershed":
            gdf['terrain_clip']=dict_paths_base['dem_clipped']
            gdf['terrain_source']=str_vrt_terrain
            gdf['hydrofabric_source']=str_url_divides
        gdf.to_file(str_gpkg_filepath, layer=layer_name, driver="GPKG")
        
    # Create a boundary condition from the terrain file
    if b_print_output:
        print('  -- STEP 5: Outflow Boundary Slope')
    str_slope = dict_all_params['outflow_boundary_slope']
    
    if int_catchment_count > 1:
        # need to use entire terrain pluss lateral watersheds
        str_bci_path = fn_raster_edge_cells_to_bci(dict_paths_lateral['dem_clipped'],str_slope)
    else:
        str_bci_path = fn_raster_edge_cells_to_bci(dict_paths_base['dem_clipped'],str_slope)
    
    src = Path(str_bci_path)
    dst = src.parents[1] / "02_lisflood_input" / src.name
    shutil.copy2(src, dst)
    
    if b_print_output:
        print('  -- STEP 6: Prep terrain for LISFLOOD')
    # Convert the geotiff to ASC for use in LISFLOOD-FP
    
    # Output ASCII Grid
    #asc_out = r'E:\cog_tile_site\cat-2402255\01_stream_delineation\dem_clipped_5070.asc'
    #str_dem_asc_clipped_filepath
    
    # Open with rioxarray
    if int_catchment_count > 1:
        dem = rioxarray.open_rasterio(dict_paths_lateral['dem_burn_roads'], masked=True)
    else:
        dem = rioxarray.open_rasterio(dict_paths_base['dem_burn_roads'], masked=True)
    
    # If the raster has multiple bands, select first band
    if dem.rio.count > 1:
        dem = dem.sel(band=1)
    
    # Set nodata if missing
    if dem.rio.nodata is None:
        dem.rio.write_nodata(-9999, inplace=True)
    
    # Write to ASCII Grid
    dem.rio.to_raster(str_dem_asc_clipped_filepath, driver='AAIGrid')
    
    # Copy from one folder to another
    src = Path(str_dem_asc_clipped_filepath)
    dst = src.parents[1] / "02_lisflood_input" / src.name
    shutil.copy2(src, dst)
    
    if b_print_output:
        print('  -- STEP 7: Determine bounding rain intensities')
    
    str_atlas_14_1000yr_5min = dict_all_params['atlas_14_1000yr_5min']
    str_atlas_14_1yr_24hr = dict_all_params['atlas_14_1yr_24hr']
    int_num_steps = int(dict_all_params['num_steps'])
    
    flt_peak_rainfall_inhr = fn_max_intensity_from_aoi_atlas_14(str_polygon_filepath,str_atlas_14_1000yr_5min)
    flt_min_rainfall_inhr = fn_min_intensity_from_aoi_atlas_14(str_polygon_filepath, str_atlas_14_1yr_24hr)
    
    flt_peak_rain_rate_mmhr = flt_peak_rainfall_inhr * 25.4
    flt_min_rain_rate_mmhr = flt_min_rainfall_inhr * 25.4
    
    fa = rioxarray.open_rasterio(dict_paths_base['dem_fa'], masked=True)
    flt_pixel_area_base_terrain, flt_max_cell_value = fn_get_fa_stats(fa)
    
    flt_max_q_cfs = fn_q_from_intensity(flt_peak_rain_rate_mmhr, flt_max_cell_value, flt_pixel_area_base_terrain)
    flt_min_q_cfs = fn_q_from_intensity(flt_min_rain_rate_mmhr, flt_max_cell_value, flt_pixel_area_base_terrain)
    
    arr_log_flows_cfs = fn_logarithmic_progression(flt_min_q_cfs, flt_max_q_cfs, int_num_steps)
    
    # Apply the function to each flow value
    arr_intensity_mm_hr = np.array([
        fn_intensity_from_q(q, flt_max_cell_value, flt_pixel_area_base_terrain)
        for q in arr_log_flows_cfs])
    
    arr_intensity_mm_hr = arr_intensity_mm_hr.round(1)
    
    if b_print_output:
        print('  -- STEP 8: Writing paramter and rainfall data')
    
    int_timesteps = int(dict_all_params['timesteps'])
    int_output_step = int(dict_all_params['output_step'])
    flt_mass_balance_step = float(dict_all_params['mass_balance_step'])
    
    str_simtime = str(int_timesteps * 3600)
    str_saveint = str(float(int_output_step * 3600))
    str_mass_step = str(flt_mass_balance_step * 3600)
    
    # Get the dem file name
    src_dem = Path(str_dem_asc_clipped_filepath)
    str_dem_filename = src_dem.name
    
    # get the bci filename
    src_bci =Path(str_bci_path)
    str_bci_filename = src_bci.name
    
    src = Path(str_bci_path)
    dst = src.parents[1] / "02_lisflood_input" / src.name
    shutil.copy2(src, dst)
    
    # create all of the rainfall netcdfs for the requested steps
    for flt_rain_mm in arr_intensity_mm_hr:
        str_nc_pnt_rainfall_out = fn_create_precip_nc(
            str_catchment,
            str_dem_asc_clipped_filepath,
            str_gpkg_filepath,
            str_out_folder_streams,
            flt_rain_mm,
            int_timesteps)
    
    
        # get the rainfall filename
        src_rainfall = Path(str_nc_pnt_rainfall_out) # still in the wrong folder
        str_rainfall_filename = src_rainfall.name
    
        dst_rainfall = src_rainfall.parents[1] / "02_lisflood_input" / src_rainfall.name
        shutil.copy2(src_rainfall, dst_rainfall)
        
        str_run_name = fn_extract_run_name(str_rainfall_filename)
        
        # Example usage
        params = [
            ("DEMfile", str_dem_filename),
            ("bcifile", str_bci_filename),
            ("resroot", str_run_name),
            ("dirroot", str_run_name),
            ("sim_time", str_simtime),
            ("initial_tstep", dict_all_params['initial_tstep']),
            ("massint", str_mass_step),
            ("saveint", str_saveint),
            ("fpfric", dict_all_params['fpfric']),
            ("elevoff", None),
            ("acceleration", None),
            ("SGC_Enable", None),
            ("dynamicrainfile", str_rainfall_filename),
            ("depththresh", dict_all_params['depththresh']),
            ("max_Froude", dict_all_params['max_Froude']),
        ]
        
        #print(fn_format_params(params))
    
        str_param_filename = str_run_name + ".par"
        
        dst_param = src_rainfall.parents[1] / "02_lisflood_input" / str_param_filename
        param_text = fn_format_params(params)
    
        with open(dst_param, "w", encoding="utf-8") as f:
            f.write(param_text)
        #print('---------------------------')
    
# .........................................................


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == '__main__':

    flt_start_run = time.time()
    
    parser = argparse.ArgumentParser(description='========= PREPARE INPUT LAYERS FOR LISFLOOD-FP =========')
    
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
    
    str_global_config_file_path = args['str_global_config_file_path']
    str_local_config_file_path = args['str_local_config_file_path']
    b_print_output = args['b_print_output']

    fn_prepare_input_layers_01(str_global_config_file_path, str_local_config_file_path, b_print_output)

    flt_end_run = time.time()
    flt_time_pass = (flt_end_run - flt_start_run) // 1
    time_pass = datetime.timedelta(seconds=flt_time_pass)
    
    print('Compute Time: ' + str(time_pass))

 #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
