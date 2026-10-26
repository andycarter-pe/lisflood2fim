# ************************************************************
# LISFLOOD-FP to FIM
# Script 02 - run_lisflood_02
#
# Created by: Andy Carter, PE
# Created - 2026.02.07
# Revised - 2026.05.08 -- adjusting Mannings n after the 'default_manning_runs' run
# Revised - 2026.10.21 -- stability check uses rain applied (from .mass) as expected
#                          outflow; startfile retry fixed; low-intensity fallback
# Revised - 2026.10.24 -- run time limits from the [lisflood_timeouts] section of
#                          the global config: cap on sim_time, wall-clock budget
#                          per run, hard kill timeout, startfile retry mode
# ************************************************************

# ************************************************************
import geopandas as gpd
import pandas as pd
import rasterio
import os
from pathlib import Path
import re
import shutil
import math
import subprocess

import argparse
import configparser
import time
import datetime
import warnings

import threading
import itertools
import sys
import numpy as np
import rasterio
# ************************************************************

# -----------------
def fn_run_with_spinner(func, *args, message="Running"):
    spinner = itertools.cycle("|/-\\")
    result = {}

    def wrapper():
        try:
            result["value"] = func(*args)
        except BaseException as e:
            result["error"] = e

    thread = threading.Thread(target=wrapper)
    thread.start()

    while thread.is_alive():
        sys.stdout.write(f"\r{message}... {next(spinner)}")
        sys.stdout.flush()
        time.sleep(0.1)

    thread.join()
    sys.stdout.write("\r" + " " * 50 + "\r")  # clear line

    # re-raise anything the worker thread hit (e.g. a LISFLOOD timeout)
    if "error" in result:
        raise result["error"]
    return result.get("value")
# -----------------

# -----------------
def fn_get_expected_outflow (flt_intensity, str_stream_folder, str_catchment):
    str_gpkg_path = os.path.join(str_stream_folder, str_catchment + ".gpkg")
    str_terrain_dem_path = os.path.join(str_stream_folder, "00_dem_clipped_5070.tif")
    
    # Read a specific layer (flow_points)
    gdf_flow_points = gpd.read_file(str_gpkg_path, layer="flow_points")

    # Sum the values in contrib_ar col = number of cells flowing out
    flt_total_cells_out = gdf_flow_points["contrib_ar"].sum()

    with rasterio.open(str_terrain_dem_path) as src:
        pixel_width, pixel_height = src.res

    flt_pixel_area = abs(pixel_width * pixel_height) # in square meters

    flt_contrib_area = flt_total_cells_out * flt_pixel_area

    flt_expected_outflow = flt_contrib_area * flt_intensity / (1000 * 3600) # in cms (cubic meters per second)
    
    return(flt_expected_outflow)
# -----------------


# -----------------
def fn_get_stable_row(str_mass_balance_filepath, flt_expected_outflow, dict_all_params):
    """
    Find the first row of a LISFLOOD .mass file where the run is at steady state.

    Revised 2026.10.21 -- expected outflow is taken from the rain actually
    applied (slope of the cumulative 'Rain-(Inf+Evap)' column) rather than
    from the flow_points contrib_ar sum, which can double count area where
    a stream segment is wrongly flagged as a headwater.  flt_expected_outflow
    is only used if the rain column is missing.

    Returns:
        (status, result)

        status:
            'ok'            -> stable row found
            'not_found'     -> mass file missing
            'invalid_file'  -> file unreadable or empty
            'missing_qout'  -> Qout column missing
            'no_match'      -> file read but no stable row found
                               (result is the LAST row, so the caller can
                               use it as a fallback)

        result:
            pandas.Series (if status == 'ok' or 'no_match')
            None otherwise
    """

    window = int(dict_all_params['window'])
    flt_min_Qout_ratio = float(dict_all_params['min_Qout_ratio'])
    flt_max_Qout_ratio = float(dict_all_params['max_Qout_ratio'])
    flt_max_rolling_avg_stability = float(dict_all_params['max_rolling_avg_stability'])

    # --- file existence ---
    if not os.path.isfile(str_mass_balance_filepath):
        return 'not_found', None

    # --- read file safely ---
    try:
        df_mass = pd.read_csv(
            str_mass_balance_filepath,
            sep=r"\s+",
            header=0
        )
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return 'invalid_file', None

    if df_mass.empty or len(df_mass) < 2:
        return 'invalid_file', None

    # --- required column ---
    if "Qout" not in df_mass.columns:
        return 'missing_qout', None

    # --- expected outflow: net rain actually applied (m3/s) ---
    str_rain_col = "Rain-(Inf+Evap)"
    if str_rain_col in df_mass.columns:
        df_mass["Q_expected"] = (
            df_mass[str_rain_col].diff() / df_mass["Time"].diff()
        ).rolling(window=window, min_periods=1).mean()
    else:
        if flt_expected_outflow is None or flt_expected_outflow == 0:
            return 'invalid_file', None
        df_mass["Q_expected"] = flt_expected_outflow

    # --- calculations ---
    df_mass["Qout_rel_change"] = (
        df_mass["Qout"].diff() / df_mass["Qout"].shift(1)
    )

    # min_periods=window so a single early change cannot pass the test
    df_mass["Qout_rel_change_rollavg"] = (
        df_mass["Qout_rel_change"]
        .abs()
        .rolling(window=window, min_periods=window)
        .mean()
    )

    df_mass["Qout_ratio"] = df_mass["Qout"] / df_mass["Q_expected"]

    # --- stability mask ---
    mask = (
        (df_mass["Qout_rel_change_rollavg"] < flt_max_rolling_avg_stability) &
        (df_mass["Qout_ratio"] > flt_min_Qout_ratio) &
        (df_mass["Qout_ratio"] < flt_max_Qout_ratio)
    )

    matches = df_mass.loc[mask]

    if not matches.empty:
        return 'ok', matches.iloc[0]

    return 'no_match', df_mass.iloc[-1]
# -----------------


# -----------------
def fn_run_par_file(str_lisflood_folder, str_par_file_to_run, str_run_label, str_message,
                    flt_timeout_sec=None):
    flt_loop_time = fn_run_with_spinner(
        fn_run_docker_lisflood,
        str_lisflood_folder,
        str_par_file_to_run,
        flt_timeout_sec,
        message=f"     -- {str_run_label} {str_message} LISFLOOD"
    )

    print(
        f"     -- {str_run_label} Run {str_par_file_to_run[:-4]} "
        f"completed in {flt_loop_time} seconds"
    )
    return flt_loop_time
# -----------------


# --------------
def fn_collect_par_values(str_lisflood_folder):
    results = []

    # regex to capture the number part like 189p4 or 88p1
    pattern = re.compile(r'_([0-9]+p[0-9]+)mm\.par$')

    for par_file in Path(str_lisflood_folder).rglob("*.par"):
        match = pattern.search(par_file.name)
        if match:
            value_str = match.group(1).replace('p', '.')
            value = float(value_str)
            results.append((str(par_file), value))

    return results
# --------------


# --------------
def fn_get_next_intensity_row(df, intensity):
    # ensure sorted
    df_sorted = df.sort_values("intensity").reset_index(drop=True)

    # find rows strictly greater than the given intensity
    next_rows = df_sorted[df_sorted["intensity"] > intensity]

    if next_rows.empty:
        return None  # no next row available

    return next_rows.iloc[0]
# --------------


# ---------------
def fn_prep_next_par_file(next_row,
                          int_time_to_stable,
                          str_stable_depth_filepath,
                          str_lisflood_folder,
                          b_use_startfile):

    if next_row is None:
        #print("No higher intensity available")
        pass
    else:
        #print(f"Next intensity: {next_row['intensity']}")

        str_parameter_to_ammend_filepath = next_row['filepath']

        # get the next parameter file
        df_par_to_ammend = pd.read_csv(
            str_parameter_to_ammend_filepath,
            sep=r"\s+",          # one or more whitespace chars
            header=None,
            comment="#"
        )

        # Convert df to dictionary
        dict_to_ammend_parameters = dict(zip(df_par_to_ammend[0], df_par_to_ammend[1]))

        # revise the sim_time and saveint of the next parameter file
        dict_to_ammend_parameters['sim_time'] = str(int_time_to_stable)
        dict_to_ammend_parameters['saveint'] = f"{float(int_time_to_stable)}"

        # copy the last stable run into the str_lisflood_folder
        shutil.copy2(str_stable_depth_filepath,str_lisflood_folder)

        # add the startfile of the last stable run, is b_use_startfile == True
        if b_use_startfile:
            dict_to_ammend_parameters['startfile'] = os.path.basename(str_stable_depth_filepath)

        # Overwrite the existing parameter file
        int_allign = 24

        with open(str_parameter_to_ammend_filepath, "w") as f:
            for key, value in dict_to_ammend_parameters.items():
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    f.write(f"{key:<{int_allign}}\n")  # leave value blank
                else:
                    f.write(f"{key:<{int_allign}}{value}\n")
        
        return(os.path.basename(str_parameter_to_ammend_filepath))
# ---------------


# --------------------
class LisfloodTimeout(Exception):
    """LISFLOOD was killed for passing the hard wall-clock timeout."""
    pass
# --------------------


# --------------------
def fn_run_docker_lisflood(str_lisflood_folder, str_par_file_to_run, flt_timeout_sec=None):
    
    '''
    str_docker_string = "lisflood-fp:augmented"

    cmd = [
        "docker", "run",
        "--rm",
        "-v", str_lisflood_folder + ":/data",
        "lisflood-fp:augmented",
        "lisflood", str_par_file_to_run,
    ]
    '''
    str_par_full_path = os.path.join(str_lisflood_folder, str_par_file_to_run)
    #print(str_par_full_path)

    cmd = [
        "lisflood", str_par_file_to_run,
    ]

    # flt_timeout_sec of None or 0 = no limit.  On a timeout subprocess.run
    # kills LISFLOOD; no depth grid (.wd) is written for that run.
    flt_start = time.time()
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=flt_timeout_sec if flt_timeout_sec else None,
        )
    except subprocess.TimeoutExpired:
        raise LisfloodTimeout(
            f"{str_par_file_to_run[:-4]} killed after {flt_timeout_sec / 60:.0f} minutes")
    
    stdout = result.stdout  # or paste the string here

    match = re.search(r"loop time\s+([0-9.]+)", stdout)
    if match:
        flt_loop_time = round(float(match.group(1)))
    else:
        # fall back to the measured wall-clock time
        flt_loop_time = round(time.time() - flt_start)

    return(flt_loop_time)
# --------------------


# ----------------------------
def fn_build_columns(row,
                     str_lisflood_folder,
                     str_stream_folder,
                     str_catchment):
    
    # add ['mass_balance_filepath', 'parameter_file', 'expected_outflow'] to df_parameter_files

    str_outfolder = f"{str_catchment}_{row['intensity_str']}mm"

    str_mass_balance_filepath = os.path.join(
        str_lisflood_folder,
        str_outfolder,
        f"{str_outfolder}.mass"
    )

    str_parameter_file = os.path.basename(row['filepath'])

    flt_expected_outflow = round(
        fn_get_expected_outflow(
            row['intensity'],
            str_stream_folder,
            str_catchment
        ),
        1
    )

    return pd.Series({
        'mass_balance_filepath': str_mass_balance_filepath,
        'parameter_file': str_parameter_file,
        'expected_outflow': flt_expected_outflow
    })
# ----------------------------

# -------------------------- 
def fn_prep_next_paramter_file(row,
                               next_row,
                               ps_first_stable_row,
                               str_lisflood_folder,
                               b_use_startfile,
                               int_buffer_sec=5400,
                               int_max_sim_time_sec=None):

    flt_time_to_stable = ps_first_stable_row['Time']
    # Round flt_time_to_stable to nearest integer
    int_time_to_stable = int(flt_time_to_stable // 1 )
    
    # buffer for stabilization (was a hard coded 5400 s / 1.5 hours)
    # Revised 2026.10.24 -- from [lisflood_timeouts] stable_buffer_hr
    int_time_to_stable += int_buffer_sec

    # cap the sim_time; without this an unstable run (which falls back to its
    # last row) makes every later run 1.5 hours longer than the one before
    if int_max_sim_time_sec:
        int_time_to_stable = min(int_time_to_stable, int(int_max_sim_time_sec))

    df_par = pd.read_csv(
        row['filepath'],
        sep=r"\s+",          # one or more whitespace characters
        header=None,
        comment="#"
    )

    # Convert df to dictionary
    dict_parameters = dict(zip(df_par[0], df_par[1]))

    # get sim_time and saveint from dictionary
    flt_sim_time = float(dict_parameters["sim_time"])
    flt_saveint = float(dict_parameters["saveint"])
    str_outfolder = dict_parameters["dirroot"]

    # determine the number of expected output files
    int_stable_outputstep = int((flt_time_to_stable // flt_saveint) + 1)
    int_stable_outputstep = 1 # override this for now -- TODO - 2025.01.25
    str_stable_outputstep = str(int_stable_outputstep).zfill(4)

    # get the filepath of the expected stable run
    str_output_depth_asc_filename = str_outfolder + "-" + str_stable_outputstep + ".wd"
    str_stable_depth_filepath = os.path.join(str_lisflood_folder, str_outfolder, str_output_depth_asc_filename)

    if next_row is not None:
        #print(f"Next intensity: {next_row['intensity']}")

        str_parameter_to_ammend_filepath = next_row['filepath']

        # get the next parameter file
        df_par_to_ammend = pd.read_csv(
            str_parameter_to_ammend_filepath,
            sep=r"\s+",          # one or more whitespace characters
            header=None,
            comment="#"
        )

        # Convert df to dictionary
        dict_to_ammend_parameters = dict(zip(df_par_to_ammend[0], df_par_to_ammend[1]))

        # revise the sim_time and saveint of the next parameter file
        dict_to_ammend_parameters['sim_time'] = str(int_time_to_stable)
        dict_to_ammend_parameters['saveint'] = f"{float(int_time_to_stable)}"

        # copy the last stable run depth output into the str_lisflood_folder
        shutil.copy2(str_stable_depth_filepath,str_lisflood_folder)

        # add the startfile of the last stable run, is b_use_startfile == True
        if b_use_startfile:
            dict_to_ammend_parameters['startfile'] = os.path.basename(str_stable_depth_filepath)

        # Overwrite the existing parameter file
        int_allign = 24 # character spacing

        with open(str_parameter_to_ammend_filepath, "w") as f:
            for key, value in dict_to_ammend_parameters.items():
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    f.write(f"{key:<{int_allign}}\n")  # leave value blank
                else:
                    f.write(f"{key:<{int_allign}}{value}\n")

        return(os.path.basename(str_parameter_to_ammend_filepath))
    else:
        #print("No higher intensity available")
        return None
# --------------------------

# --------------------------
def fn_get_par_sim_time(str_par_filepath):
    df_par = pd.read_csv(str_par_filepath, sep=r"\s+", header=None, comment="#")
    return float(dict(zip(df_par[0], df_par[1]))['sim_time'])
# --------------------------


# --------------------------
def fn_set_par_sim_time(str_par_filepath, int_sim_time):
    # set sim_time and saveint (output only at the end of the run)
    df_par = pd.read_csv(str_par_filepath, sep=r"\s+", header=None, comment="#")
    dict_parameters = dict(zip(df_par[0], df_par[1]))
    dict_parameters['sim_time'] = str(int(int_sim_time))
    dict_parameters['saveint'] = f"{float(int(int_sim_time))}"

    int_allign = 24 # character spacing
    with open(str_par_filepath, "w") as f:
        for key, value in dict_parameters.items():
            if value is None or (isinstance(value, float) and math.isnan(value)):
                f.write(f"{key:<{int_allign}}\n")
            else:
                f.write(f"{key:<{int_allign}}{value}\n")
# --------------------------


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



# ----------------------
def fn_create_modified_mannings(
    str_lisflood_folder,
    str_run_name_nth,
    str_mannings_asc,
    flt_fpfric,
    flt_depth_thresh=0.001):
    """
    After the Nth default-Manning run completes, read that run's depth
    output (.wd), identify wet cells (depth > flt_depth_thresh), and
    overwrite those cells in the Manning's n ASC with flt_fpfric.

    Saves the modified raster as 'manningsn_modified.asc' in
    str_lisflood_folder and returns the filename.

    Parameters
    ----------
    str_lisflood_folder : str   - path to 02_lisflood_input folder
    str_run_name_nth    : str   - dirroot value of the Nth run (e.g. 'cat-2420542_5p9mm')
    str_mannings_asc    : str   - filename of the original Manning's n ASC
    flt_fpfric          : float - uniform Manning's n value to assign to wet cells
    flt_depth_thresh    : float - minimum depth (m) to consider a cell wet
    """
    # --- Locate the depth output file from the Nth run ---
    str_wd_filename = str_run_name_nth + "-0001.wd"
    str_wd_path = os.path.join(str_lisflood_folder, str_run_name_nth, str_wd_filename)

    if not os.path.isfile(str_wd_path):
        raise FileNotFoundError(
            f"Depth output not found for Nth Manning run: {str_wd_path}")

    str_mannings_path = os.path.join(str_lisflood_folder, str_mannings_asc)
    str_modified_name = "manningsn_modified.asc"
    str_modified_path = os.path.join(str_lisflood_folder, str_modified_name)

    # --- Read depth raster ---
    with rasterio.open(str_wd_path) as src_wd:
        arr_depth  = src_wd.read(1)
        wd_nodata  = src_wd.nodata
        wd_profile = src_wd.profile

    # --- Read Manning's n raster ---
    with rasterio.open(str_mannings_path) as src_mn:
        arr_mannings  = src_mn.read(1).astype(np.float32)
        mn_nodata     = src_mn.nodata
        mn_profile    = src_mn.profile

    # --- Build wet mask ---
    if wd_nodata is not None:
        valid_depth = arr_depth != wd_nodata
    else:
        valid_depth = ~np.isnan(arr_depth)

    wet_mask = valid_depth & (arr_depth > flt_depth_thresh)

    # --- Overwrite wet cells with fpfric ---
    arr_modified = arr_mannings.copy()
    arr_modified[wet_mask] = flt_fpfric

    # --- Write modified ASC ---
    with rasterio.open(str_modified_path, 'w', **mn_profile) as dst:
        dst.write(arr_modified, 1)

    n_wet = int(wet_mask.sum())
    return str_modified_name, n_wet
# ----------------------


# .........................................................
def fn_run_lisflood_02(
    str_global_config_file_path,
    str_local_config_file_path,
    b_print_output
):

    #warnings.filterwarnings("ignore", category=UserWarning)

    # ---- Header output ----
    if b_print_output:
        print(f"""
+=================================================================+
|                        RUN LISFLOOD-FP                          |
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
        print(" -- Script 02: Run LISFLOOD-FP")

    # ==================================================================
    # READ GLOBAL CONFIG
    # ==================================================================
    global_config = configparser.ConfigParser()
    global_config.read(str_global_config_file_path)

    global_section_schema = {
        'lisflood_settings': [
            'fpfric',
            'default_manning_runs',
            'depththresh',
        ],
        'stable_run_paramters': [
            'window',
            'min_Qout_ratio',
            'max_Qout_ratio',
            'max_rolling_avg_stability',
            'num_unchecked_runs'
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

    # ---- run time limits (optional section; defaults keep the old behavior) ----
    # Added 2026.10.24
    flt_max_sim_time_hr   = global_config.getfloat('lisflood_timeouts', 'max_sim_time_hr', fallback=0.0)
    flt_min_sim_time_hr   = global_config.getfloat('lisflood_timeouts', 'min_sim_time_hr', fallback=0.0)
    flt_stable_buffer_hr  = global_config.getfloat('lisflood_timeouts', 'stable_buffer_hr', fallback=1.5)
    flt_max_run_wall_min  = global_config.getfloat('lisflood_timeouts', 'max_run_wall_min', fallback=0.0)
    flt_hard_timeout_min  = global_config.getfloat('lisflood_timeouts', 'hard_timeout_min', fallback=0.0)
    str_startfile_mode    = global_config.get('lisflood_timeouts', 'startfile_mode', fallback='retry').strip().lower()

    if str_startfile_mode not in ('retry', 'always', 'never'):
        raise ValueError(f"[lisflood_timeouts] startfile_mode must be retry, always or never; got '{str_startfile_mode}'")

    int_max_sim_time_sec = int(flt_max_sim_time_hr * 3600)   # 0 = no cap
    int_min_sim_time_sec = int(flt_min_sim_time_hr * 3600)
    int_buffer_sec       = int(flt_stable_buffer_hr * 3600)
    flt_hard_timeout_sec = flt_hard_timeout_min * 60          # 0 = no limit


    # *************
    # For testing and recomputing
    b_run_first_run = True
    int_process_row_index = 0
    # *************
    
    # ---------------
    # From local config
    str_catchment = dict_all_params['catchment']
    #str_out_root_folder = dict_all_params['out_root_folder']
    
    # Make the root folder absolute
    str_out_root_folder = os.path.abspath(dict_all_params['out_root_folder'])

    # Then build downstream folders
    str_stream_folder = os.path.abspath(os.path.join(str_out_root_folder, str_catchment, "01_stream_delineation"))
    str_lisflood_folder = os.path.abspath(os.path.join(str_out_root_folder, str_catchment, "02_lisflood_input"))

    str_original_cwd = os.getcwd()
    os.chdir(str_lisflood_folder)

    # create a list of all the parameter files
    list_par_files = fn_collect_par_values(str_lisflood_folder)
    
    # convert list of parameter files to a dataframe and sort
    df_parameter_files = pd.DataFrame(
        list_par_files,
        columns=["filepath", "intensity"]
    ).sort_values(by="intensity", ascending=True).reset_index(drop=True)
    
    # intetensity_str as col like '115p6'
    df_parameter_files['intensity_str'] = (
        df_parameter_files['filepath']
        .apply(lambda p: os.path.basename(p).split('_')[-1].replace('mm.par', ''))
    )
    
    str_num_runs = str(len(df_parameter_files))
    str_run_label = f"1 of {str_num_runs}: "
    
    if b_run_first_run:
        # --- First Run --- This is the lowest rainfall
        if b_print_output:
            print('  -- STEP 1: Running LISFLOOD')
        
        # Run the first model to the full time duration
        str_par_file_to_run = os.path.basename(df_parameter_files.iloc[0]['filepath'])

        # cap the first run's sim_time too
        flt_first_sim_time = fn_get_par_sim_time(str_par_file_to_run)
        if int_max_sim_time_sec and flt_first_sim_time > int_max_sim_time_sec:
            fn_set_par_sim_time(str_par_file_to_run, int_max_sim_time_sec)
            flt_first_sim_time = int_max_sim_time_sec

        try:
            flt_loop_time = fn_run_with_spinner(
                fn_run_docker_lisflood,
                str_lisflood_folder,
                str_par_file_to_run,
                flt_hard_timeout_sec,
                message=f"     -- {str_run_label} Running Initial LISFLOOD"
            )
        except LisfloodTimeout as e:
            print(f"     -- HARD TIMEOUT: {e}; no runs completed")
            os.chdir(str_original_cwd)
            return
        
        print(
                        f"     -- {str_run_label} Run {str_par_file_to_run [:-4]} "
                        f"completed in {round(flt_loop_time)} seconds")

        # model seconds per wall-clock second, used for the wall-clock budget
        flt_sim_speed = flt_first_sim_time / max(flt_loop_time, 1)
    else:
        flt_sim_speed = None
    
    
    
    # 'always' starts every run after the first from the previous run's depths
    b_use_startfile = (str_startfile_mode == 'always')

    # compute addtional cols in dataframe
    df_parameter_files[
        ['mass_balance_filepath', 'parameter_file', 'expected_outflow']
    ] = df_parameter_files.apply(
        fn_build_columns,
        axis=1,
        args=(
            str_lisflood_folder,
            str_stream_folder,
            str_catchment
        )
    )

    # Model runs in series, lowest intensity first.  For each run:
    #   1. 'ok'       -> use the first stable row to prep the next run
    #   2. not 'ok'   -> if a startfile has not been tried yet, re-run this
    #                    intensity with the previous run's depths as a startfile
    #                    and check again.  All later runs then use a startfile.
    #   3. still not  -> for the first 'num_unchecked_runs' runs, accept the
    #                    last row of the mass file (logged as a warning) and keep
    #                    going; otherwise stop.
    # Each run's sim_time is the stable time + stable_buffer_hr, capped at
    # max_sim_time_hr and at what fits in max_run_wall_min (from the speed of
    # the last run), but not below min_sim_time_hr.  A run that passes
    # hard_timeout_min is killed and the series stops there.

    # Manning's roughness transition settings
    int_default_manning_runs = int(dict_all_params['default_manning_runs'])
    flt_fpfric               = float(dict_all_params['fpfric'])
    flt_depththresh          = float(dict_all_params['depththresh'])

    str_unchecked = dict_all_params.get('num_unchecked_runs', '')
    int_num_unchecked_runs = int(str_unchecked) if str_unchecked != '' else 3

    str_mannings_asc_original = None
    str_mannings_modified     = None
    b_mannings_modified       = False

    row_prev = None
    ps_prev_stable_row = None
    list_fallback_runs = []
    list_capped_runs = []

    # -----
    def fn_sim_time_cap():
        # tightest of max_sim_time_hr and the wall-clock budget; None = no cap
        list_caps = []
        if int_max_sim_time_sec:
            list_caps.append(int_max_sim_time_sec)
        if flt_max_run_wall_min and flt_sim_speed:
            # 0.9 = margin, as runs slow down as more cells get wet
            list_caps.append(int(flt_sim_speed * flt_max_run_wall_min * 60 * 0.9))
        if not list_caps:
            return None
        return max(min(list_caps), int_min_sim_time_sec)
    # -----

    # -----
    def fn_run_capped(str_par_file, str_label, str_message, int_cap):
        # run one .par file; update the speed estimate; note capped runs
        nonlocal flt_sim_speed
        flt_sim_time = fn_get_par_sim_time(str_par_file)
        if int_cap is not None and flt_sim_time >= int_cap:
            list_capped_runs.append(str_par_file[:-4])
        flt_loop = fn_run_par_file(str_lisflood_folder, str_par_file, str_label,
                                   str_message, flt_hard_timeout_sec)
        flt_sim_speed = flt_sim_time / max(flt_loop, 1)
    # -----

    for index, row in df_parameter_files.iterrows():
        if index < int_process_row_index:
            continue

        str_run = int(index + 2)
        str_run_label = f"{str_run} of {str_num_runs}: "

        str_stable_row_status, ps_first_stable_row = fn_get_stable_row(
            row['mass_balance_filepath'],
            row['expected_outflow'],
            dict_all_params
        )

        # --- retry this intensity with a startfile from the previous run ---
        if (str_stable_row_status != 'ok'
                and str_startfile_mode == 'retry'
                and not b_use_startfile
                and row_prev is not None):

            print(f"     -- Stable Run not found: {row['parameter_file'][:-4]} "
                  f"Status: {str_stable_row_status} -- retrying with startfile")

            b_use_startfile = True

            int_cap = fn_sim_time_cap()
            str_par_file_to_run = fn_prep_next_paramter_file(row_prev,
                                                             row,
                                                             ps_prev_stable_row,
                                                             str_lisflood_folder,
                                                             b_use_startfile,
                                                             int_buffer_sec,
                                                             int_cap)

            try:
                fn_run_capped(str_par_file_to_run,
                              f"{int(index + 1)} of {str_num_runs}: ",
                              f"Re-running {row['parameter_file'][:-4]} (startfile)",
                              int_cap)
            except LisfloodTimeout as e:
                # the first attempt's depth grid is kept, but its .mass file
                # was overwritten by the killed retry, so stop here
                print(f"     -- HARD TIMEOUT: {e}; stopping the series here")
                break

            str_stable_row_status, ps_first_stable_row = fn_get_stable_row(
                row['mass_balance_filepath'],
                row['expected_outflow'],
                dict_all_params
            )

        # --- low-intensity fallback: accept the end of the run ---
        if (str_stable_row_status == 'no_match'
                and index < int_num_unchecked_runs):

            print(f"     -- WARNING: {row['parameter_file'][:-4]} did not meet the "
                  f"stability check; using final time {ps_first_stable_row['Time']:.0f} s")
            list_fallback_runs.append(row['parameter_file'][:-4])
            str_stable_row_status = 'ok'

        if str_stable_row_status != 'ok':
            print(f"     -- Stable Run not found: {row['parameter_file'][:-4]} Status: {str_stable_row_status}")
            break

        # -------------------------------------------------------
        # Manning's transition: after the Nth default-Manning run,
        # build the modified roughness file and patch remaining .par
        # -------------------------------------------------------
        if (not b_mannings_modified
                and index == int_default_manning_runs - 1):

            # Find the original manningfile name from a post-Nth .par
            next_check = fn_get_next_intensity_row(df_parameter_files, row['intensity'])
            if next_check is not None:
                df_check = pd.read_csv(
                    next_check['filepath'],
                    sep=r"\s+", header=None, comment="#"
                )
                dict_check = dict(zip(df_check[0], df_check[1]))
                str_mannings_asc_original = dict_check.get('manningfile', None)

            if str_mannings_asc_original is not None:
                # dirroot of the current (Nth) run is the run_name
                df_nth = pd.read_csv(
                    row['filepath'],
                    sep=r"\s+", header=None, comment="#"
                )
                str_nth_dirroot = dict(zip(df_nth[0], df_nth[1])).get('dirroot', '')

                if b_print_output:
                    print(f"       -- Manning\'s transition: modifying {str_mannings_asc_original} "
                          f"using wet cells from run '{str_nth_dirroot}'")

                str_mannings_modified, n_wet = fn_create_modified_mannings(
                    str_lisflood_folder  = str_lisflood_folder,
                    str_run_name_nth     = str_nth_dirroot,
                    str_mannings_asc     = str_mannings_asc_original,
                    flt_fpfric           = flt_fpfric,
                    flt_depth_thresh     = flt_depththresh
                )

                if b_print_output:
                    print(f"       -- {n_wet} wet cells set to fpfric={flt_fpfric}")
                    print(f"       -- Modified Manning\'s written: {str_mannings_modified}")

                # Patch all remaining .par files to use the modified ASC
                int_allign = 24
                for patch_idx, patch_row in df_parameter_files.iterrows():
                    if patch_idx > int_default_manning_runs - 1:
                        df_patch = pd.read_csv(
                            patch_row['filepath'],
                            sep=r"\s+", header=None, comment="#"
                        )
                        dict_patch = dict(zip(df_patch[0], df_patch[1]))
                        if 'manningfile' in dict_patch:
                            dict_patch['manningfile'] = str_mannings_modified
                            with open(patch_row['filepath'], 'w') as f:
                                for key, value in dict_patch.items():
                                    if value is None or (isinstance(value, float) and math.isnan(value)):
                                        f.write(f"{key:<{int_allign}}\n")
                                    else:
                                        f.write(f"{key:<{int_allign}}{value}\n")

                b_mannings_modified = True

        # next parameter file in sequence
        next_row = fn_get_next_intensity_row(df_parameter_files, row['intensity'])

        int_cap = fn_sim_time_cap()
        str_par_file_to_run = fn_prep_next_paramter_file(row,
                                                         next_row,
                                                         ps_first_stable_row,
                                                         str_lisflood_folder,
                                                         b_use_startfile,
                                                         int_buffer_sec,
                                                         int_cap)

        if str_par_file_to_run is not None:
            try:
                fn_run_capped(str_par_file_to_run, str_run_label,
                              f"Running {next_row['parameter_file'][:-4]}",
                              int_cap)
            except LisfloodTimeout as e:
                print(f"     -- HARD TIMEOUT: {e}; stopping the series here")
                break

        row_prev = row
        ps_prev_stable_row = ps_first_stable_row

    if list_fallback_runs:
        print(f"     -- Runs accepted without passing the stability check: {', '.join(list_fallback_runs)}")

    if list_capped_runs:
        print(f"     -- Runs shortened by [lisflood_timeouts] limits: {', '.join(list_capped_runs)}")

    os.chdir(str_original_cwd)
# .........................................................


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == '__main__':

    flt_start_run = time.time()
    
    parser = argparse.ArgumentParser(description='========= RUN LISFLOOD-FP =========')
    
    parser.add_argument('-g',
                        dest = "str_global_config_file_path",
                        help=r'REQUIRED: Global configuration filepath Example:C:\Users\civil\dev\lisflood2fim\config\global_config.ini',
                        required=False,
                        default='/app/lisflood2fim/config/demo_global_config.ini',
                        metavar='FILE',
                        type=lambda x: is_valid_file(parser, x))
    
    parser.add_argument('-c',
                        dest = "str_local_config_file_path",
                        help=r'REQUIRED: Global configuration filepath Example:C:\Users\civil\dev\lisflood2fim\config\local_config.ini',
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

    fn_run_lisflood_02(str_global_config_file_path, str_local_config_file_path, b_print_output)

    flt_end_run = time.time()
    flt_time_pass = (flt_end_run - flt_start_run) // 1
    time_pass = datetime.timedelta(seconds=flt_time_pass)
    
    print('Compute Time: ' + str(time_pass))
