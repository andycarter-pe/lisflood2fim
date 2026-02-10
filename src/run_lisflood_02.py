# ************************************************************
# LISFLOOD-FP to FIM
# Script 02 - run_lisflood_02
#
# Created by: Andy Carter, PE
# Created - 2026.02.07
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
# ************************************************************

# -----------------
def fn_run_with_spinner(func, *args, message="Running"):
    spinner = itertools.cycle("|/-\\")
    result = {}

    def wrapper():
        result["value"] = func(*args)

    thread = threading.Thread(target=wrapper)
    thread.start()

    while thread.is_alive():
        sys.stdout.write(f"\r{message}... {next(spinner)}")
        sys.stdout.flush()
        time.sleep(0.1)

    thread.join()
    sys.stdout.write("\r" + " " * 50 + "\r")  # clear line
    return result.get("value")
# -----------------

# -----------------
def fn_get_expected_outflow (flt_intensity, str_stream_folder, str_catchment):
    str_gpkg_path = os.path.join(str_stream_folder, str_catchment + ".gpkg")
    str_terrain_dem_path = os.path.join(str_stream_folder, "dem_clipped_5070.tif")
    
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
    Returns:
        (status, result)

        status:
            'ok'            -> stable row found
            'not_found'     -> mass file missing
            'invalid_file'  -> file unreadable or empty
            'missing_qout'  -> Qout column missing
            'no_match'      -> file read but no stable row found

        result:
            pandas.Series (if status == 'ok')
            None otherwise
    """
    
    window = int(dict_all_params['window'])
    flt_min_Qout_ratio = float(dict_all_params['min_Qout_ratio'])
    flt_max_Qout_ratio = float(dict_all_params['max_Qout_ratio'])
    flt_max_rolling_avg_stability = float(dict_all_params['max_rolling_avg_stability'])

    # --- file existence ---
    if not os.path.isfile(str_mass_balance_filepath):
        return 'not_found', None

    # --- invalid expected outflow ---
    if flt_expected_outflow is None or flt_expected_outflow == 0:
        return 'invalid_file', None

    # --- read file safely ---
    try:
        df_mass = pd.read_csv(
            str_mass_balance_filepath,
            sep=r"\s+",
            header=0
        )
    except (FileNotFoundError, pd.errors.EmptyDataError) as e:
        return 'invalid_file', None


    if df_mass.empty:
        return 'invalid_file', None

    # --- required column ---
    if "Qout" not in df_mass.columns:
        return 'missing_qout', None

    # --- calculations ---
    df_mass["Qout_rel_change"] = (
        df_mass["Qout"].diff() / df_mass["Qout"].shift(1)
    )

    df_mass["Qout_rel_change_rollavg"] = (
        df_mass["Qout_rel_change"]
        .abs()
        .rolling(window=window, min_periods=1)
        .mean()
    )

    df_mass["Qout_ratio"] = df_mass["Qout"] / flt_expected_outflow

    # --- stability mask ---
    mask = (
        (df_mass["Qout_rel_change_rollavg"] < flt_max_rolling_avg_stability) &
        (df_mass["Qout_ratio"] > flt_min_Qout_ratio) &
        (df_mass["Qout_ratio"] < flt_max_Qout_ratio)
    )

    matches = df_mass.loc[mask]

    if not matches.empty:
        return 'ok', matches.iloc[0]

    return 'no_match', None
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
        print("No higher intensity available")
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
def fn_run_docker_lisflood(str_lisflood_folder, str_par_file_to_run):
    
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

    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )
    
    stdout = result.stdout  # or paste the string here

    match = re.search(r"loop time\s+([0-9.]+)", stdout)
    if match:
        flt_loop_time = round(float(match.group(1)))
    else:
        flt_loop_time = None

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
                               b_use_startfile):

    flt_time_to_stable = ps_first_stable_row['Time']
    # Round flt_time_to_stable to nearest integer
    int_time_to_stable = int(flt_time_to_stable // 1 )
    
    # addiing an additional hour
    # TODO -- 2026.01.31 -- a buffer for stabilization
    int_time_to_stable += 1800

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
        print("No higher intensity available")
        return None
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
        print("Script 02: Run LISFLOOD-FP")

    # ==================================================================
    # READ GLOBAL CONFIG
    # ==================================================================
    global_config = configparser.ConfigParser()
    global_config.read(str_global_config_file_path)

    global_section_schema = {
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


    # *************
    # For testing and recomputing
    b_run_first_run = True
    int_process_row_index = 0
    # *************
    
    #print(dict_all_params)
    
    # ---------------
    # From local config
    str_catchment = dict_all_params['catchment']
    #str_out_root_folder = dict_all_params['out_root_folder']
    
    # Make the root folder absolute
    str_out_root_folder = os.path.abspath(dict_all_params['out_root_folder'])

    # Then build downstream folders
    str_stream_folder = os.path.abspath(os.path.join(str_out_root_folder, str_catchment, "01_stream_delineation"))
    str_lisflood_folder = os.path.abspath(os.path.join(str_out_root_folder, str_catchment, "02_lisflood_input"))

    #print(str_stream_folder)
    #print(str_lisflood_folder)
    str_original_cwd = os.getcwd()
    os.chdir(str_lisflood_folder)
    #print(os.getcwd())
    #print('--------------')

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
        print('  -- STEP 1: Running LISFLOOD')
        
        # Run the first model to the full time duration
        str_par_file_to_run = os.path.basename(df_parameter_files.iloc[0]['filepath'])

        flt_loop_time = fn_run_with_spinner(
            fn_run_docker_lisflood,
            str_lisflood_folder,
            str_par_file_to_run,
            message=f"     -- {str_run_label} Running Initial LISFLOOD"
        )
        
        #print(f"     -- Initial run completed in {flt_loop_time:.2f} seconds")
        
        print(
                        f"     -- {str_run_label} Run {str_par_file_to_run [:-4]} "
                        f"completed in {round(flt_loop_time)} seconds")
    
    
    
    b_use_startfile = False

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

    # Model will run in series until str_stable_row_status is not 'ok'
    # This will likely be a return of 'no_match' and will require the addition of a startfile
    # Other than 'ok' or 'not_found' for str_stable_row_status is not good
    
    for index, row in df_parameter_files.iterrows():
        if index >= int_process_row_index:
    
            str_run = int(index + 2)
            str_run_label = f"{str_run} of {str_num_runs}: "
            
            #print(row['parameter_file'][:-4])
    
            str_stable_row_status, ps_first_stable_row = fn_get_stable_row(
                row['mass_balance_filepath'],
                row['expected_outflow'],
                dict_all_params
            )
    
            #print(str_stable_row_status)
    
            if str_stable_row_status != 'ok':
                # something went wrong with this steps run
                #print(str_stable_row_status)
                print(f"     -- Stable Run not found: {row['parameter_file'][:-4]} Status: {str_stable_row_status}")
                break
            else:
                # the current run was stable... prepare the next run without the introduction of a startfile
                # revise the next row's parameter file
    
                # next parameter file in sequence
                next_row = fn_get_next_intensity_row(df_parameter_files, row['intensity'])
    
                ##need row, next_row and ps_first_stable_row
                str_par_file_to_run = fn_prep_next_paramter_file(row,
                                                                 next_row,
                                                                 ps_first_stable_row,
                                                                 str_lisflood_folder,
                                                                 b_use_startfile)
    
                if str_par_file_to_run is not None:
                    # run the Docker container of lisflood-fp

                    flt_loop_time = fn_run_with_spinner(
                        fn_run_docker_lisflood,
                        str_lisflood_folder,
                        str_par_file_to_run,
                        message=f"     -- {str_run_label} Running {next_row['parameter_file'][:-4]} LISFLOOD"
                    )
                    
                    print(
                        f"     -- {str_run_label} Run {next_row['parameter_file'][:-4]} "
                        f"completed in {round(flt_loop_time)} seconds"
                    )

    # Stable run wasn't found... if it is 'no_match' try to run with using the last
    # runs stable output at a startfile -- making it wet before running

    b_use_startfile = True
    int_process_row_index = index
    
    for index, row in df_parameter_files.iterrows():
        if index >= int_process_row_index:
    
            str_run = int(index + 2)
            str_run_label = f"{str_run} of {str_num_runs}: "
            
            #print(row['parameter_file'][:-4])
    
            str_stable_row_status, ps_first_stable_row = fn_get_stable_row(
                row['mass_balance_filepath'],
                row['expected_outflow'],
                dict_all_params
            )
    
            #print(str_stable_row_status)
    
            if str_stable_row_status != 'ok':
                # something went wrong with this steps run
                print(f"     -- Stable Run not found: {row['parameter_file'][:-4]} Status: {str_stable_row_status}")
                break
            else:
                # the current run was stable... prepare the next run with the introduction of a startfile
                # revise the next row's parameter file
    
                # next parameter file in sequence
                next_row = fn_get_next_intensity_row(df_parameter_files, row['intensity'])
    
                ##need row, next_row and ps_first_stable_row
                str_par_file_to_run = fn_prep_next_paramter_file(row,
                                                                 next_row,
                                                                 ps_first_stable_row,
                                                                 str_lisflood_folder,
                                                                 b_use_startfile)
    
                if str_par_file_to_run is not None:
                    # run the Docker container of lisflood-fp

                    flt_loop_time = fn_run_with_spinner(
                        fn_run_docker_lisflood,
                        str_lisflood_folder,
                        str_par_file_to_run,
                        message=f"     -- {str_run_label} Running {next_row['parameter_file'][:-4]} LISFLOOD"
                    )
                    
                    print(
                        f"     -- {str_run_label} Run {next_row['parameter_file'][:-4]} "
                        f"completed in {round(flt_loop_time)} seconds"
                    )

    os.chdir(str_original_cwd)
# .........................................................


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == '__main__':

    flt_start_run = time.time()
    
    parser = argparse.ArgumentParser(description='========= RUN LISFLOOD-FP =========')
    
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

    fn_run_lisflood_02(str_global_config_file_path, str_local_config_file_path, b_print_output)

    flt_end_run = time.time()
    flt_time_pass = (flt_end_run - flt_start_run) // 1
    time_pass = datetime.timedelta(seconds=flt_time_pass)
    
    print('Compute Time: ' + str(time_pass))
 #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
