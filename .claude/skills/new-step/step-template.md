<!-- Contributed by CLB Engineering for Andy Carter, PE, 2026.10.27. Distilled from
build_netcdf_03.py, heal_netcdf_04.py and lisflood2fim_00.py. -->

# lisflood2fim step template

Fill the `<...>` placeholders. Keep the spacing, rule lines and blank lines exactly as shown;
they copy Andy's files. Delete the `[GLOBAL]` lines when the step needs no global settings.

| Placeholder | Example |
|---|---|
| `<NN>` | `05` |
| `<stem>` | `check_netcdf_05` (the file name without `.py`) |
| `<Purpose>` | `check the healed netCDF global attributes` |
| `<TITLE>` | `CHECK FLOOD INUNDATION NETCDF`, centered in the 65-character banner field |
| `<Title>` | `Check Flood Inundation NetCDF` (the one-line `-- Script NN:` message) |
| `<entry>` | `fn_check_netcdf_05` (the stem: `fn_` + name + `_NN`) |
| `<section>` | a topic section such as `clean_fim_stack`, or a new one |
| `<NN_folder>` | `05_checked_nc` |
| `<YYYY.MM.DD>` | today, e.g. `2026.10.27` |

Banner line: `"|" + "<TITLE>".center(65) + "|"` gives the exact width of the `+====+` rule.
Argument lines: `(x)` for required inputs, `[x]` for optional ones (`(g)` global config,
`(c)` local config, `[r]` print output).

```python
# ************************************************************
# LISFLOOD-FP to FIM
# Script <NN> - <stem>
#
# <Purpose: what it reads (which NN_ folder) and what it writes (which NN_ folder).>
#
# Created by: Andy Carter, PE
# Created - <YYYY.MM.DD>
# ************************************************************

# ************************************************************
import os
import numpy as np

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
def fn_<helper>(str_input_filepath, str_outfolder):
    # one job: plain procedural code, prefixed names (str_, int_, flt_, b_, list_, dict_,
    # tup_, gdf_, df_, ds_, da_, arr_)
    pass
# --------------------------------------------------


# .........................................................
def <entry>(
    str_global_config_file_path,
    str_local_config_file_path,
    b_print_output
):

    # ---- Header output ----
    if b_print_output:
        print(f"""
+=================================================================+
|                          <TITLE>                                |
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
        print(" -- Script <NN>: <Title>")

    # ==================================================================
    # READ GLOBAL CONFIG
    # ==================================================================
    global_config = configparser.ConfigParser()
    global_config.read(str_global_config_file_path)

    global_section_schema = {
        '<section>': ['<key_1>', '<key_2>']      # [GLOBAL]
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
    
    # From local config
    str_out_root_folder = dict_all_params['out_root_folder']
    str_catchment = dict_all_params['catchment']
    
    # From global config (strings: cast here)
    flt_<key_1> = float(dict_all_params['<key_1>'])      # [GLOBAL]
    
    # -------------------
    # Input and output folders
    str_input_folder = os.path.join(str_out_root_folder, str_catchment, "<previous NN_ folder>")
    str_outfolder = os.path.join(str_out_root_folder, str_catchment, "<NN_folder>")
    os.makedirs(str_outfolder, exist_ok=True)
    
    if b_print_output:
        print('  -- STEP 1: <first thing>')
    
    # ... work ...
    
# .........................................................


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == '__main__':

    flt_start_run = time.time()
    
    parser = argparse.ArgumentParser(description='========= <TITLE> =========')
    
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

    <entry>(str_global_config_file_path, str_local_config_file_path, b_print_output)

    flt_end_run = time.time()
    flt_time_pass = (flt_end_run - flt_start_run) // 1
    time_pass = datetime.timedelta(seconds=flt_time_pass)
    
    print('Compute Time: ' + str(time_pass))
 #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```

Notes taken from his files:
- Comment rules vary by function (`# ----`, `# ......`, `# ^^^^`); the same rule opens and closes
  each function, two blank lines between functions. `# ......` frames the entry function.
- No tqdm and no multiprocessing in this repo, and the Docker `geo` env does not install tqdm.
  Progress is `'  -- STEP n: ...'` prints, and `fn_run_with_spinner` in `run_lisflood_02_v3.py`
  for long runs. Ask before adding tqdm (a new dependency).
- NetCDF outputs carry numbered attributes (`ds.attrs['00_stream_id']`, `'03_author'` ...) as in
  `build_netcdf_03.py`; continue that numbering when a step adds attributes.
- `is_valid_file` and `fn_str_to_bool` are copied into each script on purpose. Don't import them
  from another module.
