# ************************************************************
# LISFLOOD-FP to FIM
# Script - lisflood2fim_00 (main script)
#
# Created by: Andy Carter, PE
# Created - 2026.02.10
# ************************************************************


# ************************************************************
import os

import argparse
import time
import datetime
import warnings

# Import modules
from prepare_input_layers_01 import fn_prepare_input_layers_01
from run_lisflood_02 import fn_run_lisflood_02
from build_netcdf_03 import fn_build_netcdf_03
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


# ----------------
def fn_lisflood2fim_00(str_global_config_file_path,
                       str_local_config_file_path,
                       b_print_output):
    
    # ---- Header output ----
    if b_print_output:
        print(f"""
    +=================================================================+
    |                       RUNNING LISFLOOD2FIM                      |
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
        print("Script 00: Running LISFLOOD2FIM")
    
    fn_prepare_input_layers_01(str_global_config_file_path,
                               str_local_config_file_path,
                               b_print_output)
    
    fn_run_lisflood_02(str_global_config_file_path,
                       str_local_config_file_path,
                       b_print_output)

    fn_build_netcdf_03(str_global_config_file_path,
                       str_local_config_file_path,
                       b_print_output)
    
# ----------------


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == '__main__':

    flt_start_run = time.time()
    
    parser = argparse.ArgumentParser(description='========= BUILD FLOOD INUNDATION NETCDF =========')
    
    parser.add_argument('-g',
                        dest = "str_global_config_file_path",
                        help=r'REQUIRED: Global configuration filepath Example:/app/lisflood2fim/config/demo_global_config.ini',
                        required=True,
                        default='/app/lisflood2fim/config/demo_global_config.ini',
                        metavar='FILE',
                        type=lambda x: is_valid_file(parser, x))
    
    parser.add_argument('-c',
                        dest = "str_local_config_file_path",
                        help=r'REQUIRED: LOCAL configuration filepath Example:/app/lisflood2fim/config/demo_local_config.ini',
                        required=True,
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

    fn_build_netcdf_03(str_global_config_file_path, str_local_config_file_path, b_print_output)

    flt_end_run = time.time()
    flt_time_pass = (flt_end_run - flt_start_run) // 1
    time_pass = datetime.timedelta(seconds=flt_time_pass)
    
    print('Compute Time: ' + str(time_pass))
 #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~