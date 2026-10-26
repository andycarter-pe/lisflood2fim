# ************************************************************
# LISFLOOD-FP to FIM
# Shared WhiteboxTools runner with retry
#
# WhiteboxTools (Rust) intermittently panics (exit status 101) on
# otherwise valid inputs. Every Whitebox call in lisflood2fim should go
# through fn_run_whitebox so a transient failure is retried with partial
# outputs removed between attempts.
#
# Created by: Andy Carter, PE
# Created - 2026.10.23
# ************************************************************

import subprocess
import time
from pathlib import Path

import rasterio

# Defaults; replaced from the [whitebox] section of the global config
# via fn_set_whitebox_retry
INT_WBT_MAX_ATTEMPTS = 5
FLT_WBT_RETRY_WAIT_SEC = 5.0
INT_STDERR_TAIL_CHARS = 4000

# Sidecar files written with an ESRI Shapefile output
LIST_SHP_SIDECARS = ['.shp', '.shx', '.dbf', '.prj', '.cpg', '.sbn', '.sbx']


# ...............................
def fn_set_whitebox_retry(int_max_attempts, flt_wait_sec):
    # Set the retry defaults used by every fn_run_whitebox call
    global INT_WBT_MAX_ATTEMPTS, FLT_WBT_RETRY_WAIT_SEC

    if int_max_attempts < 1:
        raise ValueError(f'[whitebox] max_attempts must be 1 or more, got {int_max_attempts}')
    if flt_wait_sec < 0:
        raise ValueError(f'[whitebox] retry_wait_sec must be 0 or more, got {flt_wait_sec}')

    INT_WBT_MAX_ATTEMPTS = int_max_attempts
    FLT_WBT_RETRY_WAIT_SEC = flt_wait_sec
# ...............................


# ...............................
def fn_list_output_files(str_output_path):
    # Return every file that belongs to an output (all sidecars for a .shp)
    path_output = Path(str_output_path)
    if path_output.suffix.lower() == '.shp':
        return [path_output.with_suffix(ext) for ext in LIST_SHP_SIDECARS]
    return [path_output]
# ...............................


# ...............................
def fn_delete_outputs(list_output_paths):
    # Remove any (partial) output left by a previous attempt
    for str_output_path in list_output_paths:
        for path_file in fn_list_output_files(str_output_path):
            if path_file.exists():
                path_file.unlink()
# ...............................


# ...............................
def fn_check_outputs(list_output_paths):
    # Return None if every output is present and readable, else a reason string
    for str_output_path in list_output_paths:
        path_output = Path(str_output_path)

        if not path_output.exists() or path_output.stat().st_size == 0:
            return f'missing or empty output: {path_output}'

        if path_output.suffix.lower() in ('.tif', '.tiff'):
            try:
                with rasterio.open(path_output) as src:
                    _ = src.profile
            except Exception as e:
                return f'unreadable raster output: {path_output} ({e})'
    return None
# ...............................


# ...............................
def fn_tail(str_text, int_chars=INT_STDERR_TAIL_CHARS):
    str_text = (str_text or '').strip()
    if len(str_text) > int_chars:
        return '...' + str_text[-int_chars:]
    return str_text
# ...............................


# ...............................
def fn_run_whitebox(list_cmd,
                    list_output_paths,
                    int_max_attempts=None,
                    flt_wait_sec=None):
    """
    Run a WhiteboxTools command line, retrying on failure.

    An attempt fails if the process exits non-zero or any expected output
    is missing, empty or (for GeoTIFFs) unreadable. Outputs are deleted
    before every attempt so a retry never sees a partial file. The wait
    between attempts grows linearly (wait, 2*wait, 3*wait ...).

    Raises RuntimeError with the captured stdout/stderr of the last
    attempt if every attempt fails.
    """

    if int_max_attempts is None:
        int_max_attempts = INT_WBT_MAX_ATTEMPTS
    if flt_wait_sec is None:
        flt_wait_sec = FLT_WBT_RETRY_WAIT_SEC

    # list_cmd[1] is "--run=ToolName"
    str_tool = list_cmd[1].split('=', 1)[-1] if len(list_cmd) > 1 else list_cmd[0]

    str_reason = ''
    str_stdout = ''
    str_stderr = ''

    for int_attempt in range(1, int_max_attempts + 1):
        fn_delete_outputs(list_output_paths)

        try:
            result = subprocess.run(list_cmd, capture_output=True, text=True)
            str_stdout = result.stdout
            str_stderr = result.stderr

            if result.returncode != 0:
                str_reason = f'exit status {result.returncode}'
            else:
                str_reason = fn_check_outputs(list_output_paths)
        except OSError as e:
            str_stdout = ''
            str_stderr = str(e)
            str_reason = f'could not start process ({e})'

        if not str_reason:
            if int_attempt > 1:
                print(f'     -- WhiteboxTools {str_tool} succeeded on attempt {int_attempt} of {int_max_attempts}')
            return

        print(f'     -- WARNING: WhiteboxTools {str_tool} failed on attempt '
              f'{int_attempt} of {int_max_attempts}: {str_reason}')

        if int_attempt < int_max_attempts:
            time.sleep(flt_wait_sec * int_attempt)

    fn_delete_outputs(list_output_paths)

    str_msg = (f'WhiteboxTools {str_tool} failed after {int_max_attempts} attempts '
               f'(last failure: {str_reason})\n'
               f'  command: {" ".join(list_cmd)}\n'
               f'  stderr:\n{fn_tail(str_stderr)}\n'
               f'  stdout:\n{fn_tail(str_stdout)}')
    raise RuntimeError(str_msg)
# ...............................
