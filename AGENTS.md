<!-- Contributed by CLB Engineering for Andy Carter, PE, 2026.10.27. Andy owns these rules; edit freely. -->
# Agent rules for lisflood2fim
Author and maintainer: Andy Carter, PE (Center for Water and the Environment, University of Texas at Austin).
These conventions are deliberate. Match the surrounding code; do not "modernize" it.
If you cannot tell whether something is a convention or a mistake, ask.

## Pipeline
- `src/lisflood2fim_00.py` runs four steps in order, each through one imported entry function:
  `prepare_input_layers_01_v6` -> `run_lisflood_02_v3` -> `build_netcdf_03` -> `heal_netcdf_04`.
  `src/lisflood2fim_huc12_00.py` is the HUC-12 twin; it imports `prepare_input_layers_01_v3_huc12` and
  `run_lisflood_02_v2` instead. There is no step-range argument.
- Every step has the same entry signature, `fn_<stem>_NN(str_global_config_file_path, str_local_config_file_path,
  b_print_output)`, e.g. `fn_prepare_input_layers_01`, and also runs on its own through a `__main__` argparse block.
- A new version of a step is a new file beside the old ones (`prepare_input_layers_01_v7.py`) that keeps the same
  entry function name, so switching versions is one import line in the orchestrator. Keep every old version.
  Do not edit, archive or delete one unless asked.
- Steps hand off through folders under `<out_root_folder>/<catchment>/`: `01_stream_delineation`,
  `02_lisflood_input`, `03_depth_nc`, `04_healed_depth_nc`. Keep names, layout and intermediate file names
  (e.g. `dem_burn_roads_d4fixed_5070.asc`, `depth_<catchment>.nc`).
- The `huc_12` branch is a separate line of work (`prepare_input_layers_huc12_01.py`, `whitebox_utils.py`,
  `global_s3_huc12_config.ini`, image `civileng127/lisflood2fim_huc12`). Do not carry changes between `main` and
  `huc_12` unless asked.

## Code style
- File header, framed by `# ****` lines (60 `*`), as in `run_lisflood_02_v3.py`:
  ```
  # ************************************************************
  # LISFLOOD-FP to FIM
  # Script 02 - run_lisflood_02
  #
  # Created by: Andy Carter, PE
  # Created - 2026.02.07
  # Revised - 2026.05.08 -- adjusting Mannings n after the 'default_manning_runs' run
  # ************************************************************
  ```
  Imports follow, between their own pair of `# ****` lines. A `_vN` file keeps the original header and adds a
  Revised line. For a meaningful change (not whitespace) append `# Revised - YYYY.MM.DD -- <what changed>`.
  Never remove or edit existing Created/Revised/Modified lines or the `Created by` credit.
- Open and close every function with the same comment rule: `# ----------------` above `def` and again after
  the last line. Copy a rule style already in the file (`# ----`, `# ....`); `is_valid_file` uses `# ^^^^`, and
  the `__main__` block is wrapped in `#~~~~` lines.
- Prefix new names by type: `str_`, `int_`, `flt_`, `b_`, `list_`, `dict_`, `gdf_`, `df_`, `da_`, `arr_`.
  Functions are `fn_snake_case`. Do not rename existing unprefixed names.
- Plain procedural functions with `#` comments. No classes, new type hints, dataclasses, `logging` or shared
  utility modules. The helpers `is_valid_file(parser, arg)` and `fn_str_to_bool(value)` are copied into each
  script; keep copying them. Leave Andy's commented-out code in place.
- Output is `print`. With `b_print_output` True, print the banner as one f-string block:
  ```
  +=================================================================+
  |                 BUILD FLOOD INUNDATION NETCDF                   |
  |                Created by Andy Carter, PE of                    |
  |             Center for Water and the Environment                |
  |                 University of Texas at Austin                   |
  +-----------------------------------------------------------------+
    ---(g) INPUT GLOBAL CONFIGURATION FILE: {str_global_config_file_path}
    ---(c) LOCAL CONFIGURATION FILE:  {str_local_config_file_path}
    ---[r] PRINT OUTPUT: {b_print_output}
  ===================================================================
  ```
  Otherwise print `" -- Script 03: Build Flood Inundation NetCDF"`. Progress inside a step is
  `print('  -- STEP 1: Finding Depth Rasters')`, guarded by `if b_print_output:`. The step scripts use no tqdm.
  `__main__` starts with `flt_start_run = time.time()` and ends with `print('Compute Time: ' + str(time_pass))`.
- argparse: short flags `-g` (global config), `-c` (local config), `-r` (print output), one argument per call:
  `dest = "str_global_config_file_path"`, `help=r'REQUIRED: Global configuration filepath Example:C:\...\global_config.ini'`,
  `metavar='FILE'`, `type=lambda x: is_valid_file(parser, x)`; `-r` uses `help=r'OPTIONAL: Print output messages Default: True'`,
  `default=True`, `metavar='T/F'`, `type=fn_str_to_bool`.

## Configs (`config/`)
- Global variants: `global_config.ini` (Windows `E:\` paths), `global_tacc_config.ini`, `global_web_config.ini`,
  `global_web_huc12_config.ini`, `global_web_mac_test_config.ini`, `demo_global_config.ini` (Docker sample data;
  the orchestrator default). Local variants (`[run_parameters]`: `catchment`, `out_root_folder`): `local_config.ini`,
  `demo_local_config.ini`, `demo_local_mac_test_config.ini`.
- Global sections: `[datasource]`, `[lisflood_settings]`, `[flow_parameters]`, `[stable_run_paramters]`,
  `[clean_fim_stack]`. Variants differ: `[clean_fim_stack]` is only in demo, tacc and web. Add a key to every
  global variant that has the section; list the ones that do not and ask before adding a section.
- Keys are plain lowercase names, often LISFLOOD-FP parameter names (`fpfric`, `depththresh`); no type prefix.
  Put a `#` comment above each key with what it is and its units. Sections are separated by `# -----------------------`.
  To change a value, comment out the old line and add the new one; never delete old values.
- Scripts read keys through a `global_section_schema` / `local_section_schema` dict with `section.get(key, '')`
  and convert where used (`int(dict_all_params['downscale'])`). A missing key comes back as `''`, so an optional
  key must treat `''` as "use today's behaviour".
- Line endings are mixed (e.g. `global_config.ini` is CRLF, `global_web_huc12_config.ini` is LF; some `.py` files
  are CRLF). Keep each file's own line endings, comments and blank lines.

## Do not
- Rename keys, sections, functions, files, folders or NetCDF attributes, even misspelled ones. They are interfaces.
  Known examples: `[stable_run_paramters]`, and `build_netcdf_03.py` whose header says "Script 02".
- Fix a suspected bug you were not asked about (report it), or reformat, reorder imports or tidy code you were not
  asked to change.
- Add dependencies, or change the Dockerfile, its `geo` env or the Python version (3.11), without asking.
- Commit or push. Stage the change and propose a short lowercase message ("terrain name change"); Andy commits.

## Checking work
- There is no test suite. `python -m py_compile` each edited file, then run the demo
  (`civileng127/lisflood2fim:20260611`, `python /app/lisflood2fim/src/lisflood2fim_00.py` with the demo configs and
  `sample_data_in/`) and look at the numbered folders and the NetCDF.
- `out_root_folder` is relative to the working directory (`data/output` in the demo), so start runs from a folder
  outside the repo. Report exactly what you ran, what you saw, and what you could not run.

## Tools in this repo
- Project skills, if any, live in `.claude/skills/<name>/SKILL.md` (type `/` to list them). They follow this file.
