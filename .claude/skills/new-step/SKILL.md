---
name: new-step
description: Create a new numbered lisflood2fim pipeline step (src/<verb>_<noun>_NN.py with entry function fn_<verb>_<noun>_NN) or a new _vN version of an existing step, in Andy Carter's house style, then (after asking) wire it into src/lisflood2fim_00.py and/or src/lisflood2fim_huc12_00.py and add its settings to every global config variant. Use when asked to add, create, copy forward, or version a step script.
argument-hint: "<verb_noun> <NN> [purpose]  |  v<N> <existing_step.py> [what changes]"
metadata:
  contributed-by: "CLB Engineering for Andy Carter, PE, 2026.10.27"
---

# /new-step for lisflood2fim

Follow AGENTS.md if present. These rules apply even without it:
- Match the existing step files; don't "modernize". No classes, logging module, dataclasses or
  new dependencies unless Andy asks.
- Never renumber, rename or edit an existing step, key, section or output folder (even
  misspelled ones like `[stable_run_paramters]`); they are interfaces.
- Leave every line you didn't change byte-for-byte as it was, including trailing spaces on blank
  lines (many of his blank lines carry indentation). In new code, indent blank lines inside a
  block the way the surrounding file does.
- **Ask before editing either orchestrator or any config file.** Show the exact lines first.
- Keep each file's line endings: `lisflood2fim_00.py`, `build_netcdf_03.py` and several configs
  are CRLF; `lisflood2fim_huc12_00.py` and others are LF.
- Don't commit. Show `git status` and propose a short lowercase commit message; Andy commits.

Two modes. Arguments: `$ARGUMENTS`. If they don't make the mode, name, number and purpose clear,
ask once, in one message, before writing anything.

## The pipeline today (read the real files on the current branch; this is `main`)

| NN | Current file (older versions kept beside it) | Entry function | Writes to `<out_root_folder>/<catchment>/` |
|---|---|---|---|
| 01 | `prepare_input_layers_01_v6.py` (`_01`, `_v2`, `_v4`, `_v5`; `_v3_huc12` for HUC-12) | `fn_prepare_input_layers_01` | `01_stream_delineation`, `02_lisflood_input` |
| 02 | `run_lisflood_02_v3.py` (`_02`, `_v2` used by HUC-12) | `fn_run_lisflood_02` | `02_lisflood_input` |
| 03 | `build_netcdf_03.py` | `fn_build_netcdf_03` | `03_depth_nc` |
| 04 | `heal_netcdf_04.py` | `fn_heal_netcdf_04` | `04_healed_depth_nc` |

Every entry function is `fn_<name>_NN(str_global_config_file_path, str_local_config_file_path,
b_print_output)` and the version suffix never changes it, so switching versions is one import
line. The `huc_12` branch has a different layout (`prepare_input_layers_huc12_01.py`,
`whitebox_utils.py`, extra configs); on that branch, list `src/` and `config/` first.

## Mode A: a new step

1. **Name it.** `src/<verb>_<noun>_NN.py`; the next step after 04 is `05`. Between two steps use
   a letter sub-step of the earlier one (`03b`), never a renumber. If it uses multiprocessing,
   ask whether to mark it `_mp_` before the number (Andy's convention in ras2fim-2d). Confirm the
   name, entry function, output folder (`NN_<name>` under `<out_root_folder>/<catchment>/`) and
   config keys with Andy.
2. **Write it from [step-template.md](step-template.md),** reading `build_netcdf_03.py` and
   `heal_netcdf_04.py` as well. It must have: the header block between `# ****` rules
   (`# LISFLOOD-FP to FIM`, `# Script NN - <stem>`, `Created by: Andy Carter, PE`,
   `Created - YYYY.MM.DD`); the `# ****` import block; a comment rule above and below every
   function; the triple-quoted f-string banner with `---(g)`, `---(c)`, `---[r]` lines inside
   `if b_print_output:` and `print(" -- Script NN: ...")` in the `else:`; the GLOBAL / LOCAL
   `*_section_schema` config reading; copied `is_valid_file` / `fn_str_to_bool`; the `__main__`
   argparse block (`-g`, `-c`, `-r`, prefixed `dest`, `REQUIRED:` / `OPTIONAL: ... Default:`
   help); and the closing `Compute Time:` print.
3. No tqdm: this repo has none and the Docker `geo` env doesn't install it. Use
   `'  -- STEP n: ...'` prints. Ask before adding any import that isn't already used in `src/`.
4. Use LF line endings for a new file unless Andy prefers CRLF.
5. `python -m py_compile src/<new_file>.py`.
6. Wire it in (below), after asking.

## Mode B: a new version (`_vN`) of an existing step

1. Pick the name: the current newest version plus one, the suffix after the step number
   (`run_lisflood_02_v3.py` becomes `run_lisflood_02_v4.py`). Copy from the version the
   orchestrator imports today, unless Andy names another.
2. Copy forward with `cp` (keeps bytes and CRLF/LF). **Never edit, move or delete the old file.**
3. In the new file's header keep everything, and add under the last dated line:
   `# Revised - YYYY.MM.DD -- <what changed>`.
4. Make the requested change in the new file only. Keep `fn_<name>_NN` unchanged.
5. `python -m py_compile` the new file; `git status` must show the old file unmodified.
6. Offer the switch and say which orchestrator(s) import the old version; edit only if Andy says
   yes, and only the ones he names:
   ```
   -from run_lisflood_02_v3 import fn_run_lisflood_02
   +from run_lisflood_02_v4 import fn_run_lisflood_02
   ```

## Wiring into the orchestrators (ask first, show the diff)

There are two: `src/lisflood2fim_00.py` (NextGen catchments; CRLF) and
`src/lisflood2fim_huc12_00.py` (HUC-12 watersheds; LF). Ask which one gets the step: one or both.
Neither has a step range; `fn_lisflood2fim_00` runs every step in order.
- **Import:** add `from <stem> import fn_<name>_NN` under `# Import modules`, in step order.
- **Call:** add it in order inside `fn_lisflood2fim_00`, aligned like the others:
  ```
      fn_check_netcdf_05(str_global_config_file_path,
                         str_local_config_file_path,
                         b_print_output)
  ```
- Add `# Revised - YYYY.MM.DD -- add step NN <name>` under the header's last `Revised` line.
- `python -m py_compile` the orchestrator. Confirm its line endings didn't change (see below).

## Config settings (ask first)

Step settings go in the **global** configs; local configs hold only `[run_parameters]`. Sections
here are topic names (`[lisflood_settings]`, `[flow_parameters]`, `[clean_fim_stack]`), not
numbers. Ask whether the keys belong in an existing section or a new one; for a new one suggest
a topic name, or `[NN_<name>]` if Andy prefers numbered sections. Add it to every
`config/global_*.ini` and `config/demo_global_config.ini`, after the last section, in the
files' style:
```
# -----------------------
[check_netcdf]
# list of required global attributes
required_attrs = 00_stream_id
```
If the `/config-sync` skill is present, use its `-t section ... -b -` action. Otherwise use exact-text
edits. The files mix CRLF and LF and none ends with a final newline: keep each file as it was.
Check after:
```
git diff --stat -- config/
python -c "import sys;[print(f, open(f,'rb').read().count(b'\r\n'), 'CRLF of', open(f,'rb').read().count(b'\n'), open(f,'rb').read()[-1:]) for f in sys.argv[1:]]" config/*.ini
```
In a CRLF file the two counts must be equal; in an LF file the CRLF count must be 0.

## Finish

Report: files created or changed, the `py_compile` results, the diffs, the line-ending check,
what you did not run (a full run needs the Docker image or the `geo` env with LISFLOOD-FP and
WhiteboxTools), and a proposed commit message such as `add step 05 check netcdf`.
