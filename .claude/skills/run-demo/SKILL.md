---
name: run-demo
description: Run the lisflood2fim sample catchment (sample_data_in with the demo configs), in the published Docker image or the local env, then check every numbered output folder, report which LISFLOOD-FP intensity runs reached a stable outflow, read the NetCDF attributes and return a short plain summary. Use when Andy asks to run the demo or sample data, or to check that a change still runs. Never edits code.
argument-hint: "docker|docker-src|local [previous-run-folder]"
context: fork
agent: general-purpose
disallowed-tools: Edit Write NotebookEdit
metadata:
  contributed-by: "CLB Engineering for Andy Carter, PE, 2026.10.27"
---

# /run-demo for lisflood2fim

Run Andy's sample catchment the way README.md says, check the outputs and report back. You run
in your own context. Keep the long output here and return only the summary in section 7.

Follow AGENTS.md if present. These rules apply even without it:
- Never edit, create or delete anything in the repository: no code, config, README or Dockerfile
  changes. Write only inside the run folder you create.
- Do not commit, push, or remove Docker images. Remove any container you started.
- Report exactly what ran. If a step could not run, name it and say why. Never call a step
  passed unless you ran it and checked its folder.

## 1. Read the arguments
Arguments: `$ARGUMENTS`
- Word 1 is the run mode. Andy chooses it every time; there is no default.
  - `docker`: the image and `docker run` block in README.md, using the code baked into the image.
  - `docker-src`: the same image with this working copy mounted over `/app/lisflood2fim`, so it
    runs the code, configs and sample data on disk now.
  - `local`: this machine's Python env (`geo` in the Dockerfile), with `lisflood`
    (LISFLOOD-FP 8.1), WhiteboxTools and `nccopy` on PATH.
- Word 2 (optional) is a previous run folder to compare against.

If word 1 is missing or not one of the three, run nothing. Return only this question:
"Run in Docker (`docker`, or `docker-src` to test your working copy) or in your local env
(`local`)? Example: `/run-demo docker`."

## 2. Find what to run
Read these; do not assume:
- README.md: the image name and tag in `docker pull` and the `docker run` block. On `main` it is
  `civileng127/lisflood2fim:<YYYYMMDD>` running `lisflood2fim_00.py`; on the `huc_12` branch it
  is `civileng127/lisflood2fim_huc12:<YYYYMMDD>` running `lisflood2fim_huc12_00.py`.
- The orchestrator's imports, to learn which step versions run (on `main` today:
  `prepare_input_layers_01_v6`, `run_lisflood_02_v3`, `build_netcdf_03`, `heal_netcdf_04`).
- The defaults in the orchestrator's argparse: `-g /app/lisflood2fim/config/demo_global_config.ini`
  and `-c /app/lisflood2fim/config/demo_local_config.ini`. From the local config, read
  `catchment` and `out_root_folder` under `[run_parameters]` (today `cat-2420542` and
  `data/output`, relative to the working folder). From the global config, read `timesteps` and
  `[stable_run_paramters]` (the misspelling is the real section name). The intensities come from
  the precipitation rasters in step 01, so count the `.par` files rather than trust `num_steps`.

The orchestrator has no step range; it runs steps 01 to 04 in order. To rerun one step, call
that step script's own `__main__` with `-g` and `-c`.

## 3. Make the run folder
Make a new folder outside the repo, `lisflood2fim_demo_YYYYMMDD_HHMM` next to the repo, unless
Andy named one. Docker mounts it as `/mnt`. Output lands in
`<run_folder>/<out_root_folder>/<catchment>/`.

For `local`: `demo_global_config.ini` points at `/app/lisflood2fim/sample_data_in/...`. If that
path does not exist on this machine, write a copy of the demo global config into the run folder
with `/app/lisflood2fim` replaced by the repo path (use a shell command; leave `config/`
untouched), and pass it with `-g`. Run from the run folder so `data/output` lands there.

## 4. Run
Record the code that runs: `git rev-parse --short HEAD`, the branch and `git status --short` in
the repo and, for Docker, `docker run --rm <image> git -C /app/lisflood2fim log -1 --format="%h %ad %s" --date=short`.
If the image is not present, `docker pull` it first.

Docker: use README's block, without `-it` (there is no terminal here), with `--rm` and a name:
```
docker run --rm --name run-demo-lisflood2fim -v <run_folder>:/mnt <image:tag> \
  bash -c "source /opt/miniconda/etc/profile.d/conda.sh && conda activate geo && cd /mnt && python /app/lisflood2fim/src/lisflood2fim_00.py"
```
- `docker-src`: add `-e PYTHONDONTWRITEBYTECODE=1 -v <repo>:/app/lisflood2fim:ro` before the
  image name, so the run cannot write `__pycache__` or anything else into the repo.
- In Git Bash on Windows, put `MSYS_NO_PATHCONV=1` in front so `/mnt` is not rewritten.

Local: `python <repo>/src/lisflood2fim_00.py -g <config> -c <repo>/config/demo_local_config.ini`
from the run folder, in the env Andy names, with `PYTHONDONTWRITEBYTECODE=1` set.

Start the command in the background with stdout and stderr going to
`<run_folder>/run_demo_log.txt`, and time it. Wait for it to finish. Do not read the log
directly: the LISFLOOD spinner writes `\r` ten times a second, so the log is mostly spinner
frames. Read it with
`tr '\r' '\n' < log | grep -v -E '\.\.\. [|/\\-]$'`, then grep for `STEP`, `completed in`,
`Stable Run not found`, `Manning`, `Compute Time:`, `Traceback`, `Error`. If the log has not
grown for 20 minutes, stop the container and report it as hung.

After the run, `git status --short` in the repo must match what you recorded before. If it does
not, list the changed files in the summary. Do not restore them yourself.

## 5. Check the outputs
In `<run_folder>/<out_root_folder>/<catchment>/`:

| Step | Script | Expect |
|---|---|---|
| 01 | `prepare_input_layers_01_v6.py` | `01_stream_delineation/` (`<catchment>.gpkg`, `00_dem_clipped_5070.tif`, `dem_burn_roads_d4fixed_5070.asc`); `02_lisflood_input/` with one `<catchment>_<NNpN>mm.par` per intensity |
| 02 | `run_lisflood_02_v3.py` | `02_lisflood_input/<catchment>_<NNpN>mm/` per run, each with a `.mass` file and a `-0001.wd` depth grid |
| 03 | `build_netcdf_03.py` | `02_lisflood_input/<catchment>-geotiffs/`; `03_depth_nc/depth_<catchment>.nc` (the uncompressed `_big.nc` is deleted after `nccopy`) |
| 04 | `heal_netcdf_04.py` | `04_healed_depth_nc/depth_<catchment>.nc`, `_shallow_removed.nc`, `_denoised.nc`; `01_stream_delineation/04_dem_difference.tif` |

- Confirm each folder exists and is not empty. Give the file count and total size. List
  anything missing.
- **Stability report.** Count the `.par` files (intensities), the run folders with a `.mass`
  file and those with a `-0001.wd`. From the log, list each `completed in N seconds` line and any
  `Stable Run not found: <run> Status: <status>`. Then recompute the stable row for every
  intensity with Andy's own `fn_get_stable_row`, using the snippet in
  [reference.md](reference.md). Report the intensities that never stabilised; they are missing
  from the NetCDF stack.
- **NetCDF.** For `03_depth_nc/depth_<catchment>.nc` and the three healed files, read the global
  attributes (`00_stream_id` to `07_raster_folder`), the dimensions, and the `intensity` values
  and units (`mm h-1`). Snippet in [reference.md](reference.md).

## 6. Compare and clean up
- If Andy gave a previous run folder and the andy-tools plugin is installed, hand both folders to
  its `results-compare` agent. If it is not installed, say the comparison was skipped and that
  `andy-tools:results-compare` can do it.
- `docker ps -a --filter name=run-demo-` must come back empty. Remove any container you
  started. Keep the run folder; it is the result.

## 7. Return this summary (plain text, short)
```
/run-demo lisflood2fim -- <mode>, <orchestrator>, catchment <id>
Code:      repo <branch> <hash> (<clean|uncommitted changes>); image <name:tag> code <hash date>
Output:    <run_folder>/<out_root_folder>/<catchment>
Ran:       steps 01-04 | stopped at <step> because <reason>
Folders:   <folder> <n files, size> OK|MISSING   (one line each)
LISFLOOD:  <n> intensities, <n> ran, <n> stable; not stable: <list with status>; slowest run <name> <s> s
NetCDF:    <file>: <n> intensities (<min>-<max> mm h-1), attrs 00-07 present|missing <list>
Time:      Compute Time <from log>; wall <measured>
Problems:  <tracebacks, empty folders, warnings worth a look, or "none">
Compare:   <results-compare verdict, or "skipped">
Log:       <run_folder>/run_demo_log.txt
```
