<!-- Contributed by CLB Engineering for Andy Carter, PE, 2026.10.27. Supporting file for /run-demo. -->

# /run-demo reference (lisflood2fim)

Run these with the image's `geo` env, for example
`docker run --rm -v <run_folder>:/mnt <image:tag> bash -c "source /opt/miniconda/etc/profile.d/conda.sh && conda activate geo && python - <<'EOF' ... EOF"`,
or with the local env. They only read files.

## Stability report from Andy's own functions
Set `STR_MODULE` to the run step the orchestrator imports (`run_lisflood_02_v3` on `main`),
`STR_SRC` to the `src` folder that ran (`/app/lisflood2fim/src` in the image), and the paths
to the configs and the catchment output folder.

```
import configparser, os, sys
STR_SRC = '/app/lisflood2fim/src'
STR_MODULE = 'run_lisflood_02_v3'
STR_GLOBAL = '/app/lisflood2fim/config/demo_global_config.ini'
STR_CATCHMENT = 'cat-2420542'
STR_OUT = '/mnt/data/output/' + STR_CATCHMENT

sys.path.insert(0, STR_SRC)
mod = __import__(STR_MODULE)
config = configparser.ConfigParser()
config.read(STR_GLOBAL)
# configparser lowercases keys, so read them by the names the step uses
dict_params = {str_key: config['stable_run_paramters'][str_key] for str_key in
               ('window', 'min_Qout_ratio', 'max_Qout_ratio', 'max_rolling_avg_stability')}
str_stream = os.path.join(STR_OUT, '01_stream_delineation')
str_lisflood = os.path.join(STR_OUT, '02_lisflood_input')

for str_par, flt_intensity in sorted(mod.fn_collect_par_values(str_lisflood), key=lambda t: t[1]):
    str_run = os.path.basename(str_par)[:-4]
    str_mass = os.path.join(str_lisflood, str_run, str_run + '.mass')
    flt_expected = mod.fn_get_expected_outflow(flt_intensity, str_stream, STR_CATCHMENT)
    str_status, ps_row = mod.fn_get_stable_row(str_mass, flt_expected, dict_params)
    b_wd = os.path.exists(os.path.join(str_lisflood, str_run, str_run + '-0001.wd'))
    if str_status == 'ok':
        print(f"{str_run:32} {str_status:10} stable at {ps_row['Time']:9.0f} s  "
              f"Qout/Qexp={ps_row['Qout_ratio']:.3f}  wd={b_wd}")
    else:
        print(f"{str_run:32} {str_status:10} wd={b_wd}")
```

Status meanings come from `fn_get_stable_row`: `ok`, `not_found` (no `.mass` file),
`invalid_file`, `missing_qout`, `no_match` (ran, but outflow never met the
`[stable_run_paramters]` limits). If the module's function names differ (an older or newer `_vN`),
read that module and adapt the call; do not edit it.

## NetCDF attributes
```
import glob, netCDF4
STR_OUT = '/mnt/data/output/cat-2420542'
for str_nc in sorted(glob.glob(STR_OUT + '/0[34]_*/*.nc')):
    with netCDF4.Dataset(str_nc) as ds:
        print('==', str_nc)
        print('   dims:', {k: len(v) for k, v in ds.dimensions.items()})
        print('   vars:', list(ds.variables))
        for str_att in ds.ncattrs():
            print('   ', str_att, '=', str(ds.getncattr(str_att))[:200])
        if 'intensity' in ds.variables:
            var = ds.variables['intensity']
            print('   intensity:', [round(float(v), 1) for v in var[:]],
                  var.getncattr('units') if 'units' in var.ncattrs() else '')
```
