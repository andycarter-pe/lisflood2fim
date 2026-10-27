---
name: config-sync
description: Add, retire, or rename one key in every lisflood2fim config variant (config/global_*, config/demo_global_config.ini, and the *local* configs) without disturbing comments, decorative rule lines, key order, blank lines or each file's CRLF/LF line endings. Reports the files touched and every line of code that reads the key. Use when a config key must change across the configs.
argument-hint: "add|retire|rename [section] <key> [value or new_key]"
metadata:
  contributed-by: "CLB Engineering for Andy Carter, PE, 2026.10.27"
---

# /config-sync for lisflood2fim

Follow AGENTS.md if present. These rules apply even without it:
- A key change goes into **every** variant of its kind (global or local), or you say which
  variant was skipped and why.
- Never delete a line. Retired keys and old values stay in as commented-out lines.
- Never rename a key or section unless Andy asks for that rename in so many words. They are
  interfaces, including misspelled ones such as `[stable_run_paramters]`.
- Don't reorder, re-indent, re-space or "tidy" any line you were not asked to change. Keep
  trailing spaces (`fpfric = 0.06  `) and aligned `=` signs as they are.
- Don't commit. Show the diff and propose a short lowercase commit message; Andy commits.

## The config variants

List them fresh each time (`ls config/*.ini`); the `huc_12` branch has a different set (for
example `global_s3_huc12_config.ini`, `global_tacc_huc12_config.ini`). On `main` today:

| Kind | File | Line endings |
|---|---|---|
| global | `demo_global_config.ini` (Docker demo, `/app/lisflood2fim/sample_data_in/`) | CRLF |
| global | `global_config.ini` (Windows, `E:\ras2fim-2d-windows-resources\`) | CRLF |
| global | `global_tacc_config.ini` (TACC, `/work2/08140/acarter/...`) | CRLF |
| global | `global_web_config.ini` (web URLs on corral.tacc) | CRLF |
| global | `global_web_huc12_config.ini` (HUC-12 runs) | LF |
| global | `global_web_mac_test_config.ini` | LF |
| local | `demo_local_config.ini`, `local_config.ini` | CRLF |
| local | `demo_local_mac_test_config.ini` | LF |

None of them ends with a final newline. Global configs hold `[datasource]`,
`[lisflood_settings]`, `[flow_parameters]`, `[stable_run_paramters]` and (in most)
`[clean_fim_stack]`; local configs hold only `[run_parameters]` (`catchment`,
`out_root_folder`). Sections are topic names, not numbers, each preceded by a
`# -----------------------` line and one blank line. Keys are plain lowercase names
(`downscale`, `num_steps`, a few mixed case such as `max_Froude`).

The variants are **not** identical: `global_config.ini` and `global_web_huc12_config.ini` have
no `[clean_fim_stack]`, and `global_web_huc12_config.ini` has no `default_manning_runs` and has
`perct_bottom_line` commented out. Report a missing section; don't create it unless Andy says so.
When a value is a path or URL, ask for the value per variant.

## Never write the configs back with configparser

`configparser.write()` drops every comment and rule line. Edit the files as text:
1. **Preferred:** the bundled script `${CLAUDE_SKILL_DIR}/config_sync.py` (standard library only,
   runs in the `geo` env or any Python 3.8+). It reads bytes, keeps each file's CRLF or LF, its
   encoding and its missing final newline, refuses mixed-ending files, prints a diff, and checks
   that only the intended lines changed.
2. **Otherwise:** exact-text edits with the Edit tool, one file at a time, then the byte check
   below. Half of these files are CRLF; a new line with a bare LF in a CRLF file is a defect.

## Steps

1. **Pin down the request.** Action (add / retire / rename), global or local, section, key,
   value(s), and a one-line comment in Andy's voice (e.g. `# number of unstable runs allowed`).
2. **Find every reader of the key** before changing anything. Old `_vN` files still count: they
   stay importable from the orchestrators.
   ```
   grep -rn -i "<key>" src/ README.md Dockerfile
   ```
   Steps list the keys they read in `global_section_schema` / `local_section_schema` and use
   `dict_all_params['<key>']`.
3. **Check the current state:**
   ```
   python "${CLAUDE_SKILL_DIR}/config_sync.py" -t show -f "config/global_*.ini" config/demo_global_config.ini -s <section> -k <key>
   ```
4. **Dry run, then write.** The script never writes without `-w True`.
   - Add (after the last key in the section, or after `-a <key>`):
     ```
     python "${CLAUDE_SKILL_DIR}/config_sync.py" -t add -f "config/global_*.ini" config/demo_global_config.ini -s stable_run_paramters -k max_unstable_runs -v 5 -m "number of unstable runs allowed before stopping"
     ```
     Local keys: `-f "config/*local*.ini" -s run_parameters`. For different values per variant,
     run once per file with `-f <that file>`.
   - Retire (comments the line out as `#key = value`; `-m` adds `# Retired - YYYY.MM.DD -- why`):
     `-t retire -s <section> -k <key> -m "<why>"`
   - Rename (only when asked): `-t rename -s <section> -k <old> -n <new> -m "<why>"`. The old line
     stays as a comment and the new key gets the same value on the next line.
   - A whole new section: give the block, in the file's style, on stdin with `-b -`
     (no temp file needed); omit `-a` to append at the end:
     ```
     python "${CLAUDE_SKILL_DIR}/config_sync.py" -t section -f "config/global_*.ini" config/demo_global_config.ini -s check_netcdf -b - <<'EOF'
# -----------------------
[check_netcdf]
# list of required global attributes
required_attrs = 00_stream_id
EOF
     ```
   Show Andy the dry-run diff. Write with `-w True` once it is what he asked for.
5. **Retire and rename need extra care.**
   - Retire: if any step still lists the key in its schema, stop and say so. A missing key comes
     back as `''` from `section.get(key, '')`, so `int(dict_all_params['window'])` fails with a
     ValueError at run time.
   - Rename: warn first that the old name lives on in configs outside this repo (TACC, S3, web,
     older Docker images such as `civileng127/lisflood2fim:20260611`), then list every reader from
     step 2. Changing those readers is a Python edit: show each change, add a
     `# Revised - YYYY.MM.DD -- <what>` line under the file's header, and edit only after Andy
     agrees.
6. **Byte check** (required after Edit-tool edits; the script prints it for you):
   ```
   git diff --stat -- config/
   python -c "import sys;[print(f, open(f,'rb').read().count(b'\r\n'), 'CRLF', open(f,'rb').read().count(b'\n'), 'LF total', open(f,'rb').read()[-1:]) for f in sys.argv[1:]]" config/*.ini
   ```
   In a CRLF file the two counts must be equal; in an LF file the CRLF count must be 0. No file
   should gain a final newline. When a key is added after the very last line, that old last line
   shows as changed in `git diff` only because it gains a line ending; its text is unchanged.

## Report

- Each variant: added / retired / renamed / skipped (with the reason, e.g. section missing).
- The grep of readers: file, line, and the schema entry or `dict_all_params[...]` use.
- Anything Andy must still decide (values per variant, readers to update).
- A proposed commit message, e.g. `config: add max_unstable_runs to stable run parameters`.

## Supporting file

- `config_sync.py`: the byte-preserving editor used above. `python config_sync.py -h` lists the
  options.
