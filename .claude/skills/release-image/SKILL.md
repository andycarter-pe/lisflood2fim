---
name: release-image
description: Build a new dated lisflood2fim Docker image (civileng127/<image>:YYYYMMDD), smoke-check it, propose the README docker pull/run and status-date changes as a diff, and stop to ask before any docker push. Andy starts this by typing /release-image; Claude never starts it on its own.
argument-hint: "[YYYYMMDD]"
disable-model-invocation: true
metadata:
  contributed-by: "CLB Engineering for Andy Carter, PE, 2026.10.27"
---

# /release-image for lisflood2fim

Build and check a new image, update the README after Andy approves the diff, then stop before
the push. Work in this order and do not skip a stop.

Follow AGENTS.md if present. These rules apply even without it:
- **Never run `docker push`, `docker login` or `git push` unless Andy says "push" for this tag in
  this session.** A yes to the README diff is not a yes to push. Never push a `latest` tag;
  Andy does not use one.
- Never edit the Dockerfile, code, configs or env. The only file this skill changes is
  README.md, and only after Andy approves the diff.
- Do not commit. Stage README.md and propose a short lowercase message; Andy commits.
- Report exactly what you ran. If a build or check fails, stop and show the error.

## 1. Learn the current scheme
- Branch: `git branch --show-current`. `main` builds `civileng127/lisflood2fim`; the `huc_12`
  branch builds `civileng127/lisflood2fim_huc12`. Confirm from README.md, do not assume.
- From README.md read: the `docker build -t ...` line, the `docker pull civileng127/<image>:<tag>`
  line, the image in the `docker run` block, and the `**Status**:` line with its
  `(YYYY.MM.DD)` date.
- Read the published tags:
  `curl -s "https://hub.docker.com/v2/repositories/civileng127/<image>/tags?page_size=100"`.
  Andy tags by build date, `YYYYMMDD` (for example `20260611`, `20260508`), one tag per day.
- New tag: the argument if given (`$ARGUMENTS`), else today's date as `YYYYMMDD`. If that tag
  already exists on Docker Hub, stop and ask: overwrite it, or use another date?

## 2. Check that the build will contain what Andy expects
The Dockerfile does not copy this folder into the image. It runs
`git clone https://github.com/andycarter-pe/lisflood2fim.git /app/lisflood2fim` (on `huc_12`:
`git clone -b huc_12 --single-branch --depth 1 ...`). So the image gets what is on **GitHub**,
not what is on disk.
- `git fetch origin`, then report `git status --short` and `git log --oneline origin/<branch>..HEAD`.
- If `src/`, `config/` or `sample_data_in/` have uncommitted or unpushed changes, stop and tell
  Andy they will not be in the image until he pushes them. Do not push for him.
- Docker caches the `git clone` layer, so a cached build can reuse an old clone. Build with
  `--no-cache` unless Andy says a cached build is fine.

## 3. Build
Show Andy the command and the tag, then run it from the repo root:
```
docker build --no-cache -t civileng127/<image>:<tag> .
```
It compiles LISFLOOD-FP and solves the conda env, so it takes a while. Run it in the
background with output going to a log outside the repo (for example
`<temp>/release_image_lisflood2fim_<tag>.log`). Report the last 20 lines if it fails.

## 4. Smoke-check the image
Run each check and record the result:
```
docker run --rm civileng127/<image>:<tag> bash -c "source /opt/miniconda/etc/profile.d/conda.sh && conda activate geo \
 && python -c 'import numpy, pandas, geopandas, rasterio, rioxarray, shapely, fiona, pyogrio, netCDF4, whitebox; print(\"geo OK\")' \
 && lisflood -version | grep -i version && wbt --version | head -1 && which nccopy \
 && python /app/lisflood2fim/src/<orchestrator> -h | head -3 \
 && git -C /app/lisflood2fim log -1 --format='%h %ad %s' --date=short"
```
- `<orchestrator>` is `lisflood2fim_00.py` on `main`, `lisflood2fim_huc12_00.py` on `huc_12`.
- The commit printed last must equal `git rev-parse --short origin/<branch>`. If not, stop.
- Also report `docker image ls civileng127/<image>:<tag>` (ID and size), and the size of the
  previous tag if it is on this machine.
- Offer a full sample run on the new image before pushing (`/run-demo docker` after the README
  points at the new tag). Do not start it without a yes.

## 5. Propose the README change, then apply it only after a yes
Change only these:
- `docker pull civileng127/<image>:<old>` to the new tag.
- The image line in the `docker run` block (`civileng127/<image>:<old> \`) to the new tag.
- The date in the `**Status**:` line, `(YYYY.MM.DD)`, to today. Ask before changing the version
  words (for example `Version 0.1- Preliminary release.`); leave them if Andy does not say.

Write the proposed file to a temp path outside the repo (`<temp>/README_lisflood2fim_<tag>.md`) and show `diff -u README.md <temp>`.
README.md uses CRLF line endings; the proposed file must keep CRLF on every line (check with
`file`). Ask: "Apply this README change?" Copy it over README.md only after a yes. Then run
`git diff --stat README.md` and `git add README.md`, and propose a commit message in Andy's
style, for example `docker container to <tag>`.

## 6. Stop before the push
Show Andy, and then stop and wait:
```
Built:   civileng127/<image>:<tag>  (<image id>, <size>)  from GitHub <branch> <hash>
Checks:  geo OK | LISFLOOD-FP <version> | WhiteboxTools <version> | nccopy | orchestrator -h
README:  updated and staged | not changed
To publish, the command is:
  docker push civileng127/<image>:<tag>
Push now? (yes / no)
```
Run `docker push` only if Andy answers yes to this question in this session. After a push, check
the tag appears with the curl from section 1 and report it. Then remind Andy to commit and push
the README himself.
