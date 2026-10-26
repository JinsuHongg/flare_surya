# Full-cadence Surya Zarr Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a separate 512-pixel Zarr containing all timestamped Surya Bench NetCDF files from 2010 through 2024, preserving the current hourly dataset.

**Architecture:** Extend the existing S3 listing and timestamp parser with an explicit all-NetCDF selection mode while retaining the legacy hourly mode and its manifest identity. Write a fresh immutable manifest and per-year checkpoints under `/scratch/users/jhong36/surya-bench-512-lanczos-all-nc.zarr`; run a one-year pilot before submitting the full qCPU120 array, capped at five simultaneous tasks and limited to nodes with verified storage access.

**Tech Stack:** Python 3.11, Hydra/OmegaConf, urllib S3 listing, Pillow Lanczos, Zarr v2, SQLite checkpoints, SLURM qCPU120.

**Spec:** `docs/superpowers/specs/2026-10-26-surya-all-nc-zarr-design.md`

## Global Constraints

- Source range: `2010–2024` inclusive.
- Existing output `/scratch/users/jhong36/surya-bench-512-lanczos.zarr` and its checkpoints must remain unchanged.
- New output: `/scratch/users/jhong36/surya-bench-512-lanczos-all-nc.zarr`.
- Images: 13 channels, `float32`, 512 × 512, Pillow Lanczos, image chunks `(1, 13, 512, 512)`.
- Timestamps: UTC nanoseconds, strict ascending order, duplicate timestamps are fatal.
- Production array concurrency: `--array=2010-2024%5`; never run more than five year tasks from this array concurrently.
- Use only qCPU120 nodes that pass output-path visibility and write checks.
- Resume only with the same frozen manifest, output root, and immutable schema.
- Do not modify the shared builder or its config while legacy job `4736377` is running; verify it has exited before Task 1.
- Do not start implementation until the design document is approved.

## Review Focus

- An `.nc` key has a basename outside `YYYYMMDD_HHMM.nc`: the inventory must report it and stop rather than silently omit a source file.
- Two keys parse to the same UTC timestamp: manifest creation must fail with both conflicting keys identified.
- S3 listing returns a transient 429/5xx response: bounded retry must recover without writing a partial manifest.
- A year is interrupted during download, transform, array write, or promotion: resuming the same build must rewrite only incomplete samples and leave completed years valid.
- A scheduled node cannot see the shared output path: preflight/constraint must keep that node out of the production array.

---

### Task 1: Inventory and Parse Every NetCDF Timestamp

**Files:**
- Modify: `scripts/preprocessing/rebuild_surya_bench_zarr.py`
- Modify: `configs/preprocessing/rebuild_surya_bench_zarr.yaml`
- Test: `tests/preprocessing/test_rebuild_surya_bench_zarr.py`

**Interfaces:**
- `list_netcdf_keys(bucket: str, endpoint_url: str, prefix: str, start_year: int, end_year: int, timeout_seconds: int) -> list[tuple[str, int, str]]` returns `.nc` keys and listing metadata without assuming a basename format.
- `parse_source_key(key: str, source_mode: str = "hourly") -> SourceObject | None`
- `list_source_objects(..., source_mode: str = "hourly") -> list[SourceObject]`
- `inventory_only: bool` is a dry-run-only CLI/config option that summarizes every candidate `.nc` key before parsing or downloading.
- The legacy config adds `source_mode: hourly` and `inventory_only: false`; neither field changes the legacy immutable identity.
- `source_mode="all_nc"` accepts only `.nc` keys with a validated UTC timestamp basename; the initial expected form is `YYYYMMDD_HHMM.nc`.

- [ ] **Step 1: Confirm the legacy job is no longer using the shared builder**

Run: `squeue -j 4736377 -o '%.18i %.9T %.10M %.20N'`
Expected: no running task for job `4736377`. Do not edit the builder/config while that job is active.

- [ ] **Step 2: Add failing parser tests**

Add tests proving that `20100513_0100.nc` and a non-zero-minute name such as
`20100513_0112.nc` map to the correct UTC nanoseconds in `all_nc` mode; reject
invalid dates/times; retain current hourly-mode behavior; and ignore non-NetCDF
objects. Add listing tests that include both minute values, sort them
chronologically, and report duplicate timestamps with both keys.

- [ ] **Step 3: Run focused tests and confirm they fail**

Run: `conda run -n flare_surya pytest tests/preprocessing/test_rebuild_surya_bench_zarr.py -k 'parse_source_key or listing' -q`
Expected: new all-NetCDF cases fail because minute-aware selection is not yet implemented.

- [ ] **Step 4: Add a format-independent NetCDF inventory**

Implement `list_netcdf_keys(...)` over the existing paginated S3 API. It returns
all `.nc` object keys and their size/ETag metadata without parsing timestamps.
Add `inventory_only=true`, restricted to dry-run, to report candidate counts by
year and basename pattern, unsupported keys, duplicate parseable timestamps,
total source bytes, and estimated uncompressed image bytes. It must not create
the output path, manifest, or checkpoints.

- [ ] **Step 5: Run the source inventory and confirm the key format**

Run: `conda run -n flare_surya python scripts/preprocessing/rebuild_surya_bench_zarr.py inventory_only=true dry_run=true`
Expected: every `.nc` key in 2010–2024 is counted; the report identifies the
timestamp patterns and any unsupported keys. If a key pattern other than
`YYYYMMDD_HHMM.nc` appears, stop and revise the parser design before resizing.

- [ ] **Step 6: Add failing timestamp parser and ordering tests**

Test both `20100513_0100.nc` and `20100513_0112.nc`, invalid dates and minutes,
midnight/year boundaries, chronological ordering, and duplicate timestamps.
Assert duplicate errors identify both conflicting keys.

- [ ] **Step 7: Implement explicit source modes and stable ordering**

Add the `source_mode` parameter to parsing and listing. Preserve the exact
legacy hourly selection for existing output resumes. In `all_nc` mode, parse
the full date/hour/minute, convert to UTC nanoseconds, and sort by
`(timestamp_ns, key)`. Reject duplicate parsed timestamps and never silently
discard a timestamped `.nc` key.

- [ ] **Step 8: Verify focused and full preprocessing tests**

Run: `conda run -n flare_surya pytest tests/preprocessing/test_rebuild_surya_bench_zarr.py -q`
Expected: all existing hourly tests and new all-NetCDF parsing/listing tests pass.

### Task 2: Isolate the New Manifest and Preserve Resume Compatibility

**Files:**
- Modify: `configs/preprocessing/rebuild_surya_bench_zarr.yaml`
- Create: `configs/preprocessing/rebuild_surya_bench_zarr_all_nc.yaml`
- Modify: `scripts/preprocessing/rebuild_surya_bench_zarr.py`
- Test: `tests/preprocessing/test_rebuild_surya_bench_zarr.py`

**Interfaces:**
- The existing config defaults to `source_mode: hourly` and retains its current output identity.
- The new config selects `source_mode: all_nc`, `start_year: 2010`, `end_year: 2024`, `target_size: 512`, `resize_method: PIL.LANCZOS`, and `output_path: /scratch/users/jhong36/surya-bench-512-lanczos-all-nc.zarr`.
- `_config_identity(cfg: DictConfig) -> dict[str, Any]` includes source mode in the new all-NetCDF identity without changing the identity of the existing hourly build.

- [ ] **Step 1: Add failing configuration and identity tests**

Test that the legacy config retains its prior identity, the new config resolves
the separate output path and all-NetCDF mode, and changing source mode changes
the all-NetCDF manifest/config digest.

- [ ] **Step 2: Run the focused identity tests and confirm they fail**

Run: `conda run -n flare_surya pytest tests/preprocessing/test_rebuild_surya_bench_zarr.py -k 'config_identity or all_nc_config' -q`
Expected: the all-NetCDF config and its identity are not present yet.

- [ ] **Step 3: Add the new Hydra config and compatibility-safe identity**

Keep the existing config as the legacy hourly entry point. Add the all-NetCDF
config with the approved schema and distinct output root. Include source mode
in the new build identity; omit that new identity field for legacy hourly mode
so the frozen 2010–2024 hourly manifest remains resumable.

- [ ] **Step 4: Verify legacy and new identity behavior**

Run: `conda run -n flare_surya pytest tests/preprocessing/test_rebuild_surya_bench_zarr.py -k 'config_identity or all_nc_config or resume' -q`
Expected: legacy identity fixtures remain unchanged and all-NetCDF resume identity is stable.

### Task 3: Make Listing and Annual Checkpoints Resilient

**Files:**
- Modify: `configs/preprocessing/rebuild_surya_bench_zarr.yaml`
- Modify: `scripts/preprocessing/rebuild_surya_bench_zarr.py`
- Test: `tests/preprocessing/test_rebuild_surya_bench_zarr.py`

**Interfaces:**
- Add a small retry helper for S3 listing requests, retrying only transient HTTP 429 and 5xx responses with a bounded exponential backoff.
- Add runtime config values `listing_retry_count: 5`, `retry_backoff_seconds: 2`, and `max_retry_backoff_seconds: 30`; keep them out of immutable manifest identity.
- Preserve the current per-sample SQLite states and atomic annual promotion.

- [ ] **Step 1: Add failing tests for transient listing failures and resume**

Test that one transient HTTP 503 followed by success yields one complete
manifest; a persistent transient failure stops after the configured maximum;
non-retryable 4xx responses fail immediately; and an interrupted all-NetCDF
year resumes with stable timestamp slots and skips completed samples.

- [ ] **Step 2: Run the focused retry and resume tests and confirm they fail**

Run: `conda run -n flare_surya pytest tests/preprocessing/test_rebuild_surya_bench_zarr.py -k 'listing_retry or all_nc_resume' -q`
Expected: transient listing retry and all-NetCDF interruption cases fail before the implementation.

- [ ] **Step 3: Implement bounded listing retries without changing sample semantics**

Retry only HTTP 429/5xx failures up to `listing_retry_count`, sleeping
`min(retry_backoff_seconds * 2**retry_index,
max_retry_backoff_seconds)` between attempts. Keep manifest writing atomic
under the existing lock. Preserve checkpoint slot ordering from the immutable
manifest; on resume, completed rows remain valid and incomplete rows are
retried. Do not retry malformed XML, unsupported filenames, or duplicate
timestamps.

- [ ] **Step 4: Verify retry and resume behavior**

Run: `conda run -n flare_surya pytest tests/preprocessing/test_rebuild_surya_bench_zarr.py -k 'listing_retry or all_nc_resume or interruption' -q`
Expected: transient listing and interruption recovery tests pass.

### Task 4: Add the qCPU120 Array and Storage Preflight

**Files:**
- Create: `shell_scripts/artic/rebuild_surya_bench_all_nc_512_zarr.slurm`
- Modify: `tests/preprocessing/test_rebuild_surya_bench_zarr.py` (only if adding a script-rendering test fits existing conventions)

**Interfaces:**
- SLURM array years: `2010-2024`, concurrency cap: `%5`.
- Hydra config: `rebuild_surya_bench_zarr_all_nc`.
- Output path is the separate all-NetCDF root; processing remains one selected year per task.
- Submission requires a verified qCPU120 node list; the production command supplies that list through `sbatch --nodelist`.

- [ ] **Step 1: Define a node/path preflight**

Before submission, probe candidate qCPU120 nodes for `/scratch/users/jhong36`,
the new output root's parent, and read/write/traverse access. Record only nodes
that pass. Confirm the destination does not resolve to the existing hourly
output. If fewer than five eligible nodes are available, keep `%5` but allow
the scheduler to run fewer than five tasks concurrently.

- [ ] **Step 2: Add the dedicated array script**

Create a separate SLURM script that uses qCPU120, one year per task,
`--array=2010-2024%5`, 8 CPUs, 64 GB, and the all-NetCDF Hydra config. Keep the
existing hourly SLURM script unchanged. Ensure the logs include the array task
year and actual output path.

- [ ] **Step 3: Verify the script and config without submitting jobs**

Run: `bash -n shell_scripts/artic/rebuild_surya_bench_all_nc_512_zarr.slurm`
Run: `conda run -n flare_surya python scripts/preprocessing/rebuild_surya_bench_zarr.py --config-name=rebuild_surya_bench_zarr_all_nc dry_run=true`
Expected: shell syntax passes; dry-run reports 2010–2024 counts, UTC range, unique manifest digest, separate output path, and uncompressed size estimate without creating output data.

### Task 5: Pilot, Submit, Resume, and Validate the Full Build

**Files:**
- No additional source files; use the all-NetCDF config and SLURM script from Tasks 2 and 4.

**Interfaces:**
- Pilot array: year 2010 with `%1` using the same new output root and all-NetCDF manifest.
- Production array: years 2010–2024 with `%5`, constrained to nodes verified in Task 4.

- [ ] **Step 1: Recheck the active legacy build and storage quota**

Confirm the existing hourly 2011 task has exited before changing shared source
files or starting the new build. Check user quota with available cluster
storage tooling; do not rely solely on `/scratch` aggregate free space. Compare
quota against the manifest's uncompressed estimate plus a safety margin.

- [ ] **Step 2: Run and validate the 2010 pilot**

Submit only year 2010 with concurrency `%1`. Confirm the job uses a verified
node, produces the expected sample count, and publishes valid ordered UTC
timestamps, completion metadata, 13-channel schema, 512 × 512 images, and
`(1, 13, 512, 512)` image chunks.

- [ ] **Step 3: Exercise interruption recovery on the pilot**

Use the existing tested interruption fixture for automated recovery checks.
For the live pilot, do not deliberately kill a successful long-running job;
verify the same-manifest resume path with the controlled fixture and inspect
checkpoint/manifest identity before production submission.

- [ ] **Step 4: Submit the full year array**

After the pilot passes, submit `--array=2010-2024%5` with the verified node
list. Keep the same output path, source mode, channels, target size, and
manifest identity used by the pilot. Record the SLURM job ID.

- [ ] **Step 5: Monitor and resume only failed or incomplete years**

Compare `sacct` states against all 15 years. For each failure, inspect stderr
and the per-year checkpoint before retrying. Retry with the same output root
and manifest only after correcting the cause; constrain retries to verified
nodes. Do not submit duplicate active tasks for the same year.

- [ ] **Step 6: Validate the finished dataset and preserve the legacy build**

Run the builder's finished-year validation for all 15 years. Confirm annual
counts equal the frozen manifest, every timestamp is unique and strictly
increasing, all completion markers match, and the old hourly manifest/output
remain unchanged. Report final compressed storage use and the list of any
failed years.

## Verification Commands

- `conda run -n flare_surya pytest tests/preprocessing/test_rebuild_surya_bench_zarr.py -q`
- `conda run -n flare_surya ruff check scripts/preprocessing/rebuild_surya_bench_zarr.py tests/preprocessing/test_rebuild_surya_bench_zarr.py`
- `conda run -n flare_surya basedpyright --level error --pythonpath /home/user/j/jhong36/miniforge3/envs/flare_surya/bin/python scripts/preprocessing/rebuild_surya_bench_zarr.py tests/preprocessing/test_rebuild_surya_bench_zarr.py`
- `bash -n shell_scripts/artic/rebuild_surya_bench_all_nc_512_zarr.slurm`
- `git diff --check`
