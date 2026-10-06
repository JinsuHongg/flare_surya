# Surya Bench Resizable Zarr Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` to implement this plan task-by-task.

**Goal:** Rebuild the annual Surya Bench Zarr from S3 hourly NetCDF objects at a configurable square image resolution, with UTC filename timestamps and restart-safe progress.

**Architecture:** A Hydra-configured script anonymously lists S3 objects, freezes a per-year source manifest, and writes each year into a staging Zarr group. Bounded thread pools overlap downloads and transforms; the main coordinator alone owns SQLite and writes each prepared image to its stable slot before its timestamp, then marks the sample complete. A validated year is consolidated and atomically promoted to the final annual directory. Conservative defaults are two download workers, one transform worker, and two total samples in flight; worker settings are operational and excluded from resume identity.

**Tech Stack:** Python, Hydra/OmegaConf, requests, xarray/h5netcdf, OpenCV, NumPy, Zarr v2, SQLite from the Python standard library, pytest.

**Spec:** `docs/superpowers/specs/2026-10-05-surya-bench-224-rebuild-design.md`

## Global Constraints

- Use the `flare-surya` environment and avoid adding dependencies.
- Select hourly source objects whose basenames have the form `YYYYMMDD_HH00.nc`.
- Derive each timestamp from the filename and interpret it as UTC. Do not apply an 8-hour offset.
- Resize and store all 13 channels in the existing order.
- Preserve the current directory layout: each year has `/<year>/dataset/images` and `/<year>/dataset/time`.
- Default `target_size` to 224 and accept positive integer Hydra CLI overrides such as `target_size=512`; include resolution in build identity.
- Derive `images` shape `(N,13,target_size,target_size)` and chunks `(50,1,target_size,target_size)`; keep `time` chunks `(50,)`, LZ4 level 5 with shuffle 1, and time units `nanoseconds since 1970-01-01`.
- Make processing resumable across cluster walltime interruptions.
- Never modify the current Zarr in place; build to a separate output path.
- Register source NetCDF compression filters through the already-installed `hdf5plugin`; do not add dependencies.
- Do not run the full S3 corpus job; do not stage or commit changes.

## Review Focus

- Nonmatching names and non-00 minute files must be ignored; duplicate timestamps among accepted hourly objects must fail before writing; pin in Task 1.
- Truncated S3 pagination or changed object identities must not silently produce a partial or mixed manifest; pin in Tasks 1 and 2.
- Missing channels or unexpected image shapes must not create successful zero-image samples; pin in Task 2.
- Interruption after image write but before timestamp/checkpoint must cause the slot to be rewritten; pin in Task 2.
- Dry-run, existing output, and incompatible resume state must never overwrite data; pin in Task 3.

---

### Task 1: Configuration and immutable S3 manifest

**Files:**
- Create: `configs/preprocessing/rebuild_surya_bench_zarr.yaml`
- Create: `scripts/preprocessing/rebuild_surya_bench_zarr.py`
- Create: `tests/preprocessing/test_rebuild_surya_bench_zarr.py`

**Interfaces:**
- Produces `SourceObject(key: str, year: int, timestamp_ns: int, size: int, etag: str)`.
- Produces `parse_source_key(key: str) -> SourceObject | None`, returning `None` for keys outside the exact hourly filename pattern.
- Produces `list_source_objects(bucket: str, endpoint_url: str, prefix: str, start_year: int, end_year: int, timeout_seconds: int) -> list[SourceObject]`, using anonymous S3 ListObjectsV2 HTTPS pagination.
- Produces `manifest_digest(objects: list[SourceObject], config_identity: dict[str, object]) -> str`, hashing a canonical serialization of source identities and immutable output settings.

- [ ] **Step 1: Add unit tests for filename parsing, ignoring nonmatching/non-00 keys, year filtering, pagination continuation, duplicate timestamps, and stable manifest hashing.** Mock HTTP responses; do not make real S3 requests in tests.
- [ ] **Step 2: Implement the source record, strict UTC filename parser, anonymous paginated listing, candidate sorting/validation, and canonical manifest digest.** Ignore keys outside the exact hourly pattern and reject duplicate timestamps among accepted objects before any output writes.
- [ ] **Step 3: Add the Hydra configuration with bucket `nasa-surya-bench`, prefix, year bounds, null-by-default output path, ordered channels, `target_size: 224`, target-size-derived chunks, retry/timeouts, temporary directory, existing compression/time units, UTC metadata, and `dry_run: true`.** Keep environment-specific filesystem paths out of the checked-in config.
- [ ] **Step 4: Run `conda run -n flare-surya pytest tests/preprocessing/test_rebuild_surya_bench_zarr.py -q`; expect parsing, pagination, and digest cases to pass without network access.**

### Task 2: Per-year processing, Zarr writes, and resume checkpoints

**Files:**
- Modify: `scripts/preprocessing/rebuild_surya_bench_zarr.py`
- Modify: `tests/preprocessing/test_rebuild_surya_bench_zarr.py`

**Interfaces:**
- Produces `download_object(source: SourceObject, bucket: str, endpoint_url: str, temp_dir: Path, timeout_seconds: int, attempts: int) -> Path`, streaming one object to a temporary file and cleaning partial downloads on failure.
- Produces `resize_netcdf(path: Path, channels: list[str], target_size: int) -> np.ndarray`, validating all 13 variables and 2-D source shapes, resizing each channel with `cv2.INTER_AREA`, and returning `float32` `(13,target_size,target_size)` data.
- Produces `process_year(year: int, objects: list[SourceObject], bucket: str, endpoint_url: str, output_path: Path, checkpoint_root: Path, temp_dir: Path, channels: list[str], target_size: int, timeout_seconds: int, attempts: int, manifest_digest: str, config_digest: str, download_workers: int = 2, transform_workers: int = 1, max_in_flight_samples: int = 2) -> None`, using bounded preparation pools and one main coordinator as the only Zarr/SQLite owner, with a per-year staging group/checkpoint DB and dynamic image shape/chunks.

- [ ] **Step 1: Add isolated tests for all-channel ordering, 224 and 512 target dimensions, missing channel/shape errors, UTC nanosecond conversion, and checkpoint resume after an interrupted image write.** Build tiny synthetic NetCDF/Zarr fixtures locally; no S3 access.
- [ ] **Step 2: Implement streamed HTTP download with bounded retries and temporary-file cleanup; implement per-channel resize and strict shape/channel validation.**
- [ ] **Step 3: Implement per-year staging arrays with `images` float32 `(N,13,target_size,target_size)`, chunks `(50,1,target_size,target_size)`, and `time` int64 nanoseconds since epoch, with existing LZ4 settings and UTC metadata.**
- [ ] **Step 4: For each stable manifest slot, mark checkpoint `writing`, write the resized image, write its UTC timestamp only after the image write finishes, then mark it `complete`.** On restart, reprocess every slot not marked complete; verify complete timestamps against the manifest.
- [ ] **Step 5: Refuse resume if source manifest or immutable config hashes differ. Keep failed rows out of published output; retain diagnostics and leave the year in staging for a later retry.**
- [ ] **Step 6: Run `conda run -n flare-surya pytest tests/preprocessing/test_rebuild_surya_bench_zarr.py -q`; expect synthetic transform, failure, and interrupted-resume cases to pass.**

### Task 3: Safe Hydra entry point and annual promotion

**Files:**
- Modify: `scripts/preprocessing/rebuild_surya_bench_zarr.py`
- Modify: `configs/preprocessing/rebuild_surya_bench_zarr.yaml`
- Modify: `tests/preprocessing/test_rebuild_surya_bench_zarr.py`

**Interfaces:**
- Produces `run(cfg: DictConfig) -> None`, called by `@hydra.main`.
- Produces a dry-run summary with candidate counts by year, UTC range, manifest digest, resolved output path, and estimated uncompressed image bytes; dry-run performs no downloads or writes.
- Produces `validate_output_root(output_path: Path, manifest_digest: str, config_digest: str, allow_initialize: bool) -> None`, requiring an atomic `.checkpoints/build.json` identity marker for nonempty resumable output roots and allowing initialization only for a new or empty root.
- Produces `publish_year(stage: Path, final: Path, objects: list[SourceObject], channels: list[str], target_size: int, manifest_digest: str, config_digest: str) -> None`, validating the completed resolution-specific schema, consolidating metadata, recording completion identity, and atomically renaming the staged year.

- [ ] **Step 1: Add tests for dry-run no-write behavior, refusal of an existing unrelated output, valid checkpoint resume at the same resolution, rejection of a different-resolution resume, 512 output shape/chunks, completed-year skip, and atomic promotion of a complete year.**
- [ ] **Step 2: Implement the Hydra entry point, dry-run summary, explicit write gate, atomic root build marker/identity validation before any staging write, per-year dispatch, and nonzero exit when any year remains incomplete.** Refuse existing nonempty unmarked output roots, including the currently supplied Zarr.
- [ ] **Step 3: Validate each finished year before promotion: all checkpoint rows complete, `time` matches the frozen manifest and is sorted/unique, image shape/dtype/channels/chunks match the configured resolution, and metadata is consolidated.** Atomically rename the staging year on the same filesystem and preserve `.checkpoints` outside the published year group.
- [ ] **Step 4: Run `conda run -n flare-surya pytest tests/preprocessing/test_rebuild_surya_bench_zarr.py -q` and inspect a local synthetic annual Zarr tree; expect all synthetic cases to pass and the user-facing arrays/metadata to match the current layout.**
- [ ] **Step 5: Review `git diff --check` and the scoped diff; confirm the pre-existing modification to `scripts/analysis/retrieve_validation_wandb.py` is untouched.**

### Approved follow-up: Bounded download/transform pipeline

**Files:** the rebuild script, Hydra config, focused preprocessing tests, and this plan/spec.

- [x] Add event-gated local tests first for default download/transform overlap, shared in-flight and file bounds, out-of-order results at stable slots, a single main-thread Zarr/SQLite owner, failure recording and continued slots, worker cleanup/interruption/resume, and positive-integer limit validation. Prohibit real HTTP.
- [x] Use separate standard-library thread pools with default `download_workers=2`, `transform_workers=1`, and `max_in_flight_samples=2`. Count downloaded inputs, active transforms, and results awaiting the writer against the same bound; workers receive no Zarr/SQLite handles. All checkpoint state and image → timestamp → complete writes remain on the coordinator.
- [x] Treat worker limits as operational settings excluded from resume identity; wire Hydra overrides into annual processing without changing schema, publication, manifest order, or UTC time semantics.
- [x] Drain/cancel queued work on Python interruption and clean successful downloads that never reached a transform. Keep incomplete scheduled slots resumable and later slots unmodified until scheduled.
- [x] Run focused and `tests/` suites, scoped Ruff/Black, and diff review; record evidence and the active-worker shutdown caveat in the Task 3 report.
