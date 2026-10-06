# Surya Bench Resizable Zarr Rebuild Design

## Goal

Build a resumable preprocessing script that recreates the existing annual Surya Bench Zarr layout directly from public S3 NetCDF files, using UTC timestamps from object filenames and no CSV index. The output image resolution is a runtime argument with a default of 224; other positive square sizes such as 512 are supported.

## Confirmed requirements

- Use the `flare-surya` environment and avoid adding dependencies.
- Select hourly source objects whose basenames have the form `YYYYMMDD_HH00.nc`.
- Derive each timestamp from the filename and interpret it as UTC. Do not apply an 8-hour offset.
- Resize and store all 13 channels in the existing order, at the configured positive integer resolution.
- Accept the resolution as a Hydra command-line override, e.g. `target_size=224` or `target_size=512`; include it in build identity and derive image shape/chunks from it.
- Preserve the current directory layout: each year has `/<year>/dataset/images` and `/<year>/dataset/time`.
- Make processing resumable across cluster walltime interruptions.
- Overlap downloads and transforms with configurable conservative defaults: `download_workers: 2`, `transform_workers: 1`, and `max_in_flight_samples: 2` shared across both stages and results awaiting a write.
- Never modify the current Zarr in place; build to a separate output path.
- Use the installed `hdf5plugin` filter registration needed by the source NetCDF files; do not add or change dependencies.

## Data flow

1. List the configured public S3 bucket and prefix with anonymous HTTPS requests. Keep only keys matching the hourly filename pattern and configured year range; sort by timestamp and key. Validate unique timestamps and save a frozen source manifest containing each key, timestamp, object size, and ETag.
2. For each year, create a staging directory on the same filesystem as the output. Create the year dataset with image slots and a zero-filled `time` array; do not write source timestamps in advance.
3. The main coordinator marks each scheduled slot `writing` and uses separate bounded thread pools for downloads and transforms. Workers download temporary files, open them with xarray/h5netcdf, require all configured channels, and resize each 2-D channel to `target_size×target_size` using OpenCV `INTER_AREA`; transforms return the stacked `float32` image and remove their download. At most two samples are in flight by default across downloads, transforms, and prepared results awaiting the writer. Results may finish out of order; the coordinator writes each image to its stable manifest slot, then writes that row's UTC timestamp to `time`, then commits checkpoint `complete`. Only the coordinator writes Zarr or accesses SQLite.
4. If download or validation fails after bounded retries, retain the failed state and error details in the checkpoint database, leave that year unpublished, and exit nonzero after other years are handled. A failed sample is never represented by a zero image.
5. After every image and timestamp for the year has been written, validate row counts, timestamp ordering/uniqueness, dimensions, and completion state; consolidate Zarr metadata; then atomically rename the staging year directory into `/<year>`. Keep per-year progress and the source manifest under a root `.checkpoints` directory.

## Resume and safety

- Use the standard-library SQLite database for durable per-year sample states (`pending`, `writing`, `complete`, `failed`) and source/config identity. Keep a single writer for Zarr arrays.
- Worker limits are operational settings, excluded from immutable config and manifest identity; changing concurrency may resume the same build. All three worker limits must be positive integers.
- On restart, relist S3 and recompute the manifest and configuration hashes. Resume only when the output, manifest, channel order, resize settings, dtype, and schema match. If they differ, stop with an actionable error instead of mixing runs.
- Store global manifest/config identities in an atomically written `.checkpoints/build.json` before creating staging data. An existing nonempty output without this matching marker is not resumable and must be refused before any write.
- Reprocess any slot that is not marked complete. A zero timestamp means the image row is incomplete; the timestamp is the last Zarr value written for a sample, after the full resized image has been stored. Mark a slot complete only after both its image and timestamp writes finish. If a year had been atomically promoted before interruption, validate its completion metadata and skip it.
- Default to dry-run. Writing requires an explicit config override and a separate output path. Refuse to write into an existing output unless it is a valid resumable build; do not provide an in-place overwrite mode.
- Temporary NetCDF downloads are removed after processing. Keep only failed-object diagnostics and resume state needed to continue.
- On a Python interruption, close the pipeline, cancel queued work, drain active workers, and remove downloads not consumed by transforms. Incomplete scheduled slots remain resumable. Active HTTP calls cannot be forcibly canceled, so cleanup may wait for their timeout/retry policy or an active transform.

## Zarr schema

Each published year is a Zarr v2 group at `/<year>/dataset` with exactly these user-facing arrays:

- `images`: `float32`, shape `(N, 13, target_size, target_size)`, dimensions `time, channel, y, x`.
- `time`: `int64`, shape `(N,)`, nanoseconds since the Unix epoch, interpreted as UTC.

Use the existing ordered channel list (`aia94`, `aia131`, `aia171`, `aia193`, `aia211`, `aia304`, `aia335`, `aia1600`, `hmi_m`, `hmi_bx`, `hmi_by`, `hmi_bz`, `hmi_v`), image chunks `(50,1,target_size,target_size)`, time chunks `(50,)`, LZ4 compression level 5 with shuffle 1, and current time units/calendar (`nanoseconds since 1970-01-01`, `proleptic_gregorian`). Add a separate explicit UTC attribute. At the default 224 size, this matches the current on-disk Zarr metadata. Checkpoint state remains outside the year dataset so the published group retains its existing two-array schema.

## Files

- Add `scripts/preprocessing/rebuild_surya_bench_zarr.py` for listing, downloading, resizing, writing, checkpointing, and validation.
- Add `configs/preprocessing/rebuild_surya_bench_zarr.yaml` for the public `nasa-surya-bench` bucket/prefix, year range, required output path, channels, target size, retry count, worker limits, chunks/compression, temporary directory, and dry-run/write mode.
- Do not edit existing preprocessing scripts, training configs, dataset readers, or dependencies.

## Operational limits

- The script will not be run against the full S3 corpus as part of implementation; that cluster job is for the user to launch.
- The script should support a bounded year range so the operator can make an initial small run before scheduling the full rebuild.
- S3 HTTP availability, object format, and storage capacity remain runtime conditions and must be reported clearly on failure.

## Review notes

- The requested annual Zarr layout differs from the repository's `HelioNetCDFDatasetZarr` reader, which expects root-level `img` and `timestep` arrays. This work intentionally preserves the user's existing annual layout and does not change that reader.
- Timezone is represented by UTC epoch nanoseconds, which avoids local timezone conversion while preserving the existing int64 encoding.
- Resume correctness depends on checking the frozen object manifest and config identity before writing; timestamps alone are insufficient to identify an unchanged build.
