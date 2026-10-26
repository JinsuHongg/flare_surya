# Full-cadence Surya Zarr Design

## Intent

Create a 512-pixel Surya Bench Zarr from every timestamped `.nc` object in the
2010–2024 source range. Keep the existing `surya-bench-512-lanczos.zarr`
unchanged so the hourly build remains available as a separate dataset.

## Goals

- Include all supported NetCDF timestamps, including the current `HH00.nc`
  files and any other minute values in the same naming family.
- Store each year in its own Zarr group with timestamps in strictly increasing
  UTC order.
- Preserve resumability through a new immutable manifest and per-year
  checkpoints.
- Run on qCPU120 with no more than five array tasks active at once.
- Submit only to nodes where the output filesystem is visible and writable.

## Non-goals

- Modify, extend, or delete the existing hourly Zarr or its checkpoints.
- Include `.nc` objects that have no supported timestamp in their key.
- Change the 13 channels, 512 × 512 output, Pillow Lanczos interpolation, or
  channel-aligned image chunk schema.

## Selected Approach

Build a separate dataset at
`/scratch/users/jhong36/surya-bench-512-lanczos-all-nc.zarr`. Extend the
existing builder to recognize every validated timestamped NetCDF key, then
create a fresh manifest and checkpoints under that new root. This preserves
the hourly output and avoids shifting its fixed array slots in place.

An in-place migration was considered and rejected because adding observations
between existing timestamps changes annual array positions and checkpoint
slots. That would require rewriting existing arrays while retaining a reliable
rollback copy.

## Data and Schema

- Source years: 2010 through 2024 inclusive.
- Initial filename assumption to verify against a complete bucket inventory:
  `YYYYMMDD_HHMM.nc`, interpreted as UTC.
- Before accepting this pattern, inventory every `.nc` key and report any
  unsupported basename, duplicate timestamp, or missing year. Do not silently
  skip a NetCDF object.
- Images: `(sample, 13, 512, 512)`, `float32`, chunks `(1, 13, 512, 512)`,
  Pillow Lanczos, LZ4 compression.
- Time: UTC nanoseconds, one timestamp for each image, strictly increasing
  within and across annual partitions.
- New output identity: separate output path, frozen all-NetCDF manifest, and
  checkpoints keyed to the new ordered object list. Existing hourly identity
  and resume behavior remain valid.

## Constraints and Failure Handling

- Estimate required bytes from the complete object inventory before submitting
  production jobs. `/scratch` had 148 TB free at the last check; user-specific
  quota was not available from the current environment and must be checked by
  the user or cluster tooling.
- A retry must use the same output path, configuration, and frozen manifest.
  Changed source membership requires a new output root.
- Retry transient S3 listing errors such as HTTP 503 with a bounded backoff;
  malformed listing data and duplicate timestamps remain fatal.
- A node that cannot see the output path must not receive an array task. The
  previously observed `acidscn003` lacked `/scratch/users/jhong36`; `acidscn013`
  and `acidscn020` passed read/write/traverse checks.
- Keep the current 2011 hourly rebuild isolated from changes until it exits.

## Acceptance

- The preflight inventory accounts for every `.nc` object in the requested
  years or explicitly reports why the build must stop.
- The all-NetCDF dry run reports per-year counts, UTC range, unique timestamps,
  and an uncompressed size estimate.
- A one-year pilot can be resumed after interruption and passes annual schema,
  completion-marker, timestamp-order, and manifest checks.
- Production submits years 2010–2024 with `--array=2010-2024%5`, constrained
  to verified storage-visible qCPU120 nodes.
- The existing hourly Zarr and its manifest/checkpoints remain unchanged.
