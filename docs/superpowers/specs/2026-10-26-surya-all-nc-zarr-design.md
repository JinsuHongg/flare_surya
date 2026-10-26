# Full-cadence Surya Zarr Design

## Intent

Create a complete 512-pixel Surya Bench Zarr from every timestamped `.nc`
object in the 2010–2024 source range. Reuse validated hourly image samples from
the existing Zarr and download/resize only source objects not already present
there. Keep the existing `surya-bench-512-lanczos.zarr` read-only and unchanged.

## Goals

- Include all supported NetCDF timestamps, including the current `HH00.nc`
  files and any other minute values in the same naming family.
- Store each year in its own Zarr group with timestamps in strictly increasing
  UTC order.
- Preserve resumability through a new immutable manifest and per-year
  checkpoints.
- Avoid downloading and resizing hourly samples already materialized in the
  existing Zarr when their source identity and output schema match.
- Run on qCPU120 with no more than five array tasks active at once.
- Submit only to nodes where the output filesystem is visible and writable.

## Non-goals

- Modify, extend, or delete the existing hourly Zarr or its checkpoints.
- Change how the old dataset is written or checkpointed.
- Change the 13 channels, 512 × 512 output, Pillow Lanczos interpolation, or
  channel-aligned image chunk schema.

## Selected Approach

Build a separate dataset at
`/scratch/users/jhong36/surya-bench-512-lanczos-all-nc.zarr`. List all source
objects from S3 to create the authoritative full-cadence manifest. For each
year, validate the existing hourly Zarr and its frozen manifest; copy matching
512-pixel image samples into their new timestamp-ordered slots. Download and
resize remaining objects from S3. Store a new manifest and checkpoints under
the new root. The current manifest contains 121,963 hourly source objects
(68.76 TB of source object bytes). At the last check, 14 completed years
accounted for 113,647 samples and 63.93 TB of those source bytes; the 2011
hourly task was still running.

An all-S3 rebuild was considered and rejected because downloading large
NetCDF files is the current bottleneck and their matching resized images are
already available. An in-place migration was also rejected because new times
change annual array positions and checkpoint slots.

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
- New output identity: separate output path and frozen all-NetCDF manifest.
  Include the existing source manifest digest and schema identity in the new
  build identity so reuse decisions are auditable and stable on resume.
- Copy an old sample only when the old year passes finished-year validation
  and the old manifest's key, timestamp, size, and ETag exactly match the new
  bucket listing. Otherwise download and resize that source object.
- Keep each reused or downloaded sample under the same per-sample checkpoint
  rules; a copied sample is complete only after image and time writes finish.

## Constraints and Failure Handling

- Estimate required bytes from the complete object inventory before submitting
  production jobs. `/scratch` had 148 TB free at the last check; user-specific
  quota was not available from the current environment and must be checked by
  the user or cluster tooling.
- A retry must use the same output path, configuration, frozen all-NetCDF
  manifest, and reuse-source manifest digest. Changed source membership
  requires a new output root.
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
- Matching completed hourly samples are copied into the new Zarr without
  downloading their original NetCDF files.
- Unmatched, changed, or invalid old samples are downloaded from S3 and
  transformed normally.
- The existing hourly Zarr and its manifest/checkpoints remain unchanged.
