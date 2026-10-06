"""Build annual Surya Bench Zarr staging datasets with immutable resume state."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
import xml.etree.ElementTree as ET
from contextlib import closing
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import cv2
import hdf5plugin  # noqa: F401 -- registers source NetCDF HDF5 compression filters
import hydra
import numpy as np
import xarray as xr
import zarr
from loguru import logger
from numcodecs import Blosc
from omegaconf import DictConfig, OmegaConf

_HOURLY_FILENAME = re.compile(r"(?P<date>\d{8})_(?P<hour>\d{2})00\.nc\Z")


@dataclass(frozen=True)
class SourceObject:
    """Identity and listing metadata for one source NetCDF object."""

    key: str
    year: int
    timestamp_ns: int
    size: int
    etag: str


def parse_source_key(key: str) -> SourceObject | None:
    """Parse an exact hourly NetCDF basename as a UTC timestamp.

    Args:
        key: Full object key from the bucket listing.

    Returns:
        A source identity with placeholder listing metadata, or ``None`` when
        the basename does not match the required hourly filename pattern.
    """
    basename = key.rsplit("/", 1)[-1]
    match = _HOURLY_FILENAME.fullmatch(basename)
    if match is None:
        return None
    try:
        timestamp = datetime.strptime(
            f"{match.group('date')}_{match.group('hour')}", "%Y%m%d_%H"
        ).replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    timestamp_ns = int(timestamp.timestamp()) * 1_000_000_000
    return SourceObject(key, timestamp.year, timestamp_ns, 0, "")


def _xml_child(element: ET.Element, name: str) -> str | None:
    """Find a child element by local XML name, regardless of namespace."""
    for child in element:
        if child.tag.rsplit("}", 1)[-1] == name:
            return child.text
    return None


def list_source_objects(
    bucket: str,
    endpoint_url: str,
    prefix: str,
    start_year: int,
    end_year: int,
    timeout_seconds: int,
) -> list[SourceObject]:
    """List matching public S3 objects anonymously and return stable ordering.

    Args:
        bucket: Public bucket name.
        endpoint_url: HTTPS S3 endpoint, optionally including a path prefix.
        prefix: Object key prefix to list.
        start_year: First included UTC year.
        end_year: Last included UTC year.
        timeout_seconds: Per-request timeout.

    Returns:
        Matching objects sorted by timestamp and key.

    Raises:
        ValueError: If bounds are invalid, listing metadata is malformed, or
            two accepted keys map to the same timestamp.
        OSError: If an HTTP request fails.
    """
    if start_year > end_year:
        raise ValueError("start_year must be less than or equal to end_year")
    if not endpoint_url.startswith("https://"):
        raise ValueError("endpoint_url must use HTTPS")

    base_url = endpoint_url.rstrip("/") + "/" + bucket
    continuation_token: str | None = None
    accepted: list[SourceObject] = []
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if continuation_token is not None:
            params["continuation-token"] = continuation_token
        request = Request(
            base_url + "?" + urlencode(params), headers={"User-Agent": "flare-surya"}
        )
        with urlopen(request, timeout=timeout_seconds) as response:
            root = ET.fromstring(response.read())
        for entry in root.iter():
            if entry.tag.rsplit("}", 1)[-1] != "Contents":
                continue
            key = _xml_child(entry, "Key")
            size_text = _xml_child(entry, "Size")
            etag = _xml_child(entry, "ETag")
            if key is None or size_text is None or etag is None:
                raise ValueError(
                    "malformed S3 listing entry: missing Key, Size, or ETag"
                )
            parsed = parse_source_key(key)
            if parsed is None or not start_year <= parsed.year <= end_year:
                continue
            try:
                size = int(size_text)
            except ValueError as error:
                raise ValueError(f"invalid object size for {key!r}") from error
            accepted.append(
                SourceObject(key, parsed.year, parsed.timestamp_ns, size, etag)
            )

        truncated = _xml_child(root, "IsTruncated")
        if truncated != "true":
            break
        continuation_token = _xml_child(root, "NextContinuationToken")
        if not continuation_token:
            raise ValueError("truncated S3 listing omitted NextContinuationToken")

    accepted.sort(key=lambda obj: (obj.timestamp_ns, obj.key))
    timestamps: set[int] = set()
    for obj in accepted:
        if obj.timestamp_ns in timestamps:
            raise ValueError(
                f"duplicate timestamp among accepted source objects: {obj.timestamp_ns}"
            )
        timestamps.add(obj.timestamp_ns)
    return accepted


def manifest_digest(
    objects: list[SourceObject], config_identity: dict[str, object]
) -> str:
    """Hash canonical source identities and immutable output settings.

    Args:
        objects: Frozen source manifest entries.
        config_identity: Settings that determine the output identity.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """
    payload: dict[str, Any] = {
        "config_identity": config_identity,
        "objects": [
            asdict(obj)
            for obj in sorted(objects, key=lambda item: (item.timestamp_ns, item.key))
        ],
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


_REQUIRED_CHANNELS = frozenset(
    [
        "aia94",
        "aia131",
        "aia171",
        "aia193",
        "aia211",
        "aia304",
        "aia335",
        "aia1600",
        "hmi_m",
        "hmi_bx",
        "hmi_by",
        "hmi_bz",
        "hmi_v",
    ]
)


def _validate_channels(channels: list[str]) -> None:
    """Require all 13 distinct Surya channel names."""
    if len(channels) != 13 or set(channels) != _REQUIRED_CHANNELS:
        raise ValueError("channels must contain all 13 unique Surya channel names")


def download_object(
    source: SourceObject,
    bucket: str,
    endpoint_url: str,
    temp_dir: Path,
    timeout_seconds: int,
    attempts: int,
) -> Path:
    """Stream an anonymous HTTPS object to a temporary file with bounded retries.

    Args:
        source: Frozen source identity, including expected size and ETag.
        bucket: Public bucket name.
        endpoint_url: HTTPS S3 endpoint.
        temp_dir: Directory for temporary NetCDF downloads.
        timeout_seconds: HTTP timeout per attempt.
        attempts: Maximum number of download attempts.

    Returns:
        Path to the complete download; the caller owns its cleanup.

    Raises:
        ValueError: If the endpoint or retry parameters are invalid.
        OSError: If all attempts fail or the object identity changed.
    """
    if not endpoint_url.startswith("https://"):
        raise ValueError("endpoint_url must use HTTPS")
    if attempts < 1 or timeout_seconds < 1 or source.size < 0:
        raise ValueError(
            "attempts and timeout must be positive; size must be nonnegative"
        )
    temp_dir.mkdir(parents=True, exist_ok=True)
    url = (
        endpoint_url.rstrip("/")
        + "/"
        + quote(bucket, safe="")
        + "/"
        + quote(source.key, safe="/")
    )
    headers = {"User-Agent": "flare-surya"}
    if source.etag:
        headers["If-Match"] = source.etag
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        path: Path | None = None
        succeeded = False
        try:
            with tempfile.NamedTemporaryFile(
                dir=temp_dir, suffix=".nc", delete=False
            ) as target:
                path = Path(target.name)
                with urlopen(
                    Request(url, headers=headers), timeout=timeout_seconds
                ) as response:
                    etag = response.headers.get("ETag")
                    if source.etag and etag != source.etag:
                        raise OSError(f"ETag changed for {source.key}")
                    total_bytes = 0
                    while chunk := response.read(1024 * 1024):
                        target.write(chunk)
                        total_bytes += len(chunk)
                        if total_bytes > source.size:
                            raise OSError(
                                f"object size exceeds frozen manifest for {source.key}"
                            )
                    if total_bytes != source.size:
                        raise OSError(
                            f"object size differs from frozen manifest for {source.key}"
                        )
            succeeded = True
            return path
        except (OSError, ValueError) as error:
            last_error = error
            logger.warning(
                "Download attempt {}/{} failed for {}: {}",
                attempt,
                attempts,
                source.key,
                error,
            )
        finally:
            if not succeeded and path is not None:
                path.unlink(missing_ok=True)
    raise OSError(
        f"download failed after {attempts} attempts for {source.key}: {last_error}"
    ) from last_error


def resize_netcdf(path: Path, channels: list[str], target_size: int) -> np.ndarray:
    """Validate and resize the 13 source channels individually with INTER_AREA.

    Args:
        path: Local NetCDF file.
        channels: Ordered list of all 13 Surya channels.
        target_size: Positive square output size.

    Returns:
        Float32 array with shape ``(13, target_size, target_size)``.

    Raises:
        ValueError: If required channels or consistent 2-D shapes are absent.
    """
    _validate_channels(channels)
    if target_size < 1:
        raise ValueError("target_size must be positive")
    output = np.empty((len(channels), target_size, target_size), dtype=np.float32)
    with xr.open_dataset(path, engine="h5netcdf") as dataset:
        missing = [name for name in channels if name not in dataset.data_vars]
        if missing:
            raise ValueError(f"missing NetCDF channels: {missing}")
        source_shape: tuple[int, ...] | None = None
        for index, name in enumerate(channels):
            values = np.asarray(dataset[name].values, dtype=np.float32)
            # The trailing axes are spatial. Never squeeze a spatial axis to
            # disguise a non-singleton time/sample dimension as an image.
            extra_axes = tuple(range(max(values.ndim - 2, 0)))
            if any(values.shape[axis] != 1 for axis in extra_axes):
                raise ValueError(
                    f"channel {name} must be a 2-D image with singleton extra axes; got {values.shape}"
                )
            if extra_axes:
                values = np.squeeze(values, axis=extra_axes)
            if values.ndim != 2 or 0 in values.shape:
                raise ValueError(
                    f"channel {name} must be a nonempty 2-D image; got {values.shape}"
                )
            if source_shape is None:
                source_shape = values.shape
            elif values.shape != source_shape:
                raise ValueError(
                    f"channel {name} source shape {values.shape} differs from {source_shape}"
                )
            output[index] = cv2.resize(
                values, (target_size, target_size), interpolation=cv2.INTER_AREA
            )
    return output


def _checkpoint_identity(
    objects: list[SourceObject],
    channels: list[str],
    target_size: int,
    output_path: Path,
    manifest_hash: str,
    config_hash: str,
) -> dict[str, str]:
    """Bind saved progress to source rows, channel order, and the destination."""
    return {
        "manifest_digest": manifest_hash,
        "config_digest": config_hash,
        "output_path": str(output_path.resolve()),
        "source_digest": manifest_digest(objects, {}),
        "schema_identity": json.dumps(
            {"channels": channels, "target_size": target_size, "schema_version": 1},
            sort_keys=True,
        ),
    }


def _open_checkpoint(
    path: Path,
    identity: dict[str, str],
    objects: list[SourceObject],
    stage_exists: bool,
) -> sqlite3.Connection:
    """Initialize or validate a year checkpoint before opening writable arrays."""
    existing = path.exists()
    connection = sqlite3.connect(path)
    try:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if existing and not tables:
            if stage_exists:
                raise ValueError(
                    "staging has no initialized checkpoint identity; use a separate output path"
                )
            existing = False
        if existing:
            if tables != {"metadata", "samples"}:
                raise ValueError(
                    "checkpoint schema differs; use a separate output path"
                )
            saved = dict(connection.execute("SELECT key, value FROM metadata"))
            if saved != identity:
                raise ValueError(
                    "checkpoint identity differs; use a separate output path"
                )
            rows = connection.execute(
                "SELECT slot, key, timestamp_ns, size, etag FROM samples ORDER BY slot"
            ).fetchall()
            expected = [
                (slot, obj.key, obj.timestamp_ns, obj.size, obj.etag)
                for slot, obj in enumerate(objects)
            ]
            if rows != expected:
                raise ValueError(
                    "checkpoint source identity differs from the frozen manifest"
                )
        else:
            with connection:
                # SQLite DDL needs an explicit transaction to avoid a partially
                # initialized checkpoint when the job is interrupted.
                connection.execute("BEGIN")
                connection.execute(
                    "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
                connection.execute(
                    "CREATE TABLE samples (slot INTEGER PRIMARY KEY, key TEXT NOT NULL, "
                    "timestamp_ns INTEGER NOT NULL, size INTEGER NOT NULL, etag TEXT NOT NULL, "
                    "state TEXT NOT NULL CHECK(state IN ('pending','writing','complete','failed')), error TEXT)"
                )
                connection.executemany(
                    "INSERT INTO metadata VALUES (?, ?)", identity.items()
                )
                connection.executemany(
                    "INSERT INTO samples VALUES (?, ?, ?, ?, ?, 'pending', NULL)",
                    [
                        (slot, obj.key, obj.timestamp_ns, obj.size, obj.etag)
                        for slot, obj in enumerate(objects)
                    ],
                )
        return connection
    except BaseException:
        connection.close()
        raise


def _open_year_arrays(
    stage: Path, count: int, channels: list[str], can_initialize: bool, target_size: int
) -> tuple[zarr.Array, zarr.Array]:
    """Create the prescribed annual schema, or validate it before resuming."""
    dataset_path = stage / "dataset"
    if any(
        path.is_symlink()
        for path in (
            stage,
            dataset_path,
            dataset_path / "images",
            dataset_path / "time",
        )
    ):
        raise ValueError("staging dataset and array stores must not be symbolic links")
    group = zarr.open_group(str(stage), mode="a")
    dataset = group.require_group("dataset")
    compressor = Blosc(cname="lz4", clevel=5, shuffle=1)
    schemas = {
        "images": (
            (count, 13, target_size, target_size),
            (50, 1, target_size, target_size),
            np.dtype("float32"),
            {"_ARRAY_DIMENSIONS": ["time", "channel", "y", "x"], "channels": channels},
        ),
        "time": (
            (count,),
            (50,),
            np.dtype("int64"),
            {
                "_ARRAY_DIMENSIONS": ["time"],
                "units": "nanoseconds since 1970-01-01",
                "calendar": "proleptic_gregorian",
                "timezone": "UTC",
            },
        ),
    }
    if set(dataset.keys()) - set(schemas):
        raise ValueError("staging dataset has unexpected arrays or groups")
    for name, (shape, chunks, dtype, attributes) in schemas.items():
        if name not in dataset:
            if not can_initialize:
                raise ValueError(f"staging {name} array missing for completed samples")
            array = dataset.create_dataset(
                name,
                shape=shape,
                chunks=chunks,
                dtype=dtype,
                compressor=compressor,
                fill_value=0,
            )
            array.attrs.update(attributes)
        else:
            array = dataset[name]
            if (
                not isinstance(array, zarr.Array)
                or array.shape != shape
                or array.chunks != chunks
                or array.dtype != dtype
                or array.compressor != compressor
            ):
                raise ValueError(
                    f"staging {name} schema differs; use a separate output path"
                )
            missing = {
                key: value
                for key, value in attributes.items()
                if key not in array.attrs
            }
            if (missing and not can_initialize) or any(
                array.attrs[key] != value
                for key, value in attributes.items()
                if key in array.attrs
            ):
                raise ValueError(
                    f"staging {name} schema differs; use a separate output path"
                )
            if missing:
                # An interrupted metadata initialization is repairable only
                # before any rows have been marked complete.
                array.attrs.update(missing)
    return dataset["images"], dataset["time"]


def _resize_and_cleanup(
    path: Path, channels: list[str], target_size: int
) -> np.ndarray:
    """Transform a worker-owned download and remove it on success or failure."""
    try:
        return resize_netcdf(path, channels, target_size)
    finally:
        path.unlink(missing_ok=True)


def _prepared_samples(
    pending: list[tuple[int, SourceObject]],
    bucket: str,
    endpoint_url: str,
    temp_dir: Path,
    channels: list[str],
    target_size: int,
    timeout_seconds: int,
    attempts: int,
    download_workers: int,
    transform_workers: int,
    max_in_flight_samples: int,
    on_start: Callable[[int], None],
) -> Iterator[tuple[int, np.ndarray | Exception]]:
    """Yield bounded prepared results; all callbacks run on the caller thread.

    Download and transform pools never receive Zarr or SQLite objects. A slot
    counts against the shared bound until the caller finishes consuming its
    result. Closing this iterator drains running workers and removes downloads
    not consumed by a transform, including canceled or unconsumed work.
    """
    downloads = ThreadPoolExecutor(
        max_workers=download_workers, thread_name_prefix="surya-download"
    )
    transforms = ThreadPoolExecutor(
        max_workers=transform_workers, thread_name_prefix="surya-transform"
    )
    futures: dict[Future[Any], tuple[int, Path | None]] = {}
    remaining = iter(pending)

    def fill() -> None:
        while len(futures) < max_in_flight_samples:
            item = next(remaining, None)
            if item is None:
                return
            slot, source = item
            try:
                on_start(slot)
                future = downloads.submit(
                    download_object,
                    source,
                    bucket,
                    endpoint_url,
                    temp_dir,
                    timeout_seconds,
                    attempts,
                )
            except Exception as error:
                # Preparation failures use the same main-thread diagnostic
                # path as worker failures, while allowing later slots to run.
                future = Future()
                future.set_exception(error)
            futures[future] = (slot, None)

    try:
        fill()
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            future = next(iter(done))
            del done
            slot, path = futures[future]
            try:
                result = future.result()
            except Exception as error:
                futures.pop(future)
                yield slot, error
            else:
                if path is None:
                    transformed = transforms.submit(
                        _resize_and_cleanup, result, channels, target_size
                    )
                    futures.pop(future)
                    futures[transformed] = (slot, result)
                    continue
                futures.pop(future)
                yield slot, result
                del result
            fill()
    finally:
        # Running urllib requests cannot be canceled. Draining can wait for
        # their timeout/retry policy or for an active transform to finish.
        # Wait before cleaning files workers may still use.
        downloads.shutdown(wait=True, cancel_futures=True)
        transforms.shutdown(wait=True, cancel_futures=True)
        for future, (_, path) in futures.items():
            if path is None and not future.cancelled():
                try:
                    path = future.result()
                except BaseException:
                    continue
            if path is not None:
                path.unlink(missing_ok=True)


def process_year(
    year: int,
    objects: list[SourceObject],
    bucket: str,
    endpoint_url: str,
    output_path: Path,
    checkpoint_root: Path,
    temp_dir: Path,
    channels: list[str],
    target_size: int,
    timeout_seconds: int,
    attempts: int,
    manifest_digest: str,
    config_digest: str,
    download_workers: int = 2,
    transform_workers: int = 1,
    max_in_flight_samples: int = 2,
) -> None:
    """Write one year into staging with a durable SQLite completion checkpoint.

    Args:
        year: UTC source year.
        objects: Frozen manifest entries in strictly increasing timestamp order.
        bucket: Public source bucket.
        endpoint_url: Anonymous HTTPS endpoint.
        output_path: Separate output root; staging is ``.staging/<year>`` beneath it.
        checkpoint_root: Directory containing ``<year>.sqlite3`` checkpoint files.
        temp_dir: Directory for short-lived downloaded files.
        channels: Ordered 13-channel list.
        target_size: Positive square image resolution; default configuration is 224.
        timeout_seconds: Timeout for each HTTP attempt.
        attempts: Maximum download attempts per sample.
        manifest_digest: Frozen build manifest hash.
        config_digest: Immutable configuration hash.
        download_workers: Maximum concurrent downloads.
        transform_workers: Maximum concurrent NetCDF resize operations.
        max_in_flight_samples: Shared bound across downloads, transforms, and
            prepared results awaiting the single Zarr writer.

    Raises:
        ValueError: If source order, schema, or saved resume identity is invalid.
        RuntimeError: If any sample remains incomplete after processing the year.
    """
    _validate_channels(channels)
    _validate_target_size(target_size)
    _validate_worker_limits(download_workers, transform_workers, max_in_flight_samples)
    if not objects or any(obj.year != year for obj in objects):
        raise ValueError("year requires nonempty matching source objects")
    if any(
        left.timestamp_ns >= right.timestamp_ns
        for left, right in zip(objects, objects[1:])
    ):
        raise ValueError("source timestamps must be strictly increasing and unique")
    stage = output_path / ".staging" / str(year)
    checkpoint_path = checkpoint_root / f"{year}.sqlite3"
    if stage.exists() and not checkpoint_path.exists():
        raise ValueError(
            "staging exists without a checkpoint; use a separate output path"
        )
    identity = _checkpoint_identity(
        objects, channels, target_size, output_path, manifest_digest, config_digest
    )
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    connection = _open_checkpoint(checkpoint_path, identity, objects, stage.exists())
    try:
        states = dict(connection.execute("SELECT slot, state FROM samples"))
        images, times = _open_year_arrays(
            stage,
            len(objects),
            channels,
            can_initialize="complete" not in states.values(),
            target_size=target_size,
        )
        # Validate every completed row before beginning any writes.
        for slot, obj in enumerate(objects):
            if states[slot] == "complete" and int(times[slot]) != obj.timestamp_ns:
                raise ValueError(
                    f"completed timestamp differs from manifest at slot {slot}"
                )

        def start(slot: int) -> None:
            with connection:
                connection.execute(
                    "UPDATE samples SET state='writing', error=NULL WHERE slot=?",
                    (slot,),
                )
            if int(times[slot]) != 0:
                times[slot] = 0

        failures = 0
        pending = [
            (slot, obj)
            for slot, obj in enumerate(objects)
            if states[slot] != "complete"
        ]
        prepared = _prepared_samples(
            pending,
            bucket,
            endpoint_url,
            temp_dir,
            channels,
            target_size,
            timeout_seconds,
            attempts,
            download_workers,
            transform_workers,
            max_in_flight_samples,
            start,
        )
        with closing(prepared):
            for slot, result in prepared:
                obj = objects[slot]
                try:
                    if isinstance(result, Exception):
                        raise result
                    images[slot] = result
                    times[slot] = obj.timestamp_ns
                    with connection:
                        connection.execute(
                            "UPDATE samples SET state='complete', error=NULL WHERE slot=?",
                            (slot,),
                        )
                except Exception as error:
                    failures += 1
                    with connection:
                        connection.execute(
                            "UPDATE samples SET state='failed', error=? WHERE slot=?",
                            (f"{type(error).__name__}: {error}", slot),
                        )
                    logger.error(
                        "Year {} slot {} ({}) failed: {}", year, slot, obj.key, error
                    )
        if failures:
            raise RuntimeError(
                f"year {year} incomplete: {failures} failed samples; retry the same build"
            )
    finally:
        connection.close()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Durably replace one JSON file using a sibling on the same filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def validate_output_root(
    output_path: Path,
    manifest_digest: str,
    config_digest: str,
    allow_initialize: bool,
) -> None:
    """Refuse unrelated output; initialize an atomic identity only if permitted.

    Args:
        output_path: Explicit separate build destination.
        manifest_digest: Frozen global manifest identity.
        config_digest: Immutable source and output configuration identity.
        allow_initialize: Whether a new or empty root may receive a marker.

    Raises:
        ValueError: If output is unmarked, incompatible, or not a directory.
    """
    if output_path.is_symlink():
        raise ValueError("output_path must not be a symbolic link")
    output_path = output_path.resolve()
    if output_path == Path(output_path.anchor) or output_path == Path.cwd().resolve():
        raise ValueError("output_path must be a separate build directory")
    if output_path.exists() and not output_path.is_dir():
        raise ValueError("output_path must be a directory")
    marker = output_path / ".checkpoints" / "build.json"
    identity = {
        "schema_version": 1,
        "output_path": str(output_path),
        "manifest_digest": manifest_digest,
        "config_digest": config_digest,
    }
    if marker.exists():
        if marker.is_symlink() or marker.parent.is_symlink():
            raise ValueError("build marker must not be a symbolic link")
        try:
            saved = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(
                "invalid build marker; use a separate output path"
            ) from error
        if saved != identity:
            raise ValueError("build identity differs; use a separate output path")
        return
    if output_path.exists() and any(output_path.iterdir()):
        raise ValueError("nonempty unmarked output; use a separate output path")
    if allow_initialize:
        _atomic_json(marker, identity)


def _completion_identity(
    objects: list[SourceObject],
    channels: list[str],
    target_size: int,
    manifest_hash: str,
    config_hash: str,
) -> dict[str, Any]:
    """Bind a promoted dataset to its year, sources, and schema."""
    return {
        "schema_version": 1,
        "year": objects[0].year,
        "sample_count": len(objects),
        "source_digest": manifest_digest(objects, {}),
        "manifest_digest": manifest_hash,
        "config_digest": config_hash,
        "channels": channels,
        "target_size": target_size,
        "complete": True,
    }


def _validate_finished_year(
    year_path: Path,
    output_path: Path,
    objects: list[SourceObject],
    channels: list[str],
    target_size: int,
    manifest_hash: str,
    config_hash: str,
    published: bool,
) -> None:
    """Read and validate checkpoint identity and every annual schema contract."""
    _validate_channels(channels)
    _validate_target_size(target_size)
    if not objects or any(obj.year != objects[0].year for obj in objects):
        raise ValueError("annual objects must be nonempty and from one year")
    if any(a.timestamp_ns >= b.timestamp_ns for a, b in zip(objects, objects[1:])):
        raise ValueError("annual manifest timestamps must be sorted and unique")
    checkpoint = output_path / ".checkpoints" / f"{objects[0].year}.sqlite3"
    if not checkpoint.is_file() or checkpoint.is_symlink():
        raise ValueError("completed year requires its checkpoint")
    try:
        with closing(
            sqlite3.connect(checkpoint.as_uri() + "?mode=ro", uri=True)
        ) as connection:
            identity = dict(connection.execute("SELECT key, value FROM metadata"))
            rows = connection.execute(
                "SELECT slot, key, timestamp_ns, size, etag, state FROM samples ORDER BY slot"
            ).fetchall()
    except sqlite3.Error as error:
        raise ValueError("invalid annual checkpoint") from error
    expected_identity = _checkpoint_identity(
        objects, channels, target_size, output_path, manifest_hash, config_hash
    )
    expected_rows = [
        (slot, obj.key, obj.timestamp_ns, obj.size, obj.etag, "complete")
        for slot, obj in enumerate(objects)
    ]
    if identity != expected_identity or rows != expected_rows:
        raise ValueError("annual checkpoint identity differs or samples are incomplete")
    dataset_path = year_path / "dataset"
    if year_path.is_symlink() or dataset_path.is_symlink():
        raise ValueError("annual dataset must not be a symbolic link")
    dataset = zarr.open_group(str(dataset_path), mode="r")
    schemas = {
        "images": (
            (len(objects), 13, target_size, target_size),
            (50, 1, target_size, target_size),
            "float32",
            {"_ARRAY_DIMENSIONS": ["time", "channel", "y", "x"], "channels": channels},
        ),
        "time": (
            (len(objects),),
            (50,),
            "int64",
            {
                "_ARRAY_DIMENSIONS": ["time"],
                "units": "nanoseconds since 1970-01-01",
                "calendar": "proleptic_gregorian",
                "timezone": "UTC",
            },
        ),
    }
    if set(dataset.keys()) != set(schemas):
        raise ValueError("annual dataset must contain exactly images and time")
    for name, (shape, chunks, dtype, attributes) in schemas.items():
        array = dataset[name]
        if (
            not isinstance(array, zarr.Array)
            or array.shape != shape
            or array.chunks != chunks
            or array.dtype != np.dtype(dtype)
            or array.compressor != Blosc(cname="lz4", clevel=5, shuffle=1)
            or any(array.attrs.get(key) != value for key, value in attributes.items())
        ):
            raise ValueError(f"annual {name} schema differs")
    if not np.array_equal(dataset["time"][:], [obj.timestamp_ns for obj in objects]):
        raise ValueError("annual timestamps differ from the manifest")
    if published:
        expected = _completion_identity(
            objects, channels, target_size, manifest_hash, config_hash
        )
        if dataset.attrs.get("completion") != expected:
            raise ValueError("published year completion identity differs")
        consolidated = zarr.open_consolidated(str(dataset_path), mode="r")
        if consolidated.attrs.get("completion") != expected:
            raise ValueError("published consolidated completion identity differs")
        if set(consolidated.keys()) != set(dataset.keys()):
            raise ValueError("published consolidated array schema differs")
        for name in schemas:
            actual = dataset[name]
            saved = consolidated[name]
            if (
                not isinstance(saved, zarr.Array)
                or saved.shape != actual.shape
                or saved.chunks != actual.chunks
                or saved.dtype != actual.dtype
                or saved.compressor != actual.compressor
                or dict(saved.attrs) != dict(actual.attrs)
            ):
                raise ValueError("published consolidated array schema differs")


def publish_year(
    stage: Path,
    final: Path,
    objects: list[SourceObject],
    channels: list[str],
    target_size: int,
    manifest_digest: str,
    config_digest: str,
) -> None:
    """Validate a complete staging year and atomically promote its annual group.

    Args:
        stage: Year directory inside the output root's ``.staging`` directory.
        final: Final year directory directly inside the same output root.
        objects: Frozen annual source objects in timestamp order.
        channels: Ordered source channels.
        target_size: Positive square image resolution bound to the checkpoint.
        manifest_digest: Frozen global manifest identity.
        config_digest: Immutable configuration identity.

    Raises:
        ValueError: If identity, completion, schema, or destination is invalid.
    """
    output = final.parent.resolve()
    if (
        stage.is_symlink()
        or final.is_symlink()
        or stage.parent.is_symlink()
        or stage.parent.resolve() != output / ".staging"
        or not objects
        or stage.name != str(objects[0].year)
        or final.name != stage.name
        or final.exists()
    ):
        raise ValueError("promotion requires a separate unused annual destination")
    _validate_finished_year(
        stage,
        output,
        objects,
        channels,
        target_size,
        manifest_digest,
        config_digest,
        False,
    )
    if stage.stat().st_dev != output.stat().st_dev:
        raise ValueError("staging and output must be on the same filesystem")
    dataset = zarr.open_group(str(stage / "dataset"), mode="a")
    dataset.attrs["completion"] = _completion_identity(
        objects, channels, target_size, manifest_digest, config_digest
    )
    zarr.consolidate_metadata(str(stage / "dataset"))
    os.rename(stage, final)


def _validate_target_size(target_size: int) -> None:
    """Require a positive integral resolution, excluding booleans."""
    if (
        isinstance(target_size, bool)
        or not isinstance(target_size, int)
        or target_size < 1
    ):
        raise ValueError("target_size must be a positive integer")


def _validate_worker_limits(
    download_workers: int, transform_workers: int, max_in_flight_samples: int
) -> None:
    """Reject invalid operational concurrency limits before any build writes."""
    for name, value in {
        "download_workers": download_workers,
        "transform_workers": transform_workers,
        "max_in_flight_samples": max_in_flight_samples,
    }.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")


def _config_identity(cfg: DictConfig) -> dict[str, Any]:
    """Validate immutable output settings and omit operational retry settings."""
    _validate_target_size(cfg.target_size)
    _validate_worker_limits(
        cfg.download_workers, cfg.transform_workers, cfg.max_in_flight_samples
    )
    required = {
        "target_size": cfg.target_size,
        "dtype": "float32",
        "resize_method": "INTER_AREA",
        "chunks": {"images": [50, 1, cfg.target_size, cfg.target_size], "time": [50]},
        "compression": {"codec": "lz4", "clevel": 5, "shuffle": 1},
        "time_units": "nanoseconds since 1970-01-01",
        "time_calendar": "proleptic_gregorian",
        "time_timezone": "UTC",
    }
    config = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(config, dict):
        raise ValueError("build configuration must be a mapping")
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"{key} must match the approved annual schema")
    _validate_channels(list(cfg.channels))
    if int(cfg.start_year) > int(cfg.end_year):
        raise ValueError("start_year must not exceed end_year")
    for key in ("retry_count", "request_timeout_seconds", "download_timeout_seconds"):
        if int(config[key]) < 1:
            raise ValueError(f"{key} must be positive")
    return {
        key: config[key]
        for key in [
            "bucket",
            "endpoint_url",
            "prefix",
            "start_year",
            "end_year",
            "channels",
            *required,
        ]
    }


def run(cfg: DictConfig) -> None:
    """List a frozen manifest, report a dry run, or resume and publish each year.

    Args:
        cfg: Hydra build configuration; writing requires an explicit output path
            and ``dry_run=false``.

    Raises:
        ValueError: If configuration or global output identity is unsafe.
        RuntimeError: If any year remains incomplete after processing later years.
    """
    identity = _config_identity(cfg)
    raw_output = cfg.output_path
    if raw_output is not None and (
        not isinstance(raw_output, str) or not raw_output.strip()
    ):
        raise ValueError("output_path must be an explicit nonempty path")
    if not isinstance(cfg.dry_run, bool):
        raise ValueError("dry_run must be a boolean")
    if not cfg.dry_run and raw_output is None:
        raise ValueError("output_path is required before writes")
    if raw_output is not None and Path(raw_output).expanduser().is_symlink():
        raise ValueError("output_path must not be a symbolic link")
    output = Path(raw_output).expanduser().resolve() if raw_output is not None else None
    objects = list_source_objects(
        bucket=str(cfg.bucket),
        endpoint_url=str(cfg.endpoint_url),
        prefix=str(cfg.prefix),
        start_year=int(cfg.start_year),
        end_year=int(cfg.end_year),
        timeout_seconds=int(cfg.request_timeout_seconds),
    )
    manifest_hash = manifest_digest(objects, identity)
    config_hash = manifest_digest([], identity)
    years: dict[int, list[SourceObject]] = {}
    for obj in objects:
        years.setdefault(obj.year, []).append(obj)
    utc_range = (
        [
            datetime.fromtimestamp(
                obj.timestamp_ns / 1_000_000_000, timezone.utc
            ).isoformat()
            for obj in (objects[0], objects[-1])
        ]
        if objects
        else []
    )
    logger.info(
        "Build summary: {}",
        json.dumps(
            {
                "dry_run": cfg.dry_run,
                "candidate_count_by_year": {
                    year: len(rows) for year, rows in years.items()
                },
                "utc_range": utc_range,
                "manifest_digest": manifest_hash,
                "output_path": str(output) if output is not None else None,
                "estimated_uncompressed_image_bytes": len(objects)
                * len(cfg.channels)
                * int(cfg.target_size) ** 2
                * np.dtype(str(cfg.dtype)).itemsize,
            },
            sort_keys=True,
        ),
    )
    if cfg.dry_run:
        return
    assert output is not None
    validate_output_root(output, manifest_hash, config_hash, allow_initialize=True)
    _atomic_json(
        output / ".checkpoints" / "manifest.json",
        {
            "manifest_digest": manifest_hash,
            "config_digest": config_hash,
            "config_identity": identity,
            "objects": [asdict(obj) for obj in objects],
        },
    )
    failures: list[int] = []
    for year, annual in sorted(years.items()):
        final = output / str(year)
        try:
            if final.exists() or final.is_symlink():
                _validate_finished_year(
                    final,
                    output,
                    annual,
                    list(cfg.channels),
                    int(cfg.target_size),
                    manifest_hash,
                    config_hash,
                    True,
                )
                logger.info("Year {} already promoted and validated; skipping", year)
                continue
            stage = output / ".staging" / str(year)
            checkpoint = output / ".checkpoints" / f"{year}.sqlite3"
            if (
                stage.parent.is_symlink()
                or stage.is_symlink()
                or checkpoint.is_symlink()
            ):
                raise ValueError(
                    "staging and checkpoint paths must not be symbolic links"
                )
            process_year(
                year,
                annual,
                str(cfg.bucket),
                str(cfg.endpoint_url),
                output,
                output / ".checkpoints",
                Path(str(cfg.temporary_directory)).expanduser().resolve(),
                list(cfg.channels),
                int(cfg.target_size),
                int(cfg.download_timeout_seconds),
                int(cfg.retry_count),
                manifest_hash,
                config_hash,
                download_workers=int(cfg.download_workers),
                transform_workers=int(cfg.transform_workers),
                max_in_flight_samples=int(cfg.max_in_flight_samples),
            )
            publish_year(
                output / ".staging" / str(year),
                final,
                annual,
                list(cfg.channels),
                int(cfg.target_size),
                manifest_hash,
                config_hash,
            )
            logger.info("Year {} promoted ({} samples)", year, len(annual))
        except Exception as error:
            failures.append(year)
            logger.error("Year {} remains incomplete: {}", year, error)
    if failures:
        raise RuntimeError(f"incomplete years: {failures}; resume the same build")


@hydra.main(
    version_base=None,
    config_path="../../configs/preprocessing",
    config_name="rebuild_surya_bench_zarr",
)
def main(cfg: DictConfig) -> None:
    """Run the configured annual rebuild from the command line."""
    run(cfg)


if __name__ == "__main__":
    main()
