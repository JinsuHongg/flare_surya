from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr
from omegaconf import DictConfig

import scripts.preprocessing.rebuild_surya_bench_zarr as rebuild
from scripts.preprocessing.rebuild_surya_bench_zarr import (
    SourceObject,
    list_source_objects,
    manifest_digest,
    parse_source_key,
)


@pytest.fixture(autouse=True)
def _block_real_http(monkeypatch: pytest.MonkeyPatch) -> None:
    """Require every external request in this file to use an explicit fake."""

    def unexpected_request(*args: object, **kwargs: object) -> None:
        pytest.fail("real HTTP is prohibited in rebuild tests")

    monkeypatch.setattr(rebuild, "urlopen", unexpected_request)


def test_parse_exact_hourly_key_as_utc() -> None:
    result = parse_source_key("surya/20190102_0300.nc")

    assert result == SourceObject(
        key="surya/20190102_0300.nc",
        year=2019,
        timestamp_ns=1546398000000000000,
        size=0,
        etag="",
    )


@pytest.mark.parametrize(
    "key",
    [
        "surya/20190102_0301.nc",
        "surya/20190102.nc",
        "surya/20190102_0300.nc.tmp",
        "surya/20190102_0300.nc/extra",
        "surya/20190230_0300.nc",
    ],
)
def test_parse_ignores_keys_outside_exact_hourly_pattern(key: str) -> None:
    assert parse_source_key(key) is None


def _response(
    contents: list[tuple[str, int, str]], *, truncated: bool, token: str | None = None
) -> bytes:
    rows = "".join(
        f"<Contents><Key>{key}</Key><LastModified>2020-01-01T00:00:00.000Z</LastModified>"
        f"<ETag>{etag}</ETag><Size>{size}</Size></Contents>"
        for key, size, etag in contents
    )
    next_token = (
        f"<NextContinuationToken>{token}</NextContinuationToken>" if token else ""
    )
    return (
        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        f"<IsTruncated>{str(truncated).lower()}</IsTruncated>{rows}{next_token}</ListBucketResult>"
    ).encode()


class _HTTPResponse:
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self) -> _HTTPResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def test_listing_filters_years_and_paginates(monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.preprocessing.rebuild_surya_bench_zarr as module

    requests: list[str] = []
    payloads = [
        _response(
            [
                ("root/20181231_2300.nc", 10, '"a"'),
                ("root/20190101_0000.nc", 11, '"b"'),
                ("root/20190101_0001.nc", 12, '"ignored"'),
            ],
            truncated=True,
            token="continue-here",
        ),
        _response(
            [
                ("root/20201231_2300.nc", 13, '"c"'),
                ("root/20210101_0000.nc", 14, '"d"'),
            ],
            truncated=False,
        ),
    ]

    def fake_urlopen(request: object, timeout: int) -> _HTTPResponse:
        requests.append(request.full_url)  # type: ignore[attr-defined]
        assert timeout == 7
        return _HTTPResponse(payloads.pop(0))

    monkeypatch.setattr(module, "urlopen", fake_urlopen)
    objects = list_source_objects(
        "bucket", "https://s3.example", "root/", 2019, 2020, 7
    )

    assert len(requests) == 2
    assert "continuation-token=continue-here" in requests[1]
    assert [obj.key for obj in objects] == [
        "root/20190101_0000.nc",
        "root/20201231_2300.nc",
    ]
    assert [(obj.size, obj.etag) for obj in objects] == [(11, '"b"'), (13, '"c"')]


def test_listing_rejects_duplicate_accepted_timestamps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.preprocessing.rebuild_surya_bench_zarr as module

    body = _response(
        [
            ("root/a/20190101_0000.nc", 1, '"a"'),
            ("root/b/20190101_0000.nc", 2, '"b"'),
        ],
        truncated=False,
    )
    monkeypatch.setattr(module, "urlopen", lambda request, timeout: _HTTPResponse(body))

    with pytest.raises(ValueError, match="duplicate timestamp"):
        list_source_objects("bucket", "https://s3.example", "root/", 2019, 2019, 7)


def test_manifest_digest_is_stable_and_covers_identities_and_config() -> None:
    objects = [SourceObject("a.nc", 2019, 1, 10, '"etag"')]
    settings = {"channels": ["aia94", "aia131"], "size": 224}
    expected = hashlib.sha256(
        json.dumps(
            {
                "config_identity": settings,
                "objects": [
                    {
                        "key": "a.nc",
                        "year": 2019,
                        "timestamp_ns": 1,
                        "size": 10,
                        "etag": '"etag"',
                    }
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()

    assert manifest_digest(objects, settings) == expected
    assert manifest_digest(list(reversed(objects)), settings) == expected


# Local fixtures cover the transform and durable-write behavior without S3 access.
CHANNELS = [
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


def _netcdf(
    path: Path, *, omit: str | None = None, extra: int = 1, mismatch: bool = False
) -> Path:
    variables = {}
    for index, channel in enumerate(CHANNELS):
        if channel == omit:
            continue
        shape = (extra, 4, 4)
        dims = ("sample", "y", "x")
        if mismatch and channel == "hmi_v":
            shape = (extra, 3, 4)
            dims = ("sample", "other_y", "x")
        values = (
            np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + index * 100
        )
        variables[channel] = (dims, values)
    xr.Dataset(variables).to_netcdf(path, engine="h5netcdf")
    return path


def test_resize_preserves_all_channel_order_and_uses_area(tmp_path: Path) -> None:
    path = _netcdf(tmp_path / "source.nc")
    result = rebuild.resize_netcdf(path, list(reversed(CHANNELS)), 224)
    assert result.shape == (13, 224, 224)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result[:, 0, 0], np.arange(12, -1, -1) * 100)
    # Area reduction of the four corners is the mean, not the nearest pixel.
    reduced = rebuild.resize_netcdf(path, CHANNELS, 2)
    np.testing.assert_array_equal(reduced[0], [[2.5, 4.5], [10.5, 12.5]])


@pytest.mark.parametrize(
    "fixture, message",
    [
        ({"omit": "hmi_v"}, "missing"),
        ({"extra": 2}, "2-D"),
        ({"mismatch": True}, "shape"),
    ],
)
def test_resize_rejects_invalid_channels_and_shapes(
    tmp_path: Path, fixture: dict[str, object], message: str
) -> None:
    path = _netcdf(tmp_path / "source.nc", **fixture)
    with pytest.raises(ValueError, match=message):
        rebuild.resize_netcdf(path, CHANNELS, 224)


def test_resize_requires_all_thirteen_unique_channel_names(tmp_path: Path) -> None:
    path = _netcdf(tmp_path / "source.nc")
    with pytest.raises(ValueError, match="13"):
        rebuild.resize_netcdf(path, CHANNELS[:-1] + [CHANNELS[0]], 224)


class _DownloadResponse(BytesIO):
    def __init__(self, body: bytes, etag: str = '"a"') -> None:
        super().__init__(body)
        self.headers = {"ETag": etag}

    def read(self, size: int = -1) -> bytes:
        assert size > 0, "downloads must use bounded streaming reads"
        return super().read(size)


def test_download_streams_retries_and_cleans_partial_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = SourceObject("root/a b/20190101_0000.nc", 2019, 1, 4, '"a"')
    bodies = [b"bad", b"good"]

    def respond(request: object, timeout: int) -> _DownloadResponse:
        assert (
            request.full_url == "https://s3.example/bucket/root/a%20b/20190101_0000.nc"
        )
        assert timeout == 7
        return _DownloadResponse(bodies.pop(0))

    monkeypatch.setattr(rebuild, "urlopen", respond)
    result = rebuild.download_object(
        source, "bucket", "https://s3.example", tmp_path, 7, 2
    )
    assert result.read_bytes() == b"good"
    assert list(tmp_path.iterdir()) == [result]


def test_download_exhaustion_cleans_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        rebuild, "urlopen", lambda request, timeout: _DownloadResponse(b"bad")
    )
    source = SourceObject("20190101_0000.nc", 2019, 1, 4, '"a"')
    with pytest.raises(OSError, match="download"):
        rebuild.download_object(source, "bucket", "https://s3.example", tmp_path, 7, 2)
    assert not list(tmp_path.iterdir())


def test_download_rejects_changed_etag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        rebuild, "urlopen", lambda request, timeout: _DownloadResponse(b"good", '"b"')
    )
    source = SourceObject("20190101_0000.nc", 2019, 1, 4, '"a"')
    with pytest.raises(OSError, match="download"):
        rebuild.download_object(source, "bucket", "https://s3.example", tmp_path, 7, 1)
    assert not list(tmp_path.iterdir())


def _year_args(tmp_path: Path) -> dict[str, object]:
    return dict(
        year=2019,
        objects=[SourceObject("20190101_0000.nc", 2019, 1546300800000000000, 1, '"a"')],
        bucket="bucket",
        endpoint_url="https://s3.example",
        output_path=tmp_path / "output",
        checkpoint_root=tmp_path / "output" / ".checkpoints",
        temp_dir=tmp_path / "temp",
        channels=CHANNELS,
        target_size=224,
        timeout_seconds=7,
        attempts=2,
        manifest_digest="manifest-a",
        config_digest="config-a",
    )


def _local_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[str]:
    fixture = _netcdf(tmp_path / "fixture.nc")
    downloaded: list[str] = []

    def download(
        source: SourceObject,
        bucket: str,
        endpoint_url: str,
        temp_dir: Path,
        timeout_seconds: int,
        attempts: int,
    ) -> Path:
        downloaded.append(source.key)
        temp_dir.mkdir(parents=True, exist_ok=True)
        destination = temp_dir / "download.nc"
        shutil.copyfile(fixture, destination)
        return destination

    monkeypatch.setattr(rebuild, "download_object", download, raising=False)
    return downloaded


def _state(tmp_path: Path) -> list[tuple[str, str | None]]:
    with sqlite3.connect(
        tmp_path / "output" / ".checkpoints" / "2019.sqlite3"
    ) as connection:
        return connection.execute(
            "SELECT state, error FROM samples ORDER BY slot"
        ).fetchall()


def test_process_year_writes_utc_schema_and_skips_completed_slots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    downloaded = _local_download(monkeypatch, tmp_path)
    args = _year_args(tmp_path)
    rebuild.process_year(**args)
    dataset = zarr.open_group(
        str(tmp_path / "output" / ".staging" / "2019" / "dataset"), mode="r"
    )
    assert sorted(dataset.array_keys()) == ["images", "time"]
    assert dataset["images"].shape == (1, 13, 224, 224)
    assert dataset["images"].dtype == np.float32
    assert dataset["images"].chunks == (50, 1, 224, 224)
    assert dataset["time"].chunks == (50,)
    assert dataset["time"].dtype == np.int64
    assert dataset["time"][:].tolist() == [1546300800000000000]
    assert dataset["time"].attrs["units"] == "nanoseconds since 1970-01-01"
    assert dataset["time"].attrs["calendar"] == "proleptic_gregorian"
    assert dataset["time"].attrs["timezone"] == "UTC"
    assert dataset["images"].attrs["_ARRAY_DIMENSIONS"] == ["time", "channel", "y", "x"]
    assert dataset["images"].attrs["channels"] == CHANNELS
    assert dataset["images"].compressor.get_config() == {
        "id": "blosc",
        "cname": "lz4",
        "clevel": 5,
        "shuffle": 1,
        "blocksize": 0,
    }
    assert _state(tmp_path) == [("complete", None)]
    assert not list((tmp_path / "temp").iterdir())
    assert not (tmp_path / "output" / "2019").exists()
    rebuild.process_year(**args)
    assert downloaded == ["20190101_0000.nc"]


@pytest.mark.parametrize("interrupt_at", ["images", "time", "after_time"])
def test_process_year_resumes_interrupted_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupt_at: str
) -> None:
    downloaded = _local_download(monkeypatch, tmp_path)
    original = zarr.Array.__setitem__

    def interrupted(array: zarr.Array, selection: object, value: object) -> None:
        if array.basename == interrupt_at:
            raise KeyboardInterrupt("walltime")
        original(array, selection, value)
        if array.basename == "time" and interrupt_at == "after_time":
            raise KeyboardInterrupt("walltime")

    monkeypatch.setattr(zarr.Array, "__setitem__", interrupted)
    with pytest.raises(KeyboardInterrupt):
        rebuild.process_year(**_year_args(tmp_path))
    assert _state(tmp_path) == [("writing", None)]
    assert not list((tmp_path / "temp").iterdir())
    dataset = zarr.open_group(
        str(tmp_path / "output" / ".staging" / "2019" / "dataset"), mode="r"
    )
    if interrupt_at == "after_time":
        assert dataset["time"][:].tolist() == [1546300800000000000]
    else:
        assert dataset["time"][:].tolist() == [0]
    if interrupt_at in {"time", "after_time"}:
        np.testing.assert_array_equal(
            dataset["images"][0, :, 0, 0], np.arange(13) * 100
        )
    monkeypatch.setattr(zarr.Array, "__setitem__", original)
    rebuild.process_year(**_year_args(tmp_path))
    assert _state(tmp_path) == [("complete", None)]
    assert len(downloaded) == 2
    time = zarr.open_array(
        str(tmp_path / "output" / ".staging" / "2019" / "dataset" / "time")
    )
    assert time[:].tolist() == [1546300800000000000]


@pytest.mark.parametrize("field", ["manifest_digest", "config_digest"])
def test_process_year_refuses_changed_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    downloaded = _local_download(monkeypatch, tmp_path)
    args = _year_args(tmp_path)
    rebuild.process_year(**args)
    args[field] = "changed"
    with pytest.raises(ValueError, match="identity"):
        rebuild.process_year(**args)
    assert len(downloaded) == 1


def test_process_year_refuses_corrupt_complete_timestamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _local_download(monkeypatch, tmp_path)
    args = _year_args(tmp_path)
    rebuild.process_year(**args)
    time = zarr.open_array(
        str(tmp_path / "output" / ".staging" / "2019" / "dataset" / "time")
    )
    time[0] = 0
    with pytest.raises(ValueError, match="timestamp"):
        rebuild.process_year(**args)


def test_process_year_retains_failed_state_and_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: object, **kwargs: object) -> Path:
        raise OSError("source unavailable")

    monkeypatch.setattr(rebuild, "download_object", fail, raising=False)
    with pytest.raises(RuntimeError, match="incomplete"):
        rebuild.process_year(**_year_args(tmp_path))
    assert _state(tmp_path) == [("failed", "OSError: source unavailable")]
    time = zarr.open_array(
        str(tmp_path / "output" / ".staging" / "2019" / "dataset" / "time")
    )
    assert time[:].tolist() == [0]
    assert not (tmp_path / "output" / "2019").exists()
    _local_download(monkeypatch, tmp_path)
    rebuild.process_year(**_year_args(tmp_path))
    assert _state(tmp_path) == [("complete", None)]


def test_resize_does_not_squeeze_spatial_axis_to_hide_extra_samples(
    tmp_path: Path,
) -> None:
    values = np.ones((2, 1, 4), dtype=np.float32)
    path = tmp_path / "source.nc"
    xr.Dataset({name: (("sample", "y", "x"), values) for name in CHANNELS}).to_netcdf(
        path, engine="h5netcdf"
    )
    with pytest.raises(ValueError, match="2-D"):
        rebuild.resize_netcdf(path, CHANNELS, 224)


def test_process_year_continues_after_failed_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _local_download(monkeypatch, tmp_path)
    local_download = rebuild.download_object

    def download(source: SourceObject, *args: object) -> Path:
        if source.key == "20190101_0000.nc":
            raise OSError("unavailable")
        return local_download(source, *args)

    monkeypatch.setattr(rebuild, "download_object", download)
    args = _year_args(tmp_path)
    args["objects"] = [
        SourceObject("20190101_0000.nc", 2019, 1546300800000000000, 1, '"a"'),
        SourceObject("20190101_0100.nc", 2019, 1546304400000000000, 1, '"b"'),
    ]
    with pytest.raises(RuntimeError, match="incomplete"):
        rebuild.process_year(**args)
    assert _state(tmp_path) == [("failed", "OSError: unavailable"), ("complete", None)]
    time = zarr.open_array(
        str(tmp_path / "output" / ".staging" / "2019" / "dataset" / "time")
    )
    assert time[:].tolist() == [0, 1546304400000000000]


def test_process_year_cannot_resume_new_sources_even_with_same_caller_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _local_download(monkeypatch, tmp_path)
    args = _year_args(tmp_path)
    rebuild.process_year(**args)
    args["objects"] = [
        SourceObject("20190101_0000.nc", 2019, 1546300800000000000, 2, '"b"')
    ]
    with pytest.raises(ValueError, match="identity"):
        rebuild.process_year(**args)


def test_process_year_refuses_staging_without_checkpoint(tmp_path: Path) -> None:
    stage = tmp_path / "output" / ".staging" / "2019"
    stage.mkdir(parents=True)
    sentinel = stage / "existing.txt"
    sentinel.write_text("preserve")
    with pytest.raises(ValueError, match="without a checkpoint"):
        rebuild.process_year(**_year_args(tmp_path))
    assert sentinel.read_text() == "preserve"


def test_process_year_refuses_incompatible_array_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _local_download(monkeypatch, tmp_path)
    args = _year_args(tmp_path)
    rebuild.process_year(**args)
    time = zarr.open_array(
        str(tmp_path / "output" / ".staging" / "2019" / "dataset" / "time")
    )
    time.attrs["timezone"] = "UTC+08:00"
    with pytest.raises(ValueError, match="schema"):
        rebuild.process_year(**args)


def test_process_year_resumes_interrupted_array_initialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _local_download(monkeypatch, tmp_path)
    original = zarr.attrs.Attributes.update

    def interrupt(
        attributes: zarr.attrs.Attributes, *args: object, **kwargs: object
    ) -> None:
        raise KeyboardInterrupt("initialization walltime")

    monkeypatch.setattr(zarr.attrs.Attributes, "update", interrupt)
    with pytest.raises(KeyboardInterrupt):
        rebuild.process_year(**_year_args(tmp_path))
    assert _state(tmp_path) == [("pending", None)]
    monkeypatch.setattr(zarr.attrs.Attributes, "update", original)
    rebuild.process_year(**_year_args(tmp_path))
    assert _state(tmp_path) == [("complete", None)]


def test_process_year_recovers_empty_checkpoint_before_initialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _local_download(monkeypatch, tmp_path)
    checkpoint = tmp_path / "output" / ".checkpoints" / "2019.sqlite3"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    rebuild.process_year(**_year_args(tmp_path))
    assert _state(tmp_path) == [("complete", None)]


def test_rebuild_module_registers_source_lz4_hdf5_filter() -> None:
    # A fresh process prevents another reader's plugin import from masking the
    # registration required to decode the public NetCDF files.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import scripts.preprocessing.rebuild_surya_bench_zarr; "
            "import h5py; "
            "assert h5py.h5z.filter_avail(32004), 'source LZ4 HDF5 filter is unavailable'",
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def _run_config(tmp_path: Path, *, dry_run: bool = False) -> DictConfig:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(
        Path(__file__).resolve().parents[2]
        / "configs/preprocessing/rebuild_surya_bench_zarr.yaml"
    )
    cfg.output_path = str(tmp_path / "output")
    cfg.temporary_directory = str(tmp_path / "temp")
    cfg.dry_run = dry_run
    return cfg


def _mock_listing(monkeypatch: pytest.MonkeyPatch, objects: list[SourceObject]) -> None:
    monkeypatch.setattr(rebuild, "list_source_objects", lambda **kwargs: objects)


def test_root_refuses_unmarked_existing_data_without_writes(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "2019").mkdir()
    with pytest.raises(ValueError, match="unmarked|marker"):
        rebuild.validate_output_root(output, "m", "c", True)
    assert sorted(path.name for path in output.iterdir()) == ["2019"]


def test_root_marker_initialization_and_exact_resume(tmp_path: Path) -> None:
    output = tmp_path / "output"
    rebuild.validate_output_root(output, "m", "c", False)
    assert not output.exists()
    rebuild.validate_output_root(output, "m", "c", True)
    marker = output / ".checkpoints" / "build.json"
    before = marker.read_bytes()
    rebuild.validate_output_root(output, "m", "c", True)
    assert marker.read_bytes() == before
    with pytest.raises(ValueError, match="identity|differs"):
        rebuild.validate_output_root(output, "changed", "c", True)
    assert marker.read_bytes() == before


def test_dry_run_reports_summary_without_downloads_or_writes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    objects = _year_args(tmp_path)["objects"]
    _mock_listing(monkeypatch, objects)
    messages: list[str] = []
    sink = rebuild.logger.add(lambda message: messages.append(str(message)))
    try:
        rebuild.run(_run_config(tmp_path, dry_run=True))
    finally:
        rebuild.logger.remove(sink)
    assert not list(tmp_path.iterdir())
    summary = "".join(messages)
    for fragment in ["2019", "2019-01-01", "2609152", str(tmp_path / "output")]:
        assert fragment in summary
    assert "manifest" in summary


def test_run_requires_explicit_output_before_listing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = _run_config(tmp_path)
    cfg.output_path = None
    monkeypatch.setattr(
        rebuild,
        "list_source_objects",
        lambda **kwargs: pytest.fail("unexpected listing"),
    )
    with pytest.raises(ValueError, match="output_path"):
        rebuild.run(cfg)
    assert not list(tmp_path.iterdir())


def test_publish_complete_year_consolidates_and_keeps_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _local_download(monkeypatch, tmp_path)
    args = _year_args(tmp_path)
    rebuild.process_year(**args)
    stage = tmp_path / "output" / ".staging" / "2019"
    final = tmp_path / "output" / "2019"
    rebuild.publish_year(
        stage, final, args["objects"], CHANNELS, 224, "manifest-a", "config-a"
    )
    assert not stage.exists()
    dataset = zarr.open_consolidated(str(final / "dataset"), mode="r")
    assert set(dataset.array_keys()) == {"images", "time"}
    assert dataset["time"][:].tolist() == [1546300800000000000]
    assert dataset.attrs["completion"]["manifest_digest"] == "manifest-a"
    assert (tmp_path / "output" / ".checkpoints" / "2019.sqlite3").exists()


@pytest.mark.parametrize(
    "corruption", ["timestamp", "checkpoint", "channels", "compressor"]
)
def test_publish_refuses_corrupt_year_without_promotion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, corruption: str
) -> None:
    _local_download(monkeypatch, tmp_path)
    args = _year_args(tmp_path)
    rebuild.process_year(**args)
    stage = tmp_path / "output" / ".staging" / "2019"
    dataset = zarr.open_group(str(stage / "dataset"), mode="a")
    if corruption == "timestamp":
        dataset["time"][0] = 1
    elif corruption == "channels":
        dataset["images"].attrs["channels"] = list(reversed(CHANNELS))
    elif corruption == "compressor":
        metadata = stage / "dataset" / "images" / ".zarray"
        contents = json.loads(metadata.read_text())
        contents["compressor"]["clevel"] = 1
        metadata.write_text(json.dumps(contents))
    else:
        with sqlite3.connect(
            tmp_path / "output" / ".checkpoints" / "2019.sqlite3"
        ) as db:
            db.execute("UPDATE samples SET state='writing'")
    final = tmp_path / "output" / "2019"
    with pytest.raises(ValueError):
        rebuild.publish_year(
            stage, final, args["objects"], CHANNELS, 224, "manifest-a", "config-a"
        )
    assert stage.exists()
    assert not final.exists()
    assert "completion" not in dataset.attrs


def test_run_skips_only_valid_promoted_year(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _local_download(monkeypatch, tmp_path)
    _mock_listing(monkeypatch, _year_args(tmp_path)["objects"])
    cfg = _run_config(tmp_path)
    rebuild.run(cfg)
    final = tmp_path / "output" / "2019" / "dataset"
    before = {
        str(path.relative_to(final)): path.read_bytes()
        for path in final.rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(
        rebuild, "download_object", lambda *args: pytest.fail("download on resume")
    )
    rebuild.run(cfg)
    assert before == {
        str(path.relative_to(final)): path.read_bytes()
        for path in final.rglob("*")
        if path.is_file()
    }
    dataset = zarr.open_group(str(final), mode="a")
    dataset.attrs["completion"] = {"manifest_digest": "wrong"}
    with pytest.raises(RuntimeError, match="2019"):
        rebuild.run(cfg)


def test_run_continues_later_year_and_exits_nonzero_on_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _local_download(monkeypatch, tmp_path)
    original_download = rebuild.download_object
    objects = _year_args(tmp_path)["objects"] + [
        SourceObject("20200101_0000.nc", 2020, 1577836800000000000, 1, '"b"')
    ]
    _mock_listing(monkeypatch, objects)

    def download(source: SourceObject, *args: object) -> Path:
        if source.year == 2019:
            raise RuntimeError("synthetic failure")
        return original_download(source, *args)

    monkeypatch.setattr(rebuild, "download_object", download)
    with pytest.raises(RuntimeError, match="2019"):
        rebuild.run(_run_config(tmp_path))
    assert not (tmp_path / "output" / "2019").exists()
    assert (tmp_path / "output" / "2020" / "dataset" / ".zmetadata").exists()


def test_run_refuses_changed_manifest_before_staging_writes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _local_download(monkeypatch, tmp_path)
    _mock_listing(monkeypatch, _year_args(tmp_path)["objects"])
    cfg = _run_config(tmp_path)
    rebuild.run(cfg)
    before = {
        str(path.relative_to(tmp_path / "output")): path.read_bytes()
        for path in (tmp_path / "output").rglob("*")
        if path.is_file()
    }
    _mock_listing(
        monkeypatch,
        [SourceObject("20190101_0000.nc", 2019, 1546300800000000000, 1, '"changed"')],
    )
    with pytest.raises(ValueError, match="identity"):
        rebuild.run(cfg)
    assert before == {
        str(path.relative_to(tmp_path / "output")): path.read_bytes()
        for path in (tmp_path / "output").rglob("*")
        if path.is_file()
    }


def test_run_refuses_output_symlink_before_writes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destination = tmp_path / "unrelated"
    destination.mkdir()
    (tmp_path / "output").symlink_to(destination, target_is_directory=True)
    _mock_listing(monkeypatch, _year_args(tmp_path)["objects"])
    monkeypatch.setattr(
        rebuild, "download_object", lambda *args: pytest.fail("unexpected download")
    )
    with pytest.raises(ValueError, match="symbolic link"):
        rebuild.run(_run_config(tmp_path))
    assert not list(destination.iterdir())


@pytest.mark.parametrize("size", [224, 512])
def test_run_resolution_controls_published_shape_chunks_and_resume(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, size: int
) -> None:
    _local_download(monkeypatch, tmp_path)
    _mock_listing(monkeypatch, _year_args(tmp_path)["objects"])
    cfg = _run_config(tmp_path)
    if size != 224:
        cfg.target_size = size
    rebuild.run(cfg)
    dataset = zarr.open_consolidated(
        str(tmp_path / "output" / "2019" / "dataset"), mode="r"
    )
    assert dataset["images"].shape == (1, 13, size, size)
    assert dataset["images"].chunks == (50, 1, size, size)
    assert dataset.attrs["completion"]["target_size"] == size
    cfg.target_size = 512 if size == 224 else 224
    with pytest.raises(ValueError, match="identity"):
        rebuild.run(cfg)


@pytest.mark.parametrize("size", [0, -1, 224.5, True])
def test_run_rejects_invalid_resolution_before_listing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, size: object
) -> None:
    cfg = _run_config(tmp_path)
    cfg.target_size = size
    monkeypatch.setattr(
        rebuild,
        "list_source_objects",
        lambda **kwargs: pytest.fail("unexpected listing"),
    )
    with pytest.raises(ValueError, match="target_size"):
        rebuild.run(cfg)
    assert not list(tmp_path.iterdir())


def test_completed_year_refuses_stale_consolidated_schema(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _local_download(monkeypatch, tmp_path)
    _mock_listing(monkeypatch, _year_args(tmp_path)["objects"])
    cfg = _run_config(tmp_path)
    rebuild.run(cfg)
    metadata = tmp_path / "output" / "2019" / "dataset" / ".zmetadata"
    contents = json.loads(metadata.read_text())
    contents["metadata"]["images/.zarray"]["shape"] = [1, 13, 1, 1]
    metadata.write_text(json.dumps(contents))
    with pytest.raises(RuntimeError, match="2019"):
        rebuild.run(cfg)


def test_marked_root_refuses_staging_symlink_before_external_writes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _local_download(monkeypatch, tmp_path)
    _mock_listing(monkeypatch, _year_args(tmp_path)["objects"])
    cfg = _run_config(tmp_path)
    rebuild.run(cfg)
    # Switch the completed year to a new, compatible unfinished annual slot.
    shutil.rmtree(tmp_path / "output" / "2019")
    external = tmp_path / "external"
    external.mkdir()
    shutil.rmtree(tmp_path / "output" / ".staging")
    (tmp_path / "output" / ".staging").symlink_to(external, target_is_directory=True)
    with pytest.raises(RuntimeError, match="2019"):
        rebuild.run(cfg)
    assert not list(external.iterdir())


def test_hydra_cli_dry_run_512_creates_no_logs_or_output(tmp_path: Path) -> None:
    script = Path(rebuild.__file__).resolve()
    payload = _response([("20190101_0000.nc", 1, '"a"')], truncated=False)
    code = (
        "import io, runpy, sys, urllib.request\n"
        f"payload = {payload!r}\n"
        "urllib.request.urlopen = lambda *args, **kwargs: io.BytesIO(payload)\n"
        f"sys.argv = [{str(script)!r}, 'target_size=512', 'dry_run=true', "
        f"'output_path={tmp_path / 'output'}']\n"
        f"runpy.run_path({str(script)!r}, run_name='__main__')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "13631488" in result.stderr
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("linked_store", ["dataset", "images", "time"])
def test_staging_rejects_linked_stores_before_external_writes(
    tmp_path: Path, linked_store: str
) -> None:
    stage = tmp_path / "stage"
    external = tmp_path / "external"
    external.mkdir()
    (external / "sentinel").write_bytes(b"preserve")
    if linked_store == "dataset":
        stage.mkdir()
        (stage / "dataset").symlink_to(external, target_is_directory=True)
    else:
        zarr.open_group(str(stage), mode="a").require_group("dataset")
        (stage / "dataset" / linked_store).symlink_to(
            external, target_is_directory=True
        )
    before = {path.name: path.read_bytes() for path in external.iterdir()}
    with pytest.raises(ValueError, match="symbolic link"):
        rebuild._open_year_arrays(stage, 1, CHANNELS, True, 224)
    assert before == {path.name: path.read_bytes() for path in external.iterdir()}


def _pipeline_objects(count: int) -> list[SourceObject]:
    return [
        SourceObject(
            f"20190101_{slot:02d}00.nc",
            2019,
            1546300800000000000 + slot * 3600000000000,
            1,
            f'"{slot}"',
        )
        for slot in range(count)
    ]


def _pipeline_download(
    source: SourceObject,
    bucket: str,
    endpoint_url: str,
    temp_dir: Path,
    timeout_seconds: int,
    attempts: int,
) -> Path:
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / source.key
    path.write_bytes(b"local synthetic source")
    return path


def _pipeline_resize(path: Path, channels: list[str], target_size: int) -> np.ndarray:
    slot = int(path.stem.split("_")[1][:2])
    return np.full((13, target_size, target_size), slot + 1, dtype=np.float32)


def test_default_pipeline_overlaps_download_with_transform_and_bounds_temp(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import threading

    transform_started = threading.Event()
    download_overlapped = threading.Event()
    lock = threading.Lock()
    active_downloads = 0
    max_downloads = 0
    active_transforms = 0
    max_transforms = 0
    max_files = 0
    in_flight: set[str] = set()
    max_samples = 0
    original_write = zarr.Array.__setitem__

    def write(array: zarr.Array, slot: int, value: object) -> None:
        original_write(array, slot, value)
        if array.basename == "time":
            with lock:
                in_flight.remove(f"20190101_{slot:02d}00.nc")

    def download(source: SourceObject, *args: object) -> Path:
        nonlocal active_downloads, max_downloads, max_files, max_samples
        with lock:
            in_flight.add(source.key)
            max_samples = max(max_samples, len(in_flight))
            assert len(in_flight) <= 2
            active_downloads += 1
            max_downloads = max(max_downloads, active_downloads)
        try:
            if source.key == "20190101_0100.nc":
                assert transform_started.wait(5), "download did not overlap transform"
                download_overlapped.set()
            path = _pipeline_download(source, *args)
            with lock:
                max_files = max(max_files, len(list(path.parent.iterdir())))
            return path
        finally:
            with lock:
                active_downloads -= 1

    def resize(path: Path, channels: list[str], target_size: int) -> np.ndarray:
        nonlocal active_transforms, max_transforms
        with lock:
            active_transforms += 1
            max_transforms = max(max_transforms, active_transforms)
        try:
            if path.name == "20190101_0000.nc":
                transform_started.set()
                assert download_overlapped.wait(5), "transform blocked later download"
            return _pipeline_resize(path, channels, target_size)
        finally:
            with lock:
                active_transforms -= 1

    monkeypatch.setattr(rebuild, "download_object", download)
    monkeypatch.setattr(rebuild, "resize_netcdf", resize)
    monkeypatch.setattr(zarr.Array, "__setitem__", write)
    _mock_listing(monkeypatch, _pipeline_objects(5))
    rebuild.run(_run_config(tmp_path))
    assert download_overlapped.is_set()
    assert max_downloads <= 2
    assert max_transforms == 1
    assert max_files <= 2
    assert max_samples == 2
    assert not in_flight
    assert not list((tmp_path / "temp").iterdir())
    images = zarr.open_array(
        str(tmp_path / "output" / "2019" / "dataset" / "images"), mode="r"
    )
    np.testing.assert_array_equal(images[:, 0, 0, 0], [1, 2, 3, 4, 5])


def test_pipeline_out_of_order_results_use_stable_slots_and_one_owner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import threading

    owner = threading.get_ident()
    second_written = threading.Event()
    writes: list[tuple[str, int, int]] = []
    checkpoint_threads: list[int] = []
    original_write = zarr.Array.__setitem__
    original_checkpoint = rebuild._open_checkpoint

    def checkpoint(*args: object, **kwargs: object) -> sqlite3.Connection:
        connection = original_checkpoint(*args, **kwargs)
        connection.set_trace_callback(
            lambda sql: checkpoint_threads.append(threading.get_ident())
        )
        return connection

    def write(array: zarr.Array, slot: int, value: object) -> None:
        writes.append((array.basename, slot, threading.get_ident()))
        original_write(array, slot, value)
        if array.basename == "time" and slot == 1:
            second_written.set()

    def resize(path: Path, channels: list[str], target_size: int) -> np.ndarray:
        if path.name == "20190101_0000.nc":
            assert second_written.wait(5), "writer must consume completed later slots"
        return _pipeline_resize(path, channels, target_size)

    monkeypatch.setattr(rebuild, "download_object", _pipeline_download)
    monkeypatch.setattr(rebuild, "resize_netcdf", resize)
    monkeypatch.setattr(rebuild, "_open_checkpoint", checkpoint)
    monkeypatch.setattr(zarr.Array, "__setitem__", write)
    args = _year_args(tmp_path)
    args.update(
        objects=_pipeline_objects(2), transform_workers=2, max_in_flight_samples=2
    )
    rebuild.process_year(**args)
    assert writes == [
        ("images", 1, owner),
        ("time", 1, owner),
        ("images", 0, owner),
        ("time", 0, owner),
    ]
    assert checkpoint_threads and set(checkpoint_threads) == {owner}
    dataset = zarr.open_group(
        str(tmp_path / "output" / ".staging" / "2019" / "dataset"), mode="r"
    )
    np.testing.assert_array_equal(dataset["images"][:, 0, 0, 0], [1, 2])
    assert dataset["time"][:].tolist() == [1546300800000000000, 1546304400000000000]
    assert _state(tmp_path) == [("complete", None), ("complete", None)]


@pytest.mark.parametrize("failure", ["download", "transform"])
def test_pipeline_failure_cleans_temp_continues_and_resumes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    def download(source: SourceObject, *args: object) -> Path:
        if failure == "download" and source.key == "20190101_0000.nc":
            raise OSError("download unavailable")
        return _pipeline_download(source, *args)

    def resize(path: Path, channels: list[str], target_size: int) -> np.ndarray:
        if failure == "transform" and path.name == "20190101_0000.nc":
            raise ValueError("bad source")
        return _pipeline_resize(path, channels, target_size)

    monkeypatch.setattr(rebuild, "download_object", download)
    monkeypatch.setattr(rebuild, "resize_netcdf", resize)
    args = _year_args(tmp_path)
    args.update(
        objects=_pipeline_objects(3),
        download_workers=2,
        transform_workers=1,
        max_in_flight_samples=2,
    )
    with pytest.raises(RuntimeError, match="incomplete"):
        rebuild.process_year(**args)
    assert [state for state, error in _state(tmp_path)] == [
        "failed",
        "complete",
        "complete",
    ]
    assert not list((tmp_path / "temp").iterdir())
    times = zarr.open_array(
        str(tmp_path / "output" / ".staging" / "2019" / "dataset" / "time"), mode="r"
    )
    assert times[:].tolist() == [0, 1546304400000000000, 1546308000000000000]
    downloaded: list[str] = []

    def retry(source: SourceObject, *args: object) -> Path:
        downloaded.append(source.key)
        return _pipeline_download(source, *args)

    monkeypatch.setattr(rebuild, "download_object", retry)
    monkeypatch.setattr(rebuild, "resize_netcdf", _pipeline_resize)
    rebuild.process_year(**args)
    assert downloaded == ["20190101_0000.nc"]
    assert _state(tmp_path) == [("complete", None)] * 3


def test_pipeline_interruption_cleans_unconsumed_download_and_resumes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import threading

    interrupted = threading.Event()
    second_started = threading.Event()
    original_write = zarr.Array.__setitem__

    def download(source: SourceObject, *args: object) -> Path:
        if source.key == "20190101_0100.nc":
            second_started.set()
            assert interrupted.wait(5), "bounded download should be active during write"
        return _pipeline_download(source, *args)

    def write(array: zarr.Array, slot: int, value: object) -> None:
        if array.basename == "images":
            assert second_started.wait(5)
            interrupted.set()
            raise KeyboardInterrupt("walltime")
        original_write(array, slot, value)

    monkeypatch.setattr(rebuild, "download_object", download)
    monkeypatch.setattr(rebuild, "resize_netcdf", _pipeline_resize)
    monkeypatch.setattr(zarr.Array, "__setitem__", write)
    args = _year_args(tmp_path)
    args.update(
        objects=_pipeline_objects(2),
        download_workers=2,
        transform_workers=1,
        max_in_flight_samples=2,
    )
    with pytest.raises(KeyboardInterrupt):
        rebuild.process_year(**args)
    assert _state(tmp_path) == [("writing", None)] * 2
    assert not list((tmp_path / "temp").iterdir())
    monkeypatch.setattr(zarr.Array, "__setitem__", original_write)
    monkeypatch.setattr(rebuild, "download_object", _pipeline_download)
    rebuild.process_year(**args)
    assert _state(tmp_path) == [("complete", None)] * 2
    assert not list((tmp_path / "temp").iterdir())


@pytest.mark.parametrize(
    "field", ["download_workers", "transform_workers", "max_in_flight_samples"]
)
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_pipeline_worker_limits_rejected_before_listing_or_writes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, field: str, value: object
) -> None:
    cfg = _run_config(tmp_path)
    cfg[field] = value
    monkeypatch.setattr(
        rebuild,
        "list_source_objects",
        lambda **kwargs: pytest.fail("unexpected listing"),
    )
    with pytest.raises(ValueError, match=field):
        rebuild.run(cfg)
    args = _year_args(tmp_path)
    args[field] = value
    with pytest.raises(ValueError, match=field):
        rebuild.process_year(**args)
    assert not list(tmp_path.iterdir())


def test_pipeline_worker_settings_do_not_change_resume_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(rebuild, "download_object", _pipeline_download)
    monkeypatch.setattr(rebuild, "resize_netcdf", _pipeline_resize)
    _mock_listing(monkeypatch, _pipeline_objects(1))
    cfg = _run_config(tmp_path)
    rebuild.run(cfg)
    marker = tmp_path / "output" / ".checkpoints" / "build.json"
    before = marker.read_bytes()
    cfg.download_workers = 3
    cfg.transform_workers = 2
    cfg.max_in_flight_samples = 4
    rebuild.run(cfg)
    assert marker.read_bytes() == before


def test_pipeline_records_timestamp_reset_failure_and_continues(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(rebuild, "download_object", _pipeline_download)
    monkeypatch.setattr(rebuild, "resize_netcdf", _pipeline_resize)
    args = _year_args(tmp_path)
    args["objects"] = _pipeline_objects(2)
    rebuild.process_year(**args)
    with sqlite3.connect(tmp_path / "output" / ".checkpoints" / "2019.sqlite3") as db:
        db.execute("UPDATE samples SET state='writing'")
    original = zarr.Array.__setitem__

    def write(array: zarr.Array, slot: int, value: object) -> None:
        if array.basename == "time" and slot == 0 and value == 0:
            raise OSError("timestamp reset failed")
        original(array, slot, value)

    monkeypatch.setattr(zarr.Array, "__setitem__", write)
    with pytest.raises(RuntimeError, match="incomplete"):
        rebuild.process_year(**args)
    assert _state(tmp_path) == [
        ("failed", "OSError: timestamp reset failed"),
        ("complete", None),
    ]
