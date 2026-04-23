#!/usr/bin/env python3
"""
Benchmark: xarray vs h5py vs h5py-core for loading SDO NetCDF files.

Usage:
    python bench_load.py --index_csv /aifm/helio_2012_one_sample.csv
"""
import argparse
import time

import dask
import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import pandas as pd
import xarray as xr

CHANNELS = [
    "aia94", "aia131", "aia171", "aia193", "aia211",
    "aia304", "aia335", "aia1600",
    "hmi_m", "hmi_bx", "hmi_by", "hmi_bz", "hmi_v",
]

dask.config.set(scheduler='processes')

def load_xarray(path: str, channels: list[str]) -> np.ndarray:
    with xr.open_dataset(path, engine="h5netcdf", chunks="auto", cache=False) as ds:
        # return ds[list(channels)].to_array().values
        return ds[channels].to_array().to_numpy()


def load_h5py(path: str, channels: list[str]) -> np.ndarray:
    with h5py.File(path, "r", rdcc_nbytes=0) as f:
        H, W = f[channels[0]].shape
        buf = np.empty((len(channels), H, W), dtype=np.float32)
        for i, ch in enumerate(channels):
            f[ch].read_direct(buf, dest_sel=np.s_[i, :, :])
    return buf


def load_h5py_core(path: str, channels: list[str]) -> np.ndarray:
    with h5py.File(path, "r", driver="core", backing_store=False) as f:
        H, W = f[channels[0]].shape
        buf = np.empty((len(channels), H, W), dtype=np.float32)
        for i, ch in enumerate(channels):
            f[ch].read_direct(buf, dest_sel=np.s_[i, :, :])
    return buf


LOADERS = {"xarray": load_xarray, "h5py": load_h5py, "h5py_core": load_h5py_core}


def run(filepaths: list[str], channels: list[str]):
    results = {name: [] for name in LOADERS}
    sums = {name: [] for name in LOADERS}

    for i, fp in enumerate(filepaths):
        parts = []
        for name, fn in LOADERS.items():
            t0 = time.perf_counter()
            arr = fn(fp, channels)
            s = float(np.nansum(arr.astype(np.float64)))
            elapsed = time.perf_counter() - t0
            results[name].append(elapsed)
            sums[name].append(s)
            parts.append(f"{name}={elapsed:.3f}s")
        ref = sums["xarray"][-1]
        ok = all(abs(sums[n][-1] - ref) < 1e-3 * max(abs(ref), 1) for n in LOADERS)
        print(f"[{i+1}/{len(filepaths)}] {' | '.join(parts)}  sum={ref:.4e} {'✓' if ok else '✗'}")

    # summary
    print(f"\n{'method':15s} {'avg':>8s} {'min':>8s} {'max':>8s} {'total':>9s} {'speedup':>8s}")
    print("-" * 65)
    baseline = np.mean(results["xarray"])
    for name in LOADERS:
        t = np.array(results[name])
        sp = baseline / t.mean() if t.mean() > 0 else 0
        print(f"{name:15s} {t.mean():7.3f}s {t.min():7.3f}s {t.max():7.3f}s {t.sum():8.3f}s {sp:7.1f}x")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--files", nargs="+", default=None)
    p.add_argument("--index_csv", default="~/project/flare_surya/flare_surya/data/surya_input_data.csv")
    p.add_argument("--n_files", type=int, default=10)
    p.add_argument("--channels", nargs="+", default=CHANNELS)
    args = p.parse_args()

    if args.files:
        fps = args.files
    else:
        df = pd.read_csv(args.index_csv)
        fps = df.loc[df["present"] == 1, "path"][:3].tolist()[: args.n_files]

    print(f"{len(fps)} file(s), {len(args.channels)} channels\n")
    run(fps, args.channels)


if __name__ == "__main__":
    main()
