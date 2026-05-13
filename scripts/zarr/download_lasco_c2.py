"""
Downloads SOHO LASCO C2 images from the Helioviewer API, resizes them to 512x512,
and stores them in a Zarr v2 file.

Usage:
    python scripts/zarr/download_lasco_c2.py +data=lasco_c2_download
"""

import asyncio
import io
import os
import shutil
from datetime import datetime, timedelta

import aiohttp
import backoff
import hydra
import imageio.v3 as iio
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from numcodecs import Blosc
from omegaconf import OmegaConf
from PIL import Image
from loguru import logger as lgr_logger

# Constants
TIMEOUT = aiohttp.ClientTimeout(total=60)
API_BASE_URL = "https://api.helioviewer.org/v2/"


def get_session() -> aiohttp.ClientSession:
    """Create an aiohttp session."""
    return aiohttp.ClientSession(timeout=TIMEOUT, connector=aiohttp.TCPConnector(ssl=False))


@backoff.on_exception(backoff.expo, aiohttp.ClientError, max_tries=5)
async def fetch_image_id(
    session: aiohttp.ClientSession,
    date: datetime,
    source_id: int,
    tolerance_seconds: int = 600,
) -> str | None:
    """
    Query getClosestImage to find the JP2 image id nearest to `date`.

    Args:
        session: aiohttp session.
        date: The requested UTC datetime.
        source_id: Helioviewer source ID.

    Returns:
        The image id string, or None if outside tolerance or on error.
    """
    date_str = date.strftime("%Y-%m-%dT%H:%M:%SZ")

    params = {
        "date": date_str,
        "sourceId": source_id,
    }

    try:
        async with session.get(
            f"{API_BASE_URL}getClosestImage/",
            params=params,
        ) as response:
            if response.status == 200:
                data = await response.json(content_type=None)

                if "id" not in data or "date" not in data:
                    lgr_logger.warning(f"Unexpected response for {date_str}: {data}")
                    return None

                image_date = datetime.strptime(data["date"], "%Y-%m-%d %H:%M:%S")
                delta = abs((image_date - date.replace(tzinfo=None)).total_seconds())
                if delta > tolerance_seconds:
                    lgr_logger.debug(
                        f"Skipping {date_str}: closest image is {delta:.0f}s away "
                        f"(image date: {data['date']}, tolerance: {tolerance_seconds}s)"
                    )
                    return None

                return str(data["id"])
            else:
                lgr_logger.warning(
                    f"getClosestImage failed with status {response.status} for {date_str}"
                )
                return None
    except Exception as e:
        lgr_logger.error(f"Error fetching image id for {date_str}: {e}")
        raise


async def download_image(
    session: aiohttp.ClientSession,
    image_id: str,
    image_size: int = 512,
) -> np.ndarray | None:
    """
    Download a JP2 image by id via getJP2Image and resize to (image_size, image_size).

    Args:
        session: aiohttp session.
        image_id: Helioviewer image id.
        image_size: Target width and height in pixels.

    Returns:
        uint8 numpy array of shape (image_size, image_size, 1), or None if failed.
    """
    try:
        async with session.get(
            f"{API_BASE_URL}getJP2Image/",
            params={"id": image_id},
        ) as response:
            if response.status == 200:
                img_data = await response.read()

                img_array = iio.imread(io.BytesIO(img_data))

                pil_img = Image.fromarray(img_array)
                pil_img = pil_img.resize((image_size, image_size), Image.LANCZOS)
                img_array = np.array(pil_img)

                if img_array.ndim == 2:
                    img_array = img_array[:, :, np.newaxis]
                elif img_array.ndim == 3 and img_array.shape[2] != 1:
                    img_array = np.mean(
                        img_array[:, :, :3], axis=2, keepdims=True
                    ).astype(np.uint8)

                return img_array
            else:
                lgr_logger.warning(
                    f"getJP2Image failed for id={image_id}: HTTP {response.status}"
                )
                return None
    except Exception as e:
        lgr_logger.error(f"Error downloading image id={image_id}: {e}")
        return None


async def process_batch(
    session: aiohttp.ClientSession,
    dates: list[datetime],
    source_id: int,
    semaphore: asyncio.Semaphore,
    image_size: int,
    tolerance_seconds: int = 600,
) -> list[tuple[datetime, np.ndarray | None]]:
    """Process a batch of dates concurrently."""

    async def _fetch_id(date: datetime) -> tuple[datetime, str | None]:
        async with semaphore:
            return date, await fetch_image_id(
                session, date, source_id, tolerance_seconds
            )

    id_results: list[tuple[datetime, str | None]] = await asyncio.gather(
        *[_fetch_id(d) for d in dates]
    )

    unique_ids: set[str] = {img_id for _, img_id in id_results if img_id is not None}

    async def _download_unique(img_id: str) -> tuple[str, np.ndarray | None]:
        async with semaphore:
            return img_id, await download_image(session, img_id, image_size)

    downloaded: dict[str, np.ndarray | None] = dict(
        await asyncio.gather(*[_download_unique(uid) for uid in unique_ids])
    )

    return [
        (date, downloaded.get(img_id) if img_id is not None else None)
        for date, img_id in id_results
    ]


@hydra.main(
    version_base=None,
    config_path="../../configs/data/",
    config_name="lasco_c2_download",
)
def main(cfg: OmegaConf) -> None:
    """Main entry point."""
    lgr_logger.info("Starting SOHO LASCO C2 data download...")

    start_date = datetime.fromisoformat(cfg.download.start_date.replace("Z", "+00:00"))
    end_date = datetime.fromisoformat(cfg.download.end_date.replace("Z", "+00:00"))
    cadence = timedelta(minutes=cfg.download.cadence_minutes)
    image_size: int = cfg.download.image_size
    tolerance_seconds: int = cfg.download.tolerance_seconds

    dates: list[datetime] = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += cadence

    lgr_logger.info(f"Total dates to process: {len(dates)}")

    output_dir = os.path.join(cfg.output.output_dir, cfg.output.zarr_name)
    os.makedirs(output_dir, exist_ok=True)

    # Check for existing data to implement resume
    existing_timesteps = set()
    if os.path.exists(output_dir):
        try:
            # Check if it looks like a valid zarr store
            if os.path.exists(os.path.join(output_dir, ".zgroup")):
                ds_existing = xr.open_zarr(output_dir)
                existing_timesteps = set(pd.to_datetime(ds_existing.timestep.values))
                lgr_logger.info(f"Found {len(existing_timesteps)} existing timesteps. Resuming...")
        except Exception as e:
            lgr_logger.warning(f"Could not open existing zarr: {e}. Starting fresh.")

    # Filter dates
    dates = [d for d in dates if d not in existing_timesteps]
    lgr_logger.info(f"Remaining {len(dates)} dates to process.")

    total_saved = 0

    async def run_download() -> None:
        nonlocal total_saved
        # If existing timesteps, we are appending, not the first write
        is_first_write = len(existing_timesteps) == 0
        
        semaphore = asyncio.Semaphore(cfg.download.max_concurrent)
        batch_size = 100
        num_batches = (len(dates) + batch_size - 1) // batch_size

        async with get_session() as session:
            for batch_idx, i in enumerate(range(0, len(dates), batch_size)):
                batch_dates = dates[i : i + batch_size]
                lgr_logger.info(
                    f"Processing batch {batch_idx + 1} / {num_batches} "
                    f"({len(batch_dates)} dates)..."
                )

                results = await process_batch(
                    session,
                    batch_dates,
                    cfg.data_source_id,
                    semaphore,
                    image_size,
                    tolerance_seconds,
                )

                batch_imgs = [img for _, img in results if img is not None]
                batch_ts = [
                    pd.to_datetime(date).tz_localize(None) for date, img in results if img is not None
                ]

                if batch_imgs:
                    imgs_array = np.stack(batch_imgs, axis=0)  # (B, H, W, 1)

                    ds_batch = xr.Dataset(
                        {
                            "images": (["timestep", "y", "x", "channel"], imgs_array),
                        },
                        coords={
                            "timestep": pd.to_datetime(batch_ts),
                            "y": np.arange(image_size),
                            "x": np.arange(image_size),
                            "channel": ["c2"],
                        },
                    )

                    if is_first_write:
                        first_ts = batch_ts[0]
                        units = f"hours since {first_ts.strftime('%Y-%m-%d %H:%M:%S')}"
                        
                        encoding = {
                            "images": {
                                "compressor": Blosc(
                                    cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE
                                ),
                                "chunks": (100, image_size, image_size, 1),
                                "_FillValue": None,
                            },
                            "timestep": {
                                "dtype": "float64",
                                "units": units,
                                "calendar": "proleptic_gregorian",
                            },
                        }
                        
                        if os.path.exists(output_dir):
                            shutil.rmtree(output_dir)
                        os.makedirs(output_dir, exist_ok=True)

                        ds_batch.to_zarr(output_dir, mode="w", encoding=encoding)
                        is_first_write = False
                    else:
                        ds_batch.to_zarr(output_dir, append_dim="timestep")

                    total_saved += len(batch_imgs)
                    lgr_logger.info(
                        f"Batch {batch_idx + 1}: saved {len(batch_imgs)} images "
                        f"(total so far: {total_saved})"
                    )

        lgr_logger.info(
            f"Done. Saved {total_saved} images out of {len(dates)} requested "
            f"to {output_dir}"
        )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_download())
    finally:
        loop.close()

    if total_saved == 0:
        lgr_logger.warning("No images were downloaded.")
    else:
        zarr.consolidate_metadata(output_dir)
        lgr_logger.info(f"Consolidated zarr store at {output_dir}")


if __name__ == "__main__":
    main()
