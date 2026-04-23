import os
import sys

import numpy as np
import pytest
import torch
import xarray as xr

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

# ruff: noqa: E402
from flare_surya.datamodule import SolarPretrainDataModule  # noqa: E402
from flare_surya.models import (
    PretrainSolarModel,
    SolarDecoder,
    SolarEncoder,
)  # noqa: E402


# Create a toy zarr store for testing
def create_toy_zarr(path, num_samples=10):
    """Create a toy zarr store for testing."""
    if os.path.exists(path):
        return

    times = np.arange("2020-01-01", "2020-01-01 00:10", dtype="datetime64[m]")
    times = times[:num_samples]

    # Create random data - create two separate data variables for 1D case
    # Each with shape (timestep, time)
    data1 = np.random.rand(num_samples, 1440).astype(np.float32)
    data2 = np.random.rand(num_samples, 1440).astype(np.float32)

    # Create xarray dataset with separate variables
    ds = xr.Dataset(
        {
            "soft": (["timestep", "time"], data1),
            "hard": (["timestep", "time"], data2),
        },
        coords={
            "timestep": times,
            "time": np.arange(1440),
        },
    )

    # Save to zarr
    ds.to_zarr(path, mode="w")


def create_toy_index(path, num_samples=10):
    """Create a toy index CSV for testing."""
    if os.path.exists(path):
        return

    times = np.arange("2020-01-01", "2020-01-01 00:10", dtype="datetime64[m]")
    times = times[:num_samples]

    df = {"timestamp": times}
    import pandas as pd

    df = pd.DataFrame(df)
    df.to_csv(path, index=False)


@pytest.fixture
def toy_data_paths(tmp_path):
    """Create toy data paths."""
    zarr_path = tmp_path / "toy_data.zarr"
    train_index_path = tmp_path / "train_index.csv"
    val_index_path = tmp_path / "val_index.csv"
    test_index_path = tmp_path / "test_index.csv"

    create_toy_zarr(str(zarr_path))
    create_toy_index(str(train_index_path))
    create_toy_index(str(val_index_path))
    create_toy_index(str(test_index_path))

    return {
        "zarr_path": str(zarr_path),
        "train_index_path": str(train_index_path),
        "val_index_path": str(val_index_path),
        "test_index_path": str(test_index_path),
    }


def test_encoder_1d():
    """Test SolarEncoder with 1D data."""
    model = SolarEncoder(
        in_channels=2,
        seq_len=1440,
        embed_dim=128,
        depth=2,
        num_heads=4,
        data_type="1d",
    )
    x = torch.randn(2, 2, 1440)  # (batch, channel, seq_len)
    out = model(x)
    assert out.shape == (2, 1440, 128)


def test_encoder_2d():
    """Test SolarEncoder with 2D data."""
    model = SolarEncoder(
        in_channels=1,
        seq_len=196,  # 14x14 patches
        embed_dim=128,
        depth=2,
        num_heads=4,
        data_type="2d",
        image_size=224,
        patch_size=16,
    )
    x = torch.randn(2, 1, 224, 224)  # (batch, channel, H, W)
    out = model(x)
    assert out.shape == (2, 196, 128)


def test_decoder_1d():
    """Test SolarDecoder with 1D data."""
    decoder = SolarDecoder(
        in_channels=2,
        seq_len=1440,
        embed_dim=128,
        depth=2,
        num_heads=4,
        data_type="1d",
    )
    x = torch.randn(2, 1440, 128)  # (batch, seq_len, embed_dim)
    out = decoder(x)
    assert out.shape == (2, 2, 1440)


def test_decoder_2d():
    """Test SolarDecoder with 2D data."""
    decoder = SolarDecoder(
        in_channels=1,
        seq_len=196,  # 14x14 patches
        embed_dim=128,
        depth=2,
        num_heads=4,
        data_type="2d",
    )
    x = torch.randn(2, 196, 128)  # (batch, seq_len, embed_dim)
    out = decoder(x)
    # The detokenizer uses stride=2 convolutions, so output is smaller than 224x224
    # For 14x14 patches -> 7x7 after first layer -> 3x3 or 4x4 after second
    assert out.shape[0] == 2  # batch size
    assert out.shape[1] == 1  # channels


def test_pretrain_model(toy_data_paths):
    """Test PretrainSolarModel training step."""
    model = PretrainSolarModel(
        in_channels=2,
        seq_len=1440,
        embed_dim=64,
        encoder_depth=1,
        decoder_depth=1,
        num_heads=2,
        data_type="1d",
    )

    # Create a batch
    x = torch.randn(2, 2, 1440)
    y = torch.randn(2, 2, 1440)

    # Training step
    loss = model.training_step((x, y), 0)
    assert loss is not None
    assert isinstance(loss.item(), float)


def test_datamodule(toy_data_paths):
    """Test SolarPretrainDataModule."""
    datamodule = SolarPretrainDataModule(
        zarr_path=toy_data_paths["zarr_path"],
        train_index_path=toy_data_paths["train_index_path"],
        val_index_path=toy_data_paths["val_index_path"],
        test_index_path=toy_data_paths["test_index_path"],
        channels=["soft", "hard"],
        batch_size=2,
        num_workers=0,  # Use 0 for testing
        data_type="1d",
    )

    datamodule.setup("fit")

    train_loader = datamodule.train_dataloader()

    # Get one batch
    batch = next(iter(train_loader))
    x, y, ts = batch

    assert x.shape[0] == 2  # batch size
    assert x.shape[1] == 2  # channels (soft + hard)
    assert x.shape[2] == 1440  # seq_len


if __name__ == "__main__":
    # Run tests manually if needed
    print("Running manual tests...")

    # Test encoder 1D
    print("Testing encoder 1D...")
    test_encoder_1d()
    print("Passed!")

    # Test encoder 2D
    print("Testing encoder 2D...")
    test_encoder_2d()
    print("Passed!")

    # Test decoder 1D
    print("Testing decoder 1D...")
    test_decoder_1d()
    print("Passed!")

    # Test decoder 2D
    print("Testing decoder 2D...")
    test_decoder_2d()
    print("Passed!")

    print("All tests passed!")
