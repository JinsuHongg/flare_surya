import argparse
import zarr


def inspect_zarr(zarr_path):
    """
    Inspects a Zarr store to print its metadata and structure.

    Args:
        zarr_path (str): Path to the Zarr store.
    """
    print(f"Inspecting Zarr store at: {zarr_path}")
    try:
        store = zarr.open(zarr_path, mode="r")
        print("--- Store Info ---")
        print(store.info)
        print("\n--- Attributes ---")
        print(store.attrs.asdict())

        if isinstance(store, zarr.hierarchy.Group):
            print("\n--- Group Tree ---")
            print(store.tree())

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect the metadata and structure of a Zarr store."
    )
    parser.add_argument(
        "zarr_path",
        type=str,
        default="/media/jhong90/storage/surya/gong_halpha_2016-2024.zarr",
        nargs="?",
        help="Path to the Zarr store.",
    )
    args = parser.parse_args()
    inspect_zarr(args.zarr_path)
