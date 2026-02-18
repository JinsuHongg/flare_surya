import zarr
import os

# 1. Point to your specific Zarr dataset
zarr_path = "/anvil/scratch/x-jhong6/data/surya_bench_train_hour_24.zarr"

print(f"Repairing {zarr_path}...")

# 2. Use DirectoryStore
store = zarr.DirectoryStore(zarr_path)

# --- FIX IS HERE ---
# Use open_group (not group) and mode='r+' (read/write)
# If you fear the group doesn't exist at all, use mode='a' (append/create)
root = zarr.open_group(store=store, mode='r+') 

# 3. Verify 'timestep' is actually visible
if 'timestep' in root:
    print("Found 'timestep' array in raw storage.")
else:
    print("WARNING: 'timestep' array NOT found. If the folder exists, the .zarray file inside it might be missing.")

# 4. Force Re-Consolidation
print("Consolidating metadata...")
zarr.consolidate_metadata(store)

print("Success! Metadata re-consolidated.")
