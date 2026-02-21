import zarr
import os

zarr_path = "/anvil/scratch/x-jhong6/data/surya_bench_train_hour_24.zarr"
store = zarr.DirectoryStore(zarr_path)

print(f"Inspecting {zarr_path}...")

# 1. Check ROOT metadata (.zgroup)
if not os.path.exists(os.path.join(zarr_path, ".zgroup")):
    print("MISSING: Root .zgroup file. Creating it now...")
    zarr.open_group(store, mode='a')  # Creates .zgroup
else:
    print("OK: Root .zgroup exists.")

# 2. Check TIMESTEP metadata (.zarray)
ts_path = os.path.join(zarr_path, "timestep")
if not os.path.exists(os.path.join(ts_path, ".zarray")):
    print("CRITICAL ERROR: 'timestep' folder exists, but missing .zarray file!")
    print("This means your data creation script failed to initialize this array.")
    # We cannot auto-fix this easily without knowing your data shape/dtype.
    # You might need to re-run the creation script.
else:
    print("OK: 'timestep/.zarray' exists.")

# 3. Check IMG metadata (.zarray)
img_path = os.path.join(zarr_path, "img")
if not os.path.exists(os.path.join(img_path, ".zarray")):
    print("CRITICAL ERROR: 'img' folder exists, but missing .zarray file!")
else:
    print("OK: 'img/.zarray' exists.")

# 4. If everything looked OK above, FORCE CONSOLIDATION
print("-" * 30)
print("Attempting to consolidate metadata...")
try:
    zarr.consolidate_metadata(store)
    print("SUCCESS: .zmetadata created! You can now use open_consolidated().")
    
    # Verify what Zarr sees now
    root = zarr.open_group(store, mode='r')
    print(f"Visible Keys: {list(root.keys())}")
    
except Exception as e:
    print(f"FAILED to consolidate: {e}")
