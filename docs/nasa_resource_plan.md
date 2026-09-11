# NASA Resource Usage Plan

**Deadline:** NASA computational resource access ends on **October 1, 2026**.

## 1. Highest-Priority Task: Cache Surya Representations
Before the deadline, we must extract and save Surya embeddings for **every sample** in the training, validation, and test sets.

**Required Metadata with Embeddings:**
- Unique sample ID
- Observation timestamp
- Active-region ID (if available)
- Flare label and event threshold
- Forecast horizon
- Split assignment
- Input cadence / sampling configuration
- Surya checkpoint version
- Encoder layer used for feature extraction
- Preprocessing configuration

**Storage Formats:**
HDF5, Zarr, Parquet + array files, or PyTorch tensor files + metadata table. (Must support efficient partial loading).

## 2. Resource Allocation

### Tasks Requiring NASA Computing Resources (GPU-heavy)
- Full-resolution Surya inference
- Surya embedding extraction
- Partial Surya fine-tuning
- Final-block or two-block adaptation
- Generation of prediction probabilities from expensive models

### Tasks for Local Computing Resources (Post-October 1st)
- Logistic regression (linear probe)
- Small MLP training on cached features
- Metric calculation and plotting
- Bootstrapping and statistical testing
- Threshold analysis
- Manuscript table generation

## 3. Saving Outputs
For every expensive run on NASA resources, save the test and validation predictions with complete metadata (sample ID, true label, predicted probability/label, chosen threshold, etc.). Do **not** only save aggregate final metrics.
