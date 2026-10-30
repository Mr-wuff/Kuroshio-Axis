# Restore & verification workflow

1. Clone the GitHub repository. If Git LFS pointers are used, install Git LFS and run `git lfs pull`.
2. Read `manifests/experiment_summary.json`, `manifests/excluded_files_review.csv` and `manifests/split_files_restore.json` before interpreting file coverage. Use `tools/restore_split_data.py` to reconstruct any split large datasets.
3. Verify included file checksums using `manifests/uploaded_files_sha256.csv`.
4. Frozen path features are under `data/frozen/`. Accepted per-day tracks are stored in lossless year ZIPs under `data/final_tracks/` and `data/adt_axes/`; each ZIP contains per-file SHA256. Human-corrected track records are under `data/annotations/`.
5. CMEMS raw NetCDF are intentionally not scanned or distributed. Obtain original reanalysis and ADT from CMEMS independently.
6. To rerun scripts: create a WSL Python environment; check original `.ipynb` paths/configuration, replace machine-specific absolute paths, and install packages used by each Notebook. All execution outputs have been removed without altering code cells.
7. Research results are snapshots. R1–R64 model-search experiment outputs were intentionally excluded; the existence of training/evaluation code does not imply its trial data are included.
8. Human review/correction records are not equivalent to independent ground-truth observational labels; treat the source `source` field as provenance, not perfect truth.

Run `python3 tools/kuroshio_archive_publish.py --root /mnt/d/KuroshioDaily1993_2025 --repo OWNER/REPO` to refresh the archive on its source computer.
