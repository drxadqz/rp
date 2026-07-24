# Compatible Recovered Source Superset

This directory should be described as a compatible recovered source superset.

It should not be described as an exact `pre-20260709` frozen snapshot. The directory name is legacy wording retained to avoid large path churn in Git history.

Practical meaning:

- It contains source files, configs, and scripts that are useful for loading, auditing, and reconstructing the S7 lineage.
- It may include files created before and after the historical training window.
- It should be compared against current source file-by-file before claiming exact historical reproducibility.
- It is still useful as a compatibility source basis for checkpoint loading and provenance reconstruction.
