$ErrorActionPreference = "Stop"
Set-Location "E:\perception\friction_affordance_field"

$env:PYTHONPATH = "E:\perception\friction_affordance_field\src"
$Python = "D:\NMI_SPWFM_datasets\conda_envs\faf_paper\python.exe"
$Config = "configs\c3_farnet\c3_farnet_screen_s101_concrete_roughness_ordinal_s96_anchor_20260712.yaml"
$OutDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_screen_s101_concrete_roughness_ordinal_s96_anchor_20260712"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
& $Python -u train.py --config $Config 1> "$OutDir\train_stdout_$Stamp.log" 2> "$OutDir\train_stderr_$Stamp.log"
