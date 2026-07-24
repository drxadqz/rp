param(
    [ValidateSet("p0", "ablation", "lodo", "single", "baselines", "candidates", "final_lodo", "final_single", "final", "all")]
    [string] $Phase = "p0",
    [string] $Python = "D:\NMI_SPWFM_datasets\conda_envs\faf_paper\python.exe",
    [string] $LogDir = "outputs\paper_protocol_queue",
    [switch] $Force
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$TempRoot = "D:\NMI_SPWFM_datasets\tmp"
New-Item -ItemType Directory -Force -Path $TempRoot | Out-Null
$env:TEMP = $TempRoot
$env:TMP = $TempRoot
$env:TORCH_HOME = "D:\NMI_SPWFM_datasets\torch_cache"

& $Python scripts\make_paper_protocol_configs.py

$Ablation = @(
    "configs\experiments\paper_protocol\v0_global_only.yaml",
    "configs\experiments\paper_protocol\v1_physics_texture.yaml",
    "configs\experiments\paper_protocol\v2_friction_set.yaml",
    "configs\experiments\paper_protocol\v3_dg_losses.yaml",
    "configs\experiments\paper_protocol\v4_evidence_aux.yaml",
    "configs\experiments\paper_protocol\v5_full_faf.yaml"
)
$Candidates = @(
    "configs\experiments\paper_protocol\v6_full_faf_fourier.yaml",
    "configs\experiments\paper_protocol\v7_full_faf_fourier_dann.yaml",
    "configs\experiments\paper_protocol\v8_full_faf_fourier_roadprior.yaml",
    "configs\experiments\paper_protocol\v9_full_faf_roadsaw_hard_sampling.yaml",
    "configs\experiments\paper_protocol\v10_full_faf_consistency.yaml",
    "configs\experiments\paper_protocol\v11_full_faf_domain_adapter.yaml",
    "configs\experiments\paper_protocol\v12_full_faf_roi_interval_safety.yaml",
    "configs\experiments\paper_protocol\v13_lean_physics_evidence.yaml",
    "configs\experiments\paper_protocol\v14_lean_road_roi_safety.yaml",
    "configs\experiments\paper_protocol\v15_lean_bottom_square_style_safety.yaml",
    "configs\experiments\paper_protocol\v16_lean_bottom_square_color_constancy_safety.yaml",
    "configs\experiments\paper_protocol\v17_lean_quality_physics_safety.yaml",
    "configs\experiments\paper_protocol\v18_lean_mixstyle_quality_safety.yaml",
    "configs\experiments\paper_protocol\v19_lean_state_contrast_quality_safety.yaml",
    "configs\experiments\paper_protocol\v20_lean_interval_order_quality_safety.yaml",
    "configs\experiments\paper_protocol\v21_lean_quality_uncertainty_safety.yaml",
    "configs\experiments\paper_protocol\v22_lean_quality_order_contrast_safety.yaml"
)
$Lodo = @(
    "configs\experiments\paper_protocol\lodo_roadsaw_full_faf.yaml",
    "configs\experiments\paper_protocol\lodo_rscd_full_faf.yaml",
    "configs\experiments\paper_protocol\lodo_roadsc_full_faf.yaml"
)
$Single = @(
    "configs\experiments\paper_protocol\single_roadsaw_full_faf.yaml",
    "configs\experiments\paper_protocol\single_rscd_full_faf.yaml",
    "configs\experiments\paper_protocol\single_roadsc_full_faf.yaml"
)
$Baselines = @(
    "configs\experiments\paper_protocol\baseline_single_roadsaw_global_convnext.yaml",
    "configs\experiments\paper_protocol\baseline_single_rscd_global_convnext.yaml",
    "configs\experiments\paper_protocol\baseline_single_roadsc_global_convnext.yaml"
)
$FinalLodo = @(
    "configs\experiments\paper_protocol\final_lodo_roadsaw_lean_road_roi_safety.yaml",
    "configs\experiments\paper_protocol\final_lodo_rscd_lean_road_roi_safety.yaml",
    "configs\experiments\paper_protocol\final_lodo_roadsc_lean_road_roi_safety.yaml"
)
$FinalSingle = @(
    "configs\experiments\paper_protocol\final_single_roadsaw_lean_road_roi_safety.yaml",
    "configs\experiments\paper_protocol\final_single_rscd_lean_road_roi_safety.yaml",
    "configs\experiments\paper_protocol\final_single_roadsc_lean_road_roi_safety.yaml"
)

if ($Phase -eq "ablation") {
    $Configs = $Ablation
} elseif ($Phase -eq "lodo") {
    $Configs = $Lodo
} elseif ($Phase -eq "single") {
    $Configs = $Single
} elseif ($Phase -eq "baselines") {
    $Configs = $Baselines
} elseif ($Phase -eq "candidates") {
    $Configs = $Candidates
} elseif ($Phase -eq "final_lodo") {
    $Configs = $FinalLodo
} elseif ($Phase -eq "final_single") {
    $Configs = $FinalSingle
} elseif ($Phase -eq "final") {
    $Configs = $FinalLodo + $FinalSingle
} elseif ($Phase -eq "all") {
    $Configs = $Ablation + $Lodo + $Single + $Baselines + $Candidates + $FinalLodo + $FinalSingle
} else {
    $Configs = $Ablation + $Lodo
}

foreach ($Config in $Configs) {
    if (!(Test-Path $Config)) {
        throw "Config not found: $Config"
    }
    $OutputDirLine = Select-String -Path $Config -Pattern "^\s*output_dir:\s*(.+)\s*$" | Select-Object -First 1
    if (!$OutputDirLine) {
        throw "Could not infer output_dir from config: $Config"
    }
    $OutputDir = $OutputDirLine.Matches[0].Groups[1].Value.Trim()
    $Complete = (Test-Path (Join-Path $OutputDir "best.pt")) -and
        (Test-Path (Join-Path $OutputDir "detailed_test.json")) -and
        (Test-Path (Join-Path $OutputDir "interval_calibration_90.json")) -and
        (Test-Path (Join-Path $OutputDir "bootstrap_metrics.json"))
    if ($Complete -and !$Force) {
        Write-Host "Skipping completed config: $Config" -ForegroundColor DarkGreen
        continue
    }

    $Name = [IO.Path]::GetFileNameWithoutExtension($Config)
    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutLog = Join-Path $LogDir "$Name`_$Stamp.out.log"
    $ErrLog = Join-Path $LogDir "$Name`_$Stamp.err.log"
    Write-Host ""
    Write-Host "==== paper protocol config: $Config ====" -ForegroundColor Cyan
    Write-Host "logs: $OutLog $ErrLog" -ForegroundColor DarkCyan
    $PipelineArgs = @(
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        "scripts\run_config_pipeline.ps1",
        "-Config",
        $Config,
        "-Python",
        $Python
    )
    if ($Name.StartsWith("single_") -or $Name.StartsWith("baseline_single_")) {
        $PipelineArgs += "-SkipDatasetDiagnostic"
    }
    if ($Force) {
        $PipelineArgs += "-ForceTrain"
    }
    $Proc = Start-Process -FilePath "powershell.exe" `
        -ArgumentList $PipelineArgs `
        -WorkingDirectory $ProjectRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $OutLog `
        -RedirectStandardError $ErrLog `
        -Wait `
        -PassThru
    $Code = $Proc.ExitCode
    if ($Code -ne 0) {
        throw "Config failed: $Config (exit code $Code). Logs: $OutLog $ErrLog"
    }
}

& $Python scripts\postprocess_protocol_outputs.py `
    --root "D:\NMI_SPWFM_datasets\friction_affordance_outputs\paper_protocol" `
    --summary-dir "reports\paper_protocol_summary"
