param(
    [int[]]$WaitPids = @()
)

$ErrorActionPreference = "Stop"
$Repo = "E:\perception\friction_affordance_field"
$Python = "D:\NMI_SPWFM_datasets\conda_envs\faf_paper\python.exe"
$OutDir = "D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification\fast_physics_directional_texture_quality"
$LogDir = Join-Path $Repo "outputs\rscd_surface_formal_queue"
$Log = Join-Path $LogDir "fast_physics_directional_texture_quality_after_formal.log"
$Stdout = Join-Path $LogDir "fast_physics_directional_texture_quality.stdout.log"
$Stderr = Join-Path $LogDir "fast_physics_directional_texture_quality.stderr.log"

Set-Location $Repo
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

"[$(Get-Date -Format s)] waiting for formal RSCD jobs: $($WaitPids -join ', ')" | Out-File -FilePath $Log -Encoding utf8
while ($true) {
    $alive = @()
    foreach ($pidValue in $WaitPids) {
        $p = Get-Process -Id $pidValue -ErrorAction SilentlyContinue
        if ($null -ne $p) {
            $alive += $pidValue
        }
    }
    if ($alive.Count -eq 0) {
        break
    }
    "[$(Get-Date -Format s)] still waiting: $($alive -join ', ')" | Out-File -FilePath $Log -Encoding utf8 -Append
    Start-Sleep -Seconds 120
}

if (Test-Path (Join-Path $OutDir "evaluate_test.json")) {
    "[$(Get-Date -Format s)] evaluate_test.json already exists; skipping." | Out-File -FilePath $Log -Encoding utf8 -Append
    exit 0
}

"[$(Get-Date -Format s)] launching fast PhysicsTexture+DirectionalTexture RSCD-27 screen" | Out-File -FilePath $Log -Encoding utf8 -Append
Remove-Item -LiteralPath $Stdout, $Stderr -Force -ErrorAction SilentlyContinue
$args = @(
    "scripts\run_rscd_surface_classification.py",
    "--output-dir", $OutDir,
    "--epochs", "4",
    "--image-size", "192",
    "--batch-size", "12",
    "--grad-accum-steps", "2",
    "--samples-per-epoch", "10800",
    "--max-train-samples-per-class", "600",
    "--max-val-samples-per-class", "200",
    "--max-test-samples-per-class", "300",
    "--num-workers", "0",
    "--prefetch-factor", "2",
    "--use-physics-branch",
    "--physics-quality-cues",
    "--no-physics-quality-region-cues",
    "--physics-dim", "96",
    "--use-directional-texture-branch",
    "--directional-texture-dim", "64",
    "--early-stop-patience", "4",
    "--log-every-steps", "150"
)
$proc = Start-Process -FilePath $Python -ArgumentList $args -NoNewWindow -Wait -PassThru -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr
"[$(Get-Date -Format s)] python exit code $($proc.ExitCode)" | Out-File -FilePath $Log -Encoding utf8 -Append
if (Test-Path $Stdout) { Get-Content -LiteralPath $Stdout -Tail 80 | Out-File -FilePath $Log -Encoding utf8 -Append }
if (Test-Path $Stderr) { Get-Content -LiteralPath $Stderr -Tail 80 | Out-File -FilePath $Log -Encoding utf8 -Append }
if ($proc.ExitCode -ne 0) { exit $proc.ExitCode }

"[$(Get-Date -Format s)] done" | Out-File -FilePath $Log -Encoding utf8 -Append
& $Python scripts\compare_rscd_surface_candidates.py *>> $Log
& $Python scripts\compare_rscd_class_slices.py *>> $Log
& $Python scripts\write_rscd_formal_result_summary.py *>> $Log
& $Python scripts\write_rscd_training_trend_report.py *>> $Log
