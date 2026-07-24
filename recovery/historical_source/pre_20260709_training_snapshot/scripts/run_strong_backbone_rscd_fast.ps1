param()

$ErrorActionPreference = "Stop"
$Repo = "E:\perception\friction_affordance_field"
$Python = "D:\NMI_SPWFM_datasets\conda_envs\faf_paper\python.exe"
$Root = "D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification"
$LogDir = Join-Path $Repo "outputs\rscd_surface_formal_queue"
$Log = Join-Path $LogDir "fast_strong_backbone_queue.log"

Set-Location $Repo
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-QueueLog([string]$Text) {
    "[$(Get-Date -Format s)] $Text" | Out-File -FilePath $Log -Encoding utf8 -Append
}

function Wait-ClassificationJobsGone([string]$Reason) {
    while ($true) {
        $alive = @(Get-CimInstance Win32_Process | Where-Object {
            $_.Name -match "python" -and
            $_.CommandLine -match "run_rscd_surface_classification.py" -and
            $_.CommandLine -notmatch "smoke_"
        })
        if ($alive.Count -eq 0) {
            break
        }
        Write-QueueLog "waiting for classification jobs before ${Reason}: $($alive.ProcessId -join ', ')"
        Start-Sleep -Seconds 180
    }
}

function Invoke-Candidate([string]$Name, [string]$Backbone, [string[]]$ExtraArgs) {
    Wait-ClassificationJobsGone -Reason $Name
    $outDir = Join-Path $Root $Name
    $stdout = Join-Path $LogDir "$Name.stdout.log"
    $stderr = Join-Path $LogDir "$Name.stderr.log"
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    if (Test-Path (Join-Path $outDir "evaluate_test.json")) {
        Write-QueueLog "$Name already has evaluate_test.json; skipping."
        return
    }

    $baseArgs = @(
        "scripts\run_rscd_surface_classification.py",
        "--output-dir", $outDir,
        "--backbone", $Backbone,
        "--embedding-dim", "768",
        "--pretrained",
        "--epochs", "4",
        "--image-size", "224",
        "--batch-size", "8",
        "--grad-accum-steps", "3",
        "--samples-per-epoch", "10800",
        "--max-train-samples-per-class", "600",
        "--max-val-samples-per-class", "200",
        "--max-test-samples-per-class", "300",
        "--num-workers", "0",
        "--prefetch-factor", "2",
        "--early-stop-patience", "4",
        "--log-every-steps", "150"
    )
    $args = $baseArgs + $ExtraArgs
    Remove-Item -LiteralPath $stdout, $stderr -Force -ErrorAction SilentlyContinue
    Write-QueueLog "launching $Name with $Backbone"
    $proc = Start-Process -FilePath $Python -ArgumentList $args -NoNewWindow -Wait -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    Write-QueueLog "$Name python exit code $($proc.ExitCode)"
    if (Test-Path $stdout) { Get-Content -LiteralPath $stdout -Tail 80 | Out-File -FilePath $Log -Encoding utf8 -Append }
    if (Test-Path $stderr) { Get-Content -LiteralPath $stderr -Tail 80 | Out-File -FilePath $Log -Encoding utf8 -Append }
    if ($proc.ExitCode -ne 0) { throw "$Name failed with exit code $($proc.ExitCode)" }
}

Write-QueueLog "started strong-backbone RSCD fast queue"

Invoke-Candidate -Name "fast_timm_convnext_tiny_in22k_224" -Backbone "timm:convnext_tiny.fb_in22k_ft_in1k" -ExtraArgs @()

Invoke-Candidate -Name "fast_timm_convnext_tiny_in22k_physics_224" -Backbone "timm:convnext_tiny.fb_in22k_ft_in1k" -ExtraArgs @(
    "--use-physics-branch",
    "--physics-quality-cues",
    "--physics-dim", "96"
)

Invoke-Candidate -Name "fast_timm_convnextv2_tiny_224" -Backbone "timm:convnextv2_tiny.fcmae_ft_in22k_in1k" -ExtraArgs @()

& $Python scripts\compare_rscd_surface_candidates.py *>> $Log
& $Python scripts\compare_rscd_class_slices.py *>> $Log
& $Python scripts\select_final_rscd_method.py *>> $Log
& $Python scripts\write_experiment_queue_health_report.py *>> $Log
Write-QueueLog "strong-backbone RSCD fast candidates complete"
