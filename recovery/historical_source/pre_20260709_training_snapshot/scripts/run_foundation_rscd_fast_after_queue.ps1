param(
    [int[]]$WaitPids = @()
)

$ErrorActionPreference = "Stop"
$Repo = "E:\perception\friction_affordance_field"
$Python = "D:\NMI_SPWFM_datasets\conda_envs\faf_paper\python.exe"
$Root = "D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification"
$LogDir = Join-Path $Repo "outputs\rscd_surface_formal_queue"
$Log = Join-Path $LogDir "fast_foundation_after_queue.log"
$FormalRun = Join-Path $Root "formal_physics_wavelet_directional_film_gate_hier"
$TtaRun = Join-Path $Root "tta_ensemble_physics_texture_formal_hflip"
$MaterialRun = Join-Path $Root "fast_physics_material_gate_patch_quality"
$RetinexRunA = Join-Path $Root "fast_physics_retinex_texture_quality"
$RetinexRunB = Join-Path $Root "fast_physics_retinex_film_gate_hier"
$RetinexLog = Join-Path $LogDir "fast_retinex_texture_after_queue.log"

Set-Location $Repo
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-QueueLog([string]$Text) {
    "[$(Get-Date -Format s)] $Text" | Out-File -FilePath $Log -Encoding utf8 -Append
}

function Wait-ProcessIds([int[]]$Ids, [string]$Name) {
    while ($true) {
        $alive = @()
        foreach ($pidValue in $Ids) {
            $p = Get-Process -Id $pidValue -ErrorAction SilentlyContinue
            if ($null -ne $p) {
                $alive += $pidValue
            }
        }
        if ($alive.Count -eq 0) {
            break
        }
        Write-QueueLog "waiting for ${Name}: $($alive -join ', ')"
        Start-Sleep -Seconds 300
    }
}

function Wait-ClassificationJobsGone([string]$Reason) {
    while ($true) {
        $alive = @(Get-CimInstance Win32_Process | Where-Object {
            $_.CommandLine -match "run_rscd_surface_classification.py" -and
            $_.CommandLine -notmatch "smoke_"
        })
        if ($alive.Count -eq 0) {
            break
        }
        Write-QueueLog "waiting for classification jobs before ${Reason}: $($alive.ProcessId -join ', ')"
        Start-Sleep -Seconds 300
    }
}

function Wait-File([string]$Path, [string]$Name) {
    while (-not (Test-Path $Path)) {
        Write-QueueLog "waiting for ${Name}: $Path"
        Start-Sleep -Seconds 300
    }
}

function Wait-RetinexDone() {
    while ($true) {
        if ((Test-Path (Join-Path $RetinexRunA "evaluate_test.json")) -and
            (Test-Path (Join-Path $RetinexRunB "evaluate_test.json"))) {
            Write-QueueLog "Retinex fast outputs found"
            break
        }
        if ((Test-Path $RetinexLog) -and ((Get-Content -LiteralPath $RetinexLog -Tail 80) -match "Retinex texture fast candidates complete")) {
            Write-QueueLog "Retinex watcher finished"
            break
        }
        Write-QueueLog "waiting for Retinex fast queue"
        Start-Sleep -Seconds 300
    }
}

function Invoke-FastCandidate([string]$Name, [string[]]$ExtraArgs, [string]$BackboneMode = "dinov2") {
    Wait-ClassificationJobsGone -Reason $Name
    $OutDir = Join-Path $Root $Name
    $Stdout = Join-Path $LogDir "$Name.stdout.log"
    $Stderr = Join-Path $LogDir "$Name.stderr.log"
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    if (Test-Path (Join-Path $OutDir "evaluate_test.json")) {
        Write-QueueLog "$Name already has evaluate_test.json; skipping."
        return
    }

    Write-QueueLog "launching $Name"
    if ($BackboneMode -eq "convnext224") {
        $baseArgs = @(
            "scripts\run_rscd_surface_classification.py",
            "--output-dir", $OutDir,
            "--backbone", "convnext_tiny",
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
    } elseif ($BackboneMode -eq "convnext") {
        $baseArgs = @(
            "scripts\run_rscd_surface_classification.py",
            "--output-dir", $OutDir,
            "--backbone", "convnext_tiny",
            "--embedding-dim", "768",
            "--pretrained",
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
            "--early-stop-patience", "4",
            "--log-every-steps", "150"
        )
    } else {
        $baseArgs = @(
            "scripts\run_rscd_surface_classification.py",
            "--output-dir", $OutDir,
            "--backbone", "timm:vit_small_patch14_dinov2",
            "--embedding-dim", "384",
            "--pretrained",
            "--epochs", "4",
            "--image-size", "196",
            "--batch-size", "2",
            "--grad-accum-steps", "12",
            "--samples-per-epoch", "5400",
            "--max-train-samples-per-class", "600",
            "--max-val-samples-per-class", "200",
            "--max-test-samples-per-class", "300",
            "--num-workers", "0",
            "--prefetch-factor", "2",
            "--early-stop-patience", "4",
            "--log-every-steps", "150"
        )
    }
    $args = $baseArgs + $ExtraArgs
    Remove-Item -LiteralPath $Stdout, $Stderr -Force -ErrorAction SilentlyContinue
    $proc = Start-Process -FilePath $Python -ArgumentList $args -NoNewWindow -Wait -PassThru -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr
    Write-QueueLog "$Name python exit code $($proc.ExitCode)"
    if (Test-Path $Stdout) { Get-Content -LiteralPath $Stdout -Tail 80 | Out-File -FilePath $Log -Encoding utf8 -Append }
    if (Test-Path $Stderr) { Get-Content -LiteralPath $Stderr -Tail 80 | Out-File -FilePath $Log -Encoding utf8 -Append }
    if ($proc.ExitCode -ne 0) { throw "$Name failed with exit code $($proc.ExitCode)" }
}

function Get-EvaluateMacroF1([string]$Name) {
    $path = Join-Path (Join-Path $Root $Name) "evaluate_test.json"
    if (-not (Test-Path $path)) {
        return $null
    }
    $payload = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
    return [double]$payload.summary.macro_f1
}

function New-SkipMarker([string]$Name, [string]$Reason) {
    $outDir = Join-Path $Root $Name
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    $marker = Join-Path $outDir "skipped_after_global_failure.json"
    $payload = [ordered]@{
        name = $Name
        status = "skipped_pruned"
        reason = $Reason
        created_at = (Get-Date -Format s)
    }
    $payload | ConvertTo-Json -Depth 4 | Out-File -FilePath $marker -Encoding utf8
}

function Invoke-FormalPromotion() {
    Write-QueueLog "selecting foundation formal promotion"
    $decisionText = & $Python scripts\select_rscd_foundation_promotion.py
    Write-QueueLog "foundation decision: $decisionText"
    $decision = $decisionText | ConvertFrom-Json
    if ($null -eq $decision.promoted) {
        Write-QueueLog "no foundation candidate promoted"
        return
    }

    Wait-ClassificationJobsGone -Reason "foundation formal promotion"

    $outDir = [string]$decision.promoted.formal_output_dir
    if (Test-Path (Join-Path $outDir "evaluate_test.json")) {
        Write-QueueLog "formal foundation result already exists for $outDir; skipping"
        return
    }
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null

    $extra = @()
    foreach ($arg in $decision.promoted.formal_args) {
        $extra += [string]$arg
    }

    $stdout = Join-Path $LogDir "formal_foundation_candidate.stdout.log"
    $stderr = Join-Path $LogDir "formal_foundation_candidate.stderr.log"
    Write-QueueLog "launching formal foundation candidate $($decision.promoted.name)"
    $args = @(
        "scripts\run_rscd_surface_classification.py",
        "--output-dir", $outDir,
        "--epochs", "12",
        "--image-size", "196",
        "--batch-size", "2",
        "--grad-accum-steps", "12",
        "--samples-per-epoch", "7200",
        "--num-workers", "0",
        "--prefetch-factor", "2",
        "--early-stop-patience", "4",
        "--log-every-steps", "150"
    ) + $extra
    Remove-Item -LiteralPath $stdout, $stderr -Force -ErrorAction SilentlyContinue
    $proc = Start-Process -FilePath $Python -ArgumentList $args -NoNewWindow -Wait -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    Write-QueueLog "formal foundation python exit code $($proc.ExitCode)"
    if (Test-Path $stdout) { Get-Content -LiteralPath $stdout -Tail 80 | Out-File -FilePath $Log -Encoding utf8 -Append }
    if (Test-Path $stderr) { Get-Content -LiteralPath $stderr -Tail 80 | Out-File -FilePath $Log -Encoding utf8 -Append }
    if ($proc.ExitCode -ne 0) { throw "formal foundation candidate failed with exit code $($proc.ExitCode)" }

    & $Python scripts\compare_rscd_surface_candidates.py *>> $Log
    & $Python scripts\compare_rscd_class_slices.py *>> $Log
    & $Python scripts\write_rscd_formal_result_summary.py *>> $Log
    & $Python scripts\write_rscd_decision_dashboard.py *>> $Log
    & $Python scripts\select_final_rscd_method.py *>> $Log
    & $Python scripts\write_goal_completion_audit.py *>> $Log
    Write-QueueLog "formal foundation candidate complete"
}

function Invoke-PatchStatsFormalPromotion() {
    Write-QueueLog "selecting patch-invariant quality formal promotion"
    $decisionText = & $Python scripts\select_rscd_patch_quality_region_promotion.py
    Write-QueueLog "patch-quality decision: $decisionText"
    $decision = $decisionText | ConvertFrom-Json
    if ($null -eq $decision.promoted) {
        Write-QueueLog "no patch-invariant quality candidate promoted"
        return
    }

    Wait-ClassificationJobsGone -Reason "patch-invariant quality formal promotion"

    $outDir = [string]$decision.promoted.formal_output_dir
    if (Test-Path (Join-Path $outDir "evaluate_test.json")) {
        Write-QueueLog "formal patch-invariant quality result already exists for $outDir; skipping"
        return
    }
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null

    $extra = @()
    foreach ($arg in $decision.promoted.formal_args) {
        $extra += [string]$arg
    }

    $stdout = Join-Path $LogDir "formal_patch_quality_region_candidate.stdout.log"
    $stderr = Join-Path $LogDir "formal_patch_quality_region_candidate.stderr.log"
    Write-QueueLog "launching formal patch-invariant quality candidate"
    $args = @(
        "scripts\run_rscd_surface_classification.py",
        "--output-dir", $outDir,
        "--epochs", "20",
        "--image-size", "192",
        "--batch-size", "12",
        "--grad-accum-steps", "2",
        "--samples-per-epoch", "36000",
        "--num-workers", "0",
        "--prefetch-factor", "2",
        "--early-stop-patience", "5",
        "--log-every-steps", "150"
    ) + $extra
    Remove-Item -LiteralPath $stdout, $stderr -Force -ErrorAction SilentlyContinue
    $proc = Start-Process -FilePath $Python -ArgumentList $args -NoNewWindow -Wait -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    Write-QueueLog "formal patch-invariant quality python exit code $($proc.ExitCode)"
    if (Test-Path $stdout) { Get-Content -LiteralPath $stdout -Tail 80 | Out-File -FilePath $Log -Encoding utf8 -Append }
    if (Test-Path $stderr) { Get-Content -LiteralPath $stderr -Tail 80 | Out-File -FilePath $Log -Encoding utf8 -Append }
    if ($proc.ExitCode -ne 0) { throw "formal patch-invariant quality candidate failed with exit code $($proc.ExitCode)" }

    & $Python scripts\compare_rscd_surface_candidates.py *>> $Log
    & $Python scripts\compare_rscd_class_slices.py *>> $Log
    & $Python scripts\select_rscd_patch_quality_region_promotion.py *>> $Log
    & $Python scripts\write_rscd_formal_result_summary.py *>> $Log
    & $Python scripts\write_rscd_decision_dashboard.py *>> $Log
    & $Python scripts\select_final_rscd_method.py *>> $Log
    & $Python scripts\write_goal_completion_audit.py *>> $Log
    Write-QueueLog "formal patch-invariant quality candidate complete"
}

Write-QueueLog "started foundation RSCD fast watcher"
Wait-ProcessIds -Ids $WaitPids -Name "upstream process"
Wait-File -Path (Join-Path $FormalRun "evaluate_test.json") -Name "promoted formal result"
Wait-File -Path (Join-Path $TtaRun "evaluate_test.json") -Name "TTA ensemble result"
Wait-File -Path (Join-Path $MaterialRun "evaluate_test.json") -Name "material-conditioned gate result"
Wait-RetinexDone

Invoke-FastCandidate -Name "fast_physics_texture_quality_patch_stats" -ExtraArgs @(
    "--use-physics-branch",
    "--physics-quality-cues",
    "--no-physics-quality-region-cues",
    "--physics-dim", "96"
) -BackboneMode "convnext"

Invoke-FastCandidate -Name "fast_physics_texture_quality_patch_stats_224" -ExtraArgs @(
    "--use-physics-branch",
    "--physics-quality-cues",
    "--no-physics-quality-region-cues",
    "--physics-dim", "96"
) -BackboneMode "convnext224"

Invoke-PatchStatsFormalPromotion

Invoke-FastCandidate -Name "fast_dinov2_global_rscd" -ExtraArgs @()

$dinov2MacroF1 = Get-EvaluateMacroF1 -Name "fast_dinov2_global_rscd"
if ($null -ne $dinov2MacroF1 -and $dinov2MacroF1 -lt 0.50) {
    $reason = "DINOv2 global fast screen Macro-F1=$dinov2MacroF1 is far below the ConvNeXt/PhysicsTexture fast references; current end-to-end small-batch DINO protocol is pruned."
    Write-QueueLog $reason
    New-SkipMarker -Name "fast_dinov2_physics_texture_rscd" -Reason $reason
} else {
    Invoke-FastCandidate -Name "fast_dinov2_physics_texture_rscd" -ExtraArgs @(
        "--use-physics-branch",
        "--physics-quality-cues",
        "--no-physics-quality-region-cues",
        "--physics-dim", "96"
    )

    Invoke-FormalPromotion
}

& $Python scripts\compare_rscd_surface_candidates.py *>> $Log
& $Python scripts\compare_rscd_class_slices.py *>> $Log
& $Python scripts\select_rscd_patch_quality_region_promotion.py *>> $Log
& $Python scripts\select_rscd_foundation_promotion.py *>> $Log
& $Python scripts\write_rscd_decision_dashboard.py *>> $Log
& $Python scripts\select_final_rscd_method.py *>> $Log
& $Python scripts\write_goal_completion_audit.py *>> $Log
Write-QueueLog "foundation RSCD fast candidates complete"
