param(
    [int]$PriorityFastPid = 9000
)

$ErrorActionPreference = "Stop"
$Repo = "E:\perception\friction_affordance_field"
$Python = "D:\NMI_SPWFM_datasets\conda_envs\faf_paper\python.exe"
$LogDir = Join-Path $Repo "outputs\rscd_surface_formal_queue"
$Log = Join-Path $LogDir "high_priority_formal_promotion_after_fast.log"

Set-Location $Repo
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-QueueLog([string]$Text) {
    "[$(Get-Date -Format s)] $Text" | Out-File -FilePath $Log -Encoding utf8 -Append
}

function Wait-ProcessGone([int]$PidValue, [string]$Name) {
    while ($true) {
        $p = Get-Process -Id $PidValue -ErrorAction SilentlyContinue
        if ($null -eq $p) {
            break
        }
        Write-QueueLog "waiting for ${Name}: $PidValue"
        Start-Sleep -Seconds 180
    }
}

function Wait-ClassificationJobsGone() {
    while ($true) {
        $alive = @(Get-CimInstance Win32_Process | Where-Object {
            $_.CommandLine -match "run_rscd_surface_classification.py" -and
            $_.CommandLine -notmatch "smoke_"
        })
        if ($alive.Count -eq 0) {
            break
        }
        Write-QueueLog "waiting for classification jobs before high-priority formal launch: $($alive.ProcessId -join ', ')"
        Start-Sleep -Seconds 180
    }
}

function Invoke-FormalPromotion([string]$Selector, [string]$Label) {
    Write-QueueLog "selecting $Label promotion via $Selector"
    $decisionText = & $Python $Selector
    Write-QueueLog "$Label decision: $decisionText"
    $decision = $decisionText | ConvertFrom-Json
    if ($null -eq $decision.promoted) {
        Write-QueueLog "no $Label candidate promoted"
        return
    }

    Wait-ClassificationJobsGone

    $outDir = [string]$decision.promoted.formal_output_dir
    if (Test-Path (Join-Path $outDir "evaluate_test.json")) {
        Write-QueueLog "formal result already exists for $outDir; skipping $Label"
        return
    }
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null

    $extra = @()
    foreach ($arg in $decision.promoted.formal_args) {
        $extra += [string]$arg
    }

    Write-QueueLog "launching high-priority formal $Label candidate $($decision.promoted.name)"
    & $Python scripts\run_rscd_surface_classification.py `
        --output-dir $outDir `
        --epochs 20 `
        --image-size 192 `
        --batch-size 12 `
        --grad-accum-steps 2 `
        --samples-per-epoch 36000 `
        --num-workers 2 `
        --prefetch-factor 2 `
        --early-stop-patience 5 `
        --log-every-steps 150 `
        @extra *>> $Log

    Write-QueueLog "formal $Label candidate complete"
    & $Python scripts\compare_rscd_surface_candidates.py *>> $Log
    & $Python scripts\compare_rscd_class_slices.py *>> $Log
    & $Python scripts\write_rscd_formal_result_summary.py *>> $Log
    & $Python scripts\write_rscd_external_sota_gap.py *>> $Log
    & $Python scripts\write_rscd_decision_dashboard.py *>> $Log
    & $Python scripts\write_goal_completion_audit.py *>> $Log
}

Write-QueueLog "started high-priority formal promotion watcher"
Wait-ProcessGone -PidValue $PriorityFastPid -Name "high-priority fast candidate queue"

Invoke-FormalPromotion -Selector "scripts\select_rscd_residual_adapter_promotion.py" -Label "residual-adapter"
Invoke-FormalPromotion -Selector "scripts\select_rscd_texture_film_promotion.py" -Label "texture-FiLM/Wavelet"

Write-QueueLog "high-priority formal promotion watcher complete"
