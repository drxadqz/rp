param(
    [int]$PollSeconds = 180,
    [int]$RestartCooldownSeconds = 600
)

$ErrorActionPreference = "Stop"

$ProjectDir = "E:\perception\friction_affordance_field"
$PythonExe = "D:\NMI_SPWFM_datasets\conda_envs\faf_paper\python.exe"
$Config = "configs\c3_farnet\c3_farnet_formal_fullmanifest_s133c_s96_boundary_earlyphysics_b16_20260715.yaml"
$RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_s133c_s96_boundary_earlyphysics_b16_20260715"
$LogPath = Join-Path $RunDir "s133c_resume_watchdog.log"

New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

function Write-Watchdog([string]$Message) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $LogPath -Value "[$stamp] $Message"
}

function Get-ActiveS133cTraining {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object {
            $_.CommandLine -like "*train.py*" -and
            $_.CommandLine -like "*c3_farnet_formal_fullmanifest_s133c_s96_boundary_earlyphysics_b16_20260715.yaml*"
        }
}

function Get-ActiveOtherRscdTraining {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object {
            (
                ($_.CommandLine -like "*train.py*" -and $_.CommandLine -like "*c3_farnet*") -or
                ($_.CommandLine -like "*train_coupled_factor_backbone.py*")
            ) -and
            -not ($_.CommandLine -like "*c3_farnet_formal_fullmanifest_s133c_s96_boundary_earlyphysics_b16_20260715.yaml*")
        }
}

Write-Watchdog "S133c resume watchdog started. PollSeconds=$PollSeconds RestartCooldownSeconds=$RestartCooldownSeconds"
$lastRestart = [datetime]::MinValue

while ($true) {
    $metrics = Join-Path $RunDir "test_metrics.json"
    if (Test-Path $metrics) {
        Write-Watchdog "S133c test_metrics.json exists. Watchdog exiting."
        exit 0
    }

    $active = Get-ActiveS133cTraining
    if ($active) {
        $activeText = ($active | Select-Object ProcessId,CommandLine | Out-String).Trim()
        Write-Watchdog "S133c active: $activeText"
        Start-Sleep -Seconds $PollSeconds
        continue
    }

    $other = Get-ActiveOtherRscdTraining
    if ($other) {
        $otherText = ($other | Select-Object ProcessId,CommandLine | Out-String).Trim()
        Write-Watchdog "Another RSCD training process is active; not restarting S133c: $otherText"
        Start-Sleep -Seconds $PollSeconds
        continue
    }

    $now = Get-Date
    $cooldownElapsed = ($now - $lastRestart).TotalSeconds
    if ($cooldownElapsed -lt $RestartCooldownSeconds) {
        Write-Watchdog "S133c inactive but restart cooldown still active: elapsed=${cooldownElapsed}s"
        Start-Sleep -Seconds $PollSeconds
        continue
    }

    $stepCheckpoint = Join-Path $RunDir "last_step_checkpoint.pth"
    $checkpointStatus = if (Test-Path $stepCheckpoint) { "step_checkpoint_present" } else { "no_step_checkpoint_restart_from_anchor" }
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $stdout = Join-Path $RunDir "train_stdout_$stamp.log"
    $stderr = Join-Path $RunDir "train_stderr_$stamp.log"
    Write-Watchdog "Restarting S133c because no active process and no metrics. status=$checkpointStatus stdout=$stdout stderr=$stderr"
    $process = Start-Process -FilePath $PythonExe `
        -ArgumentList @("-u", "train.py", "--config", $Config) `
        -WorkingDirectory $ProjectDir `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -WindowStyle Hidden `
        -PassThru
    $lastRestart = Get-Date
    Write-Watchdog "S133c restarted. pid=$($process.Id)"
    Start-Sleep -Seconds $PollSeconds
}
