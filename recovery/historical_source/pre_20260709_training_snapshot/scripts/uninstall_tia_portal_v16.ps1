param(
    [string] $LogRoot = "E:\perception\tia_uninstall_logs",
    [switch] $CleanupInstallFolder
)

$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"

New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$MainLog = Join-Path $LogRoot "tia_portal_v16_uninstall_$stamp.log"
$SummaryJson = Join-Path $LogRoot "tia_portal_v16_uninstall_$stamp.summary.json"

function Write-Log {
    param([string] $Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -Path $MainLog -Value $line -Encoding UTF8
    Write-Host $line
}

function Test-Admin {
    $principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (!(Test-Admin)) {
    Write-Log "ERROR: This script must be run as Administrator."
    exit 740
}

Write-Log "Starting Siemens TIA Portal V16 uninstall."
Write-Log "Log root: $LogRoot"

$processPatterns = @(
    "Portal", "Siemens", "SIMATIC", "WinCC", "PLCSIM", "S7", "ALMPanelPlugin"
)

foreach ($proc in Get-Process -ErrorAction SilentlyContinue) {
    if ($processPatterns | Where-Object { $proc.ProcessName -match $_ }) {
        try {
            Write-Log "Stopping process: $($proc.ProcessName) [$($proc.Id)]"
            Stop-Process -Id $proc.Id -Force -ErrorAction Stop
        } catch {
            Write-Log "WARN: Could not stop process $($proc.ProcessName): $($_.Exception.Message)"
        }
    }
}

$servicePattern = "Siemens|SIMATIC|S7|ALM|Automation|WinCC|TIA|MSSQL\$WINCC|SQLTELEMETRY\$WINCC|SQLAgent\$WINCC"
foreach ($svc in Get-Service -ErrorAction SilentlyContinue | Where-Object { $_.Name -match $servicePattern -or $_.DisplayName -match $servicePattern }) {
    if ($svc.Status -eq "Running") {
        try {
            Write-Log "Stopping service: $($svc.Name) ($($svc.DisplayName))"
            Stop-Service -Name $svc.Name -Force -ErrorAction Stop
        } catch {
            Write-Log "WARN: Could not stop service $($svc.Name): $($_.Exception.Message)"
        }
    }
}

$uninstallPaths = @(
    "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*",
    "HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*",
    "HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*"
)

$namePattern = "Siemens|TIA|Portal|STEP 7|WinCC|SIMATIC|Startdrive|PLCSIM|Totally Integrated|Automation License Manager|S7-PCT|ProSave"
$items = Get-ItemProperty $uninstallPaths -ErrorAction SilentlyContinue |
    Where-Object { $_.DisplayName -and $_.DisplayName -match $namePattern } |
    Sort-Object DisplayName -Unique

$msiItems = @()
foreach ($item in $items) {
    $guid = $null
    if ($item.UninstallString -match "\{[0-9A-Fa-f-]{36}\}") {
        $guid = $Matches[0]
    } elseif ($item.PSChildName -match "^\{[0-9A-Fa-f-]{36}\}$") {
        $guid = $item.PSChildName
    }
    if ($guid) {
        $rank = 50
        if ($item.DisplayName -match "Runtime|Project Server|Administrator|PLCSIM|S7-PCT|ProSave") { $rank = 10 }
        if ($item.DisplayName -match "Hardware Support|Support Base|WinCC|STEP 7|HM |HMI|Simatic|Openness|Multiuser|Version Control|TIACOMPCHECK") { $rank = 20 }
        if ($item.DisplayName -match "TIA Portal Single SetupPackage|Totally Integrated Automation Portal V16$") { $rank = 30 }
        if ($item.DisplayName -match "Automation License Manager") { $rank = 90 }
        $msiItems += [pscustomobject]@{
            Rank = $rank
            DisplayName = $item.DisplayName
            Version = $item.DisplayVersion
            Guid = $guid
            UninstallString = $item.UninstallString
        }
    } else {
        Write-Log "INFO: No MSI product code found, will leave to parent uninstaller/registry: $($item.DisplayName)"
    }
}

$msiItems = $msiItems | Sort-Object Rank, DisplayName -Unique
Write-Log "MSI uninstall candidates: $($msiItems.Count)"

$results = @()
foreach ($item in $msiItems) {
    $safeName = ($item.DisplayName -replace '[^\p{L}\p{Nd}\._ -]+', '_').Trim()
    if ($safeName.Length -gt 90) {
        $safeName = $safeName.Substring(0, 90)
    }
    $msiLog = Join-Path $LogRoot ("msi_{0}_{1}.log" -f $stamp, $safeName)
    Write-Log "Uninstalling: $($item.DisplayName) [$($item.Guid)]"
    $exitCode = $null
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        & "$env:SystemRoot\System32\msiexec.exe" /x $item.Guid /qn /norestart REBOOT=ReallySuppress /L*v $msiLog
        $exitCode = $LASTEXITCODE
        Write-Log "Exit code for $($item.DisplayName), attempt ${attempt}: $exitCode"
        if ($exitCode -ne 1618) {
            break
        }
        Start-Sleep -Seconds 20
    }
    $results += [pscustomobject]@{
        DisplayName = $item.DisplayName
        Guid = $item.Guid
        ExitCode = $exitCode
        Log = $msiLog
    }
}

$remaining = Get-ItemProperty $uninstallPaths -ErrorAction SilentlyContinue |
    Where-Object { $_.DisplayName -and $_.DisplayName -match $namePattern } |
    Select-Object DisplayName, DisplayVersion, Publisher, InstallLocation, UninstallString, PSChildName |
    Sort-Object DisplayName

$successCodes = @(0, 1605, 1614, 3010)
$failures = @($results | Where-Object { $successCodes -notcontains $_.ExitCode })

if ($CleanupInstallFolder -and $failures.Count -eq 0) {
    $installFolder = "E:\PLCbotu16"
    if (Test-Path -LiteralPath $installFolder) {
        try {
            Write-Log "Removing leftover install folder: $installFolder"
            Remove-Item -LiteralPath $installFolder -Recurse -Force -ErrorAction Stop
        } catch {
            Write-Log "WARN: Could not remove leftover install folder ${installFolder}: $($_.Exception.Message)"
        }
    }
} elseif ($CleanupInstallFolder) {
    Write-Log "Skipping install folder cleanup because uninstall failures were detected."
}

$summary = [pscustomobject]@{
    StartedAt = $stamp
    MainLog = $MainLog
    Results = $results
    FailureCount = $failures.Count
    Failures = $failures
    Remaining = $remaining
}

$summary | ConvertTo-Json -Depth 6 | Set-Content -Path $SummaryJson -Encoding UTF8
Write-Log "Summary written: $SummaryJson"
Write-Log "Failure count: $($failures.Count)"
Write-Log "Remaining matching installed entries: $(@($remaining).Count)"
Write-Log "Finished. A reboot may be required."

exit $(if ($failures.Count -eq 0) { 0 } else { 1 })
