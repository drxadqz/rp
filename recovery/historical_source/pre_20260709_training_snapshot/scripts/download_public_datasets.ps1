param(
    [string]$DataRoot = "D:\NMI_SPWFM_datasets\friction_affordance_data",
    [switch]$KeepArchives
)

$ErrorActionPreference = "Stop"

function Write-Step($Message) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$stamp] $Message"
}

function Require-FreeSpace($Path, [int64]$BytesRequired) {
    $drive = (Get-Item -LiteralPath $Path).PSDrive
    if ($drive.Free -lt $BytesRequired) {
        $needGb = [math]::Round($BytesRequired / 1GB, 2)
        $freeGb = [math]::Round($drive.Free / 1GB, 2)
        throw "Insufficient free space on $($drive.Root): need about ${needGb}GB, free ${freeGb}GB"
    }
}

function Download-Extract-Remove($Name, $Url, $ArchivePath, $ExtractRoot, $ExpectedDir) {
    Write-Step "${Name}: checking destination"
    if (Test-Path -LiteralPath $ExpectedDir) {
        Write-Step "${Name}: already extracted at $ExpectedDir"
        return
    }

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $ArchivePath) | Out-Null
    New-Item -ItemType Directory -Force -Path $ExtractRoot | Out-Null

    Write-Step "${Name}: downloading to $ArchivePath"
    & curl.exe -L -C - --retry 10 --retry-delay 10 --connect-timeout 30 --speed-limit 1024 --speed-time 120 -o $ArchivePath $Url
    if ($LASTEXITCODE -ne 0) {
        throw "$Name download failed with exit code $LASTEXITCODE"
    }

    Write-Step "${Name}: extracting to $ExtractRoot"
    & tar.exe -xf $ArchivePath -C $ExtractRoot
    if ($LASTEXITCODE -ne 0) {
        throw "$Name extraction failed with exit code $LASTEXITCODE"
    }

    if (-not (Test-Path -LiteralPath $ExpectedDir)) {
        Write-Step "${Name}: expected directory not found after extraction; listing extract root"
        Get-ChildItem -LiteralPath $ExtractRoot | Select-Object Name,FullName | Format-Table
        throw "$Name extraction finished but expected directory is missing: $ExpectedDir"
    }

    if (-not $KeepArchives) {
        Write-Step "${Name}: removing archive to save disk space"
        Remove-Item -LiteralPath $ArchivePath -Force
    }
}

New-Item -ItemType Directory -Force -Path $DataRoot | Out-Null
$downloads = Join-Path $DataRoot "_downloads"
New-Item -ItemType Directory -Force -Path $downloads | Out-Null

Write-Step "Dataset root: $DataRoot"
Write-Step "Free space before download:"
Get-PSDrive -PSProvider FileSystem | Where-Object { $_.Name -in @("D", "E", "F") } |
    Select-Object Name,Used,Free,Root | Format-Table

# The three datasets below match the current project manifests/configs.
Download-Extract-Remove `
    -Name "RSCD/RSXD" `
    -Url "https://ndownloader.figshare.com/files/36625041" `
    -ArchivePath (Join-Path $downloads "RSCD.zip") `
    -ExtractRoot (Join-Path $DataRoot "RSCD_raw") `
    -ExpectedDir (Join-Path $DataRoot "RSCD_raw\RSCD dataset-1million")

Download-Extract-Remove `
    -Name "RoadSaW-150_s" `
    -Url "https://downloads.viscoda.com/research/roadsaw/RoadSaW-150_s.zip" `
    -ArchivePath (Join-Path $downloads "RoadSaW-150_s.zip") `
    -ExtractRoot $DataRoot `
    -ExpectedDir (Join-Path $DataRoot "RoadSaW-150_s")

Download-Extract-Remove `
    -Name "RoadSC aligned 150_l" `
    -Url "https://downloads.viscoda.com/research/roadsc/RoadSC-balanced_to_RoadSaW12-150_l.zip" `
    -ArchivePath (Join-Path $downloads "RoadSC-balanced_to_RoadSaW12-150_l.zip") `
    -ExtractRoot $DataRoot `
    -ExpectedDir (Join-Path $DataRoot "RoadSC-balanced_to_RoadSaW12-150_l")

Write-Step "Free space after download:"
Get-PSDrive -PSProvider FileSystem | Where-Object { $_.Name -in @("D", "E", "F") } |
    Select-Object Name,Used,Free,Root | Format-Table

Write-Step "Done. Next run scripts\audit_data.py and scripts\build_manifests.py from the project root."
