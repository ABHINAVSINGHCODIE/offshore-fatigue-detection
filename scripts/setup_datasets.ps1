# setup_datasets.ps1
# ══════════════════════════════════════════════════════════════
# Offshore Fatigue Detection — Dataset Download & Setup Script
# Run from the project root: .\scripts\setup_datasets.ps1
# ══════════════════════════════════════════════════════════════

$ErrorActionPreference = "Stop"

# ── Paths ─────────────────────────────────────────────────────
$ROOT       = Split-Path $PSScriptRoot -Parent
$RAW        = Join-Path $ROOT "data\raw"
$MRL_DIR    = Join-Path $RAW  "mrl_eye"
$MPII_DIR   = Join-Path $RAW  "mpii_pose"
$COCO_DIR   = Join-Path $RAW  "coco_pose"

# ── Utility functions ─────────────────────────────────────────
function Write-Header($msg) {
    Write-Host "`n$('═' * 60)" -ForegroundColor Cyan
    Write-Host "  $msg" -ForegroundColor Cyan
    Write-Host "$('═' * 60)" -ForegroundColor Cyan
}

function Write-OK($msg)   { Write-Host "  ✓ $msg" -ForegroundColor Green }
function Write-Skip($msg) { Write-Host "  ⏭  $msg" -ForegroundColor Yellow }
function Write-Info($msg) { Write-Host "  ℹ  $msg" -ForegroundColor White }
function Write-Err($msg)  { Write-Host "  ✗ $msg"  -ForegroundColor Red }

function DirExists($path) {
    return (Test-Path $path -PathType Container) -and
           (@(Get-ChildItem $path -Recurse -File -ErrorAction SilentlyContinue).Count -gt 0)
}

function Download-Resumable($url, $outfile) {
    Write-Info "Downloading: $url"
    Write-Info "       → $outfile"
    # curl.exe is built-in on Windows 10 1803+, supports -C - for resume
    & curl.exe -L -C - -o $outfile $url
    if ($LASTEXITCODE -ne 0) {
        throw "Download failed with exit code $LASTEXITCODE"
    }
}

# ════════════════════════════════════════════════════════════
# DATASET 1 — MRL Eye (Kaggle)
# ════════════════════════════════════════════════════════════
Write-Header "Dataset 1: MRL Eye Dataset (Kaggle)"

if (DirExists $MRL_DIR) {
    $cnt = @(Get-ChildItem $MRL_DIR -Recurse -File).Count
    Write-Skip "Already present ($cnt files). Skipping download."
} else {
    # Check for Kaggle API token
    $kaggleJson = "$env:USERPROFILE\.kaggle\kaggle.json"
    if (-not (Test-Path $kaggleJson)) {
        Write-Err "Kaggle API token NOT found at: $kaggleJson"
        Write-Host @"

  ┌─ HOW TO GET YOUR KAGGLE API TOKEN ───────────────────┐
  │  1. Go to https://www.kaggle.com                      │
  │  2. Click your profile picture → Settings            │
  │  3. Scroll to "API" section                           │
  │  4. Click "Create New Token"                          │
  │  5. A file named kaggle.json will download            │
  │  6. Move it to: $env:USERPROFILE\.kaggle\kaggle.json   │
  │  7. Re-run this script                                │
  └───────────────────────────────────────────────────────┘

"@ -ForegroundColor Yellow
        Write-Info "Skipping MRL Eye. Re-run after setting up Kaggle API."
    } else {
        Write-Info "Kaggle token found. Downloading MRL Eye dataset…"
        New-Item -ItemType Directory -Force $MRL_DIR | Out-Null
        Push-Location $MRL_DIR
        try {
            & kaggle datasets download -d akashshingha850/mrl-eye-dataset --unzip
            if ($LASTEXITCODE -eq 0) {
                Write-OK "MRL Eye downloaded and extracted."
            } else {
                Write-Err "kaggle CLI returned exit code $LASTEXITCODE"
            }
        } catch {
            Write-Err "MRL Eye download failed: $_"
        } finally {
            Pop-Location
        }
    }
}

# ════════════════════════════════════════════════════════════
# DATASET 2 — MPII Human Pose (~12.9 GB images + annotations)
# ════════════════════════════════════════════════════════════
Write-Header "Dataset 2: MPII Human Pose (~12.9 GB)"

$MPII_IMAGES_DIR = Join-Path $MPII_DIR "images"
$MPII_ANN_DIR    = Join-Path $MPII_DIR "annotations"
$MPII_TAR        = Join-Path $MPII_DIR "mpii_human_pose_v1.tar.gz"
$MPII_ANN_ZIP    = Join-Path $MPII_DIR "mpii_human_pose_v1_u12_2.zip"

New-Item -ItemType Directory -Force $MPII_IMAGES_DIR | Out-Null
New-Item -ItemType Directory -Force $MPII_ANN_DIR    | Out-Null

if (DirExists $MPII_IMAGES_DIR) {
    $cnt = @(Get-ChildItem $MPII_IMAGES_DIR -Recurse -File).Count
    Write-Skip "Images already present ($cnt files). Skipping image download."
} else {
    Write-Info "⚠  This is a 12.9 GB download. It will resume if interrupted."
    try {
        Download-Resumable `
            "https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz" `
            $MPII_TAR
        Write-Info "Extracting tar.gz (this may take several minutes)…"
        & tar.exe -xzf $MPII_TAR -C $MPII_IMAGES_DIR --strip-components=1
        Write-OK "MPII images extracted."
        Remove-Item $MPII_TAR -Force
    } catch {
        Write-Err "MPII images download/extract failed: $_"
        Write-Info "Partial download saved. Re-run to resume."
    }
}

if (DirExists $MPII_ANN_DIR) {
    Write-Skip "Annotations already present. Skipping."
} else {
    try {
        Download-Resumable `
            "https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1_u12_2.zip" `
            $MPII_ANN_ZIP
        Write-Info "Extracting annotations…"
        Expand-Archive -Path $MPII_ANN_ZIP -DestinationPath $MPII_ANN_DIR -Force
        Write-OK "MPII annotations extracted."
        Remove-Item $MPII_ANN_ZIP -Force
    } catch {
        Write-Err "MPII annotations download failed: $_"
    }
}

# ════════════════════════════════════════════════════════════
# DATASET 3 — COCO Pose 2017 val (~780 MB images + ~240 MB ann)
# ════════════════════════════════════════════════════════════
Write-Header "Dataset 3: COCO Pose 2017 (~1 GB)"

$COCO_IMAGES_DIR = Join-Path $COCO_DIR "images"
$COCO_ANN_DIR    = Join-Path $COCO_DIR "annotations"
$COCO_IMG_ZIP    = Join-Path $COCO_DIR "val2017.zip"
$COCO_ANN_ZIP    = Join-Path $COCO_DIR "annotations_trainval2017.zip"

New-Item -ItemType Directory -Force $COCO_IMAGES_DIR | Out-Null
New-Item -ItemType Directory -Force $COCO_ANN_DIR    | Out-Null

if (DirExists $COCO_IMAGES_DIR) {
    $cnt = @(Get-ChildItem $COCO_IMAGES_DIR -Recurse -File).Count
    Write-Skip "COCO images already present ($cnt files). Skipping."
} else {
    try {
        Download-Resumable `
            "http://images.cocodataset.org/zips/val2017.zip" `
            $COCO_IMG_ZIP
        Write-Info "Extracting COCO images…"
        Expand-Archive -Path $COCO_IMG_ZIP -DestinationPath $COCO_IMAGES_DIR -Force
        Write-OK "COCO images extracted."
        Remove-Item $COCO_IMG_ZIP -Force
    } catch {
        Write-Err "COCO images download failed: $_"
    }
}

if (DirExists $COCO_ANN_DIR) {
    Write-Skip "COCO annotations already present. Skipping."
} else {
    try {
        Download-Resumable `
            "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" `
            $COCO_ANN_ZIP
        Write-Info "Extracting COCO annotations…"
        Expand-Archive -Path $COCO_ANN_ZIP -DestinationPath $COCO_ANN_DIR -Force
        Write-OK "COCO annotations extracted."
        Remove-Item $COCO_ANN_ZIP -Force
    } catch {
        Write-Err "COCO annotations download failed: $_"
    }
}

# ════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ════════════════════════════════════════════════════════════
Write-Header "Dataset Summary"

$datasets = @(
    @{ Name="UTA-RLDD + YawDD (ZIP)"; Path="C:\Users\Abhinav\OneDrive\Desktop\Dataset Combined.zip"; Type="zip" }
    @{ Name="MRL Eye";               Path=$MRL_DIR;         Type="dir" }
    @{ Name="MPII Images";           Path=$MPII_IMAGES_DIR; Type="dir" }
    @{ Name="MPII Annotations";      Path=$MPII_ANN_DIR;    Type="dir" }
    @{ Name="COCO Images";           Path=$COCO_IMAGES_DIR; Type="dir" }
    @{ Name="COCO Annotations";      Path=$COCO_ANN_DIR;    Type="dir" }
)

$rows = foreach ($d in $datasets) {
    $exists = Test-Path $d.Path
    $count  = if ($exists -and $d.Type -eq "dir") {
        @(Get-ChildItem $d.Path -Recurse -File -ErrorAction SilentlyContinue).Count
    } elseif ($exists) { "1 (ZIP)" } else { "—" }
    $status = if ($exists) { "✅ Ready" } else { "❌ Missing" }

    [PSCustomObject]@{
        Dataset = $d.Name
        Status  = $status
        Files   = $count
        Path    = $d.Path
    }
}

$rows | Format-Table -AutoSize

Write-Host "`nSetup complete. Now run:" -ForegroundColor Cyan
Write-Host "  python train_models.py" -ForegroundColor White
Write-Host "  streamlit run app.py`n" -ForegroundColor White
