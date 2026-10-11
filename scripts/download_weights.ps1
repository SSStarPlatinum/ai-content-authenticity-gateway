# ============================================================
# Model Weight Download Script (Windows PowerShell)
# Usage: .\scripts\download_weights.ps1
# ============================================================

$WeightsDir = Join-Path $PSScriptRoot "..\models\weights"
New-Item -ItemType Directory -Path $WeightsDir -Force | Out-Null

Write-Host "============================================" -ForegroundColor Cyan
Write-Host " AI Content Authenticity - Model Weights" -ForegroundColor Cyan
Write-Host " Output dir: $WeightsDir" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

function Check-Model {
    param(
        [string]$Title,
        [string]$PthFile,
        [string]$OnnxFile,
        [string]$ModelId,
        [string]$RepoUrl
    )

    $pthPath = Join-Path $WeightsDir $PthFile
    $onnxPath = Join-Path $WeightsDir $OnnxFile

    Write-Host ""
    Write-Host $Title -ForegroundColor Yellow

    if (Test-Path $onnxPath) {
        $sizeMB = [math]::Round((Get-Item $onnxPath).Length / 1MB, 1)
        Write-Host ("  [OK] ONNX found: {0} ({1} MB)" -f $onnxPath, $sizeMB) -ForegroundColor Green
        return $true
    }

    if (Test-Path $pthPath) {
        $sizeMB = [math]::Round((Get-Item $pthPath).Length / 1MB, 1)
        Write-Host ("  [OK] PyTorch weight found: {0} ({1} MB)" -f $pthPath, $sizeMB) -ForegroundColor Green
        Write-Host "  Convert to ONNX:"
        Write-Host ("  python scripts/convert_to_onnx.py --model {0} --checkpoint {1} --output {2}" -f $ModelId, $pthPath, $onnxPath)
        return $true
    }

    Write-Host "  [--] Weight file not found"
    Write-Host ("  Repo: {0}" -f $RepoUrl)
    Write-Host ("  Download .pth and save as: {0}" -f $pthPath)
    return $false
}

$r1 = Check-Model -Title "[1/3] CNNSpotter (ResNet50)" -PthFile "cnn_spotter.pth" -OnnxFile "cnn_spotter.onnx" -ModelId "cnn_spotter" -RepoUrl "https://github.com/peterwang512/CNNDetection"

$r2 = Check-Model -Title "[2/3] UniversalFakeDetect (CLIP)" -PthFile "universal_fake.pth" -OnnxFile "universal_fake.onnx" -ModelId "universal_fake" -RepoUrl "https://github.com/Yuheng-Li/UniversalFakeDetect"

$r3 = Check-Model -Title "[3/3] AeroBlade (Diffusion Detection)" -PthFile "aeroblade.pth" -OnnxFile "aeroblade.onnx" -ModelId "aeroblade" -RepoUrl "https://github.com/EPFL-VILAB/AeroBlade"

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host " Check Complete" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

$ready = @()
if ($r1) { $ready += "cnn_spotter" }
if ($r2) { $ready += "universal_fake" }
if ($r3) { $ready += "aeroblade" }

if ($ready.Count -gt 0) {
    Write-Host ""
    Write-Host ("Ready deep learning models: {0}" -f ($ready -join ", ")) -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "No deep learning weights found." -ForegroundColor Yellow
    Write-Host "Download manually from the GitHub repos listed above." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "weights directory contents:"
$files = Get-ChildItem $WeightsDir -ErrorAction SilentlyContinue
if ($files) {
    foreach ($f in $files) {
        $sizeMB = [math]::Round($f.Length / 1MB, 2)
        Write-Host ("  {0} ({1} MB)" -f $f.Name, $sizeMB)
    }
} else {
    Write-Host "  (empty)"
}

Write-Host ""
Write-Host "5 weight-free models (freq/ela/noise/cfa/color) are ready to use." -ForegroundColor Green
Write-Host "Deep learning models are optional. After download, add model IDs to ENABLED_MODELS in .env" -ForegroundColor Yellow
