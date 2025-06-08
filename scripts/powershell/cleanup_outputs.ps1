# Output Directory Cleanup for Legacy Analysis System
# Usage: .\cleanup_outputs.ps1 [-DaysOld 7] [-DryRun] [-Force]

param(
    [Parameter(Mandatory=$false)]
    [int]$DaysOld = 7,
    
    [Parameter(Mandatory=$false)]
    [switch]$DryRun,
    
    [Parameter(Mandatory=$false)]
    [switch]$Force,
    
    [Parameter(Mandatory=$false)]
    [switch]$Interactive
)

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

Write-Host "Legacy Analysis System - Output Cleanup" -ForegroundColor Green
Write-Host "Cleanup Date: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
Write-Host "=" * 50 -ForegroundColor Gray

if (-not (Test-Path "outputs")) {
    Write-Host "No outputs directory found. Nothing to clean." -ForegroundColor Yellow
    exit 0
}

# Calculate cutoff date
$cutoffDate = (Get-Date).AddDays(-$DaysOld)
Write-Host "Cleaning outputs older than: $($cutoffDate.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor Cyan

# Find directories to clean
$analysisPattern = "analysis_\d{6}"
$regressionPattern = "linear_regression_\d{6}"

$oldDirs = Get-ChildItem "outputs" -Directory | Where-Object {
    ($_.Name -match $analysisPattern -or $_.Name -match $regressionPattern) -and
    $_.LastWriteTime -lt $cutoffDate
} | Sort-Object LastWriteTime

if ($oldDirs.Count -eq 0) {
    Write-Host "No old output directories found." -ForegroundColor Green
    exit 0
}

Write-Host "`nFound $($oldDirs.Count) directories to clean:" -ForegroundColor Yellow
foreach ($dir in $oldDirs) {
    $size = try {
        $sizeBytes = (Get-ChildItem $dir.FullName -Recurse -File | Measure-Object -Property Length -Sum).Sum
        if ($sizeBytes -gt 1MB) {
            "$([math]::Round($sizeBytes / 1MB, 2)) MB"
        } elseif ($sizeBytes -gt 1KB) {
            "$([math]::Round($sizeBytes / 1KB, 2)) KB"
        } else {
            "$sizeBytes bytes"
        }
    } catch {
        "Unknown size"
    }
    
    Write-Host "  $($dir.Name) - $($dir.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss')) - $size" -ForegroundColor White
}

# Calculate total space to be freed
$totalSize = try {
    $totalBytes = ($oldDirs | ForEach-Object {
        (Get-ChildItem $_.FullName -Recurse -File | Measure-Object -Property Length -Sum).Sum
    } | Measure-Object -Sum).Sum
    
    if ($totalBytes -gt 1GB) {
        "$([math]::Round($totalBytes / 1GB, 2)) GB"
    } elseif ($totalBytes -gt 1MB) {
        "$([math]::Round($totalBytes / 1MB, 2)) MB"
    } else {
        "$([math]::Round($totalBytes / 1KB, 2)) KB"
    }
} catch {
    "Unknown"
}

Write-Host "`nTotal space to be freed: $totalSize" -ForegroundColor Cyan

# Dry run mode
if ($DryRun) {
    Write-Host "`n[DRY RUN] No files will be deleted." -ForegroundColor Yellow
    Write-Host "The above directories would be removed in a real run." -ForegroundColor Yellow
    exit 0
}

# Interactive confirmation
if ($Interactive -and -not $Force) {
    $response = Read-Host "`nDo you want to delete these directories? (y/N)"
    if ($response -ne 'y' -and $response -ne 'Y') {
        Write-Host "Cleanup cancelled by user." -ForegroundColor Yellow
        exit 0
    }
}

# Safety check - don't delete if not forced and directories are very recent
if (-not $Force) {
    $recentDirs = $oldDirs | Where-Object { $_.LastWriteTime -gt (Get-Date).AddHours(-2) }
    if ($recentDirs.Count -gt 0) {
        Write-Warning "Some directories are less than 2 hours old. Use -Force to override."
        Write-Host "Recent directories:" -ForegroundColor Yellow
        foreach ($dir in $recentDirs) {
            Write-Host "  $($dir.Name) - $($dir.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor White
        }
        exit 1
    }
}

# Perform cleanup
Write-Host "`nStarting cleanup..." -ForegroundColor Yellow
$deletedCount = 0
$errors = @()

foreach ($dir in $oldDirs) {
    try {
        Write-Host "Deleting: $($dir.Name)" -ForegroundColor Gray
        Remove-Item $dir.FullName -Recurse -Force
        $deletedCount++
    } catch {
        $errors += "Failed to delete $($dir.Name): $($_.Exception.Message)"
        Write-Warning "Failed to delete $($dir.Name): $($_.Exception.Message)"
    }
}

# Summary
Write-Host "`nCleanup Summary:" -ForegroundColor Green
Write-Host "  Directories deleted: $deletedCount" -ForegroundColor White
Write-Host "  Errors: $($errors.Count)" -ForegroundColor White
Write-Host "  Space freed: $totalSize" -ForegroundColor White

if ($errors.Count -gt 0) {
    Write-Host "`nErrors encountered:" -ForegroundColor Red
    foreach ($error in $errors) {
        Write-Host "  $error" -ForegroundColor Red
    }
}

# Save cleanup log
$logFile = "outputs\logs\cleanup_log_$timestamp.log"
$logContent = @"
Legacy Analysis System - Cleanup Log
====================================
Date: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
Cutoff Date: $($cutoffDate.ToString('yyyy-MM-dd HH:mm:ss'))
Days Old Threshold: $DaysOld
Dry Run: $DryRun
Force: $Force

DIRECTORIES PROCESSED:
$(foreach ($dir in $oldDirs) { "  $($dir.Name) - $($dir.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))" })

SUMMARY:
  Total directories found: $($oldDirs.Count)
  Directories deleted: $deletedCount
  Errors: $($errors.Count)
  Space freed: $totalSize

$(if ($errors.Count -gt 0) { "ERRORS:`n$(foreach ($error in $errors) { "  $error" })" })
"@

try {
    $logContent | Out-File -FilePath $logFile -Encoding UTF8
    Write-Host "`nCleanup log saved: $logFile" -ForegroundColor Gray
} catch {
    Write-Warning "Could not save cleanup log: $($_.Exception.Message)"
}

Write-Host "`nCleanup completed." -ForegroundColor Green 