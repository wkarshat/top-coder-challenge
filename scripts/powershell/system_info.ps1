# System Information and Diagnostics for Legacy Analysis System
# Usage: .\system_info.ps1 [-SaveToFile] [-Detailed]

param(
    [Parameter(Mandatory=$false)]
    [switch]$SaveToFile,
    
    [Parameter(Mandatory=$false)]
    [switch]$Detailed
)

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

Write-Host "Legacy Analysis System - System Information" -ForegroundColor Green
Write-Host "Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
Write-Host "=" * 60 -ForegroundColor Gray

# System Information
$systemInfo = @{
    "OS Version" = (Get-CimInstance Win32_OperatingSystem).Caption
    "OS Build" = (Get-CimInstance Win32_OperatingSystem).BuildNumber
    "PowerShell Version" = $PSVersionTable.PSVersion.ToString()
    "Python Version" = try { (python --version 2>&1) } catch { "Not found" }
    "Current Directory" = Get-Location
    "User" = $env:USERNAME
    "Computer Name" = $env:COMPUTERNAME
}

Write-Host "System Information:" -ForegroundColor Yellow
foreach ($key in $systemInfo.Keys) {
    Write-Host "  $key`: $($systemInfo[$key])" -ForegroundColor White
}

# Environment Check
Write-Host "`nEnvironment Check:" -ForegroundColor Yellow

# Virtual Environment
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "  Virtual Environment: ✓ Found" -ForegroundColor Green
} else {
    Write-Host "  Virtual Environment: ✗ Not found" -ForegroundColor Red
}

# Required Files
$requiredFiles = @("public.csv", "config.yaml", "analysis.yaml", "requirements.txt")
foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "  $file`: ✓ Found" -ForegroundColor Green
    } else {
        Write-Host "  $file`: ✗ Missing" -ForegroundColor Red
    }
}

# Directory Structure
Write-Host "`nDirectory Structure:" -ForegroundColor Yellow
$directories = @("src", "scripts", "outputs", "outputs/logs")
foreach ($dir in $directories) {
    if (Test-Path $dir) {
        $itemCount = (Get-ChildItem $dir -ErrorAction SilentlyContinue).Count
        Write-Host "  $dir`: ✓ Exists ($itemCount items)" -ForegroundColor Green
    } else {
        Write-Host "  $dir`: ✗ Missing" -ForegroundColor Red
    }
}

# Recent Outputs
Write-Host "`nRecent Analysis Outputs:" -ForegroundColor Yellow
if (Test-Path "outputs") {
    $recentDirs = Get-ChildItem "outputs" -Directory | 
                  Where-Object { $_.Name -match "analysis_\d{6}" } |
                  Sort-Object LastWriteTime -Descending |
                  Select-Object -First 5
    
    if ($recentDirs) {
        foreach ($dir in $recentDirs) {
            Write-Host "  $($dir.Name): $($dir.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor White
        }
    } else {
        Write-Host "  No recent analysis outputs found" -ForegroundColor Gray
    }
} else {
    Write-Host "  Outputs directory not found" -ForegroundColor Red
}

# Detailed Information (if requested)
if ($Detailed) {
    Write-Host "`nDetailed System Information:" -ForegroundColor Yellow
    
    # Memory Information
    $memory = Get-CimInstance Win32_ComputerSystem
    Write-Host "  Total RAM: $([math]::Round($memory.TotalPhysicalMemory / 1GB, 2)) GB" -ForegroundColor White
    
    # Disk Space
    $disk = Get-CimInstance Win32_LogicalDisk | Where-Object { $_.DriveType -eq 3 }
    foreach ($drive in $disk) {
        $freeGB = [math]::Round($drive.FreeSpace / 1GB, 2)
        $totalGB = [math]::Round($drive.Size / 1GB, 2)
        Write-Host "  Drive $($drive.DeviceID) Free: $freeGB GB / $totalGB GB" -ForegroundColor White
    }
    
    # Python Packages (if virtual environment is active)
    if (Test-Path "venv\Scripts\Activate.ps1") {
        Write-Host "`nPython Packages:" -ForegroundColor Yellow
        try {
            & "venv\Scripts\Activate.ps1"
            $packages = pip list 2>$null | Select-String -Pattern "^[a-zA-Z]" | Select-Object -First 10
            foreach ($package in $packages) {
                Write-Host "  $package" -ForegroundColor White
            }
        } catch {
            Write-Host "  Could not retrieve package list" -ForegroundColor Red
        }
    }
}

# Performance Test
Write-Host "`nQuick Performance Test:" -ForegroundColor Yellow
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
1..1000 | ForEach-Object { $_ * 2 } | Out-Null
$stopwatch.Stop()
Write-Host "  Simple calculation (1000 iterations): $($stopwatch.ElapsedMilliseconds) ms" -ForegroundColor White

# Save to file if requested
if ($SaveToFile) {
    $outputFile = "outputs\logs\system_info_$timestamp.log"
    
    $report = @"
Legacy Analysis System - System Information Report
Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
==================================================

SYSTEM INFORMATION:
$(foreach ($key in $systemInfo.Keys) { "  $key`: $($systemInfo[$key])" })

ENVIRONMENT STATUS:
  Virtual Environment: $(if (Test-Path "venv\Scripts\Activate.ps1") { "✓ Found" } else { "✗ Not found" })
$(foreach ($file in $requiredFiles) { "  $file`: $(if (Test-Path $file) { "✓ Found" } else { "✗ Missing" })" })

DIRECTORY STRUCTURE:
$(foreach ($dir in $directories) { "  $dir`: $(if (Test-Path $dir) { "✓ Exists" } else { "✗ Missing" })" })

PERFORMANCE:
  Simple calculation test: $($stopwatch.ElapsedMilliseconds) ms

Report saved: $outputFile
"@
    
    $report | Out-File -FilePath $outputFile -Encoding UTF8
    Write-Host "`nSystem information saved to: $outputFile" -ForegroundColor Green
}

Write-Host "`nSystem information check completed." -ForegroundColor Green 