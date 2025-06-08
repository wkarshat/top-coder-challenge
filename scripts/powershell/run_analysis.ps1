# PowerShell Analysis Runner for Legacy Reimbursement System
# Usage: .\run_analysis.ps1 -DataSource "public.csv" -OutputDir "custom_output"

param(
    [Parameter(Mandatory=$false)]
    [string]$DataSource = "public.csv",
    
    [Parameter(Mandatory=$false)]
    [string]$OutputDir = $null,
    
    [Parameter(Mandatory=$false)]
    [switch]$QuickTest,
    
    [Parameter(Mandatory=$false)]
    [switch]$Verbose
)

# Set error handling
$ErrorActionPreference = "Stop"

# Get timestamp for logging
$timestamp = Get-Date -Format "HHmmss"
$dateStamp = Get-Date -Format "yyyyMMdd"

Write-Host "Legacy Reimbursement Analysis System - PowerShell Runner" -ForegroundColor Green
Write-Host "Timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray

try {
    # Check if virtual environment exists
    if (Test-Path "venv\Scripts\Activate.ps1") {
        Write-Host "Activating virtual environment..." -ForegroundColor Yellow
        & "venv\Scripts\Activate.ps1"
    } else {
        Write-Warning "Virtual environment not found. Using system Python."
    }
    
    # Check if data file exists
    if (-not (Test-Path $DataSource)) {
        throw "Data file '$DataSource' not found."
    }
    
    # Prepare output directory
    if (-not $OutputDir) {
        $OutputDir = "outputs\analysis_$timestamp"
    }
    
    Write-Host "Data Source: $DataSource" -ForegroundColor Cyan
    Write-Host "Output Directory: $OutputDir" -ForegroundColor Cyan
    
    # Run the analysis
    if ($QuickTest) {
        Write-Host "Running quick test..." -ForegroundColor Yellow
        python scripts\run_analysis.py --data-source $DataSource --quick-test
    } else {
        Write-Host "Running full analysis..." -ForegroundColor Yellow
        python scripts\run_analysis.py --data-source $DataSource
    }
    
    # Check if analysis completed successfully
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Analysis completed successfully!" -ForegroundColor Green
        
        # Save PowerShell log
        $logFile = "outputs\logs\powershell_log_$timestamp.log"
        $logContent = @"
PowerShell Analysis Run Log
===========================
Date: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
Data Source: $DataSource
Output Directory: $OutputDir
Quick Test: $QuickTest
Status: SUCCESS
Exit Code: $LASTEXITCODE
"@
        $logContent | Out-File -FilePath $logFile -Encoding UTF8
        Write-Host "PowerShell log saved: $logFile" -ForegroundColor Gray
        
    } else {
        throw "Analysis failed with exit code: $LASTEXITCODE"
    }
    
} catch {
    Write-Error "Error: $($_.Exception.Message)"
    
    # Save error log
    $errorLogFile = "outputs\logs\powershell_error_$timestamp.log"
    $errorContent = @"
PowerShell Analysis Error Log
============================
Date: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
Data Source: $DataSource
Error: $($_.Exception.Message)
Stack Trace: $($_.ScriptStackTrace)
"@
    $errorContent | Out-File -FilePath $errorLogFile -Encoding UTF8
    Write-Host "Error log saved: $errorLogFile" -ForegroundColor Red
    
    exit 1
}

Write-Host "PowerShell analysis runner completed." -ForegroundColor Green 