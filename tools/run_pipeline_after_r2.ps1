# Unattended pipeline for the stages after Study 2, run in the clean checkout:
#   Study 3 (student distillation) -> baselines and significance -> tables -> latent figure.
# Each stage is retried up to three times. Progress goes to runs\pipeline.log.
param(
    [string]$Final = "C:\Project\TradingSystem_final",
    [string]$Main = "C:\Project\TradingSystem"
)
$py = Join-Path $Main ".venv\Scripts\python.exe"
$log = Join-Path $Final "runs\pipeline.log"
function Note($msg) { Add-Content -Path $log -Value ("{0} {1}" -f (Get-Date -Format s), $msg) -Encoding utf8 }

Set-Location $Final
$env:PYTHONPATH = Join-Path $Final "src"
$env:PYTHONUNBUFFERED = "1"
Remove-Item Env:\TS_SMOKE_RUN -ErrorAction SilentlyContinue
Note ("pipeline start, commit " + (git rev-parse --short HEAD))

# Wait until Study 2 has written a complete result.
while ($true) {
    $m = Join-Path $Final "runs\study2\metrics.json"
    if (Test-Path $m) {
        try { $status = (Get-Content $m -Raw -Encoding utf8 | ConvertFrom-Json).status } catch { $status = "" }
        if ($status -eq "complete" -and (Test-Path (Join-Path $Final "runs\study2\teacher_signals.npz"))) { break }
    }
    Start-Sleep -Seconds 60
}
Note "study 2 result is complete"

function Run-Stage($name, $cmdline, $outlog) {
    New-Item -ItemType Directory -Force (Split-Path (Join-Path $Final $outlog)) | Out-Null
    for ($try = 1; $try -le 3; $try++) {
        Note "$name start (attempt $try)"
        cmd /c "`"$py`" $cmdline >> $outlog 2>&1"
        if ($LASTEXITCODE -eq 0) { Note "$name done"; return $true }
        Note "$name FAILED (exit $LASTEXITCODE)"
        Start-Sleep -Seconds 30
    }
    return $false
}

if (-not (Run-Stage "study 3" "scripts\run_study3.py" "runs\study3\r3.log")) { Note "pipeline stopped at study 3"; exit 1 }
if (-not (Run-Stage "baselines" "scripts\run_baselines_and_eval.py" "runs\baselines\r4.log")) { Note "pipeline stopped at baselines"; exit 1 }
if (-not (Run-Stage "tables" "scripts\make_tables.py" "runs\tables\tables.log")) { Note "pipeline stopped at tables"; exit 1 }
if (-not (Run-Stage "latent figure" "scripts\make_latent_figure.py" "runs\figures\figure.log")) { Note "pipeline stopped at latent figure"; exit 1 }
Note "pipeline complete"
