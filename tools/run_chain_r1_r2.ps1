# Runs, in order and unattended: (1) wait for the LLM sentiment scores, (2) Study 1 main run,
# (3) Study 2 main run. Everything executes in the clean checkout so that the ledger records
# are final. Each step stops the chain if it fails. Progress is appended to chain.log.
param(
    [string]$Final = "C:\Project\TradingSystem_final",
    [string]$Main = "C:\Project\TradingSystem"
)
$ErrorActionPreference = "Stop"
$py = Join-Path $Main ".venv\Scripts\python.exe"
$log = Join-Path $Final "runs\chain.log"
function Note($msg) { Add-Content -Path $log -Value ("{0} {1}" -f (Get-Date -Format s), $msg) -Encoding utf8 }

New-Item -ItemType Directory -Force (Join-Path $Final "runs\study1"), (Join-Path $Final "runs\study2") | Out-Null
Set-Location $Final
$env:PYTHONPATH = Join-Path $Final "src"
$env:PYTHONUNBUFFERED = "1"
Remove-Item Env:\TS_SMOKE_RUN -ErrorAction SilentlyContinue
Note ("chain start, commit " + (git rev-parse --short HEAD))

$summary = Join-Path $Main "data\news_sentiment_llm_summary.json"
while (-not (Test-Path $summary)) { Start-Sleep -Seconds 30 }
Note "sentiment scores present"

$dirty = git status --short | Where-Object { $_ -notmatch "ledger/" }
if ($dirty) { Note ("checkout not clean: " + ($dirty -join "; ")); exit 1 }

Note "study 1 start"
cmd /c "`"$py`" scripts\run_study1.py > runs\study1\r1_chain.log 2>&1"
if ($LASTEXITCODE -ne 0) { Note "study 1 FAILED (exit $LASTEXITCODE)"; exit 1 }
Note "study 1 done"
$dst = Join-Path $Main "runs\study1"
New-Item -ItemType Directory -Force $dst | Out-Null
Copy-Item (Join-Path $Final "runs\study1\metrics.json") $dst -Force
Copy-Item (Join-Path $Final "runs\study1\study1_oos_signals.npz") $dst -Force
Copy-Item (Join-Path $Final "runs\study1\training_logs") $dst -Recurse -Force

Note "study 2 start"
cmd /c "`"$py`" scripts\run_study2.py --debate-stride 4 --counterfactual-every 2 --edge-top-k 30 --llm-backend ollama --llm-endpoint http://127.0.0.1:11434/api/chat --llm-model deepseek-r1:32b >> runs\study2\r2_chain.log 2>&1"
if ($LASTEXITCODE -ne 0) { Note "study 2 FAILED or incomplete (exit $LASTEXITCODE)"; exit 1 }
Note "study 2 done"
