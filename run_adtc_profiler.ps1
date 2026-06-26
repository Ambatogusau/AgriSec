$ErrorActionPreference = "Stop"

$repoRoot = $PSScriptRoot
$llamaDir = Join-Path $repoRoot "tools\llama.cpp"
$llamaBench = Join-Path $llamaDir "llama-bench.exe"
$archive = Join-Path $repoRoot "tools\llama-b9803-bin-win-cpu-x64.zip"
$downloadUrl = "https://github.com/ggml-org/llama.cpp/releases/download/b9803/llama-b9803-bin-win-cpu-x64.zip"

if (-not (Test-Path -LiteralPath (Join-Path $repoRoot "model.gguf"))) {
    & (Join-Path $repoRoot "download_model.ps1")
}

if (-not (Test-Path -LiteralPath $llamaBench)) {
    New-Item -ItemType Directory -Force -Path (Join-Path $repoRoot "tools") | Out-Null
    Invoke-WebRequest -Uri $downloadUrl -OutFile $archive
    Expand-Archive -Path $archive -DestinationPath $llamaDir -Force
}

$env:PYTHONIOENCODING = "utf-8"
$env:PATH = "$llamaDir;$env:PATH"

adtc-profiler run `
    --submission $repoRoot `
    --mode participant `
    --output (Join-Path $repoRoot "submission.json") `
    --skip-accuracy
