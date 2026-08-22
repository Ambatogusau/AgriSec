$ErrorActionPreference = "Stop"

$source = Join-Path $env:USERPROFILE ".ollama\models\blobs\sha256-183715c435899236895da3869489cc30ac241476b4971a20285b1a462818a5b4"
$modelDir = Join-Path $PSScriptRoot "model"
$target = Join-Path $modelDir "model.gguf"
$rootTarget = Join-Path $PSScriptRoot "model.gguf"

if (-not (Test-Path -LiteralPath $modelDir)) {
    New-Item -ItemType Directory -Force -Path $modelDir | Out-Null
}

if (-not (Test-Path -LiteralPath $source)) {
    throw "Expected Ollama GGUF blob was not found at $source. Run: ollama pull qwen2.5:1.5b"
}

Copy-Item -LiteralPath $source -Destination $target -Force
Copy-Item -LiteralPath $source -Destination $rootTarget -Force
Write-Host "Copied Qwen2.5 1.5B GGUF model to $target and $rootTarget"
