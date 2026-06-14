param(
    [string]$OutputRoot = "release"
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$stagingRoot = Join-Path $projectRoot $OutputRoot
$packageRoot = Join-Path $stagingRoot "HTL-UAV-IDS-submission"
$sourceRoot = Join-Path $packageRoot "source_code"
$zipPath = Join-Path $stagingRoot "HTL-UAV-IDS-submission.zip"

$includeTopLevel = @(
    "src",
    "scripts",
    "weights",
    "results",
    "requirements.txt",
    "README.md",
    "README_submission.md",
    "ONLINE_DEMO_DEPLOY.md",
    "DESIGN_DOC_REVISION_GUIDE.md"
)

$excludedDirNames = @(
    ".venv",
    ".idea",
    "__pycache__",
    "node_modules",
    "dist",
    ".vite",
    "release"
)

if (Test-Path $packageRoot) {
    Remove-Item -LiteralPath $packageRoot -Recurse -Force
}

if (Test-Path $zipPath) {
    Remove-Item -LiteralPath $zipPath -Force
}

New-Item -ItemType Directory -Path $sourceRoot -Force | Out-Null

function Copy-FilteredTree {
    param(
        [string]$SourcePath,
        [string]$DestinationPath
    )

    $item = Get-Item -LiteralPath $SourcePath
    if ($item.PSIsContainer) {
        if ($excludedDirNames -contains $item.Name) {
            return
        }

        New-Item -ItemType Directory -Path $DestinationPath -Force | Out-Null

        foreach ($child in Get-ChildItem -LiteralPath $SourcePath -Force) {
            $childDest = Join-Path $DestinationPath $child.Name
            Copy-FilteredTree -SourcePath $child.FullName -DestinationPath $childDest
        }
        return
    }

    if ($item.Extension -in @(".pyc", ".pyo", ".log")) {
        return
    }

    Copy-Item -LiteralPath $SourcePath -Destination $DestinationPath -Force
}

foreach ($entry in $includeTopLevel) {
    $sourcePath = Join-Path $projectRoot $entry
    if (-not (Test-Path $sourcePath)) {
        continue
    }

    $destinationPath = Join-Path $sourceRoot (Split-Path $entry -Leaf)
    Copy-FilteredTree -SourcePath $sourcePath -DestinationPath $destinationPath
}

Compress-Archive -LiteralPath $packageRoot -DestinationPath $zipPath -Force

Write-Host "Submission package created:"
Write-Host "  Folder: $packageRoot"
Write-Host "  Zip:    $zipPath"
