param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectRoot,

    [Parameter(Mandatory = $true)]
    [string]$PythonExe,

    [Parameter(Mandatory = $true)]
    [string]$KeyFile,

    [Parameter(Mandatory = $true)]
    [string]$OutputRoot,

    [string]$TaskName = "LoL Current Season Collector",
    [string]$MinApiPatch = "16.13",
    [string[]]$Tiers = @("MASTER"),
    [double]$MaxStorageGB = 40,
    [double]$MinFreeGB = 150,
    [ValidateRange(1, 60)]
    [int]$WatchdogMinutes = 5,
    [string]$KeyRefreshCommand = "",
    [switch]$StartNow
)

$ErrorActionPreference = "Stop"

$project = (Resolve-Path -LiteralPath $ProjectRoot).Path
$python = (Resolve-Path -LiteralPath $PythonExe).Path
$key = (Resolve-Path -LiteralPath $KeyFile).Path
$script = Join-Path $project "scripts\collect_current_season.py"

if (-not (Test-Path -LiteralPath $script)) {
    throw "Collector entrypoint not found: $script"
}

$output = [System.IO.Path]::GetFullPath($OutputRoot)
$argumentParts = @(
    ('"{0}"' -f $script),
    "--key-file", ('"{0}"' -f $key),
    "--output-root", ('"{0}"' -f $output),
    "--min-api-patch", $MinApiPatch,
    "--max-storage-gb", $MaxStorageGB,
    "--min-free-gb", $MinFreeGB,
    "--tiers"
)
$argumentParts += $Tiers
if ($KeyRefreshCommand) {
    if ($KeyRefreshCommand.Contains('"')) {
        throw "KeyRefreshCommand must not contain a double quote; use a wrapper script path"
    }
    $argumentParts += @("--key-refresh-command", ('"{0}"' -f $KeyRefreshCommand))
}
$arguments = $argumentParts -join " "

$action = New-ScheduledTaskAction `
    -Execute $python `
    -Argument $arguments `
    -WorkingDirectory $project

$logonTrigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$watchdogTrigger = New-ScheduledTaskTrigger `
    -Once `
    -At (Get-Date).AddMinutes($WatchdogMinutes) `
    -RepetitionInterval (New-TimeSpan -Minutes $WatchdogMinutes)
$triggers = @($logonTrigger, $watchdogTrigger)
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 10 `
    -RestartInterval (New-TimeSpan -Minutes 5) `
    -MultipleInstances IgnoreNew

$principal = New-ScheduledTaskPrincipal `
    -UserId ("{0}\{1}" -f $env:USERDOMAIN, $env:USERNAME) `
    -LogonType Interactive `
    -RunLevel Limited

$task = New-ScheduledTask `
    -Action $action `
    -Trigger $triggers `
    -Settings $settings `
    -Principal $principal `
    -Description "Resumable Riot API collector with auth pause/hot reload and hourly rank snapshots"

Register-ScheduledTask -TaskName $TaskName -InputObject $task -Force | Out-Null
Write-Output "Installed scheduled task: $TaskName"

if ($StartNow) {
    Start-ScheduledTask -TaskName $TaskName
    Write-Output "Started scheduled task: $TaskName"
}
