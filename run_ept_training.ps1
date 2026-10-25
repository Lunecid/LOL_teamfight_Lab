[CmdletBinding()]
param(
    [ValidateSet('Check','Start','Status')][string]$Action = 'Check',
    [Parameter(Mandatory=$true)][string]$Config,
    [string]$Control,
    [string]$Python = 'C:\Users\todtj\anaconda3\python.exe',
    [switch]$AwaitJobBinding
)
$ErrorActionPreference = 'Stop'
# The controller assigns this waiting wrapper to a kill-on-close Windows Job
# before releasing one byte. EOF or a wrong byte exits before any Python call.
if ($AwaitJobBinding) {
    if ($Action -ne 'Start' -or [Console]::In.Read() -ne 71) { exit 125 }
}
$canonicalActions = @{ Check='Check'; Start='Start'; Status='Status' }
try {
    # Resolve caller-relative arguments before entering the source checkout.
    $Config = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Config)
    if ($Control) { $Control = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Control) }
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'Fixed Anaconda Python is unavailable.' }
    $Python = (Resolve-Path -LiteralPath $Python).Path
    $callArgs = @('-B','-m','c_ept.workflows.training_operator','--action',$canonicalActions[$Action],'--config',$Config)
    if ($Control) { $callArgs += @('--control',$Control) }
    $previousBytecode = $env:PYTHONDONTWRITEBYTECODE
    $previousUtf8 = $env:PYTHONUTF8
    $env:PYTHONDONTWRITEBYTECODE = '1'
    $env:PYTHONUTF8 = '1'
    Push-Location -LiteralPath $PSScriptRoot
    try {
        & $Python @callArgs
        $code = $LASTEXITCODE
    } finally {
        Pop-Location
        $env:PYTHONDONTWRITEBYTECODE = $previousBytecode
        $env:PYTHONUTF8 = $previousUtf8
    }
    exit $code
} catch {
    Write-Error -Message $_ -ErrorAction Continue
    exit 2
}
