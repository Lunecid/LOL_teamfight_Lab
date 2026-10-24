[CmdletBinding()]
param(
    [ValidateSet('Check','Start','Resume','Status')][string]$Action = 'Check',
    [string]$Config,
    [string]$Run,
    [string]$Python = 'C:\Users\todtj\anaconda3\python.exe'
)
$ErrorActionPreference = 'Stop'
# PowerShell ValidateSet accepts case variants; Python choices require this spelling.
$canonicalActions = @{
    Check = 'Check'
    Start = 'Start'
    Resume = 'Resume'
    Status = 'Status'
}
$canonicalAction = $canonicalActions[$Action]
# Resolve caller-relative inputs BEFORE changing directory. Arrays preserve spaces.
try {
    if (-not $PSBoundParameters.ContainsKey('Config')) {
        $Config = Join-Path $PSScriptRoot 'configs\s4a.local.json'
    }
    $Config = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Config)
    if ($Run) { $Run = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Run) }
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'Python 경로가 없습니다. -Python으로 기존 인터프리터를 지정하세요.' }
    $Python = (Resolve-Path -LiteralPath $Python).Path
    $callArgs = @('-B','-m','c_ept.s4a_operator','--action',$canonicalAction,'--config',$Config)
    if ($Run) { $callArgs += @('--run',$Run) }
    $previousBytecode = $env:PYTHONDONTWRITEBYTECODE
    $env:PYTHONDONTWRITEBYTECODE = '1'
    Push-Location -LiteralPath $PSScriptRoot
    try {
        & $Python @callArgs
        $code = $LASTEXITCODE
    } finally {
        Pop-Location
        $env:PYTHONDONTWRITEBYTECODE = $previousBytecode
    }
    exit $code
} catch {
    Write-Error -Message $_ -ErrorAction Continue
    exit 2
}
