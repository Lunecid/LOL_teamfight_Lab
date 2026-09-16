$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskDoc = [IO.Path]::GetFullPath((Join-Path $taskRoot 'deliverables/collaborator_specification_20260915/Research_Specification_20260915.docx'))
$taskPdf = [IO.Path]::GetFullPath((Join-Path $taskRoot 'outputs/collaborator_specification_20260915/Research_Specification_20260915.pdf'))
if (-not $taskDoc.StartsWith($taskRoot + [IO.Path]::DirectorySeparatorChar)) { throw 'Document path outside workspace' }
if (-not $taskPdf.StartsWith($taskRoot + [IO.Path]::DirectorySeparatorChar)) { throw 'PDF path outside workspace' }
$taskWord = $null
$taskDocument = $null
try {
    $taskWord = New-Object -ComObject Word.Application
    $taskWord.Visible = $false
    $taskWord.DisplayAlerts = 0
    $taskDocument = $taskWord.Documents.Open($taskDoc, $false, $true, $false)
    $taskDocument.Repaginate()
    $taskDocument.ExportAsFixedFormat($taskPdf, 17)
    Write-Output ('Rendered generated document with Microsoft Word: ' + $taskPdf)
} finally {
    if ($null -ne $taskDocument) { $taskDocument.Close(0) }
    if ($null -ne $taskWord) { $taskWord.Quit() }
}
