# Stop a run started with launch_run.ps1: kills the trainer (workers exit with it) and
# removes the scheduled task.  The latest weights are in runs\<Name>\weights\latest.pt.
param([Parameter(Mandatory = $true)] [string] $Name)
$repo = Split-Path -Parent $PSScriptRoot
$pidfile = Join-Path $repo "runs\$Name.pid"
if (Test-Path $pidfile) {
    $id = Get-Content $pidfile
    if (Get-Process -Id $id -ErrorAction SilentlyContinue) { Stop-Process -Id $id -Force; "stopped trainer $id" }
    else { "trainer $id not running" }
}
# any leftover workers of this run
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like "*multiprocessing*" -and $_.ParentProcessId -eq $id } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
schtasks /delete /tn "checkers-lm-$Name" /f 2>$null | Out-Null
