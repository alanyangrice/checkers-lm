# Stop a run started with launch_run.ps1: kills the trainer and its whole process tree (the
# venv launcher, the real python, and the 8 spawned workers -- Windows does not end children
# when a parent dies, so a plain Stop-Process leaves orphaned workers burning GPU), then
# removes the scheduled task.  Latest weights are in runs\<Name>\weights\latest.pt.
param([Parameter(Mandatory = $true)] [string] $Name)
$repo = Split-Path -Parent $PSScriptRoot
$pidfile = Join-Path $repo "runs\$Name.pid"
if (Test-Path $pidfile) {
    $id = [int](Get-Content $pidfile)
    if (Get-Process -Id $id -ErrorAction SilentlyContinue) {
        cmd /c "taskkill /PID $id /T /F >nul 2>&1"
        "stopped trainer $id and its process tree"
    } else { "trainer $id not running" }
}
# belt and braces: any python whose command line names this run and whose parent is gone
$alive = @{}; Get-Process | ForEach-Object { $alive[$_.Id] = $true }
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { -not $alive.ContainsKey([int]$_.ParentProcessId) -and $_.CommandLine -like "*multiprocessing*" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; "killed orphan worker $($_.ProcessId)" }
cmd /c "schtasks /delete /tn checkers-lm-$Name /f >nul 2>&1"
