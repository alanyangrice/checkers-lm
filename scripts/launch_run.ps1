# Launch a self-play run detached from whatever shell started it (Windows).
#
# A plain Start-Process child dies when its parent process tree is torn down (e.g. an
# editor/agent shell restarting).  Registering a one-shot scheduled task and running it
# puts the trainer under the Task Scheduler service instead, so it survives until it
# finishes, is stopped with scripts\stop_run.ps1, or the machine reboots.
#
#   powershell -File scripts\launch_run.ps1 -Name gen2 -Args "--rounds 2000 --workers 8 ... --tb data/tb"
#
# stdout/stderr go to runs\<Name>.log / runs\<Name>.err; per-round JSON to runs\<Name>\log.jsonl.
param(
    [Parameter(Mandatory = $true)] [string] $Name,
    [Parameter(Mandatory = $true)] [string] $Args
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$py = Join-Path $repo ".venv\Scripts\python.exe"
New-Item -ItemType Directory -Force (Join-Path $repo "runs") | Out-Null
$argline = $Args
$cmdfile = Join-Path $repo "runs\$Name.cmd"
@"
@echo off
cd /d "$repo"
"$py" scripts\selfplay.py runs\$Name $argline > runs\$Name.log 2> runs\$Name.err
"@ | Set-Content -Encoding ASCII $cmdfile
# run the .cmd through wscript with window style 0 => no console window appears (and none can be
# closed by accident, which the MKL/Fortran runtime inside numpy turns into a hard abort)
$vbsfile = Join-Path $repo "runs\$Name.vbs"
"CreateObject(""WScript.Shell"").Run ""cmd /c """"$cmdfile"""""", 0, False" | Set-Content -Encoding ASCII $vbsfile
$task = "checkers-lm-$Name"
cmd /c "schtasks /delete /tn $task /f >nul 2>&1"
cmd /c "schtasks /create /tn $task /tr `"wscript.exe //B `"`"$vbsfile`"`"`" /sc once /st 00:00 /f >nul" ; if ($LASTEXITCODE) { throw "schtasks create failed" }
cmd /c "schtasks /run /tn $task >nul" ; if ($LASTEXITCODE) { throw "schtasks run failed" }
Start-Sleep 3
$p = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
     Where-Object { $_.CommandLine -like "*selfplay.py runs\$Name *" } | Select-Object -First 1
if ($p) {
    $p.ProcessId | Set-Content (Join-Path $repo "runs\$Name.pid")
    "started $task  trainer PID $($p.ProcessId)"
    "  tail:  Get-Content runs\$Name\log.jsonl -Wait -Tail 3"
    "  stop:  powershell -File scripts\stop_run.ps1 -Name $Name"
} else {
    "task registered but trainer not found yet; check runs\$Name.err"
}
