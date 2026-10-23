# Run the lidraughts bot (client + matchmaker) as hidden scheduled tasks, detached from any shell.
#
#   powershell -File scripts\launch_bot.ps1 -Weights runs\gen12\weights\latest.pt [-Sims 400]
#   powershell -File scripts\launch_bot.ps1 -Stop
#
# Expects ..\lidraughts-bot (clone of AttackingOrDefending/lidraughts-bot) with config.yml (token),
# dq_engine.py copied in and `from dq_engine import DQEngine` in strategies.py.
# Logs: runs\lidraughts_bot.log / runs\lidraughts_bot.err (client), runs\matchmaker.log / .err.
param(
    [string] $Weights = "runs\gen12\weights\latest.pt",
    [int]    $Sims = 400,
    [switch] $Stop
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$botdir = Join-Path (Split-Path -Parent $repo) "lidraughts-bot"
# lidraughts-bot pins pydraughts 0.5.7 (API differs from the 0.6 the training venv uses), so the
# client runs in its own venv: lidraughts-bot\.venv-bot (requirements.txt + torch cu128 + numpy).
$pybot = Join-Path $botdir ".venv-bot\Scripts\python.exe"
$py = Join-Path $repo ".venv\Scripts\python.exe"           # matchmaker only needs requests

function Kill-Task($task, $pattern) {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like "*$pattern*" } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; "stopped $pattern pid $($_.ProcessId)" }
    cmd /c "schtasks /delete /tn $task /f >nul 2>&1"
}
if ($Stop) { Kill-Task "checkers-lm-bot" "lidraughts-bot.py"; Kill-Task "checkers-lm-matchmaker" "matchmaker.py"; return }

Copy-Item (Join-Path $repo "bot\dq_engine.py") (Join-Path $botdir "dq_engine.py") -Force
# fresh log names per launch: a just-killed process can still hold the old file for a moment,
# which makes the task's output redirect fail before python even starts
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$botlog = "$repo\runs\lidraughts_bot_$stamp.log"; $mmlog = "$repo\runs\matchmaker_$stamp.log"
Set-Content (Join-Path $repo "runs\bot_current.txt") "$botlog`n$mmlog"
$w = if ([System.IO.Path]::IsPathRooted($Weights)) { $Weights } else { Join-Path $repo $Weights }

function Start-Hidden($task, $cmdBody) {
    $cmdfile = Join-Path $repo "runs\$task.cmd"
    "@echo off`r`n$cmdBody" | Set-Content -Encoding ASCII $cmdfile
    $vbs = Join-Path $repo "runs\$task.vbs"
    "CreateObject(""WScript.Shell"").Run ""cmd /c """"$cmdfile"""""", 0, False" | Set-Content -Encoding ASCII $vbs
    cmd /c "schtasks /delete /tn $task /f >nul 2>&1"
    cmd /c "schtasks /create /tn $task /tr `"wscript.exe //B `"`"$vbs`"`"`" /sc once /st 00:00 /f >nul"; if ($LASTEXITCODE) { throw "schtasks create failed for $task" }
    cmd /c "schtasks /run /tn $task >nul"; if ($LASTEXITCODE) { throw "schtasks run failed for $task" }
    "started $task"
}

Start-Hidden "checkers-lm-bot" @"
cd /d "$botdir"
set DQ_WEIGHTS=$w
set DQ_TB=$repo\data\tb
set DQ_SIMS=$Sims
set PYTHONPATH=$repo\src
set COLUMNS=200
"$pybot" lidraughts-bot.py > "$botlog" 2>&1
"@
Start-Hidden "checkers-lm-matchmaker" @"
cd /d "$repo"
"$py" bot\matchmaker.py --token-file "$botdir\config.yml" > "$mmlog" 2>&1
"@
"weights: $w  sims: $Sims"
"logs: $botlog | $mmlog"
