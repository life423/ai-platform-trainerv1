#!/usr/bin/env pwsh
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
python $scriptPath\simple_launcher.py $args
