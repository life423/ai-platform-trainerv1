# Add the project directory to your PATH
$projectDir = "C:\Users\aiand\Documents\software-engineering\ai-platform-trainer"
$currentPath = [Environment]::GetEnvironmentVariable("PATH", "User")

# Check if the directory is already in the PATH
if (-not $currentPath.Split(";").Contains($projectDir)) {
    # Add the directory to the PATH
    $newPath = $currentPath + ";" + $projectDir
    [Environment]::SetEnvironmentVariable("PATH", $newPath, "User")
    Write-Host "Added $projectDir to your PATH environment variable."
} else {
    Write-Host "$projectDir is already in your PATH environment variable."
}

# Create a function to refresh the PATH in the current session
function Update-Environment {
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
}

# Refresh the current session's PATH
Update-Environment

Write-Host "PATH has been updated for the current session. The new PATH will be available for new PowerShell windows."
Write-Host "You should now be able to run 'ai-trainer' from any directory."
