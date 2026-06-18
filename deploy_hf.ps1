# Deploy FEDSEG-APP to Hugging Face Spaces (run after `hf auth login`)
$ErrorActionPreference = "Stop"

$SpaceId = "codehashira23/FEDSEG-APP"
$RemoteName = "huggingface"
$RemoteUrl = "https://huggingface.co/spaces/$SpaceId"

hf auth whoami
if ($LASTEXITCODE -ne 0) {
    Write-Error "Not logged in. Run: hf auth login"
}

hf repos create $SpaceId --type space --space-sdk docker --public --exist-ok

if (-not (git remote | Select-String -Pattern "^$RemoteName$" -Quiet)) {
    git remote add $RemoteName $RemoteUrl
}

git push $RemoteName main
Write-Host "Space URL: https://huggingface.co/spaces/$SpaceId"
