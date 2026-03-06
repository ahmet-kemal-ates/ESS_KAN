param(
    [Parameter(Mandatory=$true)]
    [string]$ArtifactDir,
    [string]$Config = "configs/base.yaml"
)

python -m src.eval --config $Config --artifact_dir $ArtifactDir
python -m src.export --config $Config --artifact_dir $ArtifactDir
