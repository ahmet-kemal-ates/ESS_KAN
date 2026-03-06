param(
    [string]$Config = "configs/base.yaml"
)

python -m src.train --config $Config
