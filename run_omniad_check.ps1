param(
    [string]$DataPath = "..\dataset\download\Omni-AD-30-release"
)

python .\dinomaly_omniad_uni.py `
    --mode check `
    --data_path $DataPath `
    --num_workers 0
