param(
    [Parameter(Mandatory = $true)]
    [string]$Checkpoint,
    [string]$DataPath = "..\dataset\download\Omni-AD-30-release",
    [string]$OutputDir = ".\predictions\omniad_dinomaly_uni",
    [int]$BatchSize = 4,
    [int]$NumWorkers = 4
)

python .\dinomaly_omniad_uni.py `
    --mode predict `
    --data_path $DataPath `
    --checkpoint $Checkpoint `
    --output_dir $OutputDir `
    --batch_size $BatchSize `
    --num_workers $NumWorkers
