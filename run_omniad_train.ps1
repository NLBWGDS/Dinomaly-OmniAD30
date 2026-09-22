param(
    [string]$DataPath = "..\dataset\Omni-AD-30-release",
    [string]$OutputDir = ".\saved_results\omniad_dinomaly_uni",
    [int]$BatchSize = 16,
    [int]$TotalIters = 10000,
    [int]$NumWorkers = 4
)

python .\dinomaly_omniad_uni.py `
    --mode train `
    --data_path $DataPath `
    --output_dir $OutputDir `
    --batch_size $BatchSize `
    --total_iters $TotalIters `
    --num_workers $NumWorkers
