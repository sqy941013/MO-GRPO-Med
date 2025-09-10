# Datasets

This folder contains utilities and scripts for preparing data used by MO-GRPO-Med.

## Preprocessing MIMIC-IV-Ext-iDS → SFT/RL

Use `preprocess.py` to convert MIMIC-IV-Ext-iDS into training/validation/test splits for two tasks:

- BCH (Brief Hospital Course prediction)
- DI (Discharge Instructions generation)

Outputs are subject-level isolated (no patient overlaps across splits).

### Locations

- Script: `datasets/preprocess.py`
- Default input CSV: `datasets/raw/MIMIC-IV-Ext_iDS_with_PR.csv`
- Output directory: `datasets/processed/`

Generated files:

- BCH: `BCH_train_sft.csv`, `BCH_train_rl.csv`, `BCH_dev.csv`, `BCH_test.csv`
- DI:  `DI_train_sft.csv`,  `DI_train_rl.csv`,  `DI_dev.csv`,  `DI_test.csv`

If your downloaded file has a different name (e.g., `MIMIC-IV-Ext-iDS.csv`), rename it to `MIMIC-IV-Ext_iDS_with_PR.csv` or edit the `RAW_CSV` constant inside `preprocess.py`.

### Run (PowerShell)

Run from the `datasets/` directory so relative paths resolve correctly:

```powershell
cd .\datasets

# BCH task
python .\preprocess.py --mode BCH --split 9,0.5,0.5 --rl_ratio 0.2 --seed 42

# DI task
python .\preprocess.py --mode DI  --split 9,0.5,0.5 --rl_ratio 0.2 --seed 42

# Optional: reduce size for quick experiments (row/patient caps)
python .\preprocess.py --mode DI --max_dev 1500 --max_test 1500 --max_dev_patient 0 --max_test_patient 0
```

### Arguments

- `--mode`: `BCH` or `DI`; selects input/target columns.
- `--split`: train,dev,test patient ratios (default `9,0.5,0.5` → normalized to 90/5/5).
- `--rl_ratio`: fraction of `train_sft` reused as `train_rl` (default `0.2`).
- `--max_dev`/`--max_test`: row caps for dev/test (0 = unlimited).
- `--max_dev_patient`/`--max_test_patient`: patient caps for dev/test (0 = unlimited).
- `--seed`: random seed.

### Notes

- Ensure each row has a valid `subject_id`; the script uses it for patient-level splitting.
- If you need custom columns, adapt `BCH_INPUT`, `BCH_TARGET`, `DI_INPUT`, `DI_TARGET` in the script.
