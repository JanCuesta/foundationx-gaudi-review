# Foundation X+ Gaudi review copy

Private working snapshot prepared for review of Practice II classification, localization, and segmentation on ASU SOL Gaudi. Based on Foundation X+ from https://github.com/JLianglab/Foundation_X; see LICENSE and the original README.md. This is experimental research code, not a validated port.

## Entry points
- `main_Consolidated.py`, `datasets_medical.py`, `engine.py`, `engineClsSeg.py`, and `hpu_support.py` contain the working training, preprocessing, evaluation, and HPU changes.
- `gaudi_step2/run_seg_recheck_1109.sh` and `jobs/p2_seg_seed43_1109.sbatch` describe the balanced CANDID segmentation test; the buffer-copy variant used `jobs/p2_seg_buffers42_1109.sbatch`. These scripts require the corresponding FX_SEG_* environment variables; consult the job logs or experiment notes for the complete settings.
- `gaudi_step2/run_loc_memory_1109.sh` and `jobs/p2_i_memory1109.sbatch` describe the one-epoch localization memory test. `FX_LOC_MEMORY_TEST=1` selects an experimental consistency-forward path; its gradient behavior differs from the earlier version.
- `gaudi_step2/run_seg_continue_1110.sh` and `run_loc_candid_continue_1110.sh` are checkpoint continuation experiments. Their corresponding batch files are in `gaudi_step2/jobs/`.
- `datasets_medical_before_seg_normalization.py.txt` is an earlier snapshot for comparison; it is not the active module. Besides normalization, later changes exist in the current module.

## Findings to review
The segmentation pipeline formerly normalized with `max_pixel_value=1` before augmentation and later divided the image by 255. The current paths apply augmentation first, then normalization with `max_pixel_value=255`, without that later image division. Please review the input range and assumptions against the reference implementation.

The segmentation teacher uses an epoch-wise EMA of model parameters. The optional `FX_SEG_EMA_BUFFERS=copy` experiment also copies model buffers, including running normalization statistics, from student to teacher. This is an experiment, not an established fix. Two balanced CANDID runs produced student foreground Dice of 68.70% and 68.46% at five epochs; teacher performance remained poor. One localization test completed an epoch with student mAP@0.40 12.84% and teacher 0.36%.

## Environment and limitations
The scripts contain SOL-specific paths, allocation names, container paths and data locations; adjust them before running elsewhere. A local Apptainer wrapper and additional Python dependencies from the SOL installation are referenced but not bundled; `requirements_*.txt` are inherited and not a verified environment lockfile. Dataset images, annotation files, pretrained weights, checkpoints, logs and installed dependencies are excluded. Exact reproductions require the correct datasets, split/annotation preparation, checkpoint, container and environment settings. The code has not been verified in a matched NVIDIA run. Inspect scripts before execution.
