#!/usr/bin/env python3
"""Prepare a READ-ONLY Gaudi diagnostic of Foundation X+ Experiment J.

Run this file on an ASU Sol login node after copying it there.
It creates independent script/job files under gaudi_step2/diagnostics/seg_j.
It NEVER edits training code, the original runner, or checkpoint/output directories.
"""
from pathlib import Path
import ast
import re

ROOT = Path('/scratch/jcuesta/BME/Foundation_X/gaudi_port/Foundation_X+')
BASE = ROOT / 'gaudi_step2'
DIAG = BASE / 'diagnostics' / 'seg_j'
CHECKPOINT = BASE / 'results' / 'train_j' / 'ckpt_E12_TH19.pth'
assert CHECKPOINT.is_file() and CHECKPOINT.stat().st_size > 100_000_000, 'Expected J checkpoint not present'
source = (ROOT / 'main_Consolidated.py').read_text()

# The released foundation_x5_pretraining loop skips the training loop AND
# the evaluation loop with --debugOnlyTest. Enable evaluation on the copy only.
start = source.index('        if args.taskcomponent == "foundation_x5_pretraining":')
end = source.index('            if args.modelEMA:', start)
pre = source[start:end]
assert 'did_model_train = False' in pre, 'Expected x5 did_model_train initialization missing'
pre = pre.replace('did_model_train = False', 'did_model_train = bool(args.debugOnlyTest)\n            if args.debugOnlyTest:\n                task_todo = "SEG_J_CHECKPOINT_DIAGNOSTIC"', 1)
source = source[:start] + pre + source[end:]

# Safety: even if --saveAllModel were passed, never save from a diagnostic run.
save_guard = 'if args.saveAllModel == True and did_model_train == True:'
assert source.count(save_guard) == 1, 'Expected full-Foundation-X checkpoint guard missing'
source = source.replace(save_guard, 'if args.saveAllModel == True and did_model_train == True and not args.debugOnlyTest:', 1)

# Capture predictions and ground truth from normal segmentation test routines.
# Do not retain or export patient images -- aggregate numerical metrics only.
helper = '''def _seg_j_diagnostic(y, p, dataset, model_kind):
    truth = np.asarray(y) > 0.5
    proba = np.asarray(p)
    assert truth.shape == proba.shape and truth.ndim == 4, (truth.shape, proba.shape)
    n = truth.shape[0]
    gt_pixels = truth.reshape(n, -1).sum(axis=1)
    positive_images = gt_pixels > 0
    print("SEG_J_DIAG dataset={} model={} images={} gt_positive_images={} gt_foreground_fraction={:.8f} probability_min={:.6f} probability_mean={:.6f} probability_max={:.6f}".format(dataset, model_kind, n, int(positive_images.sum()), float(truth.mean()), float(proba.min()), float(proba.mean()), float(proba.max())), flush=True)
    for thr in (0.1, 0.3, 0.5, 0.7):
        pred = proba > thr
        ppixels = pred.reshape(n, -1).sum(axis=1)
        overlap = np.logical_and(truth, pred).reshape(n, -1).sum(axis=1)
        dice_all = 2.0 * float(overlap.sum()) / max(1, int(gt_pixels.sum() + ppixels.sum()))
        positive_dice = 0.0
        if positive_images.any():
            positive_dice = 2.0 * float(overlap[positive_images].sum()) / max(1, int(gt_pixels[positive_images].sum() + ppixels[positive_images].sum()))
        missed = int(np.logical_and(positive_images, ppixels == 0).sum())
        print("SEG_J_DIAG dataset={} model={} threshold={:.1f} predicted_positive_images={} foreground_fraction={:.8f} global_dice={:.6f} positive_gt_dice={:.6f} gt_positive_images_with_empty_prediction={}".format(dataset, model_kind, thr, int((ppixels > 0).sum()), float(pred.mean()), dice_all, positive_dice, missed), flush=True)

'''
anchor = 'def evaluateSegSepFunc('
assert source.count(anchor) == 1, 'Expected segmentation evaluation function missing'
source = source.replace(anchor, helper + anchor, 1)

# Only instrument the two test calls in evaluateSegSepFunc, not every test call
# in the rest of the 11k-line trainer.
a = source.index('def evaluateSegSepFunc(')
b = source.index('\ndef train_one_epoch_SEGMENTATION(', a)
chunk = source[a:b]
needle_student = '    test_y, test_p, _ = test_SEGMENTATION(model, test_loader_loc_temp, head_number=head_number_temp, log_writter_SEGMENTATION=log_writter_DETECTION)'
needle_teacher = '        test_y, test_p, _ = test_SEGMENTATION(model_ema, test_loader_loc_temp, head_number=head_number_temp, log_writter_SEGMENTATION=log_writter_DETECTION)'
assert chunk.count(needle_student) == chunk.count(needle_teacher) == 1, 'Expected student/teacher test calls missing'
chunk = chunk.replace(needle_student, needle_student + '\n    if args.debugOnlyTest:\n        _seg_j_diagnostic(test_y, test_p, datasetname__, "student")', 1)
chunk = chunk.replace(needle_teacher, needle_teacher + '\n        if args.debugOnlyTest:\n            _seg_j_diagnostic(test_y, test_p, datasetname__, "teacher")', 1)
source = source[:a] + chunk + source[b:]
ast.parse(source)

# Keep all diagnostic files in a separate, never-used-for-training directory.
DIAG.mkdir(parents=True, exist_ok=True)
(DIAG / 'logs').mkdir(exist_ok=True)
(DIAG / 'results').mkdir(exist_ok=True)
main_file = DIAG / 'main_seg_diagnostic.py'
main_file.write_text(source)

runner = (BASE / 'run_case.sh').read_text()
old_out = 'OUT="$ROOT/gaudi_step2/results/train_$CASE"'
assert runner.count(old_out) == 1, 'Expected train output directory assignment missing'
runner = runner.replace(old_out, 'OUT="$ROOT/gaudi_step2/diagnostics/seg_j/results"', 1)
old_cmd = 'python -u main_Consolidated.py'
assert runner.count(old_cmd) == 1, 'Expected training command missing'
# Override the runner's normal "resume from diagnostic output" behavior.
cmd = 'CKPT="$ROOT/gaudi_step2/results/train_j/ckpt_E12_TH19.pth"; test -s "$CKPT"; RESUME=(--resume "$CKPT"); echo "J_DIAGNOSTIC_CHECKPOINT=$CKPT"; python -u gaudi_step2/diagnostics/seg_j/main_seg_diagnostic.py --debugOnlyTest'
runner = runner.replace(old_cmd, cmd, 1)
assert runner.count('--saveAllModel') == 1
runner = runner.replace('--saveAllModel', '', 1)
run_file = DIAG / 'run_seg_diag.sh'
run_file.write_text(runner)
run_file.chmod(0o750)

job = (BASE / 'jobs' / 'p2_step2_j_smoke.sbatch').read_text()
assert 'run_case.sh j smoke' in job, 'Original Gaudi wrapper format changed'
job = job.replace('run_case.sh j smoke', 'diagnostics/seg_j/run_seg_diag.sh j train', 1)
job = job.replace('p2_j_smoke', 'p2_j_diag')
job = re.sub(r'(?m)^#SBATCH -t \S+$', '#SBATCH -t 04:00:00', job)
job = re.sub(r'(?m)^#SBATCH -o \S+$', '#SBATCH -o ' + str(DIAG / 'logs' / 'seg_j_%j.out'), job)
job = re.sub(r'(?m)^#SBATCH -e \S+$', '#SBATCH -e ' + str(DIAG / 'logs' / 'seg_j_%j.err'), job)
job_file = DIAG / 'p2_seg_j_diagnostic.sbatch'
job_file.write_text(job)
assert 'gaudi-apptainer.sh exec bash' in job and '04:00:00' in job
print('DIAGNOSTIC_PREP_OK')
print('Checkpoint:', CHECKPOINT)
print('Generated:', main_file, run_file, job_file, sep='\n  ')
print('Source and checkpoint untouched; evaluation writes ONLY into:', DIAG)
print('Submit:', 'sbatch', str(job_file))
