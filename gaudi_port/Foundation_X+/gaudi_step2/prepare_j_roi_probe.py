#!/usr/bin/env python3
"""Create an isolated, read-only Foundation X+ J ROI-selection diagnostic on SOL.

Prerequisite: the existing epoch-12 segmentation diagnostic was prepared and ran.
Does not modify original training code, main diagnostic, or checkpoints.
"""
from pathlib import Path
import ast
import re

ROOT = Path('/scratch/jcuesta/BME/Foundation_X/gaudi_port/Foundation_X+')
BASE = ROOT / 'gaudi_step2'
PREVIOUS = BASE / 'diagnostics' / 'seg_j'
HERE = BASE / 'diagnostics' / 'seg_j_roi'
CKPT = BASE / 'results' / 'train_j' / 'ckpt_E12_TH19.pth'
assert CKPT.is_file() and CKPT.stat().st_size > 100_000_000, 'J checkpoint missing'
source = (PREVIOUS / 'main_seg_diagnostic.py').read_text()
assert 'SEG_J_DIAG' in source and '--debugOnlyTest' not in source[:300], 'Existing read-only J evaluation source missing'

old_def = 'def evaluateSegSepFunc(epoch, datasetname__, task_todo, model, model_ema, test_loader_loc_temp, head_number_temp, log_writter_DETECTION):'
new_def = 'def evaluateSegSepFunc(epoch, datasetname__, task_todo, model, model_ema, test_loader_loc_temp, head_number_temp, log_writter_DETECTION, postprocessors=None):'
assert source.count(old_def) == 1, 'Unexpected evaluator signature'
source = source.replace(old_def, new_def, 1)

# Forward read-only segmentation images through the same detector and threshold as
# train_one_epoch_SEGMENTATION_SharedLocSeg. Sample no more than 5 batches per model
# and dataset, so this is a targeted diagnostic rather than another training job.
helper = '''def _seg_j_roi_probe(model, loader, dataset, kind, postprocessors, max_batches=5):
    original_training = model.training
    original_head = model.task_DetHead
    model.eval()
    n_images = n_found = n_boxes = 0
    scores_top = []
    scores_all_count = scores_all_above = 0
    try:
        with torch.no_grad():
            for idx, (img, _mask) in enumerate(loader):
                if idx >= max_batches:
                    break
                x = img.float().to("hpu")
                outputs, _, _ = model(x)
                target_sizes = torch.tensor([[args.imgsize, args.imgsize]] * x.shape[0], device="hpu")
                results = postprocessors["bbox"](outputs, target_sizes, not_to_xyxy=True)
                for result in results:
                    scores = result["scores"].detach().float().cpu()
                    if scores.numel() == 0:
                        n_images += 1
                        continue
                    top = float(scores.max())
                    selected = int((scores > 0.3).sum().item())
                    scores_top.append(top)
                    scores_all_count += scores.numel()
                    scores_all_above += selected
                    n_images += 1
                    n_found += int(selected > 0)
                    n_boxes += selected
                del outputs, results, x
        print("SEG_J_ROI dataset={} model={} images={} images_with_boxes_score_gt_0.3={} total_boxes_score_gt_0.3={} avg_max_score={:.6f} max_score={:.6f} candidate_scores_above_threshold={}/{}".format(dataset, kind, n_images, n_found, n_boxes, sum(scores_top)/max(1,len(scores_top)), max(scores_top) if scores_top else 0., scores_all_above, scores_all_count), flush=True)
    finally:
        model.task_DetHead = original_head
        model.train(original_training)

'''
assert source.count(new_def) == 1
source = source.replace(new_def, helper + new_def + '\n    if args.debugOnlyTest and postprocessors is not None:\n        _seg_j_roi_probe(model, test_loader_loc_temp, datasetname__, "student", postprocessors)\n        if model_ema is not None:\n            _seg_j_roi_probe(model_ema, test_loader_loc_temp, datasetname__, "teacher", postprocessors)', 1)

# Pass the detector's existing postprocessor into the three J dataset calls in
# the x5 branch ONLY. Other taskcomponent branches are untouched.
x5start = source.index('        if args.taskcomponent == "foundation_x5_pretraining":')
prefix, tail = source[:x5start], source[x5start:]
old_call = 'evaluateSegSepFunc(epoch, datasetname__, task_todo, model, model_ema, test_loader_loc_temp, head_number_temp, log_writter_DETECTION)'
assert tail.count(old_call) == 3, ('Unexpected number of x5 segmentation eval calls', tail.count(old_call))
tail = tail.replace(old_call, 'evaluateSegSepFunc(epoch, datasetname__, task_todo, model, model_ema, test_loader_loc_temp, head_number_temp, log_writter_DETECTION, postprocessors=postprocessors)')
source = prefix + tail
ast.parse(source)
HERE.mkdir(parents=True, exist_ok=True)
(HERE/'logs').mkdir(exist_ok=True)
(HERE/'results').mkdir(exist_ok=True)
(HERE/'main_seg_roi_probe.py').write_text(source)

runner = (BASE/'run_case.sh').read_text()
original_root = 'ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"'
assert runner.count(original_root) == 1
runner = runner.replace(original_root, 'ROOT="' + str(ROOT) + '"', 1)
original_out = 'OUT="$ROOT/gaudi_step2/results/train_$CASE"'
assert runner.count(original_out) == 1
runner = runner.replace(original_out, 'OUT="$ROOT/gaudi_step2/diagnostics/seg_j_roi/results"', 1)
old_cmd = 'python -u main_Consolidated.py'
assert runner.count(old_cmd) == 1
runner = runner.replace(old_cmd, 'CKPT="$ROOT/gaudi_step2/results/train_j/ckpt_E12_TH19.pth"; test -s "$CKPT"; RESUME=(--resume "$CKPT"); echo "SEG_J_ROI_CHECKPOINT=$CKPT"; python -u gaudi_step2/diagnostics/seg_j_roi/main_seg_roi_probe.py --debugOnlyTest', 1)
assert runner.count('--saveAllModel') == 1
runner = runner.replace('--saveAllModel', '', 1)
runfile = HERE/'run_seg_roi_probe.sh'
runfile.write_text(runner)
runfile.chmod(0o750)
job = (BASE/'jobs'/'p2_step2_j_smoke.sbatch').read_text()
assert 'run_case.sh j smoke' in job
job = job.replace('run_case.sh j smoke', 'diagnostics/seg_j_roi/run_seg_roi_probe.sh j train', 1)
job = job.replace('p2_j_smoke','p2_j_roi')
job = re.sub(r'(?m)^#SBATCH -t \S+$', '#SBATCH -t 03:00:00', job)
job = re.sub(r'(?m)^#SBATCH -o \S+$', '#SBATCH -o '+str(HERE/'logs'/'seg_j_roi_%j.out'), job)
job = re.sub(r'(?m)^#SBATCH -e \S+$', '#SBATCH -e '+str(HERE/'logs'/'seg_j_roi_%j.err'), job)
jobfile = HERE/'p2_seg_j_roi_probe.sbatch'
jobfile.write_text(job)
print('J_ROI_PROBE_PREP_OK')
print('Unchanged checkpoint:', CKPT)
print('Created read-only ROI script:', HERE)
print('Submit:', 'sbatch', jobfile)
