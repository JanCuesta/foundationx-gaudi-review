import os
import tempfile
import numpy as np
import torch
from gaudi_step2.seg_sampling import prepare, report


class Fake:
    mode = 'train'
    def __init__(self, paths, masks):
        self.img_list = paths
        self.masks = masks
        self.img_label = ['1 1' if m.sum() else '-1' for m in masks]
    def __getitem__(self, i):
        return torch.zeros(3, 2, 2), self.masks[i]


left = torch.tensor([[[1., 0.], [0., 0.]]])
right = torch.tensor([[[0., 1.], [0., 0.]]])
zero = torch.zeros_like(left)
train = Fake(['a', 'a', 'b', 'c', 'leak'], [left, right, zero, left, left])
test = Fake(['leak', 'held'], [left, zero])
for mode in ('uniform', 'balanced'):
    os.environ['FX_SEG_SAMPLING'] = mode
    with tempfile.TemporaryDirectory() as output:
        tr, te, sampler = prepare(train, test, output)
        assert tr.paths == ['a', 'b', 'c'] and len(te) == 2
        assert tr[0][1].sum() == 2  # union, not one annotation discarded
        assert not set(tr.paths) & set(te.paths)
        weights = sampler.weights
        if mode == 'balanced':
            assert weights[torch.tensor(tr.positive)].sum() == weights[~torch.tensor(tr.positive)].sum()
        else:
            assert (weights == 1).all()
        print('PASS grouping, union, overlap removal, sampler', mode)
y = np.stack([left.numpy(), zero.numpy()])
r = report(y, y)
assert r['positive_case_mean_dice'] == 1 and r['negative_image_false_positive_rate'] == 0
print('PASS subgroup metrics')
