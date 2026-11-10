"""Image-level CANDID split cleanup and controlled sampling experiments."""
import json
import os
from collections import defaultdict
from pathlib import Path
import torch
import numpy as np


def report(target, prediction):
    truth, pred = target > .5, prediction > .5
    axes = tuple(range(1, truth.ndim))
    positive = truth.sum(axis=axes) > 0
    intersection = (truth & pred).sum(axis=axes)
    denominator = truth.sum(axis=axes) + pred.sum(axis=axes)
    per_image = np.divide(2.0 * intersection, denominator,
                          out=np.ones_like(denominator, dtype=float), where=denominator != 0)
    negative = ~positive
    result = dict(positive_images=int(positive.sum()), negative_images=int(negative.sum()),
                  positive_case_mean_dice=float(per_image[positive].mean()) if positive.any() else None,
                  negative_image_false_positive_rate=float((pred.sum(axis=axes)[negative] > 0).mean()) if negative.any() else None,
                  negative_pixel_false_positive_rate=float(pred[negative].mean()) if negative.any() else None)
    print('HELDOUT_SEG_METRICS', result, flush=True)
    return result


class ImageGroups(torch.utils.data.Dataset):
    def __init__(self, base, excluded=()):
        self.base = base
        self.base.mode = 'test'  # deterministic preprocessing in both experiments
        groups = defaultdict(list)
        for i, path in enumerate(base.img_list):
            if path not in excluded:
                groups[path].append(i)
        self.paths = sorted(groups)
        self.groups = [groups[p] for p in self.paths]
        self.positive = [any(base.img_label[i].strip() != '-1' for i in group)
                         for group in self.groups]

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        image, mask = self.base[self.groups[index][0]]
        for i in self.groups[index][1:]:
            other_image, other_mask = self.base[i]
            assert torch.equal(image, other_image), 'Annotation images differ'
            mask = torch.maximum(mask, other_mask)
        return image, mask


def prepare(train, test, output, seed=42):
    mode = os.environ['FX_SEG_SAMPLING']
    if mode not in ('uniform', 'balanced'):
        raise ValueError(mode)
    # Keep evaluation images fixed; remove matching images from training.
    overlap = set(train.img_list) & set(test.img_list)
    cleaned_train = ImageGroups(train, set(test.img_list))
    cleaned_test = ImageGroups(test)
    assert not set(cleaned_train.paths) & set(cleaned_test.paths)
    positives = sum(cleaned_train.positive)
    negatives = len(cleaned_train) - positives
    assert positives > 0 and negatives > 0
    weights = [1.0 if mode == 'uniform' else
               (1.0 / positives if positive else 1.0 / negatives)
               for positive in cleaned_train.positive]
    generator = torch.Generator().manual_seed(seed)
    # Both use replacement and the same number of draws; only weights differ.
    sampler = torch.utils.data.WeightedRandomSampler(
        weights, len(cleaned_train), replacement=True, generator=generator)
    manifest = dict(mode=mode, removed_overlap_images=sorted(overlap),
                    train_images=len(cleaned_train), train_positive=positives,
                    train_negative=negatives, test_images=len(cleaned_test),
                    test_positive=sum(cleaned_test.positive),
                    merged_train_groups=sum(len(g) > 1 for g in cleaned_train.groups),
                    merged_test_groups=sum(len(g) > 1 for g in cleaned_test.groups),
                    train_paths=cleaned_train.paths, test_paths=cleaned_test.paths,
                    preprocessing='no augmentation; corrected normalization',
                    annotation_policy='union masks per image; exclude test images from train',
                    seed=seed)
    Path(output).mkdir(parents=True, exist_ok=True)
    (Path(output) / 'split_manifest.json').write_text(json.dumps(manifest, indent=2))
    print('SAMPLING_AUDIT', {k: v for k, v in manifest.items()
                            if k not in ('train_paths', 'test_paths', 'removed_overlap_images')},
          'removed_overlap_count', len(overlap), flush=True)
    return cleaned_train, cleaned_test, sampler
