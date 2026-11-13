"""Opt-in CANDID memorization diagnostics; scores are NOT held-out metrics."""
import json
import os
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from PIL import Image


def prepare(train, test, output):
    mode = os.environ.get('FX_SEG_DIAGNOSTIC', '')
    if mode not in ('single_dice', 'single_combo', 'mixed_combo'):
        raise ValueError(mode)
    train.mode = 'test'  # deterministic geometry, no augmentation
    counts = Counter(train.img_list)
    positive, negative = [], []
    for i, label in enumerate(train.img_label):
        if counts[train.img_list[i]] != 1:
            continue  # avoid ambiguous multiple-annotation images in this test
        (negative if label.strip() == '-1' else positive).append(i)
    selected = positive[:1] if mode.startswith('single') else positive[:8] + negative[:8]
    assert len(selected) == (1 if mode.startswith('single') else 16)
    out = Path(output) / 'data_audit'
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for i in selected:
        image, mask = train[i]
        assert image.shape == (3, 224, 224) and mask.shape == (1, 224, 224)
        assert torch.isfinite(image).all() and torch.isfinite(mask).all()
        assert float(mask.min()) >= 0 and float(mask.max()) <= 1
        nonempty = bool((mask > .5).any())
        assert nonempty == (train.img_label[i].strip() != '-1')
        # Inverse ImageNet normalization; chest X-rays have identical RGB channels.
        rgb = image.numpy().transpose(1, 2, 0) * np.array([.229, .224, .225]) + np.array([.485, .456, .406])
        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        overlay = rgb.astype(float)
        foreground = mask[0].numpy() > .5
        overlay[foreground] = .5 * overlay[foreground] + .5 * np.array([255, 0, 0])
        Image.fromarray(overlay.astype(np.uint8)).save(out / f'{i}_overlay.png')
        records.append(dict(index=i, image=train.img_list[i], positive=nonempty,
                            foreground_fraction=float(foreground.mean())))
    (out / 'samples.json').write_text(json.dumps(records, indent=2))
    # Same training examples for evaluation: deliberately a memorization test.
    evaluation = torch.utils.data.Subset(train, selected)
    repeated = (selected * (128 // len(selected)))
    print('MEMORIZATION_ONLY', mode, records, 'training_samples_per_epoch', len(repeated), flush=True)
    return torch.utils.data.Subset(train, repeated), evaluation
