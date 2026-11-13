"""Isolated experiment. Never imported by the production training entry point."""
import argparse
import ast
import json
import torch


def per_image_dice_loss(y_true, y_pred, smooth=1.0):
    """Matched control for the strict experiment, averaged over images."""
    batch = y_true.shape[0] if y_true.ndim >= 3 else 1
    truth, prediction = y_true.reshape(batch, -1), y_pred.reshape(batch, -1)
    return (1.0 - (2.0 * (truth * prediction).sum(1) + smooth) / (
        truth.sum(1) + prediction.sum(1) + smooth)).mean()


def strict_empty_dice_loss(y_true, y_pred, smooth=1.0):
    """Apply the original Dice formula per image, force 1 for presence mismatch.

    Presence means any target/prediction pixel > 0.5 within each image.
    The hard override deliberately has zero gradient in those cases.
    """
    batch = y_true.shape[0] if y_true.ndim >= 3 else 1
    truth, prediction = y_true.reshape(batch, -1), y_pred.reshape(batch, -1)
    base = 1.0 - (2.0 * (truth * prediction).sum(1) + smooth) / (
        truth.sum(1) + prediction.sum(1) + smooth)
    mismatch = ((truth > .5).any(1) != (prediction > .5).any(1))
    return torch.where(mismatch, torch.ones_like(base), base).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'hpu'], default='cpu')
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    if args.device == 'hpu':
        import habana_frameworks.torch.core as htcore
        mark = htcore.mark_step
    else:
        mark = lambda: None
    tree = ast.parse(open(args.source).read())
    original_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                         and n.name == 'torch_dice_coef_loss')
    scope = {'torch': torch}
    exec(compile(ast.Module(body=[original_node], type_ignores=[]), args.source, 'exec'), scope)
    original = scope['torch_dice_coef_loss']
    cases = [
        ('both_empty', [0,0,0,0], [0,0,0,0], 0.),
        ('false_positive', [0,0,0,0], [1,1,0,0], 1.),
        ('missed_mask', [1,1,0,0], [0,0,0,0], 1.),
        ('perfect', [1,1,0,0], [1,1,0,0], 0.),
        ('partial', [1,1,0,0], [1,0,0,0], .25),
        ('wrong_location', [1,1,0,0], [0,0,1,1], .8),
    ]
    for name, truth, pred, expected in cases:
        y = torch.tensor(truth, dtype=torch.float32, device=args.device)
        p = torch.tensor(pred, dtype=torch.float32, device=args.device)
        old, new = original(y, p), strict_empty_dice_loss(y, p)
        mark()
        old_value, new_value = old.item(), new.item()
        assert abs(new_value - expected) < 1e-6, (name, new_value, expected)
        print(json.dumps(dict(case=name, original_loss=old_value, strict_loss=new_value)), flush=True)
    y = torch.tensor([[[[0., 0.], [0., 0.]]], [[[1., 1.], [0., 0.]]]], device=args.device)
    p = torch.tensor([[[[1., 1.], [0., 0.]]], [[[0., 0.], [0., 0.]]]], device=args.device)
    value = strict_empty_dice_loss(y, p)
    mark()
    assert value.item() == 1.0
    print('PASS mixed batch: each presence mismatch receives 1', flush=True)
    # Test real sigmoid probabilities, not just binary examples.
    for name, truth, initial in [('false_positive', [0,0,0,0], 2.),
                                 ('missed_mask', [1,1,0,0], -2.)]:
        y = torch.tensor(truth, dtype=torch.float32, device=args.device)
        gradients = {}
        for label, criterion in [('original', original), ('strict', strict_empty_dice_loss)]:
            logits = torch.full((4,), initial, device=args.device, requires_grad=True)
            loss = criterion(y, logits.sigmoid())
            loss.backward()
            mark()
            norm = logits.grad.abs().sum().item()
            gradients[label] = norm
        assert gradients['original'] > 0 and gradients['strict'] == 0
        print('GRADIENT_CHECK', name, gradients, flush=True)
    print('PASS: requested loss values; hard override blocks learning in both mismatch cases', flush=True)


if __name__ == '__main__':
    main()
