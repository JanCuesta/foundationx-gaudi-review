"""Opt-in localization memory experiment."""
import torch


def consistency_features(model, teacher, samples, targets, need_targets):
    was_training = model.training
    model.eval()
    try:
        student_output = model(samples, targets) if need_targets else model(samples)
    finally:
        model.train(was_training)
    with torch.no_grad():
        teacher_output = teacher(samples, targets) if need_targets else teacher(samples)
    return student_output[1], student_output[2], teacher_output[1], teacher_output[2]


def copy_teacher_buffers(model, teacher):
    student_buffers = dict(model.named_buffers())
    copied = 0
    with torch.no_grad():
        for name, target in teacher.named_buffers():
            source = student_buffers[name]
            assert source.shape == target.shape
            target.copy_(source)
            copied += 1
    return copied
