"""Apply minimal opt-in HPU changes to the SEPARATE Foundation X+ clone only.
Usage: python patch_step2_hpu.py /scratch/jcuesta/BME/Foundation_X/gaudi_port/Foundation_X+
"""
import ast
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
if ROOT is None or not (ROOT / 'main_Consolidated.py').is_file():
    raise SystemExit('Pass the path to an existing, separate Foundation_X+ source copy')
if ROOT == Path('/scratch/jcuesta/BME/Foundation_X/Foundation_X+'):
    raise SystemExit('Refusing to patch the original CUDA tree')
KIT = Path(__file__).resolve().parent

shutil.copyfile(KIT / 'hpu_support.py', ROOT / 'hpu_support.py')

# Core execution modules plus models used by the Swin-B / DINO path.
files = [ROOT / x for x in [
    'main_Consolidated.py', 'engine.py', 'engineClsSeg.py', 'util/misc.py',
    'models/dino/dn_components.py', 'models/dino/dino.py',
    'models/dino/dino_F6_consolidated.py',
    'models/dino/deformable_transformer.py',
    'models/dino/deformable_transformer_F6_consolidated.py',
    'models/dino/swin_transformer_CyclicSegmentation.py',
    'models/dino/backbone.py',
] if (ROOT / x).exists()]

for p in files:
    old = p.read_text()
    if '# FX_HPU_PORT' in old:
        print('Already patched:', p.relative_to(ROOT))
        continue
    s = old
    s = s.replace('torch.cuda.amp.autocast', 'hpu_support.autocast')
    s = s.replace('torch.cuda.amp.GradScaler', 'hpu_support.GradScaler')
    s = s.replace('torch.cuda.synchronize', 'hpu_support.sync')
    s = s.replace('torch.cuda.empty_cache', 'hpu_support.empty_cache')
    s = s.replace('torch.cuda.max_memory_allocated', 'hpu_support.max_memory_allocated')
    s = s.replace('torch.cuda.set_device', 'hpu_support.set_device')
    s = s.replace('torch.cuda.is_available', 'torch.hpu.is_available')
    s = s.replace('torch.cuda.device_count', 'torch.hpu.device_count')
    # Rework only recognized .cuda() call forms; fail for unknown ones.
    s = re.sub(r'\.cuda\(\s*non_blocking\s*=\s*True\s*\)', '.to("hpu", non_blocking=True)', s)
    s = re.sub(r'\.cuda\(\s*args\.gpu\s*,\s*non_blocking\s*=\s*True\s*\)', '.to("hpu", non_blocking=True)', s)
    s = re.sub(r'\.cuda\(\s*args\.gpu\s*\)', '.to("hpu")', s)
    s = re.sub(r'\.cuda\(\s*\)', '.to("hpu")', s)
    for quote in ('"', "'"):
        s = s.replace(quote + 'cuda' + quote, quote + 'hpu' + quote)
        s = s.replace(quote + 'cuda:0' + quote, quote + 'hpu' + quote)
    if s != old:
        # Put import at top after future imports and file header/comments.
        lines = s.splitlines(keepends=True)
        at = 0
        for i, line in enumerate(lines):
            if line.startswith('from __future__ import'):
                at = i + 1
        lines.insert(at, 'import hpu_support  # FX_HPU_PORT\n')
        s = ''.join(lines)
        if p.name == 'main_Consolidated.py':
            # Override the CUDA-dependent timm scaler, only in the HPU clone.
            s = s.replace('from timm.utils import NativeScaler, ModelEma',
                          'from timm.utils import ModelEma\nfrom hpu_support import NativeScaler')
        ast.parse(s, filename=str(p))
        backup = p.with_suffix(p.suffix + '.pre_hpu')
        if not backup.exists():
            shutil.copy2(p, backup)
        p.write_text(s)
        print('Patched:', p.relative_to(ROOT))

# Repair CUDA-only custom extension import and swap the forward path.
fn = ROOT / 'models/dino/ops/functions/ms_deform_attn_func.py'
s = fn.read_text()
if '# FX_HPU_EXTENSION_FALLBACK' not in s:
    old = 'import MultiScaleDeformableAttention as MSDA'
    guarded = 'try:\n    ' + old + '\nexcept (ImportError, OSError):\n    MSDA = None'
    if guarded in s:
        # Prior manual edit already added a guarded import; do not nest try blocks.
        s = s.replace(guarded, '# FX_HPU_EXTENSION_FALLBACK\n' + guarded, 1)
    elif '\n' + old + '\n' in s:
        s = s.replace(old, '# FX_HPU_EXTENSION_FALLBACK\n' + guarded, 1)
    else:
        raise RuntimeError('Unexpected DINO attention import; inspect manually before patching')
    ast.parse(s)
    backup = fn.with_suffix('.py.pre_hpu')
    if not backup.exists():
        shutil.copy2(fn, backup)
    fn.write_text(s)
    print('Patched:', fn.relative_to(ROOT))
else:
    print('Already patched:', fn.relative_to(ROOT))

mod = ROOT / 'models/dino/ops/modules/ms_deform_attn.py'
s = mod.read_text()
if '# FX_HPU_ATTENTION_FALLBACK' not in s:
    old = 'from ..functions import MSDeformAttnFunction'
    prior = 'from ..functions.ms_deform_attn_func import ms_deform_attn_core_pytorch'
    if old in s:
        replacement = """# FX_HPU_ATTENTION_FALLBACK
from ..functions.ms_deform_attn_func import ms_deform_attn_core_pytorch
class MSDeformAttnFunction:
    @staticmethod
    def apply(value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step):
        shapes = spatial_shapes.detach().cpu().tolist() if hasattr(spatial_shapes, 'detach') else spatial_shapes
        return ms_deform_attn_core_pytorch(value, shapes, sampling_locations, attention_weights)
"""
        s = s.replace(old, replacement, 1)
    elif prior in s and 'class MSDeformAttnFunction:' in s:
        # Already manually adapted in the separate clone; mark it without rewriting.
        s = s.replace(prior, '# FX_HPU_ATTENTION_FALLBACK\n' + prior, 1)
    else:
        raise RuntimeError('Unexpected DINO attention module; inspect manually before patching')
    ast.parse(s)
    backup = mod.with_suffix('.py.pre_hpu')
    if not backup.exists():
        shutil.copy2(mod, backup)
    mod.write_text(s)
    print('Patched:', mod.relative_to(ROOT))
else:
    print('Already patched:', mod.relative_to(ROOT))

# TSNE argument compatibility (idempotent; preserve original indentation).
for rel in ('main_Consolidated.py', 'engine.py', 'engineClsSeg.py'):
    p = ROOT / rel
    s = p.read_text()
    needle = 'tsne = TSNE(n_components=2, perplexity=30, n_iter=300)'
    if '# FX_HPU_TSNE_COMPAT' not in s and needle in s:
        lines = s.splitlines(keepends=True)
        out = []
        for line in lines:
            match = re.match(r'^(\s*)' + re.escape(needle) + r'\s*$', line)
            if match:
                pad = match.group(1)
                out.append(pad + '# FX_HPU_TSNE_COMPAT\n')
                out.append(pad + 'try:\n')
                out.append(pad + '    ' + needle + '\n')
                out.append(pad + 'except TypeError:\n')
                out.append(pad + '    tsne = TSNE(n_components=2, perplexity=30, max_iter=300)\n')
            else:
                out.append(line)
        result = ''.join(out)
        ast.parse(result)
        p.write_text(result)

# Find unresolved references in noncomment, currently relevant sources.
print('\n=== UNRESOLVED CUDA CALLS (review before training) ===')
for p in files:
    for number, line in enumerate(p.read_text().splitlines(), 1):
        code = line.split('#', 1)[0]
        if ('.cuda(' in code or 'torch.cuda.' in code or 'to("cuda")' in code or "to('cuda')" in code):
            print(f'{p.relative_to(ROOT)}:{number}: {code.strip()[:140]}')

print('HPU source patch completed; this has NOT been run on Gaudi hardware.')
