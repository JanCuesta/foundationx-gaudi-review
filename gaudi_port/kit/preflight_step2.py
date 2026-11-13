"""Run from the Foundation_X+ clone INSIDE the Gaudi container; no HPU allocation needed."""
from pathlib import Path
import importlib
import os
import sys

# When invoked as /path/to/kit/preflight_step2.py, Python otherwise searches the kit directory first.
sys.path.insert(0, str(Path.cwd()))
print('PYTHON:', sys.version.split()[0], flush=True)
modules = ['torch', 'torchvision', 'timm', 'torchmetrics', 'pydicom', 'addict',
           'pycocotools', 'memory_profiler', 'objgraph', 'yapf', 'ipdb',
           'albumentations', 'habana_frameworks.torch']
failures=[]
for mod in modules:
    try:
        importlib.import_module(mod)
        print('IMPORT OK:',mod,flush=True)
    except Exception as e:
        print('IMPORT FAIL:',mod,type(e).__name__,str(e)[:250],flush=True)
        failures.append(mod)

try:
    from config.config_datasets import DATASETS_CONFIG as c
    missing=[]
    for k,v in c.items():
        if any(x in k.lower() for x in ('candidptx','rsnapneumonia','chestxdet','siimacr')) and isinstance(v,str) and v.startswith('/') and not Path(v).exists():
            missing.append((k,v))
    print('DATASET PATHS:', 'OK' if not missing else 'MISSING '+str(missing[:15]),flush=True)
except Exception as e:
    print('CONFIG FAIL:',type(e).__name__,str(e),flush=True)

try:
    import main_Consolidated
    print('FOUNDATION_X_IMPORT_OK',flush=True)
except Exception as e:
    import traceback
    print('FOUNDATION_X_IMPORT_FAIL',flush=True)
    traceback.print_exc(limit=8)
    sys.exit(1)

if failures:
    print('Some optional or required packages missing; review list above.',flush=True)
