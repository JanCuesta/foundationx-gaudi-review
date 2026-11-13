import ast, torch, torch.nn.functional as F
import habana_frameworks.torch
from pathlib import Path
assert torch.hpu.is_available(), "No HPU available"
src = Path("/scratch/jcuesta/BME/Foundation_X/Foundation_X+/models/dino/ops/functions/ms_deform_attn_func.py").read_text()
fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "ms_deform_attn_core_pytorch")
ns = {"torch":torch, "F":F}
exec(compile(ast.Module(body=[fn], type_ignores=[]), "<fallback>", "exec"), ns)
v = torch.randn(2,80,4,8,device="hpu",requires_grad=True)
p = torch.rand(2,20,4,2,4,2,device="hpu",requires_grad=True)
w = torch.rand(2,20,4,2,4,device="hpu")
o = ns["ms_deform_attn_core_pytorch"](v, [[8,8],[4,4]], p, w)
o.square().mean().backward()
torch.hpu.synchronize()
print("HPU FALLBACK OK",tuple(o.shape),"grad_v",v.grad.abs().mean().item(),"grad_p",p.grad.abs().mean().item(),flush=True)
