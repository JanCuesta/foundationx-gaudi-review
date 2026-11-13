import torch
from models.dino.ops.functions.ms_deform_attn_func import ms_deform_attn_core_pytorch
torch.manual_seed(13)
shapes=torch.tensor([[16,16],[8,8],[4,4],[2,2]])
v=torch.randn(1,340,8,32)
loc=torch.rand(1,30,8,4,4,2)
w=torch.softmax(torch.randn(1,30,8,16),dim=-1).reshape(1,30,8,4,4)
cpu=ms_deform_attn_core_pytorch(v,shapes,loc,w)
hpu=ms_deform_attn_core_pytorch(v.to("hpu"),shapes,loc.to("hpu"),w.to("hpu")).cpu()
diff=(cpu-hpu).abs()
print("CPU_MEAN",cpu.abs().mean().item(),"HPU_MEAN",hpu.abs().mean().item(),"MAX_ERROR",diff.max().item(),"MEAN_ERROR",diff.mean().item(),flush=True)
print("ATTENTION_COMPARISON_COMPLETE",flush=True)
