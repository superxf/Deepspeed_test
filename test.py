import deepspeed
import torch

sparse_self_attention = deepspeed.ops.sparse_attention.SparseSelfAttention(
    sparsity_config = deepspeed.ops.sparse_attention.FixedSparsityConfig(
        2,
        attention="unidirectional"
    )
)

query = torch.rand((4, 2, 128, 512)).to(torch.float16).to("cuda")
key = torch.rand((4, 2, 128, 512)).to(torch.float16).to("cuda")
value = torch.rand((4, 2, 128, 512)).to(torch.float16).to("cuda")

context = sparse_self_attention(query, key, value)

print(context)