import torch
import math
from einops import rearrange, einsum

__all__ = [
    "LNO_unet"
]


ACTIVATION = {
    "Sigmoid": torch.nn.Sigmoid(),
    "Tanh": torch.nn.Tanh(),
    "ReLU": torch.nn.ReLU(),
    "LeakyReLU": torch.nn.LeakyReLU(0.1),
    "ELU": torch.nn.ELU(),
    "GELU": torch.nn.GELU()
}

#======================================================================#
def Attention_Vanilla(q, k, v):
    score = torch.softmax(torch.einsum("bhic,bhjc->bhij", q, k) / math.sqrt(k.shape[-1]), dim=-1)
    r = torch.einsum("bhij,bhjc->bhic", score, v)
    return r

class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layer, act):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layer = n_layer
        self.act = act
        self.input = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.hidden = torch.nn.ModuleList([torch.nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.n_layer)])
        self.output = torch.nn.Linear(self.hidden_dim, self.output_dim)
        
    def forward(self, x):
        r = self.act(self.input(x))
        for i in range(0, self.n_layer):
            r = r + self.act(self.hidden[i](r))
        r = self.output(r)
        return r
        
#======================================================================#
class SelfAttention(torch.nn.Module):
    def __init__(self, n_mode, n_dim, n_head, attn):
        super().__init__()
        self.n_mode = n_mode
        self.n_dim = n_dim
        self.n_head = n_head
        self.Wq = torch.nn.Linear(self.n_dim, self.n_dim)
        self.Wk = torch.nn.Linear(self.n_dim, self.n_dim)
        self.Wv = torch.nn.Linear(self.n_dim, self.n_dim)
        self.attn = attn
        self.proj = torch.nn.Linear(self.n_dim, self.n_dim)
    
    def forward(self, x):
        B, N, D = x.size()
        q = self.Wq(x).view(B, N, self.n_head, D // self.n_head).permute(0, 2, 1, 3)
        k = self.Wk(x).view(B, N, self.n_head, D // self.n_head).permute(0, 2, 1, 3)
        v = self.Wv(x).view(B, N, self.n_head, D // self.n_head).permute(0, 2, 1, 3)
        r = self.attn(q, k, v).permute(0, 2, 1, 3).contiguous().view(B, N, D)
        r = self.proj(r)
        return r

class AttentionBlock(torch.nn.Module):
    def __init__(self, n_mode, n_dim, n_head, act):
        super().__init__()
        self.n_mode = n_mode
        self.n_dim = n_dim
        self.n_head = n_head
        self.act = act
        
        self.self_attn = SelfAttention(self.n_mode, self.n_dim, self.n_head, Attention_Vanilla)
        
        self.ln1 = torch.nn.LayerNorm(self.n_dim)
        self.ln2 = torch.nn.LayerNorm(self.n_dim)
        self.drop = torch.nn.Dropout(0.0)
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.n_dim, self.n_dim*2),
            self.act,
            torch.nn.Linear(self.n_dim*2, self.n_dim),
        )

    def forward(self, y):   
        # y = y + self.drop(self.self_attn(self.ln1(y)))
        # y = y + self.mlp(self.ln2(y))
        y = y + self.drop(self.self_attn(y))
        y = self.ln1(y)
        y = y + self.mlp(y)
        y = self.ln2(y)
        return y

#======================================================================#



class LNO_down_start(torch.nn.Module):
    def __init__(self, n_block, n_mode, n_dim, n_head, n_layer, x_dim, y1_dim, y2_dim, act, model_attr):
        super().__init__()
        self.n_block = n_block
        self.n_mode = n_mode
        self.n_dim = n_dim
        self.n_head = n_head
        self.n_layer = n_layer
        self.act = ACTIVATION[act]
        
        self.x_dim = x_dim
        self.y1_dim = y1_dim
        if model_attr["time"]:
            self.y2_dim = 1
        else:
            self.y2_dim = y2_dim
        
        self.trunk_projector = MLP(self.x_dim, self.n_dim, self.n_dim, self.n_layer, self.act)
        self.branch_projector = MLP(self.y1_dim, self.n_dim, self.n_dim, self.n_layer, self.act)
        # self.out_mlp = MLP(self.n_dim, self.n_dim, self.y2_dim, self.n_layer, self.act)
        self.attention_projector = MLP(self.n_dim, self.n_dim, self.n_mode, self.n_layer, self.act)
        self.attn_blocks = torch.nn.Sequential(*[AttentionBlock(self.n_mode, self.n_dim, self.n_head, self.act) for _ in range(0, self.n_block)])
        self.ln1 = torch.nn.LayerNorm(n_dim)

    def forward(self, x, y=None):
        if y is None:
            y = x
        x = self.trunk_projector(x)
        y = self.branch_projector(y)

        score = self.attention_projector(x)
        score_encode = torch.softmax(score, dim=1)
        score_decode = torch.softmax(score, dim=-1)
        
        z = torch.einsum("bij,bic->bjc", score_encode, y)
        
        for block in self.attn_blocks:
            z = block(z)
        
        # r = torch.einsum("bij,bjc->bic", score_decode, z)
        # r = self.out_mlp(r)
        z = self.ln1(z)
        return z,score_decode


class LNO_down(torch.nn.Module):
    def __init__(self, n_block, n_mode, n_dim, n_head, n_layer, x_dim, y1_dim, y2_dim, act, model_attr):
        super().__init__()
        self.n_block = n_block
        self.n_mode = n_mode
        self.n_dim = n_dim
        self.n_head = n_head
        self.n_layer = n_layer
        self.act = ACTIVATION[act]
        
        self.x_dim = x_dim
        self.y1_dim = y1_dim
        if model_attr["time"]:
            self.y2_dim = 1
        else:
            self.y2_dim = y2_dim
        
        self.trunk_projector = MLP(self.x_dim, self.n_dim, self.n_dim, self.n_layer, self.act)
        self.branch_projector = MLP(self.y1_dim, self.n_dim, self.n_dim, self.n_layer, self.act)
        self.out_mlp = MLP(self.n_dim, self.n_dim, self.y2_dim, self.n_layer, self.act)
        self.attention_projector = MLP(self.n_dim, self.n_dim, self.n_mode, self.n_layer, self.act)
        self.attn_blocks = torch.nn.Sequential(*[AttentionBlock(self.n_mode, self.n_dim, self.n_head, self.act) for _ in range(0, self.n_block)])
        self.ln1 = torch.nn.LayerNorm(n_dim)

    def forward(self, x ,y=None):
        if y is None:
            y = x
        x = self.trunk_projector(x)
        y = self.branch_projector(y)

        score = self.attention_projector(x)
        score_encode = torch.softmax(score, dim=1)
        score_decode = torch.softmax(score, dim=-1)
        
        z = torch.einsum("bij,bic->bjc", score_encode, y)
        
        for block in self.attn_blocks:
            z = block(z)
        
        # r = torch.einsum("bij,bjc->bic", score_decode, z)
        # r = self.out_mlp(r)
        z = self.ln1(z)
        return z, score_decode

class LNO_up(torch.nn.Module):
    def __init__(self, n_block, n_mode, n_dim, n_head, n_layer, x_dim, y1_dim, y2_dim, act, model_attr):
        super().__init__()
        self.n_block = n_block
        self.n_mode = n_mode
        self.n_dim = n_dim
        self.n_head = n_head
        self.n_layer = n_layer
        self.act = ACTIVATION[act]
        
        self.x_dim = x_dim
        self.y1_dim = y1_dim
        if model_attr["time"]:
            self.y2_dim = 1
        else:
            self.y2_dim = y2_dim
        
        # self.trunk_projector = MLP(self.x_dim, self.n_dim, self.n_dim, self.n_layer, self.act)
        # self.branch_projector = MLP(self.y1_dim, self.n_dim, self.n_dim, self.n_layer, self.act)
        # self.out_mlp = MLP(self.n_dim, self.n_dim, self.y2_dim, self.n_layer, self.act)
        # self.attention_projector = MLP(self.n_dim, self.n_dim, self.n_mode, self.n_layer, self.act)
        self.attn_blocks = torch.nn.Sequential(*[AttentionBlock(self.n_mode, self.n_dim, self.n_head, self.act) for _ in range(0, self.n_block)])
        self.ln1 = torch.nn.LayerNorm(n_dim)

    def forward(self, x, score_decode ,y=None):
        if y is None:
            y = x
        # x = self.trunk_projector(x)
        # y = self.branch_projector(y)

        # score = self.attention_projector(x)
        # score_encode = torch.softmax(score, dim=1)
        # score_decode = torch.softmax(score, dim=-1)
        
        # z = torch.einsum("bij,bic->bjc", score_encode, y)
        z = x
        for block in self.attn_blocks:
            z = block(z)
        
        r = torch.einsum("bij,bjc->bic", score_decode, z)
        r = self.ln1(r)
        # r = self.out_mlp(r)
        return r

    # def _init_weights(self, module):
    #     if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
    #         module.weight.data.normal_(mean=0.0, std=0.0002)
    #         if isinstance(module, torch.nn.Linear) and module.bias is not None:
    #             module.bias.data.zero_()
    #     elif isinstance(module, torch.nn.LayerNorm):
    #         module.weight.data.fill_(1.0)
    #         module.bias.data.zero_()


class LNO_up_end(torch.nn.Module):
    def __init__(self, n_block, n_mode, n_dim, n_head, n_layer, x_dim, y1_dim, y2_dim, act, model_attr):
        super().__init__()
        self.n_block = n_block
        self.n_mode = n_mode
        self.n_dim = n_dim
        self.n_head = n_head
        self.n_layer = n_layer
        self.act = ACTIVATION[act]
        
        self.x_dim = x_dim
        self.y1_dim = y1_dim
        if model_attr["time"]:
            self.y2_dim = 1
        else:
            self.y2_dim = y2_dim
        
        # self.trunk_projector = MLP(self.x_dim, self.n_dim, self.n_dim, self.n_layer, self.act)
        # self.branch_projector = MLP(self.y1_dim, self.n_dim, self.n_dim, self.n_layer, self.act)
        self.out_mlp = MLP(self.n_dim, self.n_dim, self.y2_dim, self.n_layer, self.act)
        # self.attention_projector = MLP(self.n_dim, self.n_dim, self.n_mode, self.n_layer, self.act)
        self.attn_blocks = torch.nn.Sequential(*[AttentionBlock(self.n_mode, self.n_dim, self.n_head, self.act) for _ in range(0, self.n_block)])
        self.ln1 = torch.nn.LayerNorm(n_dim)

    def forward(self, x, score_decode ,y=None):
        if y is None:
            y = x
        # x = self.trunk_projector(x)
        # y = self.branch_projector(y)

        # score = self.attention_projector(x)
        # score_encode = torch.softmax(score, dim=1)
        # score_decode = torch.softmax(score, dim=-1)
        
        # z = torch.einsum("bij,bic->bjc", score_encode, y)
        z = x
        for block in self.attn_blocks:
            z = block(z)
        
        r = torch.einsum("bij,bjc->bic", score_decode, z)
        r = self.ln1(r)
        r = self.out_mlp(r)
        return r

    # def _init_weights(self, module):
    #     if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
    #         module.weight.data.normal_(mean=0.0, std=0.0002)
    #         if isinstance(module, torch.nn.Linear) and module.bias is not None:
    #             module.bias.data.zero_()
    #     elif isinstance(module, torch.nn.LayerNorm):
    #         module.weight.data.fill_(1.0)
    #         module.bias.data.zero_()

class LNO_unet(torch.nn.Module):
    def __init__(self, n_block, n_mode, n_dim, n_head, n_layer, x_dim, y1_dim, y2_dim, act, model_attr, n_LNO):
        super().__init__()
        
        self.n_LNO = n_LNO
        self.LNO_down_start = LNO_down_start(
                    n_block=n_block, n_mode=n_mode, n_dim=n_dim, n_head=n_head, n_layer=n_layer, act=act,
                    x_dim=x_dim, y1_dim=y1_dim, y2_dim=n_dim,
                    model_attr=model_attr
            )
        self.LNO_blocks_down =  torch.nn.Sequential(*[LNO_down(
                    n_block=n_block, n_mode=n_mode//(2**nb), n_dim=n_dim, n_head=n_head, n_layer=n_layer, act=act,
                    x_dim=n_dim, y1_dim=n_dim, y2_dim=n_dim,
                    model_attr=model_attr
            ) for nb in range(1, n_LNO+1)])
        
        self.LNO_blocks_up =  torch.nn.Sequential(*[LNO_up(
                    n_block=n_block, n_mode=n_mode//(2**nb), n_dim=n_dim, n_head=n_head, n_layer=n_layer, act=act,
                    x_dim=n_dim, y1_dim=n_dim, y2_dim=n_dim,
                    model_attr=model_attr
            ) for nb in range(n_LNO, 0, -1)])
        
        self.LNO_up_end = LNO_up_end(
                    n_block=n_block, n_mode=n_mode, n_dim=n_dim, n_head=n_head, n_layer=n_layer, act=act,
                    x_dim=n_dim, y1_dim=n_dim, y2_dim=y2_dim,
                    model_attr=model_attr
            )       
        
    

    def forward(self, x, y=None):
        
        outputs = []
        score_decode_list = []
        z,score_decode = self.LNO_down_start(x)
        outputs.append(z)
        score_decode_list.append(score_decode)

        for block in self.LNO_blocks_down:
            z,score_decode = block(z)
            outputs.append(z)
            score_decode_list.append(score_decode)

        for i, block in enumerate(self.LNO_blocks_up):
            z = z + outputs[self.n_LNO - i]
            # print('self.n_LNO - i',self.n_LNO - i)
            z = block(z,score_decode_list[self.n_LNO - i])

        z = z + outputs[0]
        r = self.LNO_up_end(z,score_decode_list[0])
        
        return r