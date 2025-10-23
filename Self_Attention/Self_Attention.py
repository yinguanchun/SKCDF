import torch
from torch import nn
from torch.nn import Dropout, Softmax, LayerNorm
from einops import rearrange, repeat

class TokenLearner_Global(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=(4,8,8), patch_size=(4,8,8), in_chans=1, embed_dim=256): # 8 512
        super().__init__()
        num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1] * (img_size[2] // patch_size[2])) #32
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.map_in = nn.Sequential(nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size),nn.GELU())

    def forward(self, x):
        x = self.proj(x) # bc,dhw,1,1,1
        x = x.flatten(2) # bc,dhw,1
        x = x.transpose(1, 2) # bc,1,dhw
        return x


class TokenLearner_Local(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=(4,8,8), patch_size=(2,2,2), in_chans=1, embed_dim=8): # 8 512
        super().__init__()
        num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1] * (img_size[2] // patch_size[2])) #32
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.map_in = nn.Sequential(nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size),nn.GELU())

    def forward(self, x): #bc,1,d,h,w
        x = self.proj(x) # bc,p1p2p3,d/p1,h/p2,w/p3
        x = x.flatten(2) # bc,p1p2p3,dhw/p1p2p3
        x = x.transpose(1, 2) # bc,dhw/p1p2p3,p1p2p3
        return x

class Self_Attention_Global(nn.Module):  # 标准：每个头的尺寸为embedding_channels/num_heads,这里为embedding_channels
    def __init__(self, num_heads, embedding_channels, attention_dropout_rate):
        super().__init__()
        self.KV_size = embedding_channels * num_heads

        self.num_heads = num_heads
        self.attention_head_size = embedding_channels
        self.q = nn.Linear(embedding_channels, embedding_channels * self.num_heads, bias=False)
        self.k = nn.Linear(embedding_channels, embedding_channels * self.num_heads, bias=False)
        self.v = nn.Linear(embedding_channels, embedding_channels * self.num_heads, bias=False)
        self.softmax = Softmax(dim=3)
        self.psi = nn.InstanceNorm2d(self.num_heads)
        self.out = nn.Linear(embedding_channels * self.num_heads, embedding_channels, bias=False)
        self.attn_dropout = Dropout(attention_dropout_rate)
        self.proj_dropout = Dropout(attention_dropout_rate)

    def multi_head_rep(self, x):  # (b/2,hw,embedding_channels * self.num_heads)
        new_x_shape = x.size()[:-1] + (
        self.num_heads, self.attention_head_size)  # (b/2,hw,self.num_heads,embedding_channels)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (b/2,self.num_heads,hw,embedding_channels)

    def forward(self, emb):
        emb_l, emb_u = torch.split(emb, emb.size(0) // 2, dim=0)
        _, N, C = emb_u.size()  # (b/2,hwd,c)



        q = self.q(emb)  # (b/2,hw,embedding_channels * self.num_heads)
        k = self.k(emb)
        v = self.v(emb)

        # convert to multi-head representation
        mh_q = self.multi_head_rep(q).transpose(-1,
                                                    -2)  ##(b/2,self.num_heads,hw,embedding_channels)->(b/2,self.num_heads,embedding_channels,hw)
        mh_k = self.multi_head_rep(k)
        mh_v = self.multi_head_rep(v).transpose(-1, -2)

        self_attn = torch.matmul(mh_q, mh_k)  # (b/2,self.num_heads,embedding_channels,embedding_channels)

        self_attn = self.attn_dropout(self.softmax(self.psi(self_attn)))
        self_attn = torch.matmul(self_attn, mh_v)  # (b/2,self.num_heads,embedding_channels,hw)

        self_attn = self_attn.permute(0, 3, 2, 1).contiguous()  # (b/2,hw,embedding_channels,self.num_heads)
        new_shape = self_attn.size()[:-2] + (self.KV_size,)  # (b/2,hw,embedding*num_heads)
        self_attn = self_attn.view(*new_shape)  # (b/2,hw,embedding_channels*num_heads)   concat然后线性层

        out = self.out(self_attn)  # (b/2,hw,embedding_channels)
        out = self.proj_dropout(out)  # (b/2,hw,embedding_channels)
        return out


class Self_Attention_Global_Block(nn.Module):  # 4,512,4,8,8
    def __init__(self, num_heads, embedding_channels, attention_dropout_rate):
        super().__init__()
        self.token_learner = TokenLearner_Global(img_size=(4, 8, 8), patch_size=(4, 8, 8), in_chans=1, embed_dim=256)
        self.attn_norm = LayerNorm(embedding_channels, eps=1e-6)
        self.attn = Self_Attention_Global(num_heads, embedding_channels, attention_dropout_rate)
        self.ffn_norm = LayerNorm(embedding_channels, eps=1e-6)
        self.map_out = nn.Sequential(nn.Conv3d(embedding_channels, embedding_channels, kernel_size=1, padding=0),
                                     nn.GELU())
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x):
        if not self.training:  # 推理模式
            x = torch.cat((x, x))

        b, c, d, h, w = x.shape
        x = x.contiguous().view(b * c, d, h, w).unsqueeze(1)  # bc,1,d,h,w
        x = self.token_learner(x)  # bc,1,dhw
        x = rearrange(x, '(b c) 1 (d h w) -> b (d h w) c', b=b, c=c, d=d, h=h, w=w)

        res = x
        x = self.attn_norm(x)
        x = self.attn(x)
        x = x + res  # residual


        x = self.ffn_norm(x)



        B, n_patch, hidden = x.size()
        x = x.permute(0, 2, 1).contiguous().view(B, hidden, d, h, w)
        x = self.map_out(x)

        if not self.training:  # 推理模式
            x = torch.split(x, x.size(0) // 2, dim=0)[0]

        return x




class Self_Attention_Local(nn.Module):
    def __init__(self, num_heads, embedding_channels, attention_dropout_rate,patch_size=(2,2,2)):
        super().__init__()
        self.KV_size = embedding_channels * num_heads
        self.patch_size=patch_size
        self.embedding_channels=embedding_channels
        self.num_heads = num_heads
        self.attention_head_size = embedding_channels
        self.q = nn.Linear(embedding_channels, embedding_channels * self.num_heads, bias=False)
        self.k = nn.Linear(embedding_channels, embedding_channels * self.num_heads, bias=False)
        self.v = nn.Linear(embedding_channels, embedding_channels * self.num_heads, bias=False)



        self.psi = nn.InstanceNorm2d(self.num_heads)
        self.softmax = Softmax(dim=3)
        self.out = nn.Linear(embedding_channels * self.num_heads, embedding_channels, bias=False)

        self.attn_dropout = Dropout(attention_dropout_rate)
        self.proj_dropout = Dropout(attention_dropout_rate)




    def multi_head_rep(self, x): #(b/2,hw,embedding_channels * self.num_heads)
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_size) #(b/2,hw,self.num_heads,embedding_channels)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3) #(b/2,self.num_heads,hw,embedding_channels)

    def forward(self, emb): # (b d/p1 h/p2 w/p3) (p1 p2 p3) c

        emb_l, emb_u = torch.split(emb, emb.size(0) // 2, dim=0)
        _, N, C = emb_u.size() # (b/2 d/p1 h/p2 w/p3) (p1 p2 p3) c



        #在不同通道的相同位置做注意力

        q = self.q(emb) #(b/2 d/p1 h/p2 w/p3) (p1 p2 p3) (c num_heads) 64,8,1024

        k = self.k(emb) #(b/2 d/p1 h/p2 w/p3) (p1 p2 p3) (c num_heads)
        v = self.v(emb) #(b/2 d/p1 h/p2 w/p3) (p1 p2 p3) (c num_heads)

        batch_size = q.size(0)



        # (2,1,embedding_channels * self.num_heads,hw)
        mh_q = rearrange(q, '(b d h w) (p1 p2 p3) (c heads) -> (b d h w) heads c (p1 p2 p3)',   #(2 d/p1 d/p2 d/p3) 1 (embedding_channels * self.num_heads) (p1,p2,p3)
                          p1=2,p2=2,p3=2,d=2,h=4,w=4,c=self.embedding_channels,heads=self.num_heads)

        mh_k = rearrange(k, '(b d h w) (p1 p2 p3) (  c heads) -> (b d h w) heads (p1 p2 p3) c',   #(2 d/p1 d/p2 d/p3) 1 (embedding_channels * self.num_heads) (p1,p2,p3)
                          p1=2,p2=2,p3=2,d=2,h=4,w=4,c=self.embedding_channels,heads=self.num_heads)

        mh_v = rearrange(v, '(b d h w) (p1 p2 p3) (  c heads) -> (b d h w) heads c (p1 p2 p3)',   #(2 d/p1 d/p2 d/p3) 1 (embedding_channels * self.num_heads) (p1,p2,p3)
                          p1=2,p2=2,p3=2,d=2,h=4,w=4,c=self.embedding_channels,heads=self.num_heads)




        self_attn = torch.matmul(mh_q, mh_k)

        #

        self_attn = self.attn_dropout(self.softmax(self.psi(self_attn)))





        self_attn = torch.matmul(self_attn, mh_v)

        self_attn = rearrange(self_attn.squeeze(1),
                                  '(b d h w) heads c (p1 p2 p3) -> b (d p1 h p2 w p3) (c heads)',
                                  p1=2, p2=2, p3=2,  d=2, h=4, w=4, c=self.embedding_channels,
                                  heads=self.num_heads)
        out = self.out(self_attn) #(b/2,hw,embedding_channels)


        return out


class Self_Attention_Local_Block(nn.Module):
    def __init__(self, num_heads, embedding_channels,
                 attention_dropout_rate,):
        super().__init__()
        self.token_learner = TokenLearner_Local(img_size=(4,8,8), patch_size=(2,2,2), in_chans=1, embed_dim=8)
        self.attn_norm = LayerNorm(embedding_channels, eps=1e-6)
        self.attn = Self_Attention_Local(num_heads, embedding_channels, attention_dropout_rate)
        self.ffn_norm = LayerNorm(embedding_channels, eps=1e-6)
        self.map_out = nn.Sequential(nn.Conv3d(embedding_channels, embedding_channels, kernel_size=1, padding=0),
                                     nn.GELU())
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x):
        if not self.training:  # 推理模式
            x = torch.cat((x, x))

        b, c, d, h, w = x.shape
        x = x.contiguous().view(b * c, d, h, w).unsqueeze(1)  # bc,1,d,h,w
        x = self.token_learner(x)  # bc,dhw/p1p2p3,p1p2p3  不同通道的相同位置进行注意力   b(dhw/p1p2p3),p1p2p3,c


        res = rearrange(x, '(b c)  (d h w)(p1 p2 p3) -> b (d p1 h p2 w p3) c', b=b, c=c, d=2, h=4, w=4,p1=2,p2=2,p3=2)


        x = rearrange(x, '(b c) (d h w) (p1 p2 p3) -> (b d h w) (p1 p2 p3) c', b=b, c=c, d=2, h=4,w=4, p1=2, p2=2, p3=2)  # b,


        x = self.attn_norm(x)
        x = self.attn(x)

        x = x + res  # residual

        x = self.ffn_norm(x)

        B, n_patch, hidden = x.size()  # 4, 256, 512

        x = x.permute(0, 2, 1).contiguous().view(B, hidden, d, h, w)
        x = self.map_out(x)

        if not self.training:  # 推理模式
            x = torch.split(x, x.size(0) // 2, dim=0)[0]

        return x





