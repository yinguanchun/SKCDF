import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Dropout, Softmax, LayerNorm
from einops import rearrange, repeat
import random
import numpy as np
import math


class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, n_channels=3, n_filters=16, normalization='none', has_dropout=False):
        super(Encoder, self).__init__()
        self.has_dropout = has_dropout

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def encoder(self, input):

        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        if self.has_dropout:
            x5 = self.dropout(x5)

        return x1, x2, x3, x4, x5

    def forward(self, image, turnoff_drop=False):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False

        features = self.encoder(image)

        if turnoff_drop:
            self.has_dropout = has_dropout

        return features


class Decoder(nn.Module):
    def __init__(self, n_classes=2, n_filters=16, normalization='none', has_dropout=False):
        super(Decoder, self).__init__()
        self.has_dropout = has_dropout

        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)

        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)

        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv9 = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.out_conv9_abc = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)



    def decoder(self, features):
        x1, x2, x3, x4, x5 = features
        # print(x1.size(), x2.size(), x3.size())
        x5_up = self.block_five_up(x5)
        # print("1",x5.size(), x5_up.size(), x4.size())
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)

        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)


        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)


        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        if self.has_dropout:
            x9 = self.dropout(x9)

        out = self.out_conv9(x9)
        out_abc = self.out_conv9_abc(x9)
        return out,out_abc

    def forward(self, features, turnoff_drop=False):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False

        out= self.decoder(features)  # logits [4,32,64,128,128]

        if turnoff_drop:
            self.has_dropout = has_dropout

        return out


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

    def forward(self, x): #bc,1,dhw,
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


class Cross_Attention_Global(nn.Module): #
    def __init__(self, num_heads, embedding_channels, attention_dropout_rate):
        super().__init__()
        self.KV_size = embedding_channels * num_heads


        self.num_heads = num_heads
        self.attention_head_size = embedding_channels
        self.q = nn.Linear(embedding_channels, embedding_channels * self.num_heads, bias=False)
        self.k = nn.Linear(embedding_channels, embedding_channels * self.num_heads, bias=False)
        self.v = nn.Linear(embedding_channels, embedding_channels * self.num_heads, bias=False)
        self.psi = nn.InstanceNorm2d(1)

        self.softmax = Softmax(dim=3)

        self.out = nn.Linear(embedding_channels * self.num_heads, embedding_channels, bias=False)

        self.attn_dropout = Dropout(attention_dropout_rate)
        self.proj_dropout = Dropout(attention_dropout_rate)

    def multi_head_rep(self, x):  # (b/2,hw,embedding_channels * self.num_heads)
        new_x_shape = x.size()[:-1] + (
        self.num_heads, self.attention_head_size)  # (b/2,hw,self.num_heads,embedding_channels)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (b/2,self.num_heads,hw,embedding_channels)



    def forward(self, emb,pred_type):
        if pred_type=="labeled":


            emb_l, emb_u = torch.split(emb, emb.size(0) // 2, dim=0)

            _, N, C = emb_u.size() #(b/2,hwd,c)



            q_u2l = self.q(emb_u.detach())  # (2,hw,embedding_channels * self.num_heads)
            k_u2l = self.k(emb_l)  # (2,hw,embedding_channels * self.num_heads)
            v_u2l = self.v(emb_l)  # (2,hw,embedding_channels * self.num_heads)

            batch_size = q_u2l.size(0)

            k_u2l = rearrange(k_u2l, 'b n c -> n (b c)')  # (hw,2 *embedding_channels * self.num_heads)
            v_u2l = rearrange(v_u2l, 'b n c -> n (b c)')  # (hw,2 *embedding_channels * self.num_heads)

            k_u2l = repeat(k_u2l, 'n bc -> r n bc', r=batch_size)  # (2,hw,2 *embedding_channels * self.num_heads)
            v_u2l = repeat(v_u2l, 'n bc -> r n bc', r=batch_size)  # (2,hw,2 *embedding_channels * self.num_heads)



            q_u2l = q_u2l.unsqueeze(1).transpose(-1, -2)  # (2,1,embedding_channels * self.num_heads,hw)
            k_u2l = k_u2l.unsqueeze(1)  # (2 , 1 , hw, 2 * embedding_channels * self.num_heads)
            v_u2l = v_u2l.unsqueeze(1).transpose(-1, -2)  # (2,1,embedding_channels * self.num_heads,hw)


            cross_attn_u2l = torch.matmul(q_u2l, k_u2l)  # (2,1,embedding_channels * self.num_heads,embedding_channels * self.num_heads)
            cross_attn_u2l = self.attn_dropout(self.softmax(self.psi(cross_attn_u2l)))

            cross_attn_u2l = torch.matmul(cross_attn_u2l, v_u2l)  # (2,self.num_heads,embedding_channels,hw)

            cross_attn_u2l = cross_attn_u2l.permute(0, 3, 2, 1).contiguous()  # (2,hw,embedding_channels,self.num_heads)
            new_shape_u2l = cross_attn_u2l.size()[:-2] + (self.KV_size,)  # (2,hw,embedding*num_heads)
            cross_attn_u2l = cross_attn_u2l.view(*new_shape_u2l)  # (2,hw,embedding*num_heads)

            out_u2l = self.out(cross_attn_u2l)
            out_u2l = self.proj_dropout(out_u2l)

            # ==========================================================

            q_l2u = self.q(emb_l)
            k_l2u = self.k(emb_u.detach())
            v_l2u = self.v(emb_u.detach())


            batch_size = q_l2u.size(0)

            k_l2u = rearrange(k_l2u, 'b n c -> n (b c)')
            v_l2u = rearrange(v_l2u, 'b n c -> n (b c)')

            k_l2u = repeat(k_l2u, 'n bc -> r n bc', r=batch_size)
            v_l2u = repeat(v_l2u, 'n bc -> r n bc', r=batch_size)

            q_l2u = q_l2u.unsqueeze(1).transpose(-1, -2)
            k_l2u = k_l2u.unsqueeze(1)
            v_l2u = v_l2u.unsqueeze(1).transpose(-1, -2)


            cross_attn_l2u = torch.matmul(q_l2u, k_l2u)
            cross_attn_l2u = self.attn_dropout(self.softmax(self.psi(cross_attn_l2u)))
            cross_attn_l2u = torch.matmul(cross_attn_l2u,v_l2u)

            cross_attn_l2u = cross_attn_l2u.permute(0, 3, 2, 1).contiguous()
            new_shape_l2u = cross_attn_l2u.size()[:-2] + (self.KV_size,)
            cross_attn_l2u = cross_attn_l2u.view(*new_shape_l2u)

            out_l2u = self.out(cross_attn_l2u)
            out_l2u = self.proj_dropout(out_l2u)

            out = torch.cat([out_l2u, out_u2l], dim=0)
            return out
        if pred_type =="unlabeled":
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


class Cross_Attention_Global_Block(nn.Module): #4,512,4,8,8
    def __init__(self, num_heads, embedding_channels,attention_dropout_rate):
        super().__init__()
        self.token_learner = TokenLearner_Global(img_size=(4,8,8), patch_size=(4,8,8), in_chans=1, embed_dim=256)
        self.attn_norm = LayerNorm(embedding_channels, eps=1e-6)
        self.attn = Cross_Attention_Global(num_heads, embedding_channels, attention_dropout_rate)
        self.ffn_norm = LayerNorm(embedding_channels, eps=1e-6)

        self.map_out = nn.Sequential(nn.Conv3d(embedding_channels,embedding_channels, kernel_size=1, padding=0),
                                     nn.GELU())
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x,pred_type):
        if not self.training: #推理模式
            x = torch.cat((x, x))


        b, c, d, h, w = x.shape
        x = x.contiguous().view(b*c, d, h, w).unsqueeze(1)  #bc,1,d,h,w
        x = self.token_learner(x) #bc,1,dhw

        x = rearrange(x, '(b c) 1 (d h w) -> b (d h w) c',b = b, c = c, d = d, h = h, w = w)

        res = x

        x = self.attn_norm(x)
        x = self.attn(x,pred_type)
        x = x + res #residual



        x = self.ffn_norm(x)


        B, n_patch, hidden = x.size() # 4, 256, 512


        x = x.permute(0, 2, 1).contiguous().view(B, hidden, d, h, w)
        x = self.map_out(x)

        if not self.training: #推理模式
            x = torch.split(x, x.size(0) // 2, dim=0)[0]

        return x

class Cross_Attention_Local(nn.Module):
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



        self.psi = nn.InstanceNorm2d(1)

        self.softmax = Softmax(dim=3)
        self.out = nn.Linear(embedding_channels * self.num_heads, embedding_channels, bias=False)

        self.attn_dropout = Dropout(attention_dropout_rate)
        self.proj_dropout = Dropout(attention_dropout_rate)




    def multi_head_rep(self, x): #(b/2,hw,embedding_channels * self.num_heads)
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_size) #(b/2,hw,self.num_heads,embedding_channels)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3) #(b/2,self.num_heads,hw,embedding_channels)

    def forward(self, emb,pred_type): # (b d/p1 h/p2 w/p3) (p1 p2 p3) c

        if pred_type=="labeled":

            emb_l, emb_u = torch.split(emb, emb.size(0) // 2, dim=0)
            _, N, C = emb_u.size() # (b/2 d/p1 h/p2 w/p3) (p1 p2 p3) c



            #在不同通道的相同位置做注意力

            q_u2l = self.q(emb_u.detach()) #(b/2 d/p1 h/p2 w/p3) (p1 p2 p3) (c num_heads) 64,8,1024

            k_u2l = self.k(emb_l) #(b/2 d/p1 h/p2 w/p3) (p1 p2 p3) (c num_heads)
            v_u2l = self.v(emb_l) #(b/2 d/p1 h/p2 w/p3) (p1 p2 p3) (c num_heads)

            batch_size = q_u2l.size(0)


            k_u2l = rearrange(k_u2l, 'b n c -> n (b c)')  # (p1 p2 p3) (b/2 d/p1 h/p2 w/p3 c num_heads)
            v_u2l = rearrange(v_u2l, 'b n c -> n (b c)')  # (p1 p2 p3) (b/2 d/p1 h/p2 w/p3 c num_heads)

            k_u2l = repeat(k_u2l, 'n bc -> r n bc', r=2)  # 2 (p1 p2 p3) (b/2 d/p1 h/p2 w/p3 c num_heads)
            v_u2l = repeat(v_u2l, 'n bc -> r n bc', r=2)  # 2 (p1 p2 p3) (b/2 d/p1 h/p2 w/p3 c num_heads)     2,1,c num_heads,

            #(b/2,self.num_head,c*(dhw/p/p/p),ppp)     (r d h w)(p1 p2 p3) (b c heads)

            # (2,1,embedding_channels * self.num_heads,hw)
            mh_q_u2l = rearrange(q_u2l, '(b d h w) (p1 p2 p3) (c heads) -> (b d h w) (c heads) (p1 p2 p3)',   #(2 d/p1 d/p2 d/p3) 1 (embedding_channels * self.num_heads) (p1,p2,p3)
                              p1=2,p2=2,p3=2,b=2,d=2,h=4,w=4,c=self.embedding_channels,heads=self.num_heads).unsqueeze(1)

            mh_k_u2l = rearrange(k_u2l, 'r (p1 p2 p3) (b d h w c heads) -> (r d h w)(p1 p2 p3)(b c heads) ',   #(2 d/p1 d/p2 d/p3) 1 (embedding_channels * self.num_heads) (p1,p2,p3)
                              r=2,p1=2,p2=2,p3=2,b=2,d=2,h=4,w=4,c=self.embedding_channels,heads=self.num_heads).unsqueeze(1)

            mh_v_u2l = rearrange(v_u2l, 'r (p1 p2 p3) (b d h w c heads) -> (r d h w) (b c heads) (p1 p2 p3)',   #(2 d/p1 d/p2 d/p3) 1 (embedding_channels * self.num_heads) (p1,p2,p3)
                              r=2,p1=2,p2=2,p3=2,b=2,d=2,h=4,w=4,c=self.embedding_channels,heads=self.num_heads).unsqueeze(1)




            self_attn_u2l = torch.matmul(mh_q_u2l, mh_k_u2l)
            #

            self_attn_u2l = self.attn_dropout(self.softmax(self.psi(self_attn_u2l)))





            self_attn_u2l = torch.matmul(self_attn_u2l, mh_v_u2l)

            self_attn_u2l = rearrange(self_attn_u2l.squeeze(1),
                                      '(b d h w) (c heads) (p1 p2 p3) -> b (d p1 h p2 w p3) (c heads)',
                                      p1=2, p2=2, p3=2, b=2, d=2, h=4, w=4, c=self.embedding_channels,
                                      heads=self.num_heads)
            out_u2l = self.out(self_attn_u2l) #(b/2,hw,embedding_channels)
            out_u2l = self.proj_dropout(out_u2l) #(b/2,hw,embedding_channels)




            q_l2u = self.q(emb_l) #(b/2,dhw,embedding_channels * self.num_heads)
            k_l2u = self.k(emb_u.detach()) #(b/2,dhw,embedding_channels * self.num_heads) 2,256,1024
            v_l2u = self.v(emb_u.detach()) #(b/2,dhw,embedding_channels * self.num_heads)

            k_l2u = rearrange(k_l2u, 'b n c -> n (b c)')  # (p1 p2 p3) (b/2 d/p1 h/p2 w/p3 c num_heads)
            v_l2u = rearrange(v_l2u, 'b n c -> n (b c)')  # (p1 p2 p3) (b/2 d/p1 h/p2 w/p3 c num_heads)

            k_l2u = repeat(k_l2u, 'n bc -> r n bc', r=2)  # 2 (p1 p2 p3) (b/2 d/p1 h/p2 w/p3 c num_heads)
            v_l2u = repeat(v_l2u, 'n bc -> r n bc',
                           r=2)  # 2 (p1 p2 p3) (b/2 d/p1 h/p2 w/p3 c num_heads)     2,1,c num_heads,

            # (b/2,self.num_head,c*(dhw/p/p/p),ppp)     (r d h w)(p1 p2 p3) (b c heads)

            # (2,1,embedding_channels * self.num_heads,hw)
            mh_q_l2u = rearrange(q_l2u, '(b d h w) (p1 p2 p3) (c heads) -> (b d h w) (c heads) (p1 p2 p3)',
                                 # (2 d/p1 d/p2 d/p3) 1 (embedding_channels * self.num_heads) (p1,p2,p3)
                                 p1=2, p2=2, p3=2, b=2, d=2, h=4, w=4, c=self.embedding_channels,
                                 heads=self.num_heads).unsqueeze(1)

            mh_k_l2u = rearrange(k_l2u, 'r (p1 p2 p3) (b d h w c heads) -> (r d h w)(p1 p2 p3)(b c heads) ',
                                 # (2 d/p1 d/p2 d/p3) 1 (embedding_channels * self.num_heads) (p1,p2,p3)
                                 r=2, p1=2, p2=2, p3=2, b=2, d=2, h=4, w=4, c=self.embedding_channels,
                                 heads=self.num_heads).unsqueeze(1)

            mh_v_l2u = rearrange(v_l2u, 'r (p1 p2 p3) (b d h w c heads) -> (r d h w) (b c heads) (p1 p2 p3)',
                                 # (2 d/p1 d/p2 d/p3) 1 (embedding_channels * self.num_heads) (p1,p2,p3)
                                 r=2, p1=2, p2=2, p3=2, b=2, d=2, h=4, w=4, c=self.embedding_channels,
                                 heads=self.num_heads).unsqueeze(1)

            self_attn_l2u = torch.matmul(mh_q_l2u, mh_k_l2u)
            #

            self_attn_l2u = self.attn_dropout(self.softmax(self.psi(self_attn_l2u)))

            self_attn_l2u = torch.matmul(self_attn_l2u, mh_v_l2u)

            self_attn_l2u = rearrange(self_attn_l2u.squeeze(1),
                                      '(b d h w) (c heads) (p1 p2 p3) -> b (d p1 h p2 w p3) (c heads)',
                                      p1=2, p2=2, p3=2, b=2, d=2, h=4, w=4, c=self.embedding_channels,
                                      heads=self.num_heads)
            out_l2u = self.out(self_attn_l2u)  # (b/2,hw,embedding_channels)
            out_l2u = self.proj_dropout(out_l2u)  # (b/2,hw,embedding_channels)

            out = torch.cat([out_l2u, out_u2l], dim=0)
            return out
        if pred_type=="unlabeled":
            emb_l, emb_u = torch.split(emb, emb.size(0) // 2, dim=0)
            _, N, C = emb_u.size()  # (b/2 d/p1 h/p2 w/p3) (p1 p2 p3) c

            # 在不同通道的相同位置做注意力

            q = self.q(emb)  # (b/2 d/p1 h/p2 w/p3) (p1 p2 p3) (c num_heads) 64,8,1024

            k = self.k(emb)  # (b/2 d/p1 h/p2 w/p3) (p1 p2 p3) (c num_heads)
            v = self.v(emb)  # (b/2 d/p1 h/p2 w/p3) (p1 p2 p3) (c num_heads)

            batch_size = q.size(0)

            # (2,1,embedding_channels * self.num_heads,hw)
            mh_q = rearrange(q, '(b d h w) (p1 p2 p3) (c heads) -> (b d h w) heads c (p1 p2 p3)',
                             # (2 d/p1 d/p2 d/p3) 1 (embedding_channels * self.num_heads) (p1,p2,p3)
                             p1=2, p2=2, p3=2, d=2, h=4, w=4, c=self.embedding_channels, heads=self.num_heads)

            mh_k = rearrange(k, '(b d h w) (p1 p2 p3) (  c heads) -> (b d h w) heads (p1 p2 p3) c',
                             # (2 d/p1 d/p2 d/p3) 1 (embedding_channels * self.num_heads) (p1,p2,p3)
                             p1=2, p2=2, p3=2, d=2, h=4, w=4, c=self.embedding_channels, heads=self.num_heads)

            mh_v = rearrange(v, '(b d h w) (p1 p2 p3) (  c heads) -> (b d h w) heads c (p1 p2 p3)',
                             # (2 d/p1 d/p2 d/p3) 1 (embedding_channels * self.num_heads) (p1,p2,p3)
                             p1=2, p2=2, p3=2, d=2, h=4, w=4, c=self.embedding_channels, heads=self.num_heads)

            self_attn = torch.matmul(mh_q, mh_k)

            #

            self_attn = self.attn_dropout(self.softmax(self.psi(self_attn)))

            self_attn = torch.matmul(self_attn, mh_v)

            self_attn = rearrange(self_attn.squeeze(1),
                                  '(b d h w) heads c (p1 p2 p3) -> b (d p1 h p2 w p3) (c heads)',
                                  p1=2, p2=2, p3=2, d=2, h=4, w=4, c=self.embedding_channels,
                                  heads=self.num_heads)
            out = self.out(self_attn)  # (b/2,hw,embedding_channels)

            return out


class Cross_Attention_Local_Block(nn.Module):
    def __init__(self, num_heads, embedding_channels,
                 attention_dropout_rate,):
        super().__init__()
        self.token_learner = TokenLearner_Local(img_size=(4,8,8), patch_size=(2,2,2), in_chans=1, embed_dim=8)
        self.attn_norm = LayerNorm(embedding_channels, eps=1e-6)
        self.attn = Cross_Attention_Local(num_heads, embedding_channels, attention_dropout_rate)
        self.ffn_norm = LayerNorm(embedding_channels, eps=1e-6)
        self.map_out = nn.Sequential(nn.Conv3d(embedding_channels, embedding_channels, kernel_size=1, padding=0),
                                     nn.GELU())
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x,pred_type):
        if not self.training:  # 推理模式
            x = torch.cat((x, x))

        b, c, d, h, w = x.shape
        x = x.contiguous().view(b * c, d, h, w).unsqueeze(1)  # bc,1,d,h,w
        x = self.token_learner(x)  # bc,dhw/p1p2p3,p1p2p3  不同通道的相同位置进行注意力   b(dhw/p1p2p3),p1p2p3,c


        res = rearrange(x, '(b c)  (d h w)(p1 p2 p3) -> b (d p1 h p2 w p3) c', b=b, c=c, d=2, h=4, w=4,p1=2,p2=2,p3=2)



        x = rearrange(x, '(b c) (d h w) (p1 p2 p3) -> (b d h w) (p1 p2 p3) c', b=b, c=c, d=2, h=4,w=4, p1=2, p2=2, p3=2)  # b,


        x = self.attn_norm(x)
        x = self.attn(x,pred_type)

        x = x + res  # residual

        x = self.ffn_norm(x)

        B, n_patch, hidden = x.size()  # 4, 256, 512

        x = x.permute(0, 2, 1).contiguous().view(B, hidden, d, h, w)
        x = self.map_out(x)

        if not self.training:  # 推理模式
            x = torch.split(x, x.size(0) // 2, dim=0)[0]

        return x




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



class VNet_Decouple_Attention_ABC(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, num_heads=2):
        super(VNet_Decouple_Attention_ABC, self).__init__()

        self.encoder = Encoder(n_channels=n_channels, n_filters=n_filters, normalization=normalization,
                               has_dropout=has_dropout)


        self.decoder_l = Decoder(n_classes=n_classes, n_filters=n_filters, normalization=normalization,
                               has_dropout=has_dropout)

        self.decoder_u = Decoder(n_classes=n_classes, n_filters=n_filters, normalization=normalization,
                               has_dropout=has_dropout)

        self.cross_attention_local = Cross_Attention_Local_Block(num_heads=num_heads,
                                                                 embedding_channels=n_filters * 16,
                                                                 attention_dropout_rate=0.1,
                                                                 )

        self.cross_attention_global = Cross_Attention_Global_Block(num_heads=num_heads,
                                                                   embedding_channels=n_filters * 16,
                                                                   attention_dropout_rate=0.1,
                                                                   )




    def forward(self, image, pred_type = None):



        if pred_type == "labeled":
            features = self.encoder(image)
            x1, x2, x3, x4, x5 = features
            x5 = self.cross_attention_local(x5, pred_type)
            x5 = self.cross_attention_global(x5, pred_type)
            features = [x1, x2, x3, x4, x5]
            out, out_abc = self.decoder_l(features)
            return out ,out_abc

        if pred_type == "unlabeled":
            features = self.encoder(image)
            x1, x2, x3, x4, x5 = features
            x5 = self.cross_attention_local(x5, pred_type)
            x5 = self.cross_attention_global(x5, pred_type)
            features = [x1, x2, x3, x4, x5]
            out,out_abc =self.decoder_u(features)
            return out, out_abc