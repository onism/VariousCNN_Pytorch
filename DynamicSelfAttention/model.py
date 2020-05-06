from __future__ import absolute_import, division, print_function

import os
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


class DSA(nn.Module):
    def __init__(self, dsa_num_attentions, dsa_input_dim, dsa_dim, dsa_r=3):
        super(DSA, self).__init__()
        dsa = []
        for i in range(dsa_num_attentions):
            dsa.append(nn.Linear(dsa_input_dim, dsa_dim))
        self.dsa = nn.ModuleList(dsa)
        self.dsa_r = dsa_r 
        self.last_dim = dsa_num_attentions * dsa_dim 
    
    def _self_attention(self, x, mask, r=3):
        '''
            x: batch * seq * dsa_dim
            mask: batch * seq
        '''
        mask = mask.to(torch.float)
        inv_mask = mask.eq(0.0) # convert to BOOL
        softmax_mask = mask.masked_fill(inv_mask, -1e20)
        q = torch.zeros(mask.shape[0], mask.shape[1], requires_grad=False).to(torch.float)
        z_list = []
        # iterative computing attention
        for idx in range(r):
            # softmax masking
            q *= softmax_mask
            a = torch.softmax(q.detach().clone(), dim=1) 
            a *= mask 
            a = a.unsqueeze(-1)
            s = (a * x).sum(1)
            z = torch.tanh(s)
            z_list.append(z)
            m = z.unsqueeze(-1)
            q += torch.matmul(x,m).squeeze(-1)
        return z_list[-1]
    
    def forward(self, x , mask):
        z_list = []
        for p in self.dsa:
            p_out = F.leaky_relu(p(x))
            z_j = self._self_attention(p_out, mask, r = self.dsa_r)
            z_list.append(z_j)
        z = torch.cat(z_list, dim=-1)
        return z