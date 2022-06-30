from torch.utils.data import Dataset
import os
import scipy.sparse
import numpy as np
from collections import defaultdict
import torch.nn as nn
import torch

class NMTM(nn.Module):
    def __init__(self, config, Map_en2cn, Map_cn2en):
        super(NMTM, self).__init__()
        self.config = config
        self.Map_en2cn = Map_en2cn.cuda()
        self.Map_cn2en = Map_cn2en.cuda()
        
        # encoder
        self.phi_cn = nn.Parameter(torch.randn(self.config['topic_num'], self.config['vocab_size_cn']))
        self.phi_en = nn.Parameter(torch.randn(self.config['topic_num'], self.config['vocab_size_en']))
        
        self.W_cn = nn.Parameter(torch.randn(self.config['vocab_size_cn'], self.config['e1']))
        self.W_en = nn.Parameter(torch.randn(self.config['vocab_size_en'], self.config['e1']))
        
        self.B_cn = nn.Parameter(torch.randn(self.config['e1']))
        self.B_en = nn.Parameter(torch.randn(self.config['e1']))
        
        self.act_fun = nn.Softplus()
        self.softmax = nn.Softmax(-1)
        self.dropout = nn.Dropout(p=0.0)

        self.batch_norm_encode_mean = nn.BatchNorm1d(self.config['topic_num'], eps=0.001, affine=False)
        self.batch_norm_encode_log_sigma_sq = nn.BatchNorm1d(self.config['topic_num'], eps=0.001, affine=False)
                
        self.W2 = nn.Parameter(torch.randn(self.config['e1'], self.config['e2']))
        self.B2 = nn.Parameter(torch.randn(self.config['e2']))
        
        self.W_m = nn.Parameter(torch.randn(self.config['e2'], self.config['topic_num']))
        self.B_m = nn.Parameter(torch.randn(self.config['topic_num']))
        
        self.W_s = nn.Parameter(torch.randn(self.config['e2'], self.config['topic_num']))
        self.B_s = nn.Parameter(torch.randn(self.config['topic_num']))
        
        self.init_params()
        # decoder
        
        self.batch_norm_decode_en = nn.BatchNorm1d(self.config['vocab_size_en'], eps=0.001, affine=False)
        self.batch_norm_decode_cn = nn.BatchNorm1d(self.config['vocab_size_cn'], eps=0.001, affine=False)
        
        # loss
        self.a = 1 * torch.ones((1, int(self.config['topic_num'])))
        self.mu_priori = nn.Parameter((torch.log(self.a).T - torch.mean(torch.log(self.a),1).T).T, requires_grad=False)
        sigma_priori = (((1.0/self.a)*(1-(2.0/self.config['topic_num']))).T + 
                            (1.0/(self.config['topic_num']*self.config['topic_num']))*torch.sum(1.0/self.a, 1)).T
        self.sigma_priori = nn.Parameter(sigma_priori, requires_grad=False)
        
    def init_params(self):
        nn.init.xavier_uniform_(self.phi_cn)
        nn.init.xavier_uniform_(self.phi_en)
        
        nn.init.xavier_uniform_(self.W_cn)
        nn.init.xavier_uniform_(self.W_en)
        nn.init.zeros_(self.B_cn)
        nn.init.zeros_(self.B_en)
        
        nn.init.xavier_uniform_(self.W2)
        nn.init.xavier_uniform_(self.W_m)     
        nn.init.xavier_uniform_(self.W_s)
        
        nn.init.zeros_(self.B2)
        nn.init.zeros_(self.B_m)     
        nn.init.zeros_(self.B_s)  
        

    def encode(self, x, lang):
        if lang == 'en': 
            h = self.act_fun(torch.matmul(x, self.W_en) + self.B_en)
        else: 
            h = self.act_fun(torch.matmul(x, self.W_cn) + self.B_cn)

        h = self.act_fun(torch.matmul(h, self.W2) + self.B2)
        
        mean = self.batch_norm_encode_mean(torch.matmul(h, self.W_m) + self.B_m)
        log_sigma_sq = self.batch_norm_encode_log_sigma_sq(torch.matmul(h, self.W_s) + self.B_s)
        val = torch.sqrt(torch.exp(log_sigma_sq))
        eps = torch.zeros_like(val).normal_()
        z = mean + torch.mul(val, eps)
        z = self.dropout(z)
        z = self.softmax(z)
        
        return z, mean, log_sigma_sq
    
    def decode(self, z, beta, lang):
        if lang == 'en': 
            batch_norm = self.batch_norm_decode_en
        else: 
            batch_norm = self.batch_norm_decode_cn
        
        x_recon = self.softmax(batch_norm(torch.matmul(z, beta)))
        return x_recon
    
    def get_loss(self, x, x_recon, z_mean, z_log_sigma_sq):
        sigma = torch.exp(z_log_sigma_sq)
        latent_loss = 0.5 * (torch.sum(torch.div(sigma, self.sigma_priori),1) + \
                        torch.sum(torch.mul(torch.div((self.mu_priori - z_mean), self.sigma_priori), (self.mu_priori - z_mean)), 1) 
                             - self.config['topic_num'] + torch.sum(torch.log(self.sigma_priori), 1) 
                             - torch.sum(z_log_sigma_sq, 1))
        recon_loss = torch.sum(-x * torch.log(x_recon), axis=1)
        loss = latent_loss + recon_loss
        return loss.mean()

    def calculate_beta(self):
        beta_cn = (self.config['lam'] * torch.matmul(self.phi_en, self.Map_en2cn) + (1-self.config['lam']) * self.phi_cn).detach()
        beta_en = (self.config['lam'] * torch.matmul(self.phi_cn, self.Map_cn2en) + (1-self.config['lam']) * self.phi_en).detach()
        return beta_cn, beta_en        
    
    def forward(self, x_cn, x_en):
        beta_cn = (self.config['lam'] * torch.matmul(self.phi_en, self.Map_en2cn) + (1-self.config['lam']) * self.phi_cn)
        beta_en = (self.config['lam'] * torch.matmul(self.phi_cn, self.Map_cn2en) + (1-self.config['lam']) * self.phi_en)

        # encode
        z_cn, z_mean_cn, z_log_sigma_sq_cn = self.encode(x_cn, 'cn')
        z_en, z_mean_en, z_log_sigma_sq_en = self.encode(x_en, 'en')
        
        # decode
        x_recon_cn = self.decode(z_cn, beta_cn, 'cn')
        x_recon_en = self.decode(z_en, beta_en, 'en')
        
        return z_cn, z_mean_cn, z_log_sigma_sq_cn, z_en, z_mean_en, z_log_sigma_sq_en, x_recon_cn, x_recon_en
