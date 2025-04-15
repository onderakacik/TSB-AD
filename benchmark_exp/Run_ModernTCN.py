'''
ModernTCN for anomaly detection in time series
Adapted from ModernTCN architecture
'''

import numpy as np
import torchinfo
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import math
import tqdm
import os

from TSB_AD.utils.torch_utility import EarlyStoppingTorch, get_gpu, adjust_learning_rate
from TSB_AD.utils.dataset import ReconstructDataset    


import pandas as pd
import numpy as np
import torch
import random, argparse, time, os, logging
from sklearn.preprocessing import MinMaxScaler

from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.models.base import BaseDetector
from TSB_AD.utils.utility import zscore

class ReparamLargeKernelConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, groups, small_kernel, small_kernel_merged=False):
        super(ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        
        # We assume the conv does not change the feature map size
        padding = kernel_size // 2
        if small_kernel_merged:
            self.lkb_reparam = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=1, groups=groups, bias=True)
        else:
            self.lkb_origin = self._conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                          stride=stride, padding=padding, dilation=1, groups=groups, bias=False)
            if small_kernel is not None:
                assert small_kernel <= kernel_size, 'The kernel size for re-param cannot be larger than the large kernel!'
                self.small_conv = self._conv_bn(in_channels=in_channels, out_channels=out_channels,
                                              kernel_size=small_kernel,
                                              stride=stride, padding=small_kernel // 2, groups=groups, dilation=1, bias=False)

    def _conv_bn(self, in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1, bias=False):
        if padding is None:
            padding = kernel_size // 2
        result = nn.Sequential()
        result.add_module('conv', nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))
        result.add_module('bn', nn.BatchNorm1d(out_channels))
        return result

    def forward(self, inputs):
        if hasattr(self, 'lkb_reparam'):
            out = self.lkb_reparam(inputs)
        else:
            out = self.lkb_origin(inputs)
            if hasattr(self, 'small_conv'):
                out += self.small_conv(inputs)
        return out

class Block(nn.Module):
    def __init__(self, large_size, small_size, dmodel, dff, nvars, small_kernel_merged=False, drop=0.1):
        super(Block, self).__init__()
        self.dw = ReparamLargeKernelConv(in_channels=nvars * dmodel, out_channels=nvars * dmodel,
                                         kernel_size=large_size, stride=1, groups=nvars * dmodel,
                                         small_kernel=small_size, small_kernel_merged=small_kernel_merged)
        self.norm = nn.BatchNorm1d(dmodel)

        # convffn1
        self.ffn1pw1 = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dff, kernel_size=1, stride=1,
                                padding=0, dilation=1, groups=nvars)
        self.ffn1act = nn.GELU()
        self.ffn1pw2 = nn.Conv1d(in_channels=nvars * dff, out_channels=nvars * dmodel, kernel_size=1, stride=1,
                                padding=0, dilation=1, groups=nvars)
        self.ffn1drop1 = nn.Dropout(drop)
        self.ffn1drop2 = nn.Dropout(drop)

        # convffn2
        self.ffn2pw1 = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dff, kernel_size=1, stride=1,
                                padding=0, dilation=1, groups=dmodel)
        self.ffn2act = nn.GELU()
        self.ffn2pw2 = nn.Conv1d(in_channels=nvars * dff, out_channels=nvars * dmodel, kernel_size=1, stride=1,
                                padding=0, dilation=1, groups=dmodel)
        self.ffn2drop1 = nn.Dropout(drop)
        self.ffn2drop2 = nn.Dropout(drop)

    def forward(self, x):
        input = x
        B, M, D, N = x.shape
        x = x.reshape(B, M*D, N)
        x = self.dw(x)
        x = x.reshape(B, M, D, N)
        x = x.reshape(B*M, D, N)
        x = self.norm(x)
        x = x.reshape(B, M, D, N)
        x = x.reshape(B, M * D, N)

        x = self.ffn1drop1(self.ffn1pw1(x))
        x = self.ffn1act(x)
        x = self.ffn1drop2(self.ffn1pw2(x))
        x = x.reshape(B, M, D, N)

        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B, D * M, N)
        x = self.ffn2drop1(self.ffn2pw1(x))
        x = self.ffn2act(x)
        x = self.ffn2drop2(self.ffn2pw2(x))
        x = x.reshape(B, D, M, N)
        x = x.permute(0, 2, 1, 3)

        x = input + x
        return x

class Stage(nn.Module):
    def __init__(self, ffn_ratio, num_blocks, large_size, small_size, dmodel, dw_model, nvars,
                small_kernel_merged=False, drop=0.1):
        super(Stage, self).__init__()
        d_ffn = dmodel * ffn_ratio
        blks = []
        for i in range(num_blocks):
            blk = Block(large_size=large_size, small_size=small_size, dmodel=dmodel, dff=d_ffn, 
                        nvars=nvars, small_kernel_merged=small_kernel_merged, drop=drop)
            blks.append(blk)
        self.blocks = nn.ModuleList(blks)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: 
            raise NotImplementedError
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight.view(1, -1, 1)
            x = x + self.affine_bias.view(1, -1, 1)
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias.view(1, -1, 1)
            x = x / (self.affine_weight.view(1, -1, 1) + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x

class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super(Flatten_Head, self).__init__()
        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])  # z: [bs x d_model * patch_num]
                z = self.linears[i](z)  # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x

class ModernTCNModel(nn.Module):
    def __init__(self,
                 seq_len=100,
                 pred_len=0,
                 enc_in=1,
                 patch_size=1,
                 patch_stride=1,
                 stem_ratio=4,
                 downsample_ratio=2,
                 ffn_ratio=1,
                 num_blocks=[1],
                 large_size=[51],
                 small_size=[5],
                 dims=[128],
                 dw_dims=[128],
                 small_kernel_merged=False,
                 backbone_dropout=0.1,
                 head_dropout=0.0,
                 use_multi_scale=False,
                 revin=True,
                 affine=False,
                 subtract_last=False):
        
        super(ModernTCNModel, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        
        # RevIN for normalization
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(enc_in, affine=affine, subtract_last=subtract_last)
        
        # Stem layer & downsampling layers
        self.downsample_layers = nn.ModuleList()
        
        # Stem layer
        stem = nn.Linear(patch_size, dims[0])
        self.downsample_layers.append(stem)
        
        # Downsampling layers
        self.num_stage = len(num_blocks)
        if self.num_stage > 1:
            for i in range(self.num_stage - 1):
                downsample_layer = nn.Sequential(
                    nn.BatchNorm1d(dims[i]),
                    nn.Conv1d(dims[i], dims[i + 1], kernel_size=downsample_ratio, stride=downsample_ratio),
                )
                self.downsample_layers.append(downsample_layer)
        
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.downsample_ratio = downsample_ratio
        
        # Backbone
        self.stages = nn.ModuleList()
        for stage_idx in range(self.num_stage):
            layer = Stage(ffn_ratio, num_blocks[stage_idx], large_size[stage_idx], small_size[stage_idx],
                          dmodel=dims[stage_idx], dw_model=dw_dims[stage_idx], nvars=enc_in,
                          small_kernel_merged=small_kernel_merged, drop=backbone_dropout)
            self.stages.append(layer)
        
        # For anomaly detection, we will use reconstruction
        self.head_detection = nn.Linear(dims[-1], patch_size)
        
    def forward_feature(self, x):
        B, M, L = x.shape
        x = x.unsqueeze(-2)  # [B, M, 1, L]
        
        for i in range(self.num_stage):
            B, M, D, N = x.shape
            x = x.reshape(B * M, D, N)
            
            if i == 0:
                if self.patch_size != self.patch_stride:
                    # Stem layer padding
                    pad_len = self.patch_size - self.patch_stride
                    pad = x[:, :, -1:].repeat(1, 1, pad_len)
                    x = torch.cat([x, pad], dim=-1)
                x = x.reshape(B, M, 1, -1).squeeze(-2)
                x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
                x = self.downsample_layers[i](x)
                x = x.permute(0, 1, 3, 2)
            else:
                if N % self.downsample_ratio != 0:
                    pad_len = self.downsample_ratio - (N % self.downsample_ratio)
                    x = torch.cat([x, x[:, :, -pad_len:]], dim=-1)
                    x = self.downsample_layers[i](x)
                    _, D_, N_ = x.shape
                    x = x.reshape(B, M, D_, N_)

            
            
            x = self.stages[i](x)
        return x
    
    def anomaly_detection(self, x_enc):
        # Apply RevIN normalization
        if self.revin:
            x_enc = x_enc.permute(0, 2, 1)  # [B, L, M] -> [B, M, L]
            x_enc = self.revin_layer(x_enc, 'norm')
            # x_enc = x_enc.permute(0, 2, 1)  # [B, M, L] -> [B, L, M]
        

        # Forward through the ModernTCN
        x = self.forward_feature(x_enc)  # [B, M, D, N]
        # Reconstruct the input
        x = x.permute(0, 1, 3, 2)  # [B, M, N, D]
        x = self.head_detection(x)  # [B, M, N, patch_size]
        B, M, _, _ = x.shape
        x = x.reshape(B, M, -1)  # [B, M, N*patch_size]
        x = x[:, :, :self.seq_len]  # Ensure output length matches input
        x = x.permute(0, 2, 1)  # [B, L, M]
        
        # Apply RevIN denormalization
        if self.revin:
            x = x.permute(0, 2, 1)  # [B, L, M] -> [B, M, L]
            x = self.revin_layer(x, 'denorm')
            x = x.permute(0, 2, 1)  # [B, M, L] -> [B, L, M]
        
        return x

    def forward(self, x_enc):
        dec_out = self.anomaly_detection(x_enc)
        return dec_out  # [B, L, M]

class ModernTCN():
    def __init__(self,
                 win_size=100,
                 enc_in=1,
                 epochs=10,
                 batch_size=128,
                 lr=0.0001,
                 patience=3,
                 features="M",
                 lradj="type1",
                 validation_size=0.2,
                 patch_size=1,
                 patch_stride=1,
                 stem_ratio=6,
                 downsample_ratio=2,
                 ffn_ratio=1,
                 num_blocks=[1],
                 large_size=[51],
                 small_size=[5],
                 dims=[128],
                 dw_dims=[128],
                 small_kernel_merged=False,
                 backbone_dropout=0.1,
                 head_dropout=0.0,
                 use_multi_scale=False,
                 revin=True,
                 affine=False,
                 subtract_last=False):
        super().__init__()

        self.win_size = win_size
        self.enc_in = enc_in
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.epochs = epochs
        self.features = features
        self.lradj = lradj
        self.validation_size = validation_size

        # ModernTCN specific parameters
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.stem_ratio = stem_ratio
        self.downsample_ratio = downsample_ratio
        self.ffn_ratio = ffn_ratio
        self.num_blocks = num_blocks
        self.large_size = large_size
        self.small_size = small_size
        self.dims = dims
        self.dw_dims = dw_dims
        self.small_kernel_merged = small_kernel_merged
        self.backbone_dropout = backbone_dropout
        self.head_dropout = head_dropout
        self.use_multi_scale = use_multi_scale
        self.revin = revin
        self.affine = affine
        self.subtract_last = subtract_last

        self.__anomaly_score = None
        
        self.cuda = True
        self.y_hats = None
        
        self.device = get_gpu(self.cuda)
            
        self.model = ModernTCNModel(
            seq_len=self.win_size,
            enc_in=self.enc_in,
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
            stem_ratio=self.stem_ratio,
            downsample_ratio=self.downsample_ratio,
            ffn_ratio=self.ffn_ratio,
            num_blocks=self.num_blocks,
            large_size=self.large_size,
            small_size=self.small_size,
            dims=self.dims,
            dw_dims=self.dw_dims,
            small_kernel_merged=self.small_kernel_merged,
            backbone_dropout=self.backbone_dropout,
            head_dropout=self.head_dropout,
            use_multi_scale=self.use_multi_scale,
            revin=self.revin,
            affine=self.affine,
            subtract_last=self.subtract_last
        ).float().to(self.device)
        
        self.model_optim = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
        self.early_stopping = EarlyStoppingTorch(None, patience=self.patience)
        
        self.input_shape = (self.batch_size, self.win_size, self.enc_in)
    
    def fit(self, data):
        tsTrain = data[:int((1-self.validation_size)*len(data))]
        tsValid = data[int((1-self.validation_size)*len(data)):]

        train_loader = DataLoader(
            dataset=ReconstructDataset(tsTrain, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=True
        )
        
        valid_loader = DataLoader(
            dataset=ReconstructDataset(tsValid, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=False
        )
        
        for epoch in range(1, self.epochs + 1):
            ## Training
            train_loss = 0
            self.model.train()
            
            loop = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
            for i, (batch_x, _) in loop:
                self.model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_x)
                
                loss.backward()
                self.model_optim.step()
                
                train_loss += loss.cpu().item()
                
                loop.set_description(f'Training Epoch [{epoch}/{self.epochs}]')
                loop.set_postfix(loss=loss.item(), avg_loss=train_loss/(i+1))
            
            ## Validation
            self.model.eval()
            total_loss = []
            
            loop = tqdm.tqdm(enumerate(valid_loader), total=len(valid_loader), leave=True)
            with torch.no_grad():
                for i, (batch_x, _) in loop:
                    batch_x = batch_x.float().to(self.device)

                    outputs = self.model(batch_x)

                    f_dim = -1 if self.features == 'MS' else 0
                    outputs = outputs[:, :, f_dim:]
                    pred = outputs.detach().cpu()
                    true = batch_x.detach().cpu()

                    loss = self.criterion(pred, true)
                    total_loss.append(loss)
                    loop.set_description(f'Valid Epoch [{epoch}/{self.epochs}]')
                    
            valid_loss = np.average(total_loss)
            loop.set_postfix(loss=loss.item(), valid_loss=valid_loss)
            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                print("   Early stopping<<<")
                break
            adjust_learning_rate(self.model_optim, epoch + 1, self.lradj, self.lr)
                        
    def decision_function(self, data):
        test_loader = DataLoader(
            dataset=ReconstructDataset(data, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=False
        )
        
        self.model.eval()
        attens_energy = []
        y_hats = []
        self.anomaly_criterion = nn.MSELoss(reduce=False)
        
        loop = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), leave=True)
        with torch.no_grad():
            for i, (batch_x, _) in loop:
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                outputs = self.model(batch_x)
                # criterion
                score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                y_hat = torch.squeeze(outputs, -1) if outputs.shape[-1] == 1 else outputs
                

                print(f"score and y_hat dims before: {score.shape}, {y_hat.shape}")
                # score = score.detach().cpu().numpy()[:, -1]
                # y_hat = y_hat.detach().cpu().numpy()[:, -1]
                score = torch.mean(score, dim=-1).detach().cpu().numpy()
                y_hat = torch.mean(y_hat, dim=-1).detach().cpu().numpy()
                print(f"score and y_hat dims after: {score.shape}, {y_hat.shape}")

                attens_energy.append(score)
                y_hats.append(y_hat)
                loop.set_description(f'Testing Phase: ')

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        scores = np.array(attens_energy)
        
        y_hats = np.concatenate(y_hats, axis=0).reshape(-1)
        y_hats = np.array(y_hats)

        assert scores.ndim == 1
        
        self.__anomaly_score = scores
        self.y_hats = y_hats

        # Pad scores if needed to match data length
        if self.__anomaly_score.shape[0] < len(data):
            self.__anomaly_score = np.array([self.__anomaly_score[0]]*math.ceil((self.win_size-1)/2) + 
                        list(self.__anomaly_score) + [self.__anomaly_score[-1]]*((self.win_size-1)//2))
        
        print('Output shape:', self.__anomaly_score.shape)
        
        return self.__anomaly_score

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score
    
    def get_y_hat(self) -> np.ndarray:
        return self.y_hats
    
    def param_statistic(self, save_file):
        model_stats = torchinfo.summary(self.model, self.input_shape, verbose=0)
        with open(save_file, 'w') as f:
            f.write(str(model_stats))


def run_ModernTCN(data_train, data_test, HP):
    HP['enc_in'] = data_test.shape[1]

    clf = ModernTCN(**HP)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

# Keep run_ModernTCN_Semisupervised as an alias
run_ModernTCN_Semisupervised = run_ModernTCN

# seeding
seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print("CUDA available: ", torch.cuda.is_available())
print("cuDNN version: ", torch.backends.cudnn.version())

if __name__ == '__main__':

    Start_T = time.time()
    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Running Custom_AD')
    parser.add_argument('--filename', type=str, default='001_NAB_id_1_Facility_tr_1007_1st_2014.csv')
    parser.add_argument('--data_direc', type=str, default='../Datasets/TSB-AD-U/')
    parser.add_argument('--AD_Name', type=str, default='ModernTCN')
    args = parser.parse_args()
    
    ModernTCN_HP = {
        'win_size': 100,
        'enc_in': 1,
        'epochs': 10, 
        'batch_size': 128,
        'lr': 0.0001,
        'patience': 3,
        'features': "M",
        'lradj': "type1",
        'validation_size': 0.2,
        'patch_size': 1,
        'patch_stride': 1,
        'ffn_ratio': 1,
        'num_blocks': [1],
        'large_size': [51],
        'small_size': [5],
        'dims': [128],
        'dw_dims': [128],
        'small_kernel_merged': False,
        'backbone_dropout': 0.1,
        'head_dropout': 0.0,
        'use_multi_scale': False,
        'revin': True,
        'affine': False,
        'subtract_last': False
    }

    df = pd.read_csv(args.data_direc + args.filename).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df['Label'].astype(int).to_numpy()
    print('data: ', data.shape)
    print('label: ', label.shape)

    slidingWindow = find_length_rank(data, rank=1)
    train_index = args.filename.split('.')[0].split('_')[-3]
    data_train = data[:int(train_index), :]


    start_time = time.time()

    output = run_ModernTCN(data_train, data, ModernTCN_HP)
    # output = run_Custom_AD_Unsupervised(data, **Custom_AD_HP)

    end_time = time.time()
    run_time = end_time - start_time

    evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow)
    print('Evaluation Result: ', evaluation_result)