import os
from argparse import ArgumentParser
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from metrics.combined import compute_metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
import utils_notears as ut
from utils import simulate_from_dag_lg
import networkx as nx

def gen_synthetic_data(n = 2000):
    nb_nodes = 10
    G = nx.DiGraph()
    for i in range(nb_nodes):
        G.add_node(i)
    G.add_edge(1,2)
    G.add_edge(1,3)
    G.add_edge(1,4)
    G.add_edge(2,5)
    G.add_edge(2,0)
    G.add_edge(3,0)
    G.add_edge(3,6)
    G.add_edge(3,7)
    G.add_edge(6,9)
    G.add_edge(0,8)
    G.add_edge(0,9)    
    
    sem_type = 'mim'
    raw_synth_data = ut.simulate_nonlinear_sem(nx.adj_matrix(G).todense(), n, sem_type)
    return MinMaxScaler().fit_transform(raw_synth_data)    
    

def DAG_adj_to_triangular(M, form = 'upper'):
    """
    Converts adjacency matrix of DAG to upper triangular form. This form is not
    unique, unless the DAG is a chain.
    Parameters
    ----------
    M : unsorted adjacency matrix
    Returns
    -------
    M : adjacency matrix in upper triangular form
    """
    
    if form == 'lower':
        M = M.T
    
    sort_indices = np.arange(M.shape[0])
    
    for i in range(M.shape[0]):
        f = np.argsort(M[i:,i:].sum(axis=0))+i
        e = np.array(range(i))
        c = np.array(np.concatenate((e,f)),dtype = 'int')
        M = M[c]
        M = M[:,c]
        sort_indices = sort_indices[c]

    if form == 'lower':
        M = M.T

    return M, sort_indices    

import torch
import scipy.linalg as slin


class TraceExpm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # detach so we can cast to NumPy
        E = slin.expm(input.detach().numpy())
        f = np.trace(E)
        E = torch.from_numpy(E)
        ctx.save_for_backward(E)
        return torch.as_tensor(f, dtype=input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        E, = ctx.saved_tensors
        grad_input = grad_output * E.t()
        return grad_input


trace_expm = TraceExpm.apply


class SynthDataset(torch.utils.data.Dataset):

    def __init__(self):
        data = np.array(gen_synthetic_data(), dtype = 'float32')
        self.x = torch.from_numpy(data)
        self.n_samples = self.x.shape[0]
        
        print("***** DATA ****")
        print("n_samples = ", self.n_samples)

    def __getitem__(self, index):
        return self.x[index]

    
    def __len__(self):
        return self.n_samples

    


class SyntheticDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = './', batch_size: int = 64, num_workers: int = 0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.dataset = SynthDataset()
        self.dims = self.dataset.x.shape[1:]
        return self.dataset.x
        
    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=self.num_workers)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        data = np.array(data, dtype = 'float32')
        self.x = torch.from_numpy(data)
        self.n_samples = self.x.shape[0]      
        print("***** DATA ****")
        print("n_samples = ", self.n_samples)
    def __getitem__(self, index):
        return self.x[index]
    def __len__(self):
        return self.n_samples
    
class DataModule(pl.LightningDataModule):

    def __init__(self, data, data_dir: str = './', batch_size: int = 64, num_workers: int = 0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = Dataset(data)

    def setup(self, stage=None):
        self.dims = self.dataset.x.shape[1:]
        return self.dataset.x
        
    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=self.num_workers)
    
    
activation_layer = nn.ReLU(inplace=True)


class Generator(nn.Module):
    def __init__(self, z_dim, x_dim, h_dim, f_scale = 0.1):
        super().__init__()
        
        def block(in_feat, out_feat, normalize=False):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(activation_layer)
            return layers

        self.model = nn.Sequential(
            *block(z_dim + x_dim, h_dim, normalize=False),
            *block(h_dim, h_dim),
            *block(h_dim, h_dim),
            nn.Linear(h_dim, x_dim),
            nn.Sigmoid()
        )
        
        for layer in self.model.parameters():
            if type(layer) == nn.Linear:
                torch.nn.init.xavier_normal_(layer)
                layer.weight.data *= f_scale
        
        
    def forward(self, x, z):
        return self.model(torch.cat([x,z],axis=1))
    


class Generator_causal(nn.Module):
    def __init__(self, z_dim, x_dim, h_dim, use_mask=False, f_scale = 0.1, dag_seed = []):
        super().__init__()
        
        self.x_dim = x_dim
        
        def block(in_feat, out_feat, normalize=False):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(activation_layer)
            return layers

        self.shared = nn.Sequential(
            *block(h_dim, h_dim),
            *block(h_dim, h_dim)
        )
        
        if use_mask:

            if len(dag_seed)>0:
                M_init = torch.rand(x_dim,x_dim)*0.0
                M_init[torch.eye(x_dim, dtype=bool)] = 0
                M_init = torch.rand(x_dim,x_dim)*0.0
                for pair in dag_seed:
                    M_init[pair[0], pair[1]] = 1
                
                
                
                self.M = torch.nn.parameter.Parameter(M_init, requires_grad = False)
                print('Initialised adjacency matrix as parsed:\n', self.M)
            else:
                M_init = torch.rand(x_dim,x_dim)*0.2
                M_init[torch.eye(x_dim, dtype=bool)] = 0
                self.M = torch.nn.parameter.Parameter(M_init)
        else:
            self.M = torch.ones(x_dim,x_dim)
        self.fc_i = nn.ModuleList([nn.Linear(x_dim+1, h_dim) for i in range(self.x_dim)])
        self.fc_f = nn.ModuleList([nn.Linear(h_dim, 1) for i in range(self.x_dim)])
    
        for layer in self.shared.parameters():
            if type(layer) == nn.Linear:
                torch.nn.init.xavier_normal_(layer.weight)
                layer.weight.data *= f_scale

        for i, layer in enumerate(self.fc_i):
            torch.nn.init.xavier_normal_(layer.weight)
            layer.weight.data *= f_scale
            layer.weight.data[:,i] = 1e-16
        
        for i, layer in enumerate(self.fc_f):
            torch.nn.init.xavier_normal_(layer.weight)
            layer.weight.data *= f_scale
    
        
    def sequential(self, x, z, gen_order=None, biased_edges = {}):
        out = x.clone().detach()
        
        if gen_order is None:
            gen_order = range(self.x_dim)
        

        for i in gen_order:
            x_masked = out.clone() * self.M[:,i]
            x_masked[:,i] = 0.
            if i in biased_edges:
                for j in biased_edges[i]:       
                    x_j = x_masked[:,j].detach().numpy()   
                    np.random.shuffle(x_j)
                    x_masked[:,j] = torch.from_numpy(x_j)  
            out_i = activation_layer(self.fc_i[i](torch.cat([x_masked,z[:,i].unsqueeze(1)],axis=1)))
            out[:,i] = nn.Sigmoid()(self.fc_f[i](self.shared(out_i))).squeeze()
        return out
        
        
        
class Discriminator(nn.Module):
    def __init__(self, x_dim, h_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            activation_layer,
            nn.Linear(h_dim, h_dim),
            activation_layer,
            nn.Linear(h_dim, 1)
        )

        for layer in self.model.parameters():
            if type(layer) == nn.Linear:
                torch.nn.init.xavier_normal_(layer)

    def forward(self, x_hat):
        return self.model(x_hat)

    
class DECAF(pl.LightningModule):

    def __init__(
        self,
        dm, # datamodule
        dag_seed = [],
        h_dim: int = 200,
        lr: float = 1e-3,
        b1: float = 0.5,
        b2: float = 0.999,
        batch_size: int = 32,
        lambda_gp: float = 10,
        lambda_privacy: float = 1,
        d_updates: int = 5,
        eps: float = 1e-8,
        causal: bool = False,
        alpha: float = 1,
        rho: float = 1,
        weight_decay: float = 1e-2,
        grad_dag_loss: bool = False,
        l1_g: float = 0,
        l1_W: float = 1,
        p_gen: float = -1,
        use_mask: bool = False, 

    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.epoch_no = 0
        self.iterations_d = 0
        self.iterations_g = 0
        
        
        print(dag_seed)

        self.x_dim = dm.dims[0]
        self.z_dim = self.x_dim 
        self.orig_data = []
        
        print("Setting up network with x_dim, z_dim, h_dim = ", self.x_dim, self.z_dim, h_dim)
        # networks
        if causal:
            self.generator = Generator_causal(z_dim=self.z_dim, x_dim=self.x_dim, h_dim=h_dim, use_mask=use_mask, dag_seed = dag_seed)
        else:
            self.generator = Generator(z_dim=self.z_dim, x_dim=self.x_dim, h_dim=h_dim)
        self.discriminator = Discriminator(x_dim=self.x_dim, h_dim=h_dim)

        self.dag_seed = dag_seed
        
    def forward(self, x, z):
        return self.generator(x, z)

    
    def gradient_dag_loss(self, x, z):
        """
        Calculates the gradient of the output wrt the input. This is a better way to compute the DAG loss, 
        but fairly slow atm
        """
        x.requires_grad=True
        z.requires_grad=True
        gen_x = self.generator(x,z)
        dummy = torch.ones(x.size(0))
        dummy = dummy.type_as(x)
        
        W = torch.zeros(x.shape[1], x.shape[1])
        W = W.type_as(x)
        
        
        for i in range(x.shape[1]):    
            gradients = torch.autograd.grad(
                outputs=gen_x[:,i],
                inputs=x,
                grad_outputs=dummy,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            W[i] = torch.sum(torch.abs(gradients),axis=0)
        
        h = trace_expm(W**2) - self.hparams.x_dim
        
        return 0.5 * self.hparams.rho * h * h + self.hparams.alpha * h        

    
    
    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand(real_samples.size(0),1)
        alpha = alpha.expand(real_samples.size())
        alpha = alpha.type_as(real_samples)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates)
        fake = torch.ones(real_samples.size(0), 1)
        fake = fake.type_as(real_samples)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty        
    
    
    def privacy_loss(self, real_samples, fake_samples):
        return -torch.mean(torch.sqrt(torch.mean((real_samples-fake_samples)**2,axis=1) + self.hparams.eps))        
    
    
    def get_W(self):
        if self.hparams.use_mask:
            return self.generator.M
        else:
            W_0 = []
            for i in range(self.x_dim):
                weights = self.generator.fc_i[i].weight[:,:-1] # don't take the noise variable's weights
                W_0.append(torch.sqrt(torch.sum((weights)**2,axis=0,keepdim=True)+ self.hparams.eps))
            return torch.cat(W_0,axis=0).T
    
        
    def dag_loss(self):
        W = self.get_W()
        h = trace_expm(W**2) - self.x_dim
        l1_loss = torch.norm(W,1)  
        return 0.5 * self.hparams.rho * h**2 + self.hparams.alpha * h + self.hparams.l1_W * l1_loss
        
        
    
    def sample_z(self, n):
        return torch.rand(n, self.z_dim)*2-1
    
    @staticmethod
    def l1_reg(model):
        l1 = torch.tensor(0., requires_grad=True)
        for name, layer in model.named_parameters():
            if 'weight' in name:
                l1 = l1 + layer.norm(p=1)
        return l1
            
        
    def gen_synthetic(self, x, gen_order = [], biased_edges = {}):
        if len(gen_order) != 0:
            return self.generator.sequential(x, self.sample_z(x.shape[0]).type_as(x), gen_order, biased_edges)
        else:
            return self.generator.sequential(x, self.sample_z(x.shape[0]).type_as(x), biased_edges)
    
    
    def get_dag(self): 
        return np.round(self.get_W().detach().numpy(), 3)
    
    def get_bi_dag(self): 
        dag = np.round(self.get_W().detach().numpy(), 3)
        bi_dag = np.zeros_like(dag)
        for i in range(len(dag)):
            for j in range(i, len(dag)):
                bi_dag[i][j] = dag[i][j] + dag[j][i]
        return np.round(bi_dag, 3)
    
    def get_gen_order(self):
        dense_dag = np.array(self.get_dag())
        dense_dag[dense_dag > 0.5] = 1
        dense_dag[dense_dag <= 0.5] = 0 
        G = nx.from_numpy_matrix(dense_dag, create_using = nx.DiGraph)
        gen_order = list(nx.algorithms.dag.topological_sort(G))
        return gen_order
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        # sample noise
        z = self.sample_z(batch.shape[0])
        z = z.type_as(batch)

        if self.hparams.p_gen < 0:  
            generated_batch = self.generator.sequential(batch, z, self.get_gen_order()) 
        else:# train simultaneously
            raise ValueError("we're not allowing simultaneous generation no more. Set p_gen negative")
            to_gen = torch.rand_like(batch) < self.hparams.p_gen
            to_gen = to_gen.type_as(batch)
            generated_batch = self.generator.sequential(batch, z)* to_gen + batch * (1-to_gen)
        # train generator
        if optimizer_idx == 0:
            self.iterations_d += 1
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            real_loss = torch.mean(self.discriminator(batch))
            fake_loss = torch.mean(self.discriminator(generated_batch.detach())) 

            # discriminator loss 
            d_loss = fake_loss - real_loss
            
            # add the gradient penalty
            d_loss += self.hparams.lambda_gp *self.compute_gradient_penalty(batch, generated_batch)
            
            
            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output
        
        if optimizer_idx == 1:
            # sanity check: keep track of G updates
            self.iterations_g += 1
        
            # adversarial loss (negative D fake loss)
            g_loss = -torch.mean(self.discriminator(generated_batch)) #self.adversarial_loss(self.discriminator(self.generated_batch), valid)
            
            # add privacy loss of ADS-GAN
            g_loss += self.hparams.lambda_privacy * self.privacy_loss(batch, generated_batch)
            
            # add l1 regularization loss
            g_loss += self.hparams.l1_g * self.l1_reg(self.generator)
            
            if len(self.dag_seed) == 0:
                if self.hparams.causal:
                    if self.hparams.grad_dag_loss:
                        g_loss += self.gradient_dag_loss(batch, z)
                    else:
                        g_loss += self.dag_loss()

            tqdm_dict = {'g_loss': g_loss}
            
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
           
            })
            
            return output


    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        weight_decay = self.hparams.weight_decay

        opt_g = torch.optim.AdamW(self.generator.parameters(), lr=lr, betas=(b1, b2), weight_decay=weight_decay)
        opt_d = torch.optim.AdamW(self.discriminator.parameters(), lr=lr, betas=(b1, b2), weight_decay=weight_decay)
        return (
            {'optimizer': opt_d, 'frequency': self.hparams.d_updates},
            {'optimizer': opt_g, 'frequency': 1}
            )

    def set_val_data(self, orig_data):
        self.orig_data = orig_data
    
    def on_epoch_end(self, log=True):
        if not log or self.epoch_no%30==0:
            if len(self.orig_data) > 0:
                print(self.get_gen_order())
                synth_data = self.gen_synthetic(self.orig_data, self.get_gen_order()).detach().numpy()
                which_metric = [['PRDC', 'FD','ID'],['OC']]
                results_metrics = compute_metrics(self.orig_data.numpy(), synth_data, 
                                                  which_metric = which_metric, 
                                                  wd_params = {},
                                                  model = None,
                                                  verbose = True # not log
                                         )
                if log:
                    for key in results_metrics:
                        self.log(key, results_metrics[key])
            else:
                print("Debug mode off")

        self.epoch_no += 1
