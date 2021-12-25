from collections import OrderedDict
from typing import Any, Optional, Union

import networkx as nx
import numpy as np
import pytorch_lightning as pl
import scipy.linalg as slin
import torch
import torch.nn as nn

import decaf.logger as log


class TraceExpm(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor) -> torch.Tensor:
        # detach so we can cast to NumPy
        E = slin.expm(input.detach().numpy())
        f = np.trace(E)
        E = torch.from_numpy(E)
        ctx.save_for_backward(E)
        return torch.as_tensor(f, dtype=input.dtype)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        (E,) = ctx.saved_tensors
        grad_input = grad_output * E.t()
        return grad_input


trace_expm = TraceExpm.apply

activation_layer = nn.ReLU(inplace=True)


class Generator_causal(nn.Module):
    def __init__(
        self,
        variables: dict,
        z_dim: int,
        x_dim: int,
        h_dim: int,
        f_scale: float = 0.1,
    ) -> None:
        super().__init__()

        # Internalise parameters.
        self.x_dim = x_dim
        self.variables = variables

        # Calculate parents indexing.
        self.variables_idxs = np.zeros((len(variables), 2), dtype='int')
        idx = 0
        for (var, var_info) in sorted(variables.items()):
            self.variables_idxs[var,0] = idx # lower idx
            idx += var_info['size']
            self.variables_idxs[var,1] = idx # upper idx
        # print(self.variables_idxs)

        # Create mask to filter parents values.
        M_init = torch.rand(len(variables), x_dim) * 0.0
        for (var, var_info) in sorted(variables.items()):
            for parent in var_info['parents']:
                for idx in range(self.variables_idxs[parent][0],
                                 self.variables_idxs[parent][1]):
                    M_init[var, idx] = 1.0
        self.M = torch.nn.parameter.Parameter(M_init, requires_grad=False)
        print("Initialised adjacency matrix as parsed:\n", self.M)

        # def block(in_feat: int, out_feat: int, normalize: bool = False) -> list:
        #     layers = [nn.Linear(in_feat, out_feat)]
        #     if normalize:
        #         layers.append(nn.BatchNorm1d(out_feat, 0.8))
        #     layers.append(activation_layer)
        #     return layers
        # self.shared = nn.Sequential(*block(h_dim, h_dim), *block(h_dim, h_dim))

        self.fc_i = nn.ModuleList(
            [nn.Linear(x_dim + 1, h_dim) for i in range(len(variables))]
        )
        self.fc_f = nn.ModuleList([nn.Linear(h_dim, var_info['size'])
                            for var, var_info in sorted(variables.items())])

        # for layer in self.shared.parameters():
        #     if type(layer) == nn.Linear:
        #         torch.nn.init.xavier_normal_(layer.weight)
        #         layer.weight.data *= f_scale

        for i, layer in enumerate(self.fc_i):
            torch.nn.init.xavier_normal_(layer.weight)
            layer.weight.data *= f_scale
            layer.weight.data[:, i] = 1e-16

        for i, layer in enumerate(self.fc_f):
            torch.nn.init.xavier_normal_(layer.weight)
            layer.weight.data *= f_scale

    def sequential(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        gen_order: Union[list, dict, None] = None,
        biased_edges: dict = {},
    ) -> torch.Tensor:
        out = x.clone().detach()
        num_points = x.shape[0]

        for i in gen_order:
            # print('i=',i)

            # Select input variables (parents) using mask.
            x_masked = out.clone() * self.M[i, :]

            # Optionally remove edges.
            if i in biased_edges:
                print('Debiasing...')
                for parent in biased_edges[i]:

                    # Randomize parent.
                    random_order = np.random.permutation(num_points)
                    for p_idx in range(self.variables_idxs[parent][0],
                                   self.variables_idxs[parent][1]):
                        x_p = x_masked[:, p_idx].detach().numpy()
                        x_p = x_p[random_order]
                        x_masked[:, p_idx] = torch.from_numpy(x_p)

            # print('x_masked=', x_masked)

            if self.variables[i]['type'] == 'continuous':
                out_i = activation_layer(
                    self.fc_i[i](torch.cat([x_masked, z[:, i].unsqueeze(1)], axis=1))
                )
                out_i = nn.Tanh()(self.fc_f[i](out_i))
                out[:, self.variables_idxs[i][0]:self.variables_idxs[i][1]] = out_i

            else: # type = 'discrete'
                out_i = activation_layer(
                    self.fc_i[i](torch.cat([x_masked, z[:, i].unsqueeze(1)], axis=1))
                )
                out_i = nn.functional.gumbel_softmax(self.fc_f[i](out_i), hard=True)
                out[:, self.variables_idxs[i][0]:self.variables_idxs[i][1]] = out_i

        # print('out', out)
        return out


class Discriminator(nn.Module):
    def __init__(self, x_dim: int, h_dim: int) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            activation_layer,
            nn.Linear(h_dim, h_dim),
            activation_layer,
            nn.Linear(h_dim, 1),
        )

        for layer in self.model.parameters():
            if type(layer) == nn.Linear:
                torch.nn.init.xavier_normal_(layer)

    def forward(self, x_hat: torch.Tensor) -> torch.Tensor:
        return self.model(x_hat)


class GAN(pl.LightningModule):
    def __init__(
        self,
        variables: dict,
        gen_order: list,
        h_dim: int = 200,
        batch_size: int = 32,
        lr: float = 1e-3, # optimizer h.p.
        b1: float = 0.5, # optimizer h.p.
        b2: float = 0.999, # optimizer h.p.
        weight_decay: float = 1e-2, # optimizer h.p.
        lambda_gp: float = 10.0, # WGAN gradient penalty h.p.
        d_updates: int = 5, # discriminator updates per generator updates
        l1_g: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.iterations_d = 0
        self.iterations_g = 0

        self.x_dim = sum([v['size'] for v in variables.values()])
        self.z_dim = len(variables)

        log.info(
            f"Setting up networks with x_dim = {self.x_dim}, z_dim = {self.z_dim}, h_dim = {h_dim}"
        )
        # Create networks.
        self.generator = Generator_causal(
            variables=variables,
            z_dim=self.z_dim,
            x_dim=self.x_dim,
            h_dim=h_dim,
        )
        self.discriminator = Discriminator(x_dim=self.x_dim, h_dim=h_dim)

        # Internalise parameters.
        self.gen_order = gen_order
        self.variables = variables

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.generator(x, z)

    def compute_gradient_penalty(
        self, real_samples: torch.Tensor, fake_samples: torch.Tensor
    ) -> torch.Tensor:
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand(real_samples.size(0), 1)
        alpha = alpha.expand(real_samples.size())
        alpha = alpha.type_as(real_samples)
        # Get random interpolation between real and fake samples
        interpolates = (
            alpha * real_samples + ((1 - alpha) * fake_samples)
        ).requires_grad_(True)
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

    def sample_z(self, n: int) -> torch.Tensor:
        return torch.rand(n, self.z_dim) * 2 - 1

    @staticmethod
    def l1_reg(model: nn.Module) -> float:
        l1 = torch.tensor(0.0, requires_grad=True)
        for name, layer in model.named_parameters():
            if "weight" in name:
                l1 = l1 + layer.norm(p=1)
        return l1

    def gen_synthetic(
        self, x: torch.Tensor, gen_order: Optional[list] = None, biased_edges: dict = {}
    ) -> torch.Tensor:
        return self.generator.sequential(
            x,
            self.sample_z(x.shape[0]).type_as(x),
            gen_order=gen_order,
            biased_edges=biased_edges,
        )

    def get_gen_order(self) -> list:
        return self.gen_order

    def training_step(
        self, batch: torch.Tensor, batch_idx: int, optimizer_idx: int
    ) -> OrderedDict:
        # sample noise
        z = self.sample_z(batch.shape[0])
        z = z.type_as(batch)

        generated_batch = self.generator.sequential(batch, z, self.get_gen_order())

        if optimizer_idx == 0:
            # Train discriminator.
            self.iterations_d += 1
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            real_loss = torch.mean(self.discriminator(batch))
            fake_loss = torch.mean(self.discriminator(generated_batch.detach()))

            # discriminator loss
            d_loss = fake_loss - real_loss

            # add the gradient penalty
            d_loss += self.hparams.lambda_gp * self.compute_gradient_penalty(
                batch, generated_batch
            )

            tqdm_dict = {"d_loss": d_loss.detach()}
            self.log("d_loss", tqdm_dict)
            self.log("d_loss_fake", {'d_loss_fake': fake_loss.detach()})
            self.log("d_loss_real", {'d_loss_real': -real_loss.detach()})
            output = OrderedDict(
                {"loss": d_loss, "progress_bar": tqdm_dict}
            )
            return output

        elif optimizer_idx == 1:
            # Train generator.
            self.iterations_g += 1

            # adversarial loss (negative D fake loss)
            g_loss = -torch.mean(
                self.discriminator(generated_batch)
            )

            # add l1 regularization loss
            g_loss += self.hparams.l1_g * self.l1_reg(self.generator)

            tqdm_dict = {"g_loss": g_loss.detach()}
            self.log("g_loss", tqdm_dict)
            output = OrderedDict(
                {"loss": g_loss, "progress_bar": tqdm_dict}
            )
            return output
        else:
            raise ValueError("should not get here")

    def configure_optimizers(self) -> tuple:
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        weight_decay = self.hparams.weight_decay

        opt_g = torch.optim.AdamW(
            self.generator.parameters(),
            lr=lr,
            betas=(b1, b2),
            weight_decay=weight_decay,
        )
        opt_d = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=lr,
            betas=(b1, b2),
            weight_decay=weight_decay,
        )
        return (
            {"optimizer": opt_d, "frequency": self.hparams.d_updates},
            {"optimizer": opt_g, "frequency": 1},
        )
