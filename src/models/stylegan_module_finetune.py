import copy
import logging
import math
import os
from pathlib import Path
from typing import Any, List

import numpy as np
import torch
import torchvision
import wandb
from omegaconf import OmegaConf
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.utilities import rank_zero_only

from src.metrics.frechet_inception_distance import calculate_fretchet
from src.models.components.ada_layers.augments_factory import get_augments
from src.models.components.ada_layers.torch_utils import misc
from src.models.components.generators.stylegan import Generator
from src.models.components.losses.id_loss import IDLoss, IDMSELoss
from src.models.components.losses.loss_wrapper import LossWrapper
from src.models.components.utils.init_net import init_net
from src.models.stylegan_module import StyleGANModule
from src.utils import get_pylogger
from src.utils.instantiate import instantiate
from src.utils.load_pl_dict import load_dict

logger = get_pylogger('StyleGANFinetuneModule')

class StyleGANFinetuneModule(StyleGANModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    GENERATOR_LOAD_NAME = 'netG_ema'

    def __init__(
        self,
        hparams,
        k_blending_layers=0,
        is_train=True
    ):
        super().__init__(hparams, is_train)

        self.k_blending_layers = k_blending_layers
        self.netG_k_layered = copy.deepcopy(self.netG).eval()
        if is_train:
            # Copy of model for generating samples
            self.netG_samples_generator = copy.deepcopy(self.netG).eval()

            # Id loss
            self.id_loss = LossWrapper(IDMSELoss(ir_se50_weights=self.hparams.train.losses.facial_recognition.ir_se50_weights,
                                              empty_scale=self.hparams.train.losses.facial_recognition.empty_scale),
                                       self.hparams.train.losses.weights.id_weight)
            self.register_buffer('id_loss_mean',
                                 torch.tensor(self.hparams.train.losses.facial_recognition.norm.mean).view(1, -1, 1, 1))
            self.register_buffer('id_loss_std',
                                 torch.tensor(self.hparams.train.losses.facial_recognition.norm.std).view(1, -1, 1, 1))

            self.face_crop = list(self.hparams.train.losses.facial_recognition.face_crop)

    def id_loss_mapping(self, real_tensor):
        return (real_tensor - self.id_loss_mean) / self.id_loss_std

    def forward(self, z):
        return self.netG_k_layered(z, None, noise_mode='const')

    def forward_ema(self, z):
        return self.netG_ema(z, None, noise_mode='const')

    def forward_base(self, z):
        return self.netG_samples_generator(z, None, noise_mode='const')
    def run_G(self, z):
        ws = self.netG.mapping(z, None)
        img = self.netG.synthesis(ws, noise_mode=self.hparams.train.noise_mode)
        return img, ws

    def run_G_base(self, z):
        ws = self.netG_samples_generator.mapping(z, None)
        img = self.netG_samples_generator.synthesis(ws, noise_mode=self.hparams.train.noise_mode)
        return img, ws



    def update_k_layered_gen(self, k=None):
        if k is None:
            k = self.k_blending_layers
        # From https://arxiv.org/pdf/2010.05334.pdf
        partial_state_dict = self.get_k_layers_stylegan(k, self.netG_samples_generator)
        self.netG_k_layered.load_state_dict({**self.netG_ema.state_dict(), **partial_state_dict}, strict=True)


    def get_k_layers_stylegan(self, k, netG: Generator):
        new_state_dict = {**netG.mapping.state_dict(prefix='mapping.')}
        syntesis = netG.synthesis
        blocks = syntesis.block_resolutions[:k]
        # print('Blocks from  base', blocks)
        for block in blocks:
            new_state_dict.update(getattr(syntesis, f'b{block}').state_dict(prefix=f'synthesis.b{block}.'))
        return new_state_dict

    def generator_loss(self, gen_z):
        gen_img, _gen_ws = self.run_G(gen_z)  # May get synced by Gpl.
        gen_logits = self.run_D(gen_img)
        loss_Gmain = torch.nn.functional.softplus(-gen_logits)  # -log(sigmoid(gen_logits))
        loss_Gmain = loss_Gmain.mean()
        self.log('loss_Gmain', loss_Gmain)

        denormalized_fake = self.backward_mapping(gen_img)
        fake_normalized_to_id_loss = self.id_loss_mapping(denormalized_fake)

        with torch.no_grad():
            gen_base_img, _gen_base_ws = self.run_G_base(gen_z)
        denormalized_base = self.backward_mapping(gen_base_img)
        base_normalized_to_id_loss = self.id_loss_mapping(denormalized_base)

        crop = self.face_crop
        id_loss = self.id_loss(fake_normalized_to_id_loss, base_normalized_to_id_loss, crop)
        self.log('id_loss', id_loss / self.id_loss.weight)

        loss_Gpl = 0
        if self.global_step % self.G_reg_interval in (0,1):
            # print('pl_grads', self.global_step)
            batch_size = gen_z.shape[0] // self.pl_batch_shrink
            gen_img, gen_ws = self.run_G(gen_z[:batch_size])
            pl_noise = torch.randn_like(gen_img) / math.sqrt(gen_img.shape[2] * gen_img.shape[3])
            pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True,
                                               only_inputs=True)[0]
            pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
            pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
            self.pl_mean.copy_(pl_mean.detach())
            pl_penalty = (pl_lengths - pl_mean).square()
            loss_Gpl = pl_penalty * self.pl_weight

            loss_Gpl = (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(self.G_reg_interval)
            self.log('loss_Gpl', loss_Gpl)


        return loss_Gmain + loss_Gpl + id_loss

    def on_validation_start(self):
        super(StyleGANFinetuneModule, self).on_validation_start()

    @rank_zero_only
    def log_images(self, real, fake):
        self.update_k_layered_gen()
        # tensors [self.real, self.fake, preds, self.cartoon, self.edge_fake]
        if self.global_step // 2 % self.hparams.train.logging.img_log_freq in (0,1):
            if self.gen_z is None:
                bs = real.shape[0]
                gen_z = torch.randn([bs, self.z_dim]).type_as(real)
                self.gen_z = gen_z
            else:
                gen_z = self.gen_z
            with torch.no_grad():
                fake_ema_k_layered = self(gen_z.type_as(real))
                fake_ema = self.forward_ema(gen_z.type_as(real))
                fake_base = self.forward_base(gen_z.type_as(real))
            out_image = torch.cat([real, fake, fake_ema_k_layered, fake_base, fake_ema], dim=0)
            grid = torchvision.utils.make_grid(out_image, nrow=len(real))
            grid = self.backward_mapping(grid)[0]
            grid = torch.clamp(grid, 0.0, 1.0)

            for logger in self.loggers:
                print('Log image', self.global_step // 2)  # To avoid segmentation fault
                if isinstance(logger, TensorBoardLogger):
                    logger.experiment.add_image('train_image', grid, self.global_step // 2)
                elif isinstance(logger, WandbLogger):
                    logger.experiment.log({'train_image': [wandb.Image(grid)]})



    def validation_step(self, batch, batch_nb):

        real_idx = batch
        bs = real_idx.shape[0]
        gen_z = torch.randn([bs, self.z_dim]).type_as(real_idx)

        base, _ = self.run_G_base(gen_z)
        syntesis = self.netG_ema.synthesis
        count_blocks = len(syntesis.block_resolutions)
        fakes = []
        for k in range(count_blocks):
            self.update_k_layered_gen(k)
            fake = self(gen_z)
            fakes.append(fake)


        grid = torchvision.utils.make_grid(torch.cat([base, *fakes], dim=0))
        grid = grid * torch.tensor(self.hparams.norm.std, dtype=grid.dtype, device=grid.device).view(-1, 1, 1) + torch.tensor(self.hparams.norm.mean, dtype=grid.dtype, device=grid.device).view(-1, 1, 1)

        torchvision.utils.save_image(grid, str(self.val_folder / (str(round(real_idx[0].item())) + '.png')), nrow=1)
        return {}

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        if self.hparams.train.train_mapper:
            params_G = self.netG.parameters()
        else:
            params_G = self.netG.synthesis.parameters()
        params_D = self.netD.parameters()
        optimizer_G = instantiate({'params': params_G, **self.hparams.train.optimizing.optimizers.optimizer_G})
        optimizer_D = instantiate({'params': params_D, **self.hparams.train.optimizing.optimizers.optimizer_D})
        interval = self.hparams.train.optimizing.schedulers.interval
        return [optimizer_G, optimizer_D], [{'scheduler': self.get_scheduler(optimizer_G, self.hparams.train.optimizing.schedulers.scheduler_G), 'interval': interval},
                                            {'scheduler': self.get_scheduler(optimizer_D, self.hparams.train.optimizing.schedulers.scheduler_D), 'interval': interval}]
if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "mnist.yaml")
    _ = hydra.utils.instantiate(cfg)
