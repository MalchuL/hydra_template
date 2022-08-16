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

from src.models.components.ada_layers.augments_factory import get_augments
from src.models.components.ada_layers.torch_utils import misc
from src.models.components.utils.init_net import init_net
from src.utils import get_pylogger
from src.utils.instantiate import instantiate
from src.utils.load_pl_dict import load_dict

logger = get_pylogger('StyleGANModule')

class StyleGANModule(LightningModule):
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

    def __init__(
        self,
        hparams,
        is_train=True
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(hparams)

        self.netG = self.create_generator()
        self.netG_ema = copy.deepcopy(self.netG).eval()

        logger.info(self.netG)

        self.register_buffer('mean', torch.tensor(self.hparams.norm.mean).view(1, -1, 1, 1))
        self.register_buffer('std', torch.tensor(self.hparams.norm.std).view(1, -1, 1, 1))

        if is_train:
            torch.backends.cudnn.benchmark = self.hparams.train.speed.cudnn_benchmark  # Improves training speed.
            torch.backends.cuda.matmul.allow_tf32 = self.hparams.train.speed.allow_tf32  # Allow PyTorch to internally use tf32 for matmul
            torch.backends.cudnn.allow_tf32 = self.hparams.train.speed.allow_tf32  # Allow PyTorch to internally use tf32 for convolutions

            self.augment_pipe = get_augments(self.hparams.train.ada_augs.name)

            self.netD = self.create_discriminator()
            logging.info(self.netD)


            self.G_reg_interval = self.hparams.train.params.G_reg_interval
            self.D_reg_interval = self.hparams.train.params.D_reg_interval

            # Weights block
            # TODO add mb_ratio = reg_interval / (reg_interval + 1)
            self.register_buffer('pl_mean', torch.zeros([]))

            # Additional params
            self.z_dim = self.hparams.z_dim
            self.style_mixing_prob = self.hparams.train.params.style_mixing_prob

            self.ada_target = self.hparams.train.params.ada_target
            self.register_buffer('ada_stats', torch.zeros([]))
            self.ada_gamma = self.hparams.train.params.ada_gamma
            self.ada_interval = self.hparams.train.params.ada_interval
            self.ada_kimg = self.hparams.train.params.ada_kimg

            self.r1_gamma = self.hparams.train.losses.r1_gamma

            self.pl_batch_shrink = self.hparams.train.params.pl_batch_shrink
            self.pl_weight = self.hparams.train.params.pl_weight
            self.pl_decay = self.hparams.train.params.pl_decay


    def create_generator(self):
        netG = instantiate(self.hparams.netG)
        if self.hparams.train.initialization.pretrain_checkpoint_G:
            load_dict(netG, 'netG', self.hparams.train.initialization.pretrain_checkpoint_G)
        else:
            init_net(netG, **self.hparams.train.initialization.init_G)
        return netG

    def create_discriminator(self):
        netD = instantiate(self.hparams.train.netD)
        init_net(netD, **self.hparams.train.initialization.init_D)
        return netD

    def backward_mapping(self, real_tensor):
        return (real_tensor * self.std + self.mean)

    def forward(self, z):
        return self.netG_ema(z, None, noise_mode='const')

    def run_G(self, z):
        ws = self.netG.mapping(z, None)
        if self.style_mixing_prob > 0:
            cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
            cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
            ws[:, cutoff:] = self.netG.mapping(torch.randn_like(z), None, skip_w_avg_update=True)[:, cutoff:]

        img = self.netG.synthesis(ws)
        return img, ws

    def run_D(self, img):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        logits = self.netD(img, None)
        return logits

    def generator_loss(self, gen_z):
        gen_img, _gen_ws = self.run_G(gen_z)  # May get synced by Gpl.
        gen_logits = self.run_D(gen_img)
        loss_Gmain = torch.nn.functional.softplus(-gen_logits)  # -log(sigmoid(gen_logits))
        loss_Gmain = loss_Gmain.mean()

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



        return loss_Gmain + loss_Gpl

    def update_aug_probs(self, D_logits):
        # Execute ADA heuristic.
        if self.global_step % self.ada_interval in (0,1):
            # print('update_aug_probs', self.global_step)
            signs = D_logits.sign().mean()
            batch_size = D_logits.shape[0]
            self.ada_stats = self.ada_stats * self.ada_gamma + (1 - self.ada_gamma) * signs
            adjust = torch.sign(self.ada_stats - self.ada_target) * (batch_size * self.ada_interval) / (
                    self.ada_kimg * 1000)
            self.augment_pipe.p.copy_((self.augment_pipe.p + adjust).max(misc.constant(0, device=self.device)))
        self.log('augment_pipe_p', self.augment_pipe.p)
        self.log('ada_stats', self.ada_stats)

    def discriminator_loss(self, real_img, gen_z):
        gen_img, _gen_ws = self.run_G(gen_z)
        gen_logits = self.run_D(gen_img) # Gets synced by loss_Dreal.

        loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
        loss_Dgen = loss_Dgen.mean()
        self.log('loss_Dgen', loss_Dgen)
        self.log('gen_logits', gen_logits.mean())

        real_img_tmp = real_img.detach().requires_grad_(True)
        real_logits = self.run_D(real_img_tmp)
        self.update_aug_probs(real_logits)

        loss_Dreal = torch.nn.functional.softplus(-real_logits)  # -log(sigmoid(real_logits))
        loss_Dreal = loss_Dreal.mean()
        self.log('loss_Dreal', loss_Dreal)
        self.log('real_logits', real_logits.mean())

        loss_Dr1 = 0
        if self.global_step % self.D_reg_interval in (0,1):
            # print('r1_grads', self.global_step)
            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True,
                                           only_inputs=True)[0]

            r1_penalty = r1_grads.square().sum([1, 2, 3])
            loss_Dr1 = r1_penalty * (self.r1_gamma / 2)

            loss_Dr1 = (real_logits * 0 + loss_Dr1).mean().mul(self.D_reg_interval)
            self.log('loss_Dr1', loss_Dr1)


        return loss_Dgen + loss_Dreal + loss_Dr1, gen_img


    def training_step(self, batch: Any, batch_idx: int, optimizer_idx):
        self.augment_pipe.train().requires_grad_(False)

        if optimizer_idx == 0:
            real = batch['real']
            bs = real.shape[0]
            gen_z = torch.randn([bs, self.z_dim]).type_as(real)

            # This correct else does not converge
            self.netG.train()
            self.netD.train()

            self.requires_grad(self.netG, True)
            self.requires_grad(self.netD, False)

            loss = self.generator_loss(gen_z)

            self.log('loss_G', loss, prog_bar=True)


        elif optimizer_idx == 1:
            real = batch['real']
            bs = real.shape[0]
            gen_z = torch.randn([bs, self.z_dim]).type_as(real)
            # This correct
            self.netD.train()
            self.netG.eval()

            self.requires_grad(self.netG, False)
            self.requires_grad(self.netD, True)

            loss, fake_img = self.discriminator_loss(real, gen_z)
            self.log_images(real, fake_img)
            self.log('loss_D', loss, prog_bar=True)

        else:
            raise ValueError(f"Optimizer with id {optimizer_idx} can't exists")

        return {'loss': loss}

    @rank_zero_only
    def log_images(self, real, fake):
        # tensors [self.real, self.fake, preds, self.cartoon, self.edge_fake]
        if self.global_step // 2 % self.hparams.train.logging.img_log_freq in (0,1):
            with torch.no_grad():
                bs = real.shape[0]
                gen_z = torch.randn([bs, self.z_dim]).type_as(real)
                fake_ema = self(gen_z)
            out_image = torch.cat([real, fake, fake_ema], dim=0)
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

        self.netG.eval()
        real = batch
        bs = real.shape[0]
        gen_z = torch.randn([bs, self.z_dim]).type_as(real)
        fake = self(gen_z)

        grid = torchvision.utils.make_grid(torch.cat([fake], dim=0))
        grid = grid * torch.tensor(self.hparams.norm.std, dtype=grid.dtype, device=grid.device).view(-1, 1, 1) + torch.tensor(self.hparams.norm.mean, dtype=grid.dtype, device=grid.device).view(-1, 1, 1)

        torchvision.utils.save_image(grid, str(self.val_folder / (str(batch_nb) + '.png')), nrow=1)
        print(self.val_folder.absolute())
        return {}

    def on_train_batch_end(self, outputs, batch, batch_idx: int, dataloader_idx: int):

        with torch.no_grad():
            ema_kimg = self.hparams.train.ema.ema_kimg
            ema_rampup = self.hparams.train.ema.ema_rampup
            ema_nimg = ema_kimg * 1000
            ema_nimg = min(ema_nimg, self.global_step * ema_rampup)
            ema_beta = 0.5 ** (1 / max(ema_nimg, 1e-8))
            for p_ema, p in zip(self.netG_ema.parameters(), self.netG.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(self.netG_ema.buffers(), self.netG.buffers()):
                b_ema.copy_(b)

    def on_validation_start(self) -> None:
        self.val_folder = Path('output')
        if not os.path.exists('output'):
            self.val_folder.mkdir(exist_ok=True, parents=True)



    def get_scheduler(self, optimizer, scheduler_params):
        if scheduler_params is not None:
            args = OmegaConf.to_container(scheduler_params, resolve=True)
            if 'SequentialLR' in args['_target_']:
                for inner_scheduler_args in args['schedulers']:
                    inner_scheduler_args['optimizer'] = optimizer
            args['optimizer'] = optimizer
            return instantiate(args)
        else:
            return None

    def requires_grad(self, model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        params_G = self.netG.parameters()
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
