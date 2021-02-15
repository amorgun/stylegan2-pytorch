import argparse
import math
import random
import os
import pathlib
import shutil

import hydra
import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
import pytorch_lightning as pl
import torchvision
from torchvision import transforms, utils
from tqdm import tqdm

try:
    import wandb

except ImportError:
    wandb = None

from model import Generator, Discriminator
from dataset import MultiResolutionDataset, FolderDataset
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from non_leaking import augment, AdaptiveAugment


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device, rand=None):
    rand = rand if rand is not None else random.random()
    if prob > 0 and rand < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


class DataModule(pl.LightningDataModule):
    def __init__(self, hparams, transform):
        super().__init__()
        self.hparams = hparams
        self.transform = transform

    def prepare_data(self):
        # download
        hydra.utils.instantiate(self.hparams.dataset)

    def train_dataloader(self):
        return hydra.utils.instantiate(
            self.hparams.dataloader,
            dataset=hydra.utils.instantiate(
                self.hparams.dataset,
                transform=self.transform,
            )
        )                
         

class StyleGAN2(pl.LightningModule):
    def __init__(self, hparams) -> None:
        super().__init__()
        self.hparams = hparams
        
        self.generator = Generator(
            self.hparams.size_h, self.hparams.size_w, self.hparams.log_size,
            self.hparams.latent, self.hparams.n_mlp, channel_multiplier=self.hparams.channel_multiplier
        )
        self.discriminator = Discriminator(
            self.hparams.size_h, self.hparams.size_w, self.hparams.log_size,
            channel_multiplier=self.hparams.channel_multiplier
        )
        self.g_ema = Generator(
            self.hparams.size_h, self.hparams.size_w, self.hparams.log_size,
            self.hparams.latent, self.hparams.n_mlp, channel_multiplier=self.hparams.channel_multiplier
        )
        self.img_dim = (3, self.hparams.size_h, self.hparams.size_w) 
        self.accum = 0.5 ** (32 / (10 * 1000))
        self.mean_path_length = 0
        self.ada_aug_p = max(self.hparams.augment_p, 0)
        self.use_ada_augment = self.hparams.augment and self.hparams.augment_p == 0
        if self.use_ada_augment:
            self.ada_augment = AdaptiveAugment(self.hparams.ada_target, self.hparams.ada_length, self.hparams.ada_every)
        self.g_ema.eval()
        accumulate(self.g_ema, self.generator, 0)

    def configure_optimizers(self):
        cfg = self.hparams
        g_reg_ratio = cfg.g_reg_every / (cfg.g_reg_every + 1)
        d_reg_ratio = cfg.d_reg_every / (cfg.d_reg_every + 1)
        
        g_optim = optim.Adam(
            self.generator.parameters(),
            lr=cfg.lr * g_reg_ratio,
            betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
        )
        d_optim = optim.Adam(
            self.discriminator.parameters(),
            lr=cfg.lr * d_reg_ratio,
            betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
        )
        return (
            {'optimizer': d_optim},
            {'optimizer': d_optim},
            {'optimizer': g_optim},
            {'optimizer': g_optim},
        )
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        real_img = batch[0]
        batch_size = real_img.shape[0]
        if optimizer_idx == 0:
            requires_grad(self.generator, False)
            requires_grad(self.discriminator, True)
            
            
            noise = mixing_noise(batch_size, self.hparams.latent, self.hparams.mixing, self.device)
            fake_img, _ = self.generator(noise)

            if self.hparams.augment:
                real_img_aug, _ = augment(real_img, self.ada_aug_p)
                fake_img, _ = augment(fake_img, self.ada_aug_p)

            else:
                real_img_aug = real_img

            fake_pred = self.discriminator(fake_img)
            real_pred = self.discriminator(real_img_aug)
            d_loss = d_logistic_loss(real_pred, fake_pred)

            self.log('real_score', real_pred.mean())
            self.log('fake_score', fake_pred.mean())
            self.log('d', d_loss, prog_bar=True)

            if self.use_ada_augment:
                self.ada_aug_p = self.ada_augment.tune(real_pred)
            if self.hparams.augment:
                self.log('ada_aug_p', self.ada_aug_p, prog_bar=True)
            
            return {'loss': d_loss}
        if optimizer_idx == 1:
            if self.trainer.global_step % self.hparams.d_reg_every != 0:
                return
    
            requires_grad(self.generator, False)
            requires_grad(self.discriminator, True)
            real_img.requires_grad = True
            if args.augment:
                real_img_aug, _ = augment(real_img, self.ada_aug_p)
            else:
                real_img_aug = real_img
            real_pred = self.discriminator(real_img_aug)
            r1_loss = d_r1_loss(real_pred, real_img)
            
            self.log('r1', r1_loss, prog_bar=True)
            return {'loss': (self.hparams.r1 / 2 * r1_loss * self.hparams.d_reg_every + 0 * real_pred[0])}
        if optimizer_idx == 2:
            requires_grad(self.generator, True)
            requires_grad(self.discriminator, False)

            if self.hparams.top_k_batches > 0:
                with torch.no_grad():
                    noises, scores = [], []
                    rand = random.random()
                    for _ in range(self.hparams.top_k_batches):
                        noise = mixing_noise(self.hparams.top_k_batch_size, self.hparams.latent, self.hparams.mixing, self.device, rand=rand)
                        fake_img, _ = self.generator(noise)
                        if self.hparams.augment:
                            fake_img, _ = augment(fake_img, self.ada_aug_p)
                        score = self.discriminator(fake_img)
                        noises.append(noise)
                        scores.append(score)
                    scores = torch.cat(scores)
                    best_score_ids = torch.argsort(scores, descending=True)[:batch_size].squeeze(1)
                    noise = [torch.cat([n[idx] for n in noises], dim=0)[best_score_ids] for idx in range(len(noises[0]))]
            else:
                noise = mixing_noise(batch_size, self.hparams.latent, self.hparams.mixing, self.device)

            fake_img, _ = self.generator(noise)

            if self.hparams.augment:
                fake_img, _ = augment(fake_img, self.ada_aug_p)

            fake_pred = self.discriminator(fake_img)
            g_loss = g_nonsaturating_loss(fake_pred)
            
            self.log('g', g_loss, prog_bar=True)
            return {'loss': g_loss}
        if optimizer_idx == 3:
            if self.trainer.global_step % self.hparams.g_reg_every != 0:
                return
            
            requires_grad(self.generator, True)
            requires_grad(self.discriminator, False)
            path_batch_size = max(1, batch_size // self.hparams.path_batch_shrink)
            noise = mixing_noise(path_batch_size, self.hparams.latent, self.hparams.mixing, self.device)
            fake_img, latents = self.generator(noise, return_latents=True)

            path_loss, self.mean_path_length, path_lengths = g_path_regularize(
                fake_img, latents, self.mean_path_length
            )

            weighted_path_loss = self.hparams.path_regularize * self.hparams.g_reg_every * path_loss

            if self.hparams.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            self.log('path', path_loss, prog_bar=True)
            self.log('mean_path', self.mean_path_length, prog_bar=True)
            self.log('path_length', path_lengths.mean())
            return {'loss': weighted_path_loss}
            
    
    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        accumulate(self.g_ema, self.generator, self.accum)
        
    def forward(self, batch):
        return self.g_ema([batch])
        
        
class GanImageSampler(pl.Callback):
    def __init__(
        self,
        frequency=1,
        num_samples=3,
        nrow=8,
        padding=2,
        normalize=False,
        norm_range=None,
        scale_each=False,
        pad_value=0,
        seed=42,
    ) -> None:
        super().__init__()
        self.frequency = frequency
        self.num_samples = num_samples
        self.nrow = nrow
        self.padding = padding
        self.normalize = normalize
        self.norm_range = norm_range
        self.scale_each = scale_each
        self.pad_value = pad_value
        self.seed = seed

    def on_batch_end(self, trainer, pl_module):
        if trainer.global_step % self.frequency != 0:
            return
        dim = (self.num_samples, pl_module.hparams.latent)  # type: ignore[union-attr]
        z = torch.normal(mean=0.0, std=1.0, size=dim, generator=torch.manual_seed(self.seed)).to(pl_module.device)

        # generate images
        with torch.no_grad():
            pl_module.eval()
            images, _ = pl_module(z)
            pl_module.train()

        if len(images.size()) == 2:
            img_dim = pl_module.img_dim
            images = images.view(self.num_samples, *img_dim)

        grid = torchvision.utils.make_grid(
            tensor=images,
            nrow=self.nrow,
            padding=self.padding,
            normalize=self.normalize,
            range=tuple(self.norm_range),
            scale_each=self.scale_each,
            pad_value=self.pad_value,
        )
        str_title = f"{pl_module.__class__.__name__}_images"
        trainer.logger.experiment.add_image(str_title, grid, global_step=trainer.global_step)

        
class ModelCheckpoint(pl.Callback):
    FILE_EXTENSION = ".ckpt"
    LATEST_FILENAME = "latest"
    
    def __init__(
        self,
        frequency=1,
        folder='checkpoint',
        save_weights_only: bool = False
    ) -> None:
        super().__init__()
        self.frequency = frequency
        self.folder = pathlib.Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.save_weights_only = save_weights_only
        
    @property
    def latest_path(self):
        return self.folder / f'{self.LATEST_FILENAME}{self.FILE_EXTENSION}'

    def on_batch_end(self, trainer, pl_module):
        if trainer.global_step % self.frequency != 0:
            return
        if (
            trainer.fast_dev_run
            or self.frequency < 1
            or trainer.running_sanity_check
        ):
            return
        
        filepath = self.folder / f'{pl_module.__class__.__name__}_{trainer.global_step}{self.FILE_EXTENSION}'

        accelerator_backend = trainer.accelerator_backend
        if accelerator_backend is not None and accelerator_backend.rpc_enabled:
            # RPCPlugin manages saving all model states
            accelerator_backend.ddp_plugin.rpc_save_model(self._save_model, filepath, trainer, pl_module)
        else:
            self._save_model(filepath, trainer, pl_module)
        shutil.copy2(filepath, self.latest_path)
        
    def on_pretrain_routine_start(self, trainer, pl_module):
        self.save_function = trainer.save_checkpoint
        
    def _save_model(self, filepath: str, trainer, pl_module):
        # in debugging, track when we save checkpoints
        trainer.dev_debugger.track_checkpointing_history(filepath)

        # delegate the saving to the trainer
        if self.save_function is not None:
            self.save_function(filepath, self.save_weights_only)
        else:
            raise ValueError(".save_function() not set")


@hydra.main(config_name="config/config")
def main(cfg):
    model = StyleGAN2(cfg.model)
    trainer = pl.Trainer(**cfg.trainer, callbacks=[
        GanImageSampler(**cfg.image_sample),
        ModelCheckpoint(**cfg.checkpoint),
    ])
    transform = transforms.Compose([
        hydra.utils.instantiate(t)
        for t in cfg.transforms
    ])
    if 'init_ckpt' in cfg:
        print(f'Load model from {cfg["init_ckpt"]}')
        model = StyleGAN2.load_from_checkpoint(checkpoint_path=cfg["init_ckpt"], **cfg.model)
#         ckpt = torch.load(cfg['init_ckpt'])
#         if 'g' in ckpt:
#             model.generator.load_state_dict(ckpt["g"])
#         if 'd' in ckpt:
#             model.discriminator.load_state_dict(ckpt["d"])
#         if 'g_ema' in ckpt:
#             model.g_ema.load_state_dict(ckpt["g_ema"])

    data = DataModule(cfg, transform)
    trainer.fit(model, data)
    return


if __name__ == "__main__":
    main()
