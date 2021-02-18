import pytorch_lightning as pl
from torchvision import transforms
import hydra

from lightning import StyleGAN2, GanImageSampler, ModelCheckpoint, DataModule


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
