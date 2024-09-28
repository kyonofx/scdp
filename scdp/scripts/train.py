import logging
import json
from typing import List
from pathlib import Path

import hydra
import lightning.pytorch as pl
import omegaconf
import torch
from lightning.pytorch import Callback
from omegaconf import DictConfig, ListConfig

from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger

from scdp.common.system import log_hyperparameters, PROJECT_ROOT

# NOTE: disable slurm detection of lightning
from lightning.pytorch.plugins.environments import SLURMEnvironment
SLURMEnvironment.detect = lambda: False

pylogger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")

def build_callbacks(cfg: ListConfig, *args: Callback) -> List[Callback]:
    """Instantiate the callbacks given their configuration.

    Args:
        cfg: a list of callbacks instantiable configuration
        *args: a list of extra callbacks already instantiated

    Returns:
        the complete list of callbacks to use
    """
    callbacks: List[Callback] = list(args)

    for callback in cfg:
        pylogger.info(f"Adding callback <{callback['_target_'].split('.')[-1]}>")
        callbacks.append(hydra.utils.instantiate(callback, _recursive_=False))

    return callbacks


def run(cfg: DictConfig) -> str:
    """Generic train loop.

    Args:
        cfg: run configuration, defined by Hydra in /conf

    Returns:
        the run directory inside the storage_dir used by the current experiment
    """
    if cfg.train.deterministic:
        seed_everything(cfg.train.seed)

    # Instantiate datamodule
    pylogger.info(f"Instantiating <{cfg.data['_target_']}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data, _recursive_=False
    )
    datamodule.setup(stage="fit")
        
    metadata = getattr(datamodule, "metadata", None)
    if metadata is None:
        pylogger.warning(
            f"No 'metadata' attribute found in datamodule <{datamodule.__class__.__name__}>"
        )

    # Instantiate model
    pylogger.info(f"Instantiating <{cfg.model['_target_']}>")
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model, train=cfg.train, _recursive_=False, metadata=metadata,
    )

    callbacks: List[Callback] = build_callbacks(cfg.train.callbacks)

    storage_dir: str = cfg.core.storage_dir

    if "wandb" in cfg.train.logging:
        pylogger.info("Instantiating <WandbLogger>")
        wandb_config = cfg.train.logging.wandb
        logger = WandbLogger(**wandb_config)
        pylogger.info(f"W&B is now watching <{cfg.train.logging.wandb_watch.log}>!")
        logger.watch(
            model,
            log=cfg.train.logging.wandb_watch.log,
            log_freq=cfg.train.logging.wandb_watch.log_freq,
        )
    else:
        logger = TensorBoardLogger(**cfg.train.logging.tensorboard)
        pylogger.info(
            f"TensorBoard Logger logs into <{cfg.train.logging.tensorboard.save_dir}>."
        )

    ckpt = None    
    
    trainer = pl.Trainer(
        default_root_dir=storage_dir,
        logger=logger,
        callbacks=callbacks,
        **cfg.train.trainer,
    )

    # save the config yaml file.
    yaml_conf: str = omegaconf.OmegaConf.to_yaml(cfg)
    Path(storage_dir).mkdir(parents=True, exist_ok=True)
    (Path(storage_dir) / "config.yaml").write_text(yaml_conf)
    log_hyperparameters(cfg, model, trainer)
    with open(Path(storage_dir) / "metadata.json", "w") as f:
        json.dump(metadata, f)
    
    if (Path(storage_dir) / 'last.ckpt').exists():
        ckpt = Path(storage_dir) / 'last.ckpt'
        pylogger.info(f"found checkpoint: {ckpt}")
    else:
        import numpy as np
        ckpts = list(Path(storage_dir).glob("*epoch*.ckpt"))
        if len(ckpts) > 0:
            ckpt_epochs = np.array(
                [int(ckpt.parts[-1].split("-")[0].split("=")[1]) for ckpt in ckpts]
            )
            ckpt_ix = ckpt_epochs.argsort()[-1]
            ckpt = str(ckpts[ckpt_ix])
        else:
            ckpt = None
    
    if cfg.model.expo_trainable:
        model.load_state_dict(torch.load(ckpt)["state_dict"], strict=False)
        ckpt = None
            
    pylogger.info("starting training.")
    trainer.fit(
        model, datamodule.train_dataloader(), datamodule.val_dataloader(), ckpt_path=ckpt
    )

    if (
        datamodule.test_dataset is not None
        and trainer.checkpoint_callback.best_model_path is not None
    ):
        pylogger.info("starting testing.")
        trainer.test(dataloaders=[datamodule.test_dataloader()])

    # Logger closing to release resources/avoid multi-run conflicts
    if logger is not None:
        logger.experiment.finish()

@hydra.main(config_path=str(PROJECT_ROOT / "scdp" / "config"), config_name="default", version_base="1.1")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    # pylint: disable=E1120
    main()
