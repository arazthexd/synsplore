import numpy as np
import torch
from torch import nn
from synsplore.dataset.dataset import SynsploreDataset
from synsplore.dataset import _get_random_input
from synsplore.model.main import SynModule
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torchsummary import summary
from synsplore.train.training import Training
from synsplore.dataset.transforms import *
import logging
from hydra.core.hydra_config import HydraConfig
from mlflow.entities import Run
import hydraflow

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Print the configuration
    logger_dir = HydraConfig.get().runtime.output_dir
    log.info(f"Logging Directory: {logger_dir}")
    OmegaConf.resolve(cfg)
    cfg_model_dict = dict(cfg.model)
    cfg_model_dict["d_model"] = cfg_model_dict["global"]["d_model"]
    cfg_model_dict.pop("global")

    ## Load the dataset
    # dataset = SynsploreDataset(cfg.dataset))
    def clear_data(dataset: SynsploreDataset) -> SynsploreDataset:
        """Clear the dataset."""
        dataset = SynRightAlign().transform(dataset)
        dataset = NumericsToDType(torch.float).transform(dataset)
        return dataset
    
    num_batches = 2
    dataset = [clear_data(_get_random_input(10)) for _ in range(num_batches)]

    ## Load the model
    model = SynModule(**{
        k: instantiate(v) if k != "d_model" else v for k, v in cfg_model_dict.items()
    })

    trainer = Training(
        model=model,
        data=dataset,
        # optimizer=instantiate(cfg.optimizer),
        # scheduler=instantiate(cfg.scheduler) if "scheduler" in cfg else None
        optimizer=torch.optim.Adam(params=model.parameters(), lr=0.001),
        scheduler=None,  # instantiate(cfg.scheduler) if "scheduler" in cfg else None
    )
    trainer.train(epochs=cfg.train.training.epochs)
    


if __name__ == "__main__":
    main()