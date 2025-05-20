import numpy as np
import torch
from synsplore.dataset.dataset import SynsploreDataset
from synsplore.dataset import _get_random_input
from synsplore.model.main import SynModule
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torchsummary import summary
from synsplore.dataset.transforms import *

@hydra.main(version_base=None, config_path="../../configs/model", config_name="default")
def main(cfg: DictConfig) -> None:
    # Print the configuration
    OmegaConf.resolve(cfg)

    print(cfg.r_enc)
    cfg_dict = dict(cfg)
    print(cfg_dict)

    # Load the dataset
    # dataset = SynsploreDataset()
    sample = _get_random_input(10)
    sample = SynRightAlign().transform(sample)
    sample = NumericsToDType(torch.float).transform(sample)

    print(sample)

    # # Load the model
    cfg_dict["d_model"] = cfg_dict["global"]["d_model"]
    cfg_dict.pop("global")
    model = SynModule(**{
        k: instantiate(v) if k != "d_model" else v for k, v in cfg_dict.items()
    })
    output = model(sample["syndata"])
    print(f"output: {len(output)}")
    print(output[0].shape)
    print(output[1])
    print(output[2])
    print(model.get_loss(sample["syndata"]))


if __name__ == "__main__":
    main()